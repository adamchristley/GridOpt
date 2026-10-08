"""Linear-program battery dispatch and real-world replay with no grid export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float = 600.0
    max_power_kw: float = 250.0
    min_soc_fraction: float = 0.10
    max_soc_fraction: float = 0.90
    initial_soc_fraction: float = 0.50
    round_trip_efficiency: float = 0.90
    degradation_cost_per_kwh: float = 0.01

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0 or self.max_power_kw < 0:
            raise ValueError("Battery capacity must be positive and power nonnegative")
        if not (0 <= self.min_soc_fraction <= self.initial_soc_fraction
                <= self.max_soc_fraction <= 1):
            raise ValueError("SOC bounds must enclose initial state and lie in [0, 1]")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("Round-trip efficiency must be in (0, 1]")
        if self.degradation_cost_per_kwh < 0:
            raise ValueError("Degradation cost cannot be negative")

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))


def synthetic_tariff(timestamps: pd.Series) -> np.ndarray:
    """ILLUSTRATIVE TOU tariff, not a real utility rate schedule.

    16:00-20:59 weekdays = $0.28/kWh; midnight-05:59 = $0.08;
    other hours = $0.14. Demand charge defaults to $24/kW per 30-day cycle.
    """
    hour = timestamps.dt.hour.to_numpy()
    weekday = timestamps.dt.dayofweek.to_numpy()
    rate = np.full(len(timestamps), 0.14)
    rate[hour < 6] = 0.08
    rate[(weekday < 5) & (hour >= 16) & (hour < 21)] = 0.28
    return rate


def optimize_day(
    forecast_kw: np.ndarray, price_per_kwh: np.ndarray, battery: Battery,
    previous_billing_peak_kw: float, demand_charge_per_kw: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one 24-hour LP with charge/discharge, SOC, import, and demand peak constraints.

    All quantities are per 1-hour interval. Final SOC must equal initial SOC.
    Billing peak includes previous realized grid peaks, not predictions.
    """
    load = np.asarray(forecast_kw, dtype=float)
    rates = np.asarray(price_per_kwh, dtype=float)
    if load.shape != (24,) or rates.shape != (24,):
        raise ValueError("Expected 24 hourly load forecasts and 24 prices")
    if (load < 0).any() or (rates < 0).any() or not np.isfinite(load).all() or not np.isfinite(rates).all():
        raise ValueError("Demand and prices must be nonnegative and finite")
    if previous_billing_peak_kw < 0 or demand_charge_per_kw < 0:
        raise ValueError("Demand peak and charge must be nonnegative")

    # Decision vector: charge[t] (24), discharge[t] (24), billing_peak (1).
    n = 24
    c = np.concatenate([
        rates + battery.degradation_cost_per_kwh / 2,
        -rates + battery.degradation_cost_per_kwh / 2,
        [demand_charge_per_kw],
    ])
    A, b = [], []
    for t in range(n):
        upper = np.zeros(2 * n + 1)
        upper[t], upper[n + t], upper[-1] = 1, -1, -1
        A.append(upper)
        b.append(-load[t])
        no_export = np.zeros(2 * n + 1)
        no_export[t], no_export[n + t] = -1, 1
        A.append(no_export)
        b.append(load[t])

    e0 = battery.capacity_kwh * battery.initial_soc_fraction
    low = battery.capacity_kwh * battery.min_soc_fraction
    high = battery.capacity_kwh * battery.max_soc_fraction
    for k in range(1, n + 1):
        soc_delta = np.zeros(2 * n + 1)
        soc_delta[:k] = battery.eta
        soc_delta[n:n + k] = -1 / battery.eta
        A.extend([soc_delta, -soc_delta])
        b.extend([high - e0, e0 - low])
    terminal = np.zeros(2 * n + 1)
    terminal[:n] = battery.eta
    terminal[n:2 * n] = -1 / battery.eta
    bounds = [(0, battery.max_power_kw)] * (2 * n) + [(previous_billing_peak_kw, None)]
    result = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=terminal[None, :],
                     b_eq=np.array([0.0]), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Dispatch LP failed: {result.message}")
    return result.x[:n], result.x[n:2 * n]


@dataclass
class DispatchResult:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    grid_kw: np.ndarray
    battery_soc_kwh: np.ndarray
    baseline_bill_usd: float
    optimized_bill_usd: float
    degradation_usd: float
    savings_usd: float
    savings_pct: float
    baseline_peak_kw: float
    optimized_peak_kw: float
    cycles: float


def simulate_month(
    actual_kw: np.ndarray, forecast_kw: np.ndarray, rates: np.ndarray,
    battery: Battery, demand_charge_per_kw: float = 24.0,
) -> DispatchResult:
    """Re-optimize each morning, execute on actual demand and track the *realized* peak.

    Discharge is clipped to avoid exporting if actual demand is lower than the
    prediction. Degradation is an approximate throughput charge, not a warranty
    or lifetime economics calculation. This is an open-loop, hourly simulation.
    """
    actual = np.asarray(actual_kw, dtype=float)
    predicted = np.asarray(forecast_kw, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if len(actual) == 0 or len(actual) % 24 or actual.shape != predicted.shape or actual.shape != rates.shape:
        raise ValueError("Load, forecast and tariff must share a nonempty whole-day horizon")
    if (actual < 0).any() or (rates < 0).any():
        raise ValueError("Load and tariffs must be nonnegative")

    charge = np.zeros_like(actual)
    discharge = np.zeros_like(actual)
    soc = np.zeros_like(actual)
    grid = np.zeros_like(actual)
    billed_peak = 0.0
    energy = battery.capacity_kwh * battery.initial_soc_fraction
    lower = battery.capacity_kwh * battery.min_soc_fraction
    upper = battery.capacity_kwh * battery.max_soc_fraction
    for start in range(0, len(actual), 24):
        c, d = optimize_day(predicted[start:start + 24], rates[start:start + 24],
                            battery, billed_peak, demand_charge_per_kw)
        # Fast feedback: clip surprise realized demand to the day-ahead target
        # when physically possible. Replanning still occurs only daily.
        target_grid_kw = float(np.max(predicted[start:start + 24] + c - d))
        for t in range(24):
            i = start + t
            # Stay within physical SOC bounds under numerical solver tolerance.
            c_actual = min(max(c[t], 0.0), max(0.0, (upper - energy) / battery.eta))
            needed_for_peak = max(0.0, actual[i] + c_actual - target_grid_kw)
            d_request = max(float(d[t]), needed_for_peak)
            d_actual = min(max(d_request, 0.0), max(0.0, (energy + battery.eta * c_actual - lower) * battery.eta),
                           actual[i] + c_actual, battery.max_power_kw)
            energy += battery.eta * c_actual - d_actual / battery.eta
            energy = float(np.clip(energy, lower, upper))
            charge[i], discharge[i], soc[i] = c_actual, d_actual, energy
            grid[i] = actual[i] + c_actual - d_actual
            billed_peak = max(billed_peak, grid[i])

    baseline_peak = float(np.max(actual))
    baseline_bill = float(np.dot(actual, rates) + demand_charge_per_kw * baseline_peak)
    optimized_bill = float(np.dot(grid, rates) + demand_charge_per_kw * billed_peak)
    degradation = float(battery.degradation_cost_per_kwh * (charge + discharge).sum() / 2)
    net_savings = baseline_bill - optimized_bill - degradation
    return DispatchResult(
        charge_kw=charge, discharge_kw=discharge, grid_kw=grid, battery_soc_kwh=soc,
        baseline_bill_usd=baseline_bill, optimized_bill_usd=optimized_bill,
        degradation_usd=degradation, savings_usd=net_savings,
        savings_pct=100.0 * net_savings / baseline_bill if baseline_bill else 0.0,
        baseline_peak_kw=baseline_peak, optimized_peak_kw=billed_peak,
        cycles=float(discharge.sum() / (battery.capacity_kwh * (upper / battery.capacity_kwh - lower / battery.capacity_kwh)))
        if upper > lower else 0.0,
    )
