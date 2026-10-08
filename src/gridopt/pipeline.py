"""Command-line experiment: honest day-ahead forecast and operational dispatch replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridopt.data import demo_data, read_csv, validate_hourly
from gridopt.forecasting import forecast_day_ahead
from gridopt.optimization import Battery, simulate_month, synthetic_tariff


def run(
    output_dir: Path, input_csv: Path | None = None, seed: int = 42,
    capacity_kwh: float = 600, power_kw: float = 250,
    demand_charge_per_kw: float = 24,
) -> dict:
    if demand_charge_per_kw < 0:
        raise ValueError("demand charge must be nonnegative")
    data = read_csv(input_csv) if input_csv is not None else validate_hourly(demo_data(seed=seed))
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast = forecast_day_ahead(data, seed=seed)
    heldout = data.iloc[-30 * 24:].copy().reset_index(drop=True)
    rates = synthetic_tariff(heldout["timestamp"])
    battery = Battery(capacity_kwh=capacity_kwh, max_power_kw=power_kw)
    result = simulate_month(forecast.actual_kw, forecast.forecast_kw, rates, battery,
                            demand_charge_per_kw=demand_charge_per_kw)

    timeline = heldout.assign(
        forecast_kw=np.round(forecast.forecast_kw, 3),
        naive_forecast_kw=np.round(forecast.seasonal_naive_kw, 3),
        rate_usd_per_kwh=rates,
        battery_charge_kw=np.round(result.charge_kw, 3),
        battery_discharge_kw=np.round(result.discharge_kw, 3),
        grid_import_kw=np.round(result.grid_kw, 3),
        state_of_charge_kwh=np.round(result.battery_soc_kwh, 3),
    )
    timeline.to_csv(output_dir / "dispatch_timeseries.csv", index=False)

    metrics = {
        "dataset": "user_supplied_hourly_csv" if input_csv else "synthetic_120_day_example",
        "method": "rolling 24-hour day-ahead recursive ML forecast; daily LP dispatch + peak-clipping feedback, realized-load replay",
        "simulation_notice": "ILLUSTRATIVE; 30-day billing cycle; synthetic TOU and demand tariff; not utility-validated",
        "training_days": int(len(data) / 24 - 30),
        "test_days": 30,
        "forecast_mae_kw": round(forecast.mae_kw, 3),
        "forecast_rmse_kw": round(forecast.rmse_kw, 3),
        "seasonal_naive_mae_kw": round(forecast.naive_mae_kw, 3),
        "baseline_bill_usd": round(result.baseline_bill_usd, 2),
        "optimized_bill_usd": round(result.optimized_bill_usd, 2),
        "degradation_usd": round(result.degradation_usd, 2),
        "net_savings_usd": round(result.savings_usd, 2),
        "net_savings_pct": round(result.savings_pct, 2),
        "baseline_peak_kw": round(result.baseline_peak_kw, 3),
        "optimized_peak_kw": round(result.optimized_peak_kw, 3),
        "peak_reduction_pct": round(100 * (1 - result.optimized_peak_kw / result.baseline_peak_kw), 2)
        if result.baseline_peak_kw else 0.0,
        "discharged_usable_capacity_cycles": round(result.cycles, 3),
        "battery": asdict(battery),
        "tariff": {
            "weekday_peak_16_to_21_usd_per_kwh": 0.28,
            "overnight_00_to_06_usd_per_kwh": 0.08,
            "other_usd_per_kwh": 0.14,
            "demand_charge_usd_per_kw_per_30_days": demand_charge_per_kw,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    preview = timeline.iloc[:7 * 24]
    t = preview["timestamp"]
    axes[0].plot(t, preview["load_kw"], label="Actual load", linewidth=1.5)
    axes[0].plot(t, preview["forecast_kw"], label="Day-ahead forecast", linewidth=1.0)
    axes[0].set_ylabel("kW")
    axes[0].set_title("7-day sample: load and day-ahead forecast")
    axes[0].legend()
    axes[1].plot(t, preview["load_kw"], label="No storage", linewidth=1.3)
    axes[1].plot(t, preview["grid_import_kw"], label="With optimized battery", linewidth=1.3)
    axes[1].set_ylabel("Grid import (kW)")
    axes[1].set_title("Battery peak shaving and energy shifting")
    axes[1].legend()
    axes[2].plot(t, preview["state_of_charge_kwh"])
    axes[2].set_ylabel("Stored energy (kWh)")
    axes[2].set_title("Battery state of charge")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.savefig(output_dir / "overview.png", dpi=150)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="GridOpt forecasting + dispatch experiment")
    parser.add_argument("--input-csv", type=Path, help="Hourly timestamp,load_kw[,temperature_c] CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity-kwh", type=float, default=600)
    parser.add_argument("--power-kw", type=float, default=250)
    parser.add_argument("--demand-charge", type=float, default=24)
    args = parser.parse_args()
    metrics = run(args.output_dir, args.input_csv, seed=args.seed,
                  capacity_kwh=args.capacity_kwh, power_kw=args.power_kw,
                  demand_charge_per_kw=args.demand_charge)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
