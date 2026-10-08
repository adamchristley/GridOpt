import numpy as np
import pytest

from gridopt.optimization import Battery, optimize_day, simulate_month


def test_dispatch_meets_physics_and_terminal_soc():
    battery = Battery(capacity_kwh=100, max_power_kw=40)
    forecast = np.full(24, 80.0)
    forecast[15:20] = 170
    rates = np.full(24, 0.09)
    rates[15:20] = 0.32
    charge, discharge = optimize_day(forecast, rates, battery, 0.0, 30)
    soc = 50 + np.cumsum(battery.eta * charge - discharge / battery.eta)
    assert np.max(charge) <= 40.0001
    assert np.max(discharge) <= 40.0001
    assert np.min(soc) >= 10 - 1e-5
    assert np.max(soc) <= 90 + 1e-5
    assert soc[-1] == pytest.approx(50, abs=1e-5)
    assert np.min(forecast + charge - discharge) >= -1e-5
    assert np.max(forecast + charge - discharge) < 170


def test_zero_power_matches_no_battery():
    load = np.tile(np.array([100.0] * 16 + [160.0] * 8), 2)
    rates = np.tile(np.array([0.1] * 16 + [0.3] * 8), 2)
    result = simulate_month(load, load, rates, Battery(max_power_kw=0))
    assert result.savings_usd == pytest.approx(0, abs=1e-5)
    assert np.allclose(result.grid_kw, load)


def test_realized_grid_never_exports_under_forecast_error():
    load = np.full(24, 0.0)
    pred = np.full(24, 200.0)
    pred[17:20] = 500
    rates = np.full(24, 0.2)
    result = simulate_month(load, pred, rates, Battery())
    assert np.min(result.grid_kw) >= -1e-5
    assert np.min(result.battery_soc_kwh) >= 60 - 1e-5


def test_invalid_battery_is_rejected():
    with pytest.raises(ValueError):
        Battery(round_trip_efficiency=1.2)
