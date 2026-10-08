"""Validated hourly input data and deterministic illustrative load profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "load_kw")


def demo_data(days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate *synthetic* site data, not measured utility or asset observations.

    Temperature, load, and demand spikes are stylized and deliberately do not
    model a specific location, utility tariff, or physical asset.
    """
    if days < 65:
        raise ValueError("days must be at least 65 for forecasting and 30-day holdout")
    rng = np.random.default_rng(seed)
    n = days * 24
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    day_index = np.arange(n) / 24

    # Seasonal temperature with a predictable diurnal cycle and random deviations.
    temp_c = (
        11 + 9 * np.sin(2 * np.pi * day_index / 365 - 0.9)
        + 5.5 * np.sin(2 * np.pi * (hour - 8) / 24)
        + rng.normal(0, 1.8, n)
    )
    office_hours = ((hour >= 7) & (hour < 20)).astype(float)
    weekday_mask = (weekday < 5).astype(float)
    profile = 365 + 135 * office_hours * weekday_mask + 42 * office_hours * (1 - weekday_mask)
    daily_wave = 35 * np.sin(2 * np.pi * (hour - 10) / 24)
    cooling = 4.6 * np.maximum(temp_c - 16, 0)
    heating = 2.8 * np.maximum(9 - temp_c, 0)
    noise = rng.normal(0, 13, n)
    demand_spikes = (rng.random(n) < 0.014) * rng.uniform(45, 115, n)
    load_kw = np.maximum(50, profile + daily_wave + cooling + heating + noise + demand_spikes)

    return pd.DataFrame({
        "timestamp": timestamps,
        "load_kw": np.round(load_kw, 3),
        "temperature_c": np.round(temp_c, 3),
    })


def validate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Require continuous hourly timestamps, numeric nonnegative load, and no missing values."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV column(s): {', '.join(sorted(missing))}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    if out["timestamp"].dt.tz is not None:
        raise ValueError("Use local timestamps without timezone offsets; handle DST externally")
    out = out.sort_values("timestamp").reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps are not supported")
    if not out["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Input must be continuous at exactly one-hour intervals; check DST/gaps")
    out["load_kw"] = pd.to_numeric(out["load_kw"], errors="raise")
    if not np.isfinite(out["load_kw"]).all() or (out["load_kw"] < 0).any():
        raise ValueError("load_kw must contain finite, nonnegative numeric values")
    if "temperature_c" not in out:
        # Support a useful load-only baseline without inventing weather observations.
        out["temperature_c"] = 0.0
    out["temperature_c"] = pd.to_numeric(out["temperature_c"], errors="raise")
    if not np.isfinite(out["temperature_c"]).all():
        raise ValueError("temperature_c must be finite if included")
    if len(out) < 65 * 24:
        raise ValueError("At least 65 continuous days are required (35 days train + 30 test)")
    if len(out) % 24 != 0:
        raise ValueError("Input must contain a whole number of 24-hour days")
    return out


def read_csv(path: str | Path) -> pd.DataFrame:
    return validate_hourly(pd.read_csv(path))
