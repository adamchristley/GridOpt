"""Chronological, day-ahead load forecasting with a fixed trained model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = (
    "lag_1", "lag_24", "lag_168", "mean_24", "temperature_c",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend",
)
WARMUP_HOURS = 168


def features_for_hour(timestamp: pd.Timestamp, temperature: float, history: list[float]) -> list[float]:
    """Every load lag uses values strictly prior to the prediction timestamp."""
    if len(history) < WARMUP_HOURS:
        raise ValueError("Need at least 168 prior hours")
    hour, weekday = timestamp.hour, timestamp.dayofweek
    return [
        history[-1], history[-24], history[-168], float(np.mean(history[-24:])),
        temperature,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
        float(weekday >= 5),
    ]


@dataclass
class ForecastResult:
    actual_kw: np.ndarray
    forecast_kw: np.ndarray
    seasonal_naive_kw: np.ndarray
    mae_kw: float
    rmse_kw: float
    naive_mae_kw: float


def forecast_day_ahead(frame: pd.DataFrame, test_days: int = 30, seed: int = 42) -> ForecastResult:
    """Forecast 24 hours at each day boundary; future load is never read.

    The synthetic weather series is treated as a known 24-hour temperature
    forecast. If supplied from a CSV, its values likewise represent exogenous
    future inputs; real deployments must substitute a *weather forecast*.
    """
    n = len(frame)
    test_hours = test_days * 24
    start = n - test_hours
    if start < 35 * 24:
        raise ValueError("At least 35 days must remain for training")
    ts = pd.DatetimeIndex(frame["timestamp"])
    load = frame["load_kw"].to_numpy(dtype=float)
    temperature = frame["temperature_c"].to_numpy(dtype=float)

    train_features = [features_for_hour(ts[i], temperature[i], list(load[:i]))
                      for i in range(WARMUP_HOURS, start)]
    train_targets = load[WARMUP_HOURS:start]
    model = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.07, max_leaf_nodes=24,
        l2_regularization=2.0, random_state=seed,
    )
    model.fit(np.asarray(train_features), train_targets)

    predictions = np.empty(test_hours)
    naive = np.empty(test_hours)
    for day_start in range(start, n, 24):
        # At 00:00 this morning, observed load is known only through yesterday.
        history = load[:day_start].tolist()
        for i in range(day_start, day_start + 24):
            x = features_for_hour(ts[i], temperature[i], history)
            prediction = max(0.0, float(model.predict(np.array([x]))[0]))
            predictions[i - start] = prediction
            naive[i - start] = load[i - 24]  # yesterday's same hour is already known
            history.append(prediction)  # recursively forecast the rest of the day

    actual = load[start:]
    return ForecastResult(
        actual_kw=actual,
        forecast_kw=predictions,
        seasonal_naive_kw=naive,
        mae_kw=float(mean_absolute_error(actual, predictions)),
        rmse_kw=float(np.sqrt(mean_squared_error(actual, predictions))),
        naive_mae_kw=float(mean_absolute_error(actual, naive)),
    )
