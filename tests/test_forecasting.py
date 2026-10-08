import numpy as np

from gridopt.data import demo_data
from gridopt.forecasting import features_for_hour, forecast_day_ahead


def test_lags_use_only_prior_history():
    data = demo_data()
    h = list(np.arange(168, dtype=float))
    f = features_for_hour(data.timestamp.iloc[168], 8.0, h)
    assert f[0] == 167
    assert f[1] == 144
    assert f[2] == 0
    assert f[3] == np.mean(np.arange(144, 168))


def test_forecast_returns_30_day_holdout():
    result = forecast_day_ahead(demo_data(days=67))
    assert len(result.actual_kw) == len(result.forecast_kw) == 720
    assert np.isfinite(result.forecast_kw).all()
    assert result.rmse_kw >= 0
    assert result.naive_mae_kw >= 0
