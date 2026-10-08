import pandas as pd
import pytest

from gridopt.data import demo_data, validate_hourly


def test_demo_is_reproducible_and_complete():
    a, b = demo_data(), demo_data()
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 120 * 24
    assert (a.load_kw > 0).all()
    assert validate_hourly(a).equals(a)


def test_gaps_are_rejected():
    broken = demo_data().drop(index=15)
    with pytest.raises(ValueError, match="continuous"):
        validate_hourly(broken)


def test_duplicate_hours_are_rejected():
    broken = demo_data()
    broken.loc[15, "timestamp"] = broken.loc[14, "timestamp"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_hourly(broken)


def test_missing_load_and_negative_values():
    data = demo_data()
    with pytest.raises(ValueError, match="required"):
        validate_hourly(data.drop(columns="load_kw"))
    data.loc[2, "load_kw"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_hourly(data)
