import json

from gridopt.pipeline import run


def test_end_to_end_synthetic(tmp_path):
    metrics = run(tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "dispatch_timeseries.csv").exists()
    assert (tmp_path / "overview.png").stat().st_size > 1000
    assert metrics["test_days"] == 30
    assert metrics["dataset"].startswith("synthetic")
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
