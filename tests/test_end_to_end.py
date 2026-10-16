import pandas as pd, pytest
from pathlib import Path
from common.harness import run_stream
from common.validate_results import validate
from classical.models import SVRRegressor, OCSVMAnomaly, RiverHT, BatchTreeClassifier

P = Path("data/processed")
pytestmark = pytest.mark.skipif(not (P / "T1_stream.parquet").exists(), reason="run data/make_mock.py")


def load(t):
    d = pd.read_parquet(P / f"{t}_stream.parquet"); return d.filter(like="f").to_numpy(), d["y"].to_numpy()


@pytest.mark.parametrize("task,model,kw", [
    ("T1", SVRRegressor(), dict(W=100, H=50)),
    ("T2", OCSVMAnomaly(), dict(W=200, H=100)),
    ("T3", RiverHT(), dict(W=100, H=100)),
    ("T3", BatchTreeClassifier(), dict(W=100, H=100)),
])
def test_runs_and_validates(task, model, kw, tmp_path):
    X, y = load(task); df = run_stream(model, X, y, task, max_steps=6, **kw)
    assert len(df) > 0 and df["metric_value"].notna().any()
    p = tmp_path / "r.csv"; df.to_csv(p, index=False); assert validate(p) == []
