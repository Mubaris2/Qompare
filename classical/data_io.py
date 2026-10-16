import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def load_task(task, data_dir=PROC):
    """Return X, y, meta for T1/T2/T3 from data/processed (real or mock)."""
    data_dir = Path(data_dir)
    df = pd.read_parquet(data_dir / f"{task}_stream.parquet")
    X = df.filter(regex=r"^f\d+$").to_numpy(float); y = df["y"].to_numpy()
    mp = data_dir / f"{task}_stream.meta.json"
    return X, y, (json.loads(mp.read_text()) if mp.exists() else {})
