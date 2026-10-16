"""InfluxDB 2.x sink matching Section 3 schema.
One point per step per metric.
Tags: task, model, variant, W, H, drift, seed, family, metric_name
Fields: metric_value, step, n_seen, t_encode, t_fit, t_predict,
        n_params, n_qubits, n_circuit_evals, peak_mem_mb, drift_flag, latency
Timestamp: BASE_EPOCH_S + n_seen (seconds)
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Fixed base timestamp (2026-01-01T00:00:00Z) so offline backfill and live runs align on x-axis
BASE_EPOCH_S = int(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
QUANTUM_MODELS = {"vqr", "qsvr", "qsvc", "qnn"}


def infer_family(model_name: str, n_qubits: int = 0) -> str:
    if n_qubits and int(n_qubits) > 0:
        return "quantum"
    return "quantum" if model_name.lower() in QUANTUM_MODELS else "classical"


class MetricsSink:
    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "supersecrettoken123",
        org: str = "course_project",
        bucket: str = "metrics",
    ):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = bucket
        self.org = org

    def row_to_point(self, row: dict, family: str | None = None, latency: float | None = None) -> Point:
        model_name = str(row["model"])
        n_qubits = int(row.get("n_qubits", 0) or 0)
        fam = family or infer_family(model_name, n_qubits)
        n_seen = int(row["n_seen"])
        ts_sec = BASE_EPOCH_S + n_seen

        p = (
            Point("stream_eval")
            .tag("task", str(row["task"]))
            .tag("model", model_name)
            .tag("variant", str(row.get("variant", "") or "default"))
            .tag("W", str(row["W"]))
            .tag("H", str(row["H"]))
            .tag("drift", str(row["drift"]))
            .tag("seed", str(row["seed"]))
            .tag("family", fam)
            .tag("metric_name", str(row["metric_name"]))
            .field("metric_value", float(row["metric_value"]))
            .field("step", int(row["step"]))
            .field("n_seen", n_seen)
            .field("t_encode", float(row.get("t_encode", 0.0) or 0.0))
            .field("t_fit", float(row.get("t_fit", 0.0) or 0.0))
            .field("t_predict", float(row.get("t_predict", 0.0) or 0.0))
            .field("n_params", int(row.get("n_params", 0) or 0))
            .field("n_qubits", n_qubits)
            .field("n_circuit_evals", int(row.get("n_circuit_evals", 0) or 0))
            .field("drift_flag", int(row.get("drift_flag", 0) or 0))
            .time(ts_sec, WritePrecision.S)
        )

        peak_mem = row.get("peak_mem_mb")
        if peak_mem is not None and not (isinstance(peak_mem, float) and math.isnan(peak_mem)):
            p.field("peak_mem_mb", float(peak_mem))

        if latency is not None and not math.isnan(float(latency)):
            p.field("latency", float(latency))

        return p

    def write_rows(self, rows: list[dict], family: str | None = None, latency: float | None = None, batch_size: int = 2000):
        points = [self.row_to_point(r, family=family, latency=latency) for r in rows if not math.isnan(float(r["metric_value"]))]
        for i in range(0, len(points), batch_size):
            self.write_api.write(bucket=self.bucket, org=self.org, record=points[i : i + batch_size])

    def write_metric(self, task, model, W, H, drift, seed, step, metric_name, metric_value, timings=None, latency=None, n_seen=None, variant="default"):
        """Legacy helper used by spark_job.py and quantum/stream_consumer.py."""
        tm = timings or {}
        row = {
            "task": task,
            "model": model,
            "variant": variant,
            "W": W,
            "H": H,
            "drift": drift,
            "seed": seed,
            "step": step,
            "n_seen": n_seen if n_seen is not None else int(step) * int(H),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "t_encode": tm.get("t_encode", 0.0),
            "t_fit": tm.get("t_fit", 0.0),
            "t_predict": tm.get("t_predict", 0.0),
            "n_params": tm.get("n_params", 0),
            "n_qubits": tm.get("n_qubits", 0),
            "n_circuit_evals": tm.get("n_circuit_evals", 0),
            "peak_mem_mb": tm.get("peak_mem_mb", float("nan")),
            "drift_flag": tm.get("drift_flag", 0),
        }
        self.write_rows([row], latency=latency)

    def close(self):
        self.client.close()