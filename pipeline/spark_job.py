"""Spark Structured Streaming job for the faculty pipeline.
Reads micro-batches of size H from Kafka, applies FrozenTransform if raw features are sent,
runs StreamRunner in foreachBatch, and writes all 19-column metrics + latency to InfluxDB.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType

from common.config import STREAM_START, H as DEFAULT_H
from common.stream_runner import StreamRunner
from common.influx_sink import MetricsSink
from common.transform import FrozenTransform


def build_classical_model(task: str, model_name: str, seed: int, best_params_path: str):
    """Load classical model from classical.registry with tuned parameters if available."""
    from classical import registry as R

    params = {}
    p_path = Path(best_params_path)
    if p_path.exists():
        all_params = json.loads(p_path.read_text())
        params = all_params.get(task, {}).get(model_name, {})

    if hasattr(R, "make_model"):
        return R.make_model(task, model_name, seed=seed, **params)
    if hasattr(R, "get_model"):
        return R.get_model(model_name, seed=seed, **params)
    if hasattr(R, "MODELS"):
        cls_or_fn = R.MODELS[model_name]
        return cls_or_fn(seed=seed, **params) if callable(cls_or_fn) else cls_or_fn
    raise RuntimeError("Could not locate model factory in classical.registry")


def build_spark_schema(n_features: int) -> StructType:
    fields = [StructField(f"f{i}", DoubleType(), True) for i in range(n_features)]
    fields.extend([
        StructField("y", DoubleType(), True),
        StructField("step", IntegerType(), True),
        StructField("task", StringType(), True),
        StructField("kafka_timestamp", DoubleType(), True),
    ])
    return StructType(fields)


def run_spark_job(args):
    parquet_path = Path(args.data) if args.data else Path(f"data/processed/{args.task}_stream.parquet")
    df_proc = pd.read_parquet(parquet_path)
    feat_cols = [c for c in df_proc.columns if c.startswith("f")]
    X_full = df_proc[feat_cols].to_numpy(dtype=float)
    y_full = df_proc["y"].to_numpy()
    h_step = args.H or DEFAULT_H[args.task]

    tf_path = Path(f"data/processed/{args.task}_transform.json")
    frozen_tf = FrozenTransform.from_json(tf_path) if (args.raw_input and tf_path.exists()) else None

    model = build_classical_model(args.task, args.model, args.seed, args.params)
    runner = StreamRunner(
        model=model, task=args.task, W=args.W, H=h_step, seed=args.seed, drift=args.drift
    )
    runner.seed_window(
        X_full[STREAM_START - args.W : STREAM_START],
        y_full[STREAM_START - args.W : STREAM_START],
        n_seen_init=STREAM_START,
    )
    sink = MetricsSink()

    spark = (
        SparkSession.builder.appName(f"FacultyPipeline_{args.task}_{args.model}")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    schema = build_spark_schema(len(feat_cols))

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", h_step)
        .load()
    )
    parsed_df = raw_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    def foreach_batch_fn(batch_df, epoch_id):
        pdf = batch_df.toPandas()
        if pdf.empty:
            return
        pdf = pdf.sort_values("step")
        pdf = pdf[pdf["step"] >= STREAM_START]
        if pdf.empty:
            return

        t0 = time.time()
        xb = pdf[feat_cols].to_numpy(dtype=float)
        yb = pdf["y"].to_numpy()
        if frozen_tf is not None:
            xb = frozen_tf.transform_features(xb)
            yb = frozen_tf.transform_target(yb)

        step_rows = runner.ingest_batch(xb, yb)
        latency = time.time() - float(pdf["kafka_timestamp"].min())
        sink.write_rows(step_rows, family="classical", latency=latency)
        print(f"[Spark Epoch {epoch_id}] step={runner.step - 1} n_seen={runner.n_seen} latency={latency:.3f}s")

    query = (
        parsed_df.writeStream.foreachBatch(foreach_batch_fn)
        .trigger(processingTime="1 seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["T1", "T2", "T3"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", default=None, help="Defaults to data/processed/<task>_stream.parquet")
    p.add_argument("--params", default="configs/best_params_classical.json")
    p.add_argument("--W", type=int, default=200)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drift", default="none")
    p.add_argument("--raw-input", action="store_true", help="Apply <task>_transform.json in Spark")
    p.add_argument("--topic", default="stream_data")
    p.add_argument("--bootstrap-servers", default="localhost:9092")
    args = p.parse_args()
    run_spark_job(args)