"""Backfill offline experiment CSVs (results/classical and results/quantum) into InfluxDB."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from common.influx_sink import MetricsSink
from common.stream_runner import COLUMNS


def backfill_directory(results_dir: Path, sink: MetricsSink, family_override: str | None = None):
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return 0

    total_points = 0
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            print(f"Skipping {csv_path.name}: missing columns {missing}")
            continue

        records = df.to_dict(orient="records")
        sink.write_rows(records, family=family_override)
        total_points += len(records)
        print(f"Backfilled {csv_path.name}: {len(records)} rows")

    return total_points


def main():
    parser = argparse.ArgumentParser(description="Backfill offline result CSVs into InfluxDB.")
    parser.add_argument("--dirs", nargs="+", default=["results/classical", "results/quantum"])
    parser.add_argument("--url", default="http://localhost:8086")
    parser.add_argument("--token", default="supersecrettoken123")
    parser.add_argument("--org", default="course_project")
    parser.add_argument("--bucket", default="metrics")
    args = parser.parse_args()

    sink = MetricsSink(url=args.url, token=args.token, org=args.org, bucket=args.bucket)
    try:
        grand_total = 0
        for d in args.dirs:
            path = Path(d)
            if not path.exists():
                print(f"Directory {path} does not exist, skipping.")
                continue
            fam = "quantum" if "quantum" in path.name.lower() else ("classical" if "classical" in path.name.lower() else None)
            grand_total += backfill_directory(path, sink, family_override=fam)
        print(f"Backfill complete. Total points written: {grand_total}")
    finally:
        sink.close()


if __name__ == "__main__":
    main()