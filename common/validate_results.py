import sys
import pandas as pd

REQUIRED_COLUMNS = [
    "task", "model", "variant", "seed", "W", "H", "drift", "step", "n_seen",
    "metric_name", "metric_value", "t_encode", "t_fit", "t_predict",
    "n_params", "n_qubits", "n_circuit_evals", "peak_mem_mb"
]

def validate_csv(filepath: str) -> bool:
    df = pd.read_csv(filepath)
    if list(df.columns) != REQUIRED_COLUMNS:
        print(f"[FAIL] {filepath}: Columns do not match schema.\nExpected: {REQUIRED_COLUMNS}\nFound: {list(df.columns)}")
        return False
    if df.isnull().any().any():
        print(f"[FAIL] {filepath}: Contains NaN/null values.")
        return False
    print(f"[PASS] {filepath} ({len(df)} rows valid)")
    return True

if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print("Usage: python common/validate_results.py <path_to_csv>")
        sys.exit(1)
    success = all(validate_csv(f) for f in files)
    sys.exit(0 if success else 1)

    