# CONTRACT (Day 0, frozen after Day 3)

Both members read this. Change only by PR touching `common/`, approved by both.

## Tasks
| Task | Data | Features | Target | Metrics |
|---|---|---|---|---|
| T1 regression | Seoul Bike | 7 PCA comps | demand in [0,1] | mse, mae |
| T2 anomaly | AD benchmark (ODDS/ADBench) | 4-6 PCA comps | y=1 anomaly | auc_pooled, ap_pooled |
| T3 classification | ELEC2 (first 5-10k rows for quantum) | 4 PCA comps | y in {0,1} | accuracy, f1_macro, acc_nochange, kappa_plus |

## Data files (A produces, B consumes)
`data/processed/T*_stream.parquet`: columns `f0..f{d-1}` in [0, pi], and `y`. Row order is final.
Sidecar `T*_stream.meta.json`: scaler/PCA info. Mock versions ship Day 1 (`python data/make_mock.py`).
Scaler and PCA fit on first 400 rows only. No leakage.
**Deviation to confirm (B):** T1 target is min-max scaled globally to [0,1] (as in P2), because a bounded-output
variational model cannot predict outside its range. Features stay leak-free.

## Protocol
Train on last W samples, test on next H, slide by H. Test then train.
W in {100, 200, 400}. H=20 (T1, T3), H=50 (T2). Seeds 0..4 (quantum may use 0..2, disclosed).
T2 is semisupervised: harness passes only normal samples (y==0) to `fit`; `y` is None.
T2 AUC is pooled over the last 500 scored samples (per-block AUC is undefined when a block has no anomalies).
Drift scenarios: `none`, `abrupt`, `gradual` (injected by `common/drift.py` AFTER scaling), `natural` (ELEC2).

## Model interface (`common/base.py`)
`fit(X, y)` retrains from scratch on the window unless `warm_start=True`.
`predict(X)` for T1/T3. `score_anomaly(X)` for T2 (higher = more anomalous).
`timings`: t_encode, t_fit, t_predict (seconds; harness falls back to wall time if None).
`meta`: n_params, n_qubits, circuit_depth, n_circuit_evals.

## Result CSV
Path: `results/<owner>/<task>_<model>*.csv` (owner = classical | quantum). Columns, in order:
task, model, variant, seed, W, H, drift, step, n_seen, metric_name, metric_value, t_encode, t_fit,
t_predict, n_params, n_qubits, n_circuit_evals, peak_mem_mb, drift_flag.
Long format: one row per step x metric. Validate: `python -m common.validate_results results/*/*.csv`.

## Rules
Same data, order, window, seeds for all models. Equal tuning budget per model, tuned on a prefix, never on the test stream.
Branches `a/*` and `b/*`. PRs only for `common/`. Pin versions in `requirements.txt`.

## T3 metric note (added Week 1)
`acc_nochange` = accuracy of predicting the previous label on the same test block. `kappa_plus = (acc - acc_nochange) / (1 - acc_nochange)`:
0 = no better than persistence, < 0 = worse. ELEC2 labels are strongly autocorrelated, so raw accuracy alone is misleading.
Per-block values are noisy: pool over the stream (recompute from mean accuracy and mean acc_nochange) for tables.
