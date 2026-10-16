# Quantum vs Classical Streaming Analytics

Layout: `common/` (A, frozen Day 3) | `data/` (A) | `classical/` (A) | `quantum/` (B) | `results/{classical,quantum}/` | `analysis/` (A) | `report/`.

## Quick start (Member A)
```
pip install -r requirements.txt
make mock && make test            # mock parquet + tests (all must pass)
make data                         # download ALL raw data into data/raw/ (inside the project)
python data/download.py --verify  # same bytes as MANIFEST.json?
make prep                         # data/raw -> data/processed/T{1,2,3}_stream.parquet
make ht                           # P1 reproduction on data/raw/elec2/electricity.csv
```
Everything reads from `data/raw/`. Nothing is read from `~/river_data`.
If a host is unreachable, `download.py` prints the URL and the exact path to drop the file in; re-run it to record the hash.
Commit `data/raw/MANIFEST.json`. Member B runs `python data/download.py --verify` to confirm identical data.

| Raw file | Source | Used for |
|---|---|---|
| data/raw/elec2/electricity.csv | River's mirror of NSW electricity data | T3, P1 reproduction |
| data/raw/seoul_bike/SeoulBikeData.csv | UCI id 560 | T1 |
| data/raw/ad/{32_shuttle,23_mammography,6_cardio}.npz | ADBench classical datasets (GitHub) | T2 |

`prep.py` options: `python data/prep.py t2 --file data/raw/ad/6_cardio.npz --n-rows 10000 --n-components 5`.

## Week 2 (classical baselines, Member A)
```
python -m classical.tune                      # 20-trial random search per tunable model, prefix rows only -> configs/best_params_classical.json
python -m classical.batch_parity p2           # P2 parity split (T1 400 train / 250 test)
python -m classical.batch_parity p3 --draws 50   # P3 protocol on ADBench mammography -> parity/classical_parity.csv + parity/p3_splits_*.npz
python -m classical.run_experiments --workers 4  # full grid -> results/classical/*.csv  (add --resume to continue)
python -m classical.run_experiments --models svr_rbf --W 200 --seeds 0 --drifts none --max-steps 20   # quick smoke
make validate
```
Grid = task x model x W {100,200,400} x drift {none, abrupt, gradual} x seeds 0..4. Constants: `common/config.py`.

## Member B entry point
Read `common/CONTRACT.md`. Use `data/processed/*` (mock now, real later). Subclass `common.base.Model`.
Run with `common.harness.run_stream(model, X, y, task, W, H, seed)`. Save to `results/quantum/`.
