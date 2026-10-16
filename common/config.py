"""Single source of truth for experiment constants. Owned by A, frozen with common/."""
TUNE_ROWS = 800            # rows [0, TUNE_ROWS) are the tuning prefix: hyperparameter search ONLY, never test data
TUNE_W = 200               # window used while tuning
N_TRIALS = 20              # random-search trials per tunable model (same budget for quantum models)
TUNE_SEED = 0
STREAM_START = TUNE_ROWS   # streaming evaluation (first test block) starts here for every model
W_GRID = [100, 200, 400]
H = {"T1": 20, "T2": 50, "T3": 20}
SEEDS = [0, 1, 2, 3, 4]
DRIFTS = ["none", "abrupt", "gradual"]
DRIFT_START_FRAC = 0.5
DRIFT_WIDTH_FRAC = 0.2     # gradual ramp length
DRIFT_MAGNITUDE = 1.0      # in feature std units
STREAM_FIT_ROWS = 800      # scaler/PCA fit rows (data/prep.py). Must be <= TUNE_ROWS: never fit on test rows
