"""Central configuration: paths, cleaning thresholds, feature lists and model settings."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PREDICTIONS_CSV = ROOT / "validation_predictions.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON = REPORTS_DIR / "metrics.json"

TARGET = "posted_rate"
EQUIPMENT = ("Dry Van", "Flatbed", "Reefer")
RANDOM_STATE = 42

# --- calendar -----------------------------------------------------------------------------
EPOCH = pd.Timestamp("2025-01-01")          # origin of the linear trend
DAYS_PER_MONTH = 30.4375
QUARTER_END_MONTHS = (3, 6, 9, 12)          # months in which rates ramp up, then reset

# --- label-noise filter ---------------------------------------------------------------------
# Clean labels sit within [0.80, 1.30] of the structural baseline; corrupted ones are
# below 0.50 or above 2.00. Any threshold inside the two empty gaps gives the same split.
N_DISTANCE_BINS = 20
CLEAN_RATIO_BAND = (0.6, 1.6)

# --- features -------------------------------------------------------------------------------
COORDS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

# Stage 1 (linear, in log space): everything that has to extrapolate beyond the training window.
LINEAR_FEATURES = [
    "eq_flatbed", "eq_reefer", "log_distance", "log_distance_sq",
    "weight_k", "weight_missing", *COORDS,
    "market_index", "t_months", "qe_ramp",
]
# Stage 2 (boosted trees on the stage-1 residual): non-linear structure. No time index on purpose.
# Day-of-week is left out as well: it adds nothing once market_index is present (see ablations).
GBM_FEATURES = [
    "log_distance", "equipment_code", "weight", "weight_missing", *COORDS,
    "market_index", "qe_ramp",
]

# Chosen on forward-chaining folds from a 3 x 3 grid (leaves x rounds); see reports/cv_gbm_grid.csv.
GBM_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=800,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=RANDOM_STATE,
)

# --- validation -----------------------------------------------------------------------------
# Forward-chaining folds: train on Jan..(m-1), test on the next HORIZON months.
# HORIZON = 2 mirrors the real task (labels end 31 Oct, predictions cover Nov + Dec).
FIRST_TEST_MONTH = 5
LAST_TEST_MONTH = 9
HORIZON_MONTHS = 2
N_UNSEEN_CITIES = 8                          # validation.csv contains 8 cities never seen in training
N_CITY_REPEATS = 5
GBM_GRID = {"max_leaf_nodes": (7, 15, 31), "max_iter": (200, 400, 800)}
