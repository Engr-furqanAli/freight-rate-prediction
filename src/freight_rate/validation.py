"""Validation: forward-chaining folds, an unseen-city hold-out, and a random-split contrast."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config as C
from .data import LabelNoiseFilter, clean_features, daily_market_index
from .features import add_features
from .model import HybridRateModel

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NAIVE = "A. Naive: median $/mile by equipment x distance"


@dataclass(frozen=True)
class Variant:
    name: str
    linear: tuple[str, ...] | None
    gbm: tuple[str, ...] | None


_PLAIN = ["log_distance", "equipment_code", "weight", "weight_missing", *C.COORDS, "market_index", "dow"]
_CALENDAR = ["t_months", "month", "day_of_month"]
FINAL = Variant("E. Hybrid: trend + quarter-end ramp (final)", tuple(C.LINEAR_FEATURES), tuple(C.GBM_FEATURES))
VARIANTS = [
    Variant("B. GBM with quote_signal + calendar", None, tuple(_PLAIN + _CALENDAR + ["quote_signal"])),
    Variant("C. GBM without quote_signal + calendar", None, tuple(_PLAIN + _CALENDAR)),
    Variant(
        "D. Hybrid: trend only (no quarter-end ramp)",
        tuple(f for f in C.LINEAR_FEATURES if f != "qe_ramp"),
        tuple(f for f in C.GBM_FEATURES if f != "qe_ramp"),
    ),
    FINAL,
]


_L, _G = list(C.LINEAR_FEATURES), list(C.GBM_FEATURES)
ABLATIONS = [
    FINAL,
    Variant("+ day-of-week in the trees", tuple(_L), tuple(_G + ["dow"])),
    Variant("+ slow market level (7-day mean) in the linear stage", tuple(_L + ["market_level"]), tuple(_G)),
    Variant("coordinates in the linear stage only", tuple(_L), tuple(f for f in _G if f not in C.COORDS)),
    Variant("linear stage only (no trees)", tuple(_L), None),
    Variant("without the trend", tuple(f for f in _L if f != "t_months"), tuple(_G)),
    Variant("without market_index", tuple(f for f in _L if f != "market_index"), tuple(f for f in _G if f != "market_index")),
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(100 * np.mean(np.abs(err) / y_true)),
        "bias_pct": float(100 * np.mean(err / y_true)),
        "n": int(len(y_true)),
    }


def prepare(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Fit every data-dependent step on the training side only, then apply it to both sides.

    The day-level market index is the one exception: it is built from the *features* of both
    sides, exactly as at prediction time, and never touches a label.
    """
    noise = LabelNoiseFilter().fit(train_raw)
    weight_fill = float(train_raw["weight"].abs().median())
    daily = daily_market_index(train_raw, test_raw)
    train = add_features(clean_features(train_raw[~noise.is_corrupt(train_raw)], daily, weight_fill))
    test = add_features(clean_features(test_raw, daily, weight_fill))
    return train, test, noise


def score_split(train_raw: pd.DataFrame, test_raw: pd.DataFrame, variants: list[Variant], naive: bool = True) -> list[dict]:
    """Train each variant on ``train_raw`` and score it on ``test_raw``.

    Reported on clean labels (corrupted test labels are unpredictable by construction) and,
    for transparency, on all labels.
    """
    train, test, noise = prepare(train_raw, test_raw)
    y = test_raw[C.TARGET].to_numpy()
    clean = ~noise.is_corrupt(test_raw)
    predictions: dict[str, np.ndarray] = {}
    if naive:
        predictions[NAIVE] = noise.baseline_rpm(test_raw) * test_raw["distance"].to_numpy()
    for v in variants:
        predictions[v.name] = HybridRateModel(v.linear, v.gbm).fit(train, train[C.TARGET]).predict(test)

    month = test_raw["date"].dt.month.to_numpy()
    rows = []
    for name, pred in predictions.items():
        row = {"variant": name, **metrics(y[clean], pred[clean])}
        row["mae_all_labels"] = metrics(y, pred)["mae"]
        row["rmse_all_labels"] = metrics(y, pred)["rmse"]
        for m in np.unique(month):                       # per test month: exposes horizon effects
            sel = clean & (month == m)
            row[f"mape_{MONTHS[m - 1]}"] = metrics(y[sel], pred[sel])["mape"]
            row[f"bias_{MONTHS[m - 1]}"] = metrics(y[sel], pred[sel])["bias_pct"]
        rows.append(row)
    return rows


def forward_chaining_cv(dev: pd.DataFrame, variants: list[Variant] = VARIANTS) -> pd.DataFrame:
    """Expanding-window folds: train Jan..(m-1), test the following ``HORIZON_MONTHS`` months."""
    month = dev["date"].dt.month
    out = []
    for m in range(C.FIRST_TEST_MONTH, C.LAST_TEST_MONTH + 1):
        last = m + C.HORIZON_MONTHS - 1
        fold = f"Jan-{MONTHS[m - 2]} -> {MONTHS[m - 1]}+{MONTHS[last - 1]}"
        for row in score_split(dev[month < m], dev[(month >= m) & (month <= last)], variants):
            out.append({"fold": fold, **row})
    return pd.DataFrame(out)


def random_kfold_cv(dev: pd.DataFrame, variants: list[Variant], n_splits: int = 5) -> pd.DataFrame:
    """Shuffled K-fold. Only here to show how optimistic a random split is on this data."""
    out = []
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    for i, (tr, te) in enumerate(folds.split(dev), start=1):
        for row in score_split(dev.iloc[tr], dev.iloc[te], variants, naive=False):
            out.append({"fold": f"random {i}", **row})
    return pd.DataFrame(out)


def unseen_city_cv(dev: pd.DataFrame, variants: list[Variant]) -> pd.DataFrame:
    """Time x space hold-out, mirroring the 8 never-seen cities in validation.csv.

    On the last forward fold (train Jan-Aug, test Sep+Oct) a random set of cities is removed
    from training entirely; test loads are then scored separately depending on whether they
    touch a held-out city.
    """
    month = dev["date"].dt.month
    train_all, test = dev[month <= 8], dev[month >= 9]
    cities = np.array(sorted(set(train_all["pickup"]) | set(train_all["delivery"])))
    rng = np.random.default_rng(C.RANDOM_STATE)
    out = []
    for rep in range(1, C.N_CITY_REPEATS + 1):
        held = set(rng.choice(cities, size=C.N_UNSEEN_CITIES, replace=False))
        train = train_all[~(train_all["pickup"].isin(held) | train_all["delivery"].isin(held))]
        touches = (test["pickup"].isin(held) | test["delivery"].isin(held)).to_numpy()
        for label, part in (("touches unseen city", test[touches]), ("known cities only", test[~touches])):
            for row in score_split(train, part, variants, naive=False):
                out.append({"repeat": rep, "segment": label, **row})
    return pd.DataFrame(out)


def gbm_grid_cv(dev: pd.DataFrame) -> pd.DataFrame:
    """Forward-chaining score of the final model for each tree size x number of rounds."""
    month = dev["date"].dt.month
    folds = []
    for m in range(C.FIRST_TEST_MONTH, C.LAST_TEST_MONTH + 1):
        test_raw = dev[(month >= m) & (month <= m + C.HORIZON_MONTHS - 1)]
        train, test, noise = prepare(dev[month < m], test_raw)
        folds.append((train, test, test_raw[C.TARGET].to_numpy(), ~noise.is_corrupt(test_raw)))
    out = []
    for leaves in C.GBM_GRID["max_leaf_nodes"]:
        for rounds in C.GBM_GRID["max_iter"]:
            scores = []
            for train, test, y, clean in folds:
                model = HybridRateModel(gbm_params={"max_leaf_nodes": leaves, "max_iter": rounds})
                pred = model.fit(train, train[C.TARGET]).predict(test)
                scores.append(metrics(y[clean], pred[clean])["mape"])
            out.append({"max_leaf_nodes": leaves, "max_iter": rounds, "mape": np.mean(scores), "worst_fold_mape": np.max(scores)})
    return pd.DataFrame(out)
