"""Data-quality audit. Every number quoted in the report is computed here, not typed by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import LabelNoiseFilter

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def _pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 2)


def audit(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """Counts behind every cleaning rule, for the development and the validation set."""
    noise = LabelNoiseFilter().fit(train)
    ratio = noise.ratio(train)
    low, high = C.CLEAN_RATIO_BAND
    down, up = ratio < low, ratio > high
    clean = ~(down | up)

    known = set(train["pickup"]) | set(train["delivery"])
    new = sorted((set(valid["pickup"]) | set(valid["delivery"])) - known)
    touches_new = valid["pickup"].isin(new) | valid["delivery"].isin(new)

    def weight_stats(df: pd.DataFrame) -> dict:
        w = df["weight"]
        return {
            "missing": int(w.isna().sum()), "missing_pct": _pct(w.isna().sum(), len(df)),
            "negative": int((w < 0).sum()), "negative_pct": _pct((w < 0).sum(), len(df)),
            "at_cap_47500": int((w.abs() == 47500).sum()),
            "median_abs_negative": float(w[w < 0].abs().median()),
            "median_positive": float(w[w > 0].median()),
        }

    def index_stats(df: pd.DataFrame) -> dict:
        mi = df["market_index"]
        return {
            "missing": int(mi.isna().sum()), "missing_pct": _pct(mi.isna().sum(), len(df)),
            "mean": round(float(mi.mean()), 4), "std": round(float(mi.std()), 4),
            "min": round(float(mi.min()), 4), "max": round(float(mi.max()), 4),
            "within_day_std": round(float(df.groupby("date")["market_index"].std().mean()), 4),
            "between_day_std": round(float(df.groupby("date")["market_index"].median().std()), 4),
        }

    return {
        "rows": {"train": len(train), "validation": len(valid)},
        "date_range": {
            "train": [str(train["date"].min().date()), str(train["date"].max().date())],
            "validation": [str(valid["date"].min().date()), str(valid["date"].max().date())],
        },
        "validation_rows_by_month": {MONTHS[m - 1]: int(n) for m, n in valid["date"].dt.month.value_counts().sort_index().items()},
        "duplicates": {
            "train_load_id": int(train["load_id"].duplicated().sum()),
            "validation_load_id": int(valid["load_id"].duplicated().sum()),
            "train_full_rows": int(train.drop(columns="load_id").duplicated().sum()),
        },
        "weight": {"train": weight_stats(train), "validation": weight_stats(valid)},
        "market_index": {"train": index_stats(train), "validation": index_stats(valid)},
        "labels": {
            "corrupt": int((~clean).sum()), "corrupt_pct": _pct((~clean).sum(), len(train)),
            "scaled_down": int(down.sum()), "scaled_up": int(up.sum()),
            "scaled_down_range": [round(float(ratio[down].min()), 2), round(float(ratio[down].max()), 2)],
            "scaled_up_range": [round(float(ratio[up].min()), 2), round(float(ratio[up].max()), 2)],
            "clean_range": [round(float(ratio[clean].min()), 2), round(float(ratio[clean].max()), 2)],
            "corrupt_by_month": {MONTHS[m - 1]: int(n) for m, n in train.loc[~clean, "date"].dt.month.value_counts().sort_index().items()},
        },
        "unseen_cities": {
            "names": new, "validation_rows_touching": int(touches_new.sum()),
            "validation_share_pct": _pct(touches_new.sum(), len(valid)),
            "cities_in_train": len(known),
        },
        "distance": {
            "min": float(train["distance"].min()), "max": float(train["distance"].max()),
            "rows_at_floor_70": int((train["distance"] == 70).sum()),
        },
    }


def quote_signal_regimes(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """How ``quote_signal`` relates to the rate over time, with and without labels.

    * ``corr_with_target``   rank correlation with the realised rate-per-mile (needs labels).
    * ``corr_with_baseline`` rank correlation with a label-free structural baseline, which can
      also be computed for validation.csv. It tracks the labelled diagnostic closely on the
      development data, so it tells us which regime the unlabelled period is in.
    """
    noise = LabelNoiseFilter().fit(train)
    corrupt = noise.is_corrupt(train)
    parts = []
    for name, df in (("train", train), ("validation", valid)):
        d = pd.DataFrame({"set": name, "date": df["date"].to_numpy(), "quote_signal": df["quote_signal"].to_numpy(),
                          "baseline": noise.baseline_rpm(df)})
        d["rpm"] = (df[C.TARGET] / df["distance"]).to_numpy() if name == "train" else np.nan
        d["clean"] = ~corrupt if name == "train" else True
        parts.append(d)
    rows = pd.concat(parts, ignore_index=True)

    def per_group(g: pd.DataFrame) -> pd.Series:
        labelled = g[g["clean"] & g["rpm"].notna()]
        return pd.Series({
            "corr_with_baseline": _rank_corr(g["quote_signal"], g["baseline"]),
            "corr_with_target": _rank_corr(labelled["quote_signal"], labelled["rpm"]) if len(labelled) > 2 else np.nan,
            "n": len(g),
        })

    daily = rows.groupby(["set", "date"], sort=True)[["quote_signal", "baseline", "rpm", "clean"]].apply(per_group).reset_index()
    rows["month"] = rows["date"].dt.month
    monthly = rows.groupby(["set", "month"], sort=True)[["quote_signal", "baseline", "rpm", "clean"]].apply(per_group).reset_index()
    monthly["regime"] = np.select(
        [monthly["corr_with_baseline"] > 0.5, monthly["corr_with_baseline"] < -0.5],
        ["tracks the rate", "mirrors the rate"], default="noise")
    monthly["month"] = monthly["month"].map(lambda m: MONTHS[m - 1])

    # Independent confirmation that flagged labels are corrupt: in the months where the signal
    # tracks the rate, quote_signal x distance reproduces clean labels almost exactly.
    plus_months = [MONTHS.index(m) + 1 for m in monthly.loc[(monthly["set"] == "train") & (monthly["regime"] == "tracks the rate"), "month"]]
    in_plus = train["date"].dt.month.isin(plus_months).to_numpy()
    implied = (train["quote_signal"] * train["distance"]).to_numpy()
    gap = np.abs(train[C.TARGET].to_numpy() / implied - 1)
    check = {
        "months_where_signal_tracks_rate": [MONTHS[m - 1] for m in plus_months],
        "clean_labels_median_gap_pct": round(float(100 * np.median(gap[in_plus & ~corrupt])), 2),
        "clean_labels_within_10pct": _pct((gap[in_plus & ~corrupt] < 0.10).sum(), (in_plus & ~corrupt).sum()),
        "flagged_labels_within_10pct": _pct((gap[in_plus & corrupt] < 0.10).sum(), max((in_plus & corrupt).sum(), 1)),
        "flagged_labels_in_those_months": int((in_plus & corrupt).sum()),
    }
    return monthly, daily, check
