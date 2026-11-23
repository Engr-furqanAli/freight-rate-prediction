"""Loading and cleaning.

Every rule here is backed by a finding from the data audit (see the report / README):

* ``weight``        ~0.6-1.2% negative (sign-flipped: |w| has the same distribution and the same
                    rate response as positive weights) and ~0.6-1.4% missing.
* ``market_index``  ~0.8-2.1% missing; it is a day-level quantity (within-day sd 0.025 against
                    a between-day sd of ~0.17), so the same-day median is an accurate fill.
* ``posted_rate``   ~1.4% of labels are corrupted by a factor of 0.17-0.47x or 2.0-5.3x.
* city names        8 cities in validation.csv never occur in training, so geography enters the
                    model through coordinates only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def load_raw(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (development data, validation loads, December chart inputs)."""
    d = Path(data_dir) if data_dir else C.DATA_DIR
    train = pd.read_csv(d / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(d / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(d / "december_chart_inputs.csv", parse_dates=["date"])
    return train, valid, december


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) pair per city, collected from the pickup and delivery columns of ``frames``."""
    parts = [
        f[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for f in frames for side in ("pickup", "delivery")
    ]
    coords = pd.concat(parts, ignore_index=True).drop_duplicates()
    if coords["city"].duplicated().any():
        raise ValueError("a city maps to more than one coordinate pair")
    return coords.set_index("city")


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Median ``market_index`` per calendar day. Uses features only, never labels."""
    both = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    return both.groupby("date")["market_index"].median()


def market_index_for(dates: pd.Series, daily: pd.Series) -> pd.Series:
    """Day-level market index for each date.

    Days absent from ``daily`` fall back to the mean of the same weekday over the latest four
    observed weeks, which preserves the strong weekly cycle of the index.
    """
    out = dates.map(daily)
    if out.isna().any():
        recent = daily.sort_index().iloc[-28:]
        by_dow = recent.groupby(recent.index.dayofweek).mean()
        out = out.fillna(dates.dt.dayofweek.map(by_dow))
    return out


def clean_features(
    df: pd.DataFrame,
    daily_index: pd.Series,
    weight_fill: float,
    coords: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply the feature-cleaning rules. Works for frames with or without labels.

    ``coords`` is only needed for inputs that carry city names but no coordinates
    (the December chart file).
    """
    out = df.copy()

    if "pickup_lat" not in out.columns:
        if coords is None:
            raise ValueError("city coordinates are required for inputs without lat/lon columns")
        for side in ("pickup", "delivery"):
            unknown = sorted(set(out[side]) - set(coords.index))
            if unknown:
                raise ValueError(f"no coordinates known for {side} cities: {unknown}")
            out[f"{side}_lat"] = out[side].map(coords["lat"])
            out[f"{side}_lon"] = out[side].map(coords["lon"])

    # weight: undo sign flips, flag and fill the missing ones
    weight = out["weight"].abs()
    out["weight_missing"] = weight.isna().astype(int)
    out["weight"] = weight.fillna(weight_fill)

    # market index: row value when present, otherwise the day-level value
    day_level = market_index_for(out["date"], daily_index)
    out["market_index"] = out["market_index"].fillna(day_level) if "market_index" in out.columns else day_level
    if out["market_index"].isna().any():
        raise ValueError("market_index could not be resolved for some rows")
    # slow market level: trailing 7-day mean of the day-level index (averages out the weekly cycle)
    level = daily_index.sort_index().rolling(7, min_periods=1).mean()
    out["market_level"] = market_index_for(out["date"], level)
    return out


class LabelNoiseFilter:
    """Flags corrupted targets using a robust structural baseline.

    Baseline = median rate-per-mile of each (equipment, distance-bin) cell. Medians are immune
    to the ~1.4% contamination, so the filter can be fitted on the raw training rows. A label
    is corrupt when its rate-per-mile is outside ``CLEAN_RATIO_BAND`` times the baseline.
    """

    def fit(self, df: pd.DataFrame) -> "LabelNoiseFilter":
        self.edges_ = np.quantile(df["distance"], np.linspace(0, 1, C.N_DISTANCE_BINS + 1))[1:-1]
        rpm = df[C.TARGET] / df["distance"]
        self.table_ = rpm.groupby([df["equipment"].to_numpy(), self._bins(df)]).median()
        self.fallback_ = float(rpm.median())
        return self

    def _bins(self, df: pd.DataFrame) -> np.ndarray:
        return np.digitize(df["distance"].to_numpy(), self.edges_)

    def baseline_rpm(self, df: pd.DataFrame) -> np.ndarray:
        """Label-free structural expectation of rate-per-mile for each row."""
        index = pd.MultiIndex.from_arrays([df["equipment"].to_numpy(), self._bins(df)])
        return self.table_.reindex(index).fillna(self.fallback_).to_numpy()

    def ratio(self, df: pd.DataFrame) -> np.ndarray:
        return (df[C.TARGET] / df["distance"]).to_numpy() / self.baseline_rpm(df)

    def is_corrupt(self, df: pd.DataFrame) -> np.ndarray:
        low, high = C.CLEAN_RATIO_BAND
        ratio = self.ratio(df)
        return (ratio < low) | (ratio > high)
