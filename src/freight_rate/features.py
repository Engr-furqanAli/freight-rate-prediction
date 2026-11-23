"""Feature engineering. Pure functions of a cleaned frame; nothing is fitted here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def quarter_end_ramp(dates: pd.Series) -> pd.Series:
    """0 -> 1 progress through a quarter-end month (Mar/Jun/Sep/Dec); 0 in every other month.

    Rates climb ~4% through each quarter-end month and reset on the first day of the next
    quarter. Expressing this as a calendar *position* rather than a month id is what lets the
    pattern learned in March, June and September carry over to December.
    """
    is_quarter_end = dates.dt.month.isin(C.QUARTER_END_MONTHS)
    progress = (dates.dt.day - 1) / (dates.dt.days_in_month - 1)
    return progress.where(is_quarter_end, 0.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every model feature to a frame that went through ``clean_features``."""
    out = df.copy()
    unknown = set(out["equipment"]) - set(C.EQUIPMENT)
    if unknown:
        raise ValueError(f"unknown equipment types: {sorted(unknown)}")

    out["log_distance"] = np.log(out["distance"])
    out["log_distance_sq"] = out["log_distance"] ** 2
    out["weight_k"] = out["weight"] / 1e4
    out["eq_flatbed"] = (out["equipment"] == "Flatbed").astype(float)
    out["eq_reefer"] = (out["equipment"] == "Reefer").astype(float)
    out["equipment_code"] = out["equipment"].map({name: i for i, name in enumerate(C.EQUIPMENT)}).astype(int)

    dates = out["date"]
    out["t_months"] = (dates - C.EPOCH).dt.days / C.DAYS_PER_MONTH
    out["qe_ramp"] = quarter_end_ramp(dates)
    out["dow"] = dates.dt.dayofweek
    # Only used by the "typical GBM" ablations, never by the final model:
    out["month"] = dates.dt.month
    out["day_of_month"] = dates.dt.day
    return out
