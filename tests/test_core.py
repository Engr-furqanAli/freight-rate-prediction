"""Unit tests for the cleaning rules, calendar features, label filter, model and output files."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freight_rate import config as C
from freight_rate.data import LabelNoiseFilter, city_coordinates, clean_features, daily_market_index, market_index_for
from freight_rate.features import add_features, quarter_end_ramp
from freight_rate.model import HybridRateModel

FAST = {"max_iter": 60, "max_leaf_nodes": 7}


def make_loads(n: int = 3000, seed: int = 0, trend: float = 0.01) -> pd.DataFrame:
    """Synthetic loads with a known structure: power-law in distance, a linear trend in log rate."""
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 300, n)
    df = pd.DataFrame({
        "load_id": [f"X-{i:05d}" for i in range(n)],
        "pickup": rng.choice(["A", "B"], n), "delivery": rng.choice(["C", "D"], n),
        "distance": rng.uniform(100, 2500, n),
        "equipment": rng.choice(C.EQUIPMENT, n),
        "weight": rng.uniform(10_000, 45_000, n),
        "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(days, unit="D"),
        "market_index": rng.normal(1.0, 0.05, n),
        "quote_signal": rng.normal(2.0, 0.2, n),
    })
    lat = {"A": 30.0, "B": 40.0, "C": 33.0, "D": 42.0}
    lon = {"A": -95.0, "B": -75.0, "C": -110.0, "D": -85.0}
    for side in ("pickup", "delivery"):
        df[f"{side}_lat"] = df[side].map(lat)
        df[f"{side}_lon"] = df[side].map(lon)
    log_rate = np.log(4.0) + 0.9 * np.log(df["distance"]) + trend * days / C.DAYS_PER_MONTH + rng.normal(0, 0.02, n)
    df[C.TARGET] = np.exp(log_rate)
    return df


def test_quarter_end_ramp_is_a_calendar_position():
    dates = pd.Series(pd.to_datetime(["2025-03-01", "2025-03-31", "2025-04-01", "2025-11-30", "2025-12-16", "2025-12-31"]))
    assert quarter_end_ramp(dates).tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.5, 1.0])


def test_weight_sign_flips_and_gaps_are_repaired():
    df = make_loads(6)
    df.loc[0, "weight"] = -30_000.0
    df.loc[1, "weight"] = np.nan
    out = clean_features(df, daily_market_index(df), weight_fill=31_000.0)
    assert out.loc[0, "weight"] == 30_000.0 and out.loc[0, "weight_missing"] == 0
    assert out.loc[1, "weight"] == 31_000.0 and out.loc[1, "weight_missing"] == 1
    assert (out["weight"] > 0).all()


def test_market_index_gap_takes_the_same_day_median():
    df = make_loads(200)
    df["date"] = pd.Timestamp("2025-05-05")
    expected = df["market_index"].iloc[1:].median()
    df.loc[0, "market_index"] = np.nan
    out = clean_features(df, daily_market_index(df), weight_fill=31_000.0)
    assert out.loc[0, "market_index"] == pytest.approx(expected)


def test_unknown_day_falls_back_to_recent_same_weekday():
    days = pd.date_range("2025-10-01", "2025-10-28")
    daily = pd.Series(np.where(days.dayofweek == 2, 1.2, 0.9), index=days)          # Wednesdays are high
    wanted = pd.Series(pd.to_datetime(["2025-11-05", "2025-11-06"]))                   # a Wednesday, a Thursday
    assert market_index_for(wanted, daily).tolist() == pytest.approx([1.2, 0.9])


def test_inputs_without_coordinates_use_the_city_lookup():
    df = make_loads(50)
    coords = city_coordinates(df)
    bare = df.drop(columns=[*C.COORDS, "market_index", "quote_signal", C.TARGET])
    with pytest.raises(ValueError):
        clean_features(bare, daily_market_index(df), 31_000.0)
    out = clean_features(bare, daily_market_index(df), 31_000.0, coords)
    assert out[C.COORDS].notna().all().all() and out["market_index"].notna().all()
    assert (out["pickup_lat"].to_numpy() == df["pickup_lat"].to_numpy()).all()


def test_label_noise_filter_flags_exactly_the_scaled_labels():
    df = make_loads(4000, seed=1)
    bad = np.arange(0, 4000, 100)
    df.loc[bad[::2], C.TARGET] *= 4.0
    df.loc[bad[1::2], C.TARGET] *= 0.25
    flagged = LabelNoiseFilter().fit(df).is_corrupt(df)
    assert set(np.flatnonzero(flagged)) == set(bad)


def test_hybrid_extrapolates_a_trend_that_plain_trees_cannot():
    df = make_loads(6000, seed=2, trend=0.01)
    month = df["date"].dt.month
    X = add_features(clean_features(df, daily_market_index(df), 31_000.0))
    train, test = X[(month <= 7).to_numpy()], X[(month >= 9).to_numpy()]

    def bias(model: HybridRateModel) -> float:
        pred = model.fit(train, train[C.TARGET]).predict(test)
        return float(np.mean(pred / test[C.TARGET] - 1))

    hybrid = bias(HybridRateModel(gbm_params=FAST))
    plain = bias(HybridRateModel(None, ["log_distance", "equipment_code", "weight", "t_months"], gbm_params=FAST))
    assert abs(hybrid) < 0.006            # the linear stage carries the trend forward
    assert plain < -0.012                 # trees hold the last level and fall behind


def test_final_model_never_sees_quote_signal_or_city_names():
    used = set(C.LINEAR_FEATURES) | set(C.GBM_FEATURES)
    assert not used & {"quote_signal", "pickup", "delivery", "load_id", "month", C.TARGET}
    assert "t_months" not in C.GBM_FEATURES


def test_submission_file_matches_the_template():
    predictions, template = C.PREDICTIONS_CSV, C.DATA_DIR / "validation_predictions_template.csv"
    if not (Path(predictions).exists() and template.exists()):
        pytest.skip("run `python -m freight_rate.run` first")
    out, ref = pd.read_csv(predictions), pd.read_csv(template)
    assert list(out.columns) == ["load_id", "predicted_rate"]
    assert out["load_id"].tolist() == ref["load_id"].tolist()
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
