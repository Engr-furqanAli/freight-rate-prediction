"""End-to-end pipeline: audit -> validation -> final fit -> prediction files.

    python -m freight_rate.run                    # everything (about 1-2 minutes)
    python -m freight_rate.run --skip-validation  # only refit and rewrite the prediction files
    python -m freight_rate.run --ablations        # also rerun feature ablations and the GBM grid
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import validation as V
from .data import LabelNoiseFilter, city_coordinates, clean_features, daily_market_index, load_raw
from .features import add_features
from .model import HybridRateModel
from .quality import audit, quote_signal_regimes


def fit_final(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[HybridRateModel, dict]:
    """Fit the final model on every clean development row."""
    noise = LabelNoiseFilter().fit(train)
    context = {
        "weight_fill": float(train["weight"].abs().median()),
        "daily": daily_market_index(train, valid),          # features only
        "coords": city_coordinates(train, valid),
    }
    X = add_features(clean_features(train[~noise.is_corrupt(train)], context["daily"], context["weight_fill"]))
    model = HybridRateModel().fit(X, X[C.TARGET])
    context["n_train"] = len(X)
    return model, context


def predict(model: HybridRateModel, frame: pd.DataFrame, context: dict) -> np.ndarray:
    X = add_features(clean_features(frame, context["daily"], context["weight_fill"], context["coords"]))
    return model.predict(X)


def write_validation_predictions(valid: pd.DataFrame, pred: np.ndarray, data_dir: Path) -> pd.DataFrame:
    """Fill the provided template (its order, its ids) and save ``validation_predictions.csv``."""
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    by_id = pd.Series(np.round(pred, 2), index=valid["load_id"].to_numpy())
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(by_id)
    if out["predicted_rate"].isna().any() or (out["predicted_rate"] <= 0).any():
        raise ValueError("missing or non-positive predictions")
    if len(out) != len(valid) or out["load_id"].duplicated().any():
        raise ValueError("template ids do not match validation.csv")
    out.to_csv(C.PREDICTIONS_CSV, index=False)
    return out


def write_december(pred: np.ndarray, data_dir: Path) -> pd.DataFrame:
    """Fill ``predicted_rate`` in place, leaving every other column byte-for-byte as provided."""
    path = data_dir / "december_chart_inputs.csv"
    raw = pd.read_csv(path)
    raw["predicted_rate"] = np.round(pred, 2)
    raw.to_csv(path, index=False)
    return raw


def _summarise_validation(train: pd.DataFrame) -> dict:
    forward = V.forward_chaining_cv(train)
    random = V.random_kfold_cv(train, [V.VARIANTS[0], V.FINAL])
    cities = V.unseen_city_cv(train, [V.FINAL])
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    forward.to_csv(C.REPORTS_DIR / "cv_forward.csv", index=False)
    random.to_csv(C.REPORTS_DIR / "cv_random.csv", index=False)
    cities.to_csv(C.REPORTS_DIR / "cv_unseen_city.csv", index=False)

    keep = ["mape", "mae", "rmse", "bias_pct", "mae_all_labels", "rmse_all_labels"]
    bias_cols = [c for c in forward.columns if c.startswith("bias_") and c != "bias_pct"]
    return {
        "forward_mean": forward.groupby("variant")[keep].mean().round(3).to_dict("index"),
        "forward_worst_fold_mape": forward.groupby("variant")["mape"].max().round(3).to_dict(),
        "forward_mape_by_fold": forward.pivot_table(index="variant", columns="fold", values="mape", sort=False).round(3).to_dict("index"),
        "forward_worst_month_abs_bias": forward.groupby("variant")[bias_cols].mean().abs().max(axis=1).round(2).to_dict(),
        "forward_bias_by_month_final": forward[forward["variant"] == V.FINAL.name][bias_cols].mean().round(2).to_dict(),
        "random_mean": random.groupby("variant")[keep].mean().round(3).to_dict("index"),
        "unseen_city_mean": cities.groupby("segment")[["mape", "mae", "bias_pct", "n"]].mean().round(3).to_dict("index"),
    }


def _summarise_ablations(train: pd.DataFrame) -> dict:
    ablations = V.forward_chaining_cv(train, V.ABLATIONS)
    ablations = ablations[ablations["variant"] != V.NAIVE]
    grid = V.gbm_grid_cv(train)
    ablations.to_csv(C.REPORTS_DIR / "cv_ablations.csv", index=False)
    grid.to_csv(C.REPORTS_DIR / "cv_gbm_grid.csv", index=False)
    bias_cols = [c for c in ablations.columns if c.startswith("bias_") and c != "bias_pct"]
    by_variant = ablations.groupby("variant", sort=False)
    return {
        "ablation_mape": by_variant["mape"].mean().round(3).to_dict(),
        "ablation_folds_better_than_final": {
            name: int((g["mape"].to_numpy() < ablations.loc[ablations["variant"] == V.FINAL.name, "mape"].to_numpy()).sum())
            for name, g in by_variant
        },
        "ablation_worst_month_abs_bias": by_variant[bias_cols].mean().abs().max(axis=1).round(2).to_dict(),
        "gbm_grid": grid.round(3).to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="folder holding the provided CSV files (default: ./data)")
    parser.add_argument("--skip-validation", action="store_true", help="skip cross-validation; only refit and predict")
    parser.add_argument("--ablations", action="store_true", help="also run feature ablations and the GBM grid (adds a few minutes)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else C.DATA_DIR
    started = time.time()

    train, valid, december = load_raw(data_dir)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else {}

    print("1/4  auditing data ...")
    results["audit"] = audit(train, valid)
    monthly, daily, check = quote_signal_regimes(train, valid)
    daily.to_csv(C.REPORTS_DIR / "quote_signal_daily.csv", index=False)
    results["quote_signal"] = {"monthly": monthly.round(3).to_dict("records"), "label_check": check}

    if not args.skip_validation:
        print("2/4  validating (forward-chaining, random-split contrast, unseen cities) ...")
        results["validation"] = _summarise_validation(train)
    else:
        print("2/4  validation skipped")
    if args.ablations:
        print("     ablations and GBM grid ...")
        results["ablations"] = _summarise_ablations(train)

    print("3/4  fitting the final model on all clean development rows ...")
    model, context = fit_final(train, valid)
    effects = model.linear_effects()
    pred_valid = predict(model, valid, context)
    pred_december = predict(model, december, context)

    print("4/4  writing prediction files ...")
    submission = write_validation_predictions(valid, pred_valid, data_dir)
    chart = write_december(pred_december, data_dir)

    month = valid["date"].dt.month.to_numpy()
    results["final_model"] = {
        "training_rows": context["n_train"],
        "trend_pct_per_month": round(100 * (np.exp(effects["t_months"]) - 1), 3),
        "quarter_end_ramp_pct": round(100 * (np.exp(effects["qe_ramp"]) - 1), 3),
        "market_index_log_points_per_unit": round(effects["market_index"], 4),
        "flatbed_premium_pct": round(100 * (np.exp(effects["eq_flatbed"]) - 1), 2),
        "reefer_premium_pct": round(100 * (np.exp(effects["eq_reefer"]) - 1), 2),
        "weight_pct_per_10k_lb": round(100 * (np.exp(effects["weight_k"]) - 1), 2),
        "linear_effects": {k: round(v, 5) for k, v in effects.items()},
        "validation_predictions": {
            "rows": len(submission), "min": float(submission["predicted_rate"].min()), "max": float(submission["predicted_rate"].max()),
            "mean_rate_per_mile_Nov": round(float(np.mean(pred_valid[month == 11] / valid["distance"].to_numpy()[month == 11])), 4),
            "mean_rate_per_mile_Dec": round(float(np.mean(pred_valid[month == 12] / valid["distance"].to_numpy()[month == 12])), 4),
            "train_clean_mean_rate_per_mile_Oct": round(float((train[C.TARGET] / train["distance"])[(train["date"].dt.month == 10).to_numpy() & ~LabelNoiseFilter().fit(train).is_corrupt(train)].mean()), 4),
        },
        "december_chart": {
            "first": float(chart["predicted_rate"].iloc[0]), "last": float(chart["predicted_rate"].iloc[-1]),
            "min": float(chart["predicted_rate"].min()), "max": float(chart["predicted_rate"].max()),
        },
    }
    C.METRICS_JSON.write_text(json.dumps(results, indent=2))

    print(f"\nDone in {time.time() - started:.0f}s")
    print(f"  {C.PREDICTIONS_CSV.name}: {len(submission):,} rows")
    print(f"  {(data_dir / 'december_chart_inputs.csv')}: 31 rows, ${chart['predicted_rate'].min():,.2f} - ${chart['predicted_rate'].max():,.2f}")
    if "validation" in results:
        final = results["validation"]["forward_mean"][V.FINAL.name]
        print(f"  forward-chaining CV (final model): MAPE {final['mape']:.2f}%  MAE ${final['mae']:.2f}  RMSE ${final['rmse']:.2f}")
    print(f"  metrics: {C.METRICS_JSON}")


if __name__ == "__main__":
    main()
