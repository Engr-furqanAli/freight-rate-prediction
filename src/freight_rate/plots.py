"""Figures for the report. Run after the pipeline:  python -m freight_rate.plots"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as C
from . import validation as V
from .data import LabelNoiseFilter, clean_features, daily_market_index, load_raw
from .features import add_features
from .model import HybridRateModel

MONTHS = V.MONTHS
INK, INK_2, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#e8e7e3", "#c9c8c3", "#f1f0ec"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b9b8b2"      # validated pair + de-emphasis grey


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def _head(ax, title: str, subtitle: str) -> None:
    ax.text(0, 1.17, title, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0, 1.07, subtitle, transform=ax.transAxes, fontsize=8.8, color=INK_2, ha="left", va="bottom")


def _month_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def fig_quote_signal() -> None:
    d = pd.read_csv(C.REPORTS_DIR / "quote_signal_daily.csv", parse_dates=["date"])
    train, valid = d[d["set"] == "train"], d[d["set"] == "validation"]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.axvspan(valid["date"].min(), valid["date"].max(), color=BAND, lw=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(train["date"], train["corr_with_target"], color=BLUE, lw=2, solid_joinstyle="round")
    ax.plot(d["date"], d["corr_with_baseline"], color=ORANGE, lw=2, solid_joinstyle="round")
    ax.set_ylim(-1.18, 1.18)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Rank correlation")
    _month_axis(ax)
    note = dict(fontsize=8.5, color=INK_2, ha="center", va="center")
    ax.text(pd.Timestamp("2025-02-14"), 0.66, "tracks the rate", **note)
    ax.text(pd.Timestamp("2025-05-01"), -0.66, "mirrors the rate", **note)
    ax.text(pd.Timestamp("2025-08-16"), 0.26, "noise", **note)
    ax.text(pd.Timestamp("2025-12-01"), 0.42, "validation period\nnoise on all 61 days", **note)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=2, label="vs. realised rate per mile (needs labels)"),
        Line2D([], [], color=ORANGE, lw=2, label="vs. label-free structural baseline (works on validation.csv)"),
    ], loc="upper left", bbox_to_anchor=(0, -0.13), ncol=1, handlelength=1.6)
    _head(ax, "quote_signal flips sign between months, then carries no signal",
          "Daily rank correlation between quote_signal and the rate per mile")
    fig.savefig(C.FIGURES_DIR / "quote_signal_regimes.png")
    plt.close(fig)


def fig_quarter_end(train: pd.DataFrame) -> dict:
    """Daily rate level once lane structure, market index and trend are removed."""
    noise = LabelNoiseFilter().fit(train)
    X = add_features(clean_features(train[~noise.is_corrupt(train)], daily_market_index(train), float(train["weight"].abs().median())))
    # Linear stage only, without the ramp term: a transparent residual. (Trees are left out here
    # because in-sample they can use market_index as a partial proxy for the period.)
    model = HybridRateModel([f for f in C.LINEAR_FEATURES if f != "qe_ramp"], None).fit(X, X[C.TARGET])
    resid = pd.Series(100 * (np.log(X[C.TARGET].to_numpy()) - model.predict_log(X)), index=X.index)
    by_day = resid.groupby(X["date"]).median()
    ordinary = ~by_day.index.month.isin(C.QUARTER_END_MONTHS)
    offset = by_day[ordinary].median()
    by_day = by_day - offset

    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ramps = {}
    for m in (3, 6, 9):
        start = pd.Timestamp(2025, m, 1)
        end = start + pd.offsets.MonthEnd(0)
        ax.axvspan(start, end, color=BAND, lw=0)
        rows = (X["date"].dt.month == m).to_numpy()
        ramps[MONTHS[m - 1]] = float(np.polyfit(X.loc[rows, "qe_ramp"], resid[rows], 1)[0])
        ax.text(start + (end - start) / 2, 5.55, f"{MONTHS[m - 1]}: +{ramps[MONTHS[m - 1]]:.1f}%", fontsize=8.5, color=INK_2, ha="center", va="center")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(by_day.index, by_day.to_numpy(), color=BLUE, lw=2, solid_joinstyle="round")
    ax.set_ylim(-2.2, 6.2)
    ax.set_ylabel("Rate vs. ordinary days (%)")
    _month_axis(ax)
    _head(ax, "Rates climb about 4% through each quarter-end month, then reset",
          "Daily median rate after removing lane structure, market index and trend; shaded = quarter-end months")
    fig.savefig(C.FIGURES_DIR / "quarter_end_ramp.png")
    plt.close(fig)
    return ramps


def fig_validation_design(metrics: dict) -> None:
    by_fold = metrics["validation"]["forward_mape_by_fold"][V.FINAL.name]
    spans = [(m, m + C.HORIZON_MONTHS - 1) for m in range(C.FIRST_TEST_MONTH, C.LAST_TEST_MONTH + 1)] + [(11, 12)]
    labels = [f"Fold {i + 1}" for i in range(len(spans) - 1)] + ["Final model"]
    fig, ax = plt.subplots(figsize=(7.4, 2.75))
    for i, (first, last) in enumerate(spans):
        y = len(spans) - 1 - i
        ax.barh(y, first - 1, left=0, height=0.52, color=BLUE, edgecolor="white", linewidth=2)
        ax.barh(y, last - first + 1, left=first - 1, height=0.52, color=ORANGE, edgecolor="white", linewidth=2)
        key = f"Jan-{MONTHS[first - 2]} -> {MONTHS[first - 1]}+{MONTHS[last - 1]}"
        text = f"{by_fold[key]:.2f}% MAPE" if key in by_fold else "scored by Spotter"
        ax.text(12.2, y, text, va="center", ha="left", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 12)
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(MONTHS)
    for tick, m in zip(ax.get_xticklabels(), range(1, 13)):
        if m in C.QUARTER_END_MONTHS:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels(labels[::-1])
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(color=BLUE, label="Train"), Patch(color=ORANGE, label="Test: the next two unseen months")],
              loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, handlelength=1.2)
    _head(ax, "Every fold predicts the next two unseen months, as the real task does",
          "Expanding-window forward-chaining; bold months are quarter-ends; right: error of the final model")
    fig.savefig(C.FIGURES_DIR / "validation_design.png")
    plt.close(fig)


def fig_model_comparison(metrics: dict) -> None:
    mean = metrics["validation"]["forward_mean"]
    worst = metrics["validation"]["forward_worst_fold_mape"]
    names = sorted(mean)
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.barh(y, [mean[n]["mape"] for n in names], height=0.5, color=[BLUE if n == V.FINAL.name else NEUTRAL for n in names])
    ax.scatter([worst[n] for n in names], y, s=46, color=INK_2, edgecolor="white", linewidth=1.6, zorder=3)
    for yi, n in zip(y, names):
        ax.text(worst[n] + 0.14, yi, f"{mean[n]['mape']:.2f}%  (worst {worst[n]:.2f}%)", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(worst.values()) + 2.1)
    ax.set_xlabel("MAPE on the next two unseen months (%), clean labels")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Patch(color=NEUTRAL, label="Mean of 5 forward folds"),
        Line2D([], [], marker="o", color="none", markerfacecolor=INK_2, markeredgecolor="white", markersize=7, label="Worst fold"),
    ], loc="lower right", ncol=1)
    _head(ax, "The final hybrid is the most accurate and the most stable model",
          "Forward-chaining error by model; the final model is highlighted")
    fig.savefig(C.FIGURES_DIR / "model_comparison.png")
    plt.close(fig)


def main() -> None:
    _style()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(C.METRICS_JSON.read_text())
    train, _, _ = load_raw()
    fig_quote_signal()
    metrics["quarter_end_ramp_by_month_pct"] = {k: round(v, 2) for k, v in fig_quarter_end(train).items()}
    fig_validation_design(metrics)
    fig_model_comparison(metrics)
    C.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"figures written to {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
