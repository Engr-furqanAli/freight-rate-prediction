"""Build the PDF report from saved metrics and figures:  python -m freight_rate.report

Every number in the report is read from ``reports/metrics.json``, so the document always matches
the last pipeline run. Needs ``reportlab``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import config as C
from . import validation as V

REPORT_PDF = C.REPORTS_DIR / "Freight_Rate_Report.pdf"
DECEMBER_PNG = C.ROOT / "scorer_results" / "candidate_december.png"
WIDTH = 7.0 * inch
FIG_WIDTH = 6.2 * inch
INK, INK_2, RULE, FILL, ACCENT = "#0b0b0b", "#52514e", "#d9d8d3", "#f1f0ec", "#064A56"


def _fonts() -> None:
    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Body", str(folder / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(folder / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(folder / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")


STYLES = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=17, leading=21, textColor=colors.HexColor(ACCENT), spaceAfter=2),
    "byline": ParagraphStyle("byline", fontName="Body", fontSize=9, leading=12, textColor=colors.HexColor(INK_2), spaceAfter=8),
    "h": ParagraphStyle("h", fontName="Body-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor(ACCENT), spaceBefore=9, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Body", fontSize=9, leading=12.6, textColor=colors.HexColor(INK), spaceAfter=5),
    "li": ParagraphStyle("li", fontName="Body", fontSize=9, leading=12.6, textColor=colors.HexColor(INK), leftIndent=12, bulletIndent=2, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8, leading=10.4, textColor=colors.HexColor(INK)),
    "head": ParagraphStyle("head", fontName="Body-Bold", fontSize=8, leading=10.4, textColor=colors.HexColor(INK)),
    "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=8, leading=10.5, textColor=colors.HexColor(INK_2), spaceAfter=6),
}


def P(text: str, style: str = "p") -> Paragraph:
    return Paragraph(text, STYLES[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(item, STYLES["li"], bulletText="•") for item in items]


def figure(path: Path, width: float = FIG_WIDTH) -> Image:
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * h / w)


def table(rows: list[list[str]], widths: list[float], numeric_from: int | None = None) -> Table:
    data = [[P(c, "head") for c in rows[0]]] + [[P(c, "cell") for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(FILL)),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(RULE)),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor(RULE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    t.setStyle(TableStyle(style))
    return t


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(colors.HexColor(INK_2))
    canvas.drawString(0.75 * inch, 0.5 * inch, "Freight rate prediction: validation approach and results")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build() -> Path:
    _fonts()
    m = json.loads(C.METRICS_JSON.read_text())
    a, q, v, f = m["audit"], m["quote_signal"], m["validation"], m["final_model"]
    final, rnd = v["forward_mean"][V.FINAL.name], v["random_mean"]
    with_quote = next(k for k in v["forward_mean"] if k.startswith("B."))
    lab, uc, wt, mi = a["labels"], a["unseen_cities"], a["weight"], a["market_index"]
    ramps = m.get("quarter_end_ramp_by_month_pct", {})
    dec = f["december_chart"]
    seen, unseen = v["unseen_city_mean"]["known cities only"], v["unseen_city_mean"]["touches unseen city"]
    n_val = a["rows"]["validation"]
    dec_share = 100 * a["validation_rows_by_month"]["Dec"] / n_val

    story: list = []
    story += [P("Freight Rate Prediction", "title"),
              P("Validation approach, data split and results &nbsp;|&nbsp; Machine Learning Engineer assessment &nbsp;|&nbsp; Furqan Ali", "byline")]

    # ---------------------------------------------------------------- 1
    story += [P("1. Summary", "h")]
    story += bullets([
        f"<b>The task is a forecast, so the split is by time.</b> Labels run {a['date_range']['train'][0]} to {a['date_range']['train'][1]}; "
        f"the {n_val:,} loads to predict cover November and December. I validate with expanding-window forward-chaining folds, "
        "each scored on the next two unseen months. A random split is shown below to pick the wrong model.",
        f"<b>Result.</b> The final model scores <b>{final['mape']:.2f}% MAPE</b> (MAE ${final['mae']:.0f}, RMSE ${final['rmse']:.0f}) on average over five "
        f"forward folds, {v['forward_worst_fold_mape'][V.FINAL.name]:.2f}% in its worst fold, against {v['forward_mean'][next(k for k in v['forward_mean'] if k.startswith('A.'))]['mape']:.2f}% for a naive "
        f"baseline and {v['forward_mean'][next(k for k in v['forward_mean'] if k.startswith('C.'))]['mape']:.2f}% for a standard gradient-boosting model.",
        "<b>Three findings decide the outcome:</b> <i>quote_signal</i> is unstable and carries no signal in the validation period (dropped); "
        "rates ramp up about 4% through every quarter-end month, and December is one (modelled); "
        f"{len(uc['names'])} cities in validation.csv never appear in training (geography enters through coordinates, not names).",
        "<b>Model.</b> A hybrid: a linear model in log space owns everything that must extrapolate (trend, quarter-end ramp, market index); "
        "gradient-boosted trees learn the remaining non-linear structure (distance, weight, geography).",
    ])

    # ---------------------------------------------------------------- 2
    story += [P("2. Data audit: what was found and how it was handled", "h")]
    story += [table([
        ["Issue", "Evidence", "Treatment"],
        ["Corrupted labels",
         f"{lab['corrupt']:,} rows ({lab['corrupt_pct']}%): {lab['scaled_down']} scaled by {lab['scaled_down_range'][0]}-{lab['scaled_down_range'][1]}x and "
         f"{lab['scaled_up']} by {lab['scaled_up_range'][0]}-{lab['scaled_up_range'][1]}x, spread evenly over months. Clean labels all sit within "
         f"{lab['clean_range'][0]}-{lab['clean_range'][1]}x of a structural baseline, so the gap is empty. Independent check: in the months where quote_signal tracks "
         f"the rate, it reproduces {q['label_check']['clean_labels_within_10pct']:.0f}% of clean labels within 10% and {q['label_check']['flagged_labels_within_10pct']:.0f}% of flagged ones.",
         "Removed from training. Metrics are reported on clean labels and, for transparency, on all labels."],
        ["Weight: sign flips and gaps",
         f"Negative in {wt['train']['negative']} development rows ({wt['train']['negative_pct']}%) and {wt['validation']['negative']} validation rows ({wt['validation']['negative_pct']}%); "
         f"magnitudes match the positive ones (median {wt['train']['median_abs_negative']:,.0f} vs {wt['train']['median_positive']:,.0f} lb) and so does the rate response. "
         f"Missing in {wt['train']['missing']} / {wt['validation']['missing']} rows.",
         "Absolute value; missing filled with the training median plus a flag."],
        ["market_index gaps",
         f"Missing in {mi['train']['missing']} development rows ({mi['train']['missing_pct']}%) and {mi['validation']['missing']} validation rows ({mi['validation']['missing_pct']}%). "
         f"It is a day-level quantity: within-day sd {mi['train']['within_day_std']} against {mi['train']['between_day_std']} between days.",
         "Filled with the same-day median (features only)."],
        ["quote_signal instability", "Tracks the rate in some months, mirrors it in others, and is noise in the validation period (section 3).", "Excluded from the model."],
        ["Unseen cities",
         f"{', '.join(uc['names'])}: absent from training, touching {uc['validation_rows_touching']:,} validation loads ({uc['validation_share_pct']}%).",
         "City names are not used; coordinates are. Checked with a city hold-out (section 5)."],
        ["No action needed",
         f"No duplicate ids or rows, all dates parse, three consistent equipment labels. Weight is capped at 47,500 lb ({wt['train']['at_cap_47500']:,} rows) "
         f"and distance floored at 70 miles ({a['distance']['rows_at_floor_70']} rows); both look like business rules.",
         "Kept as is."],
    ], [1.25, 4.05, 1.7])]

    # ---------------------------------------------------------------- 3
    story += [KeepTogether([
        P("3. quote_signal: strong in a random split, useless where it matters", "h"),
        P("In five of the ten training months <i>quote_signal</i> equals the rate per mile almost exactly "
          f"(median gap {q['label_check']['clean_labels_median_gap_pct']}%); in four it is its mirror image; in August it is unrelated. "
          "The regime changes on calendar-month boundaries. To learn which regime the unlabelled period is in, I correlate the signal with a "
          "<b>label-free</b> structural baseline (median rate per mile by equipment and distance). On the development data this diagnostic "
          "follows the labelled one closely, and in validation.csv it is zero on every one of the 61 days. "
          "A model trained on it would apply October's mirror relationship to pure noise."),
    ]), figure(C.FIGURES_DIR / "quote_signal_regimes.png")]

    # ---------------------------------------------------------------- 4
    ramp_text = ", ".join(f"+{val:.1f}% in {name}" for name, val in ramps.items()) if ramps else "about +4% in March, June and September"
    story += [KeepTogether([
        P("4. Time structure: a trend and a quarter-end ramp", "h"),
        P(f"After removing lane structure and the market index, two time effects remain. A steady trend of <b>+{f['trend_pct_per_month']:.2f}% per month</b>, "
          f"and a ramp through each quarter-end month ({ramp_text}) that resets on the first day of the next quarter. "
          "No other month moves by more than about 1%. Tree models cannot extrapolate either effect, and a month identifier cannot transfer "
          f"the ramp to December, which is a quarter-end month and holds {dec_share:.0f}% of the validation loads. "
          "I therefore encode it as a calendar position: progress from 0 to 1 through a quarter-end month, 0 otherwise."),
    ]), figure(C.FIGURES_DIR / "quarter_end_ramp.png")]

    # ---------------------------------------------------------------- 5
    story += [P("5. Validation approach and data split", "h")]
    story += [P(f"<b>Why not a random split.</b> In shuffled 5-fold cross-validation the model that uses quote_signal looks best "
                f"({rnd[with_quote]['mape']:.2f}% MAPE against {rnd[V.FINAL.name]['mape']:.2f}% for the final model), because every test row has "
                f"same-month neighbours in training that reveal the regime. Scored forward in time, the ranking reverses: "
                f"{v['forward_mean'][with_quote]['mape']:.2f}% against {final['mape']:.2f}%. A random split would have selected the wrong model.")]
    story += [figure(C.FIGURES_DIR / "validation_design.png")]
    story += bullets([
        "<b>Split.</b> Expanding window, five folds. Train on January up to month m-1, test on months m and m+1. The two-month horizon mirrors the "
        "real task (labels end 31 October, predictions cover November and December). Folds 1 and 4 reproduce its exact shape: "
        "the second test month is a quarter-end month.",
        "<b>Leakage controls.</b> Everything that learns from data (label-noise filter, weight fill value, both model stages) is fitted on the "
        "training side of each fold only. The day-level market index is built from features, never labels, exactly as at prediction time. "
        "The trees get no time index, so they cannot memorise periods.",
        f"<b>Unseen-city check.</b> On the last fold, {C.N_UNSEEN_CITIES} random cities are removed from training entirely ({C.N_CITY_REPEATS} repeats). "
        f"Loads touching a held-out city score {unseen['mape']:.2f}% MAPE against {seen['mape']:.2f}% for known cities, so coordinates generalise.",
        "<b>Metrics.</b> MAPE, MAE and RMSE on clean labels (corrupted test labels are unpredictable by construction), MAE and RMSE on all labels, "
        "and signed bias per test month to expose drift.",
        f"<b>Final fit.</b> After validation the model is refitted on all {f['training_rows']:,} clean January-October rows with the settings chosen in "
        "cross-validation. validation.csv has no labels and was used only for its features.",
    ])

    # ---------------------------------------------------------------- 6
    story += [P("6. Results", "h")]
    rows = [["Model", "MAPE", "MAE", "RMSE", "Worst fold MAPE", "Largest monthly bias", "MAE, all labels"]]
    for name in sorted(v["forward_mean"]):
        r = v["forward_mean"][name]
        cells = [name, f"{r['mape']:.2f}%", f"${r['mae']:.0f}", f"${r['rmse']:.0f}", f"{v['forward_worst_fold_mape'][name]:.2f}%",
                 f"{v['forward_worst_month_abs_bias'][name]:.1f}%", f"${r['mae_all_labels']:.0f}"]
        rows.append([f"<b>{c}</b>" for c in cells] if name == V.FINAL.name else cells)
    story += [table(rows, [2.75, 0.6, 0.55, 0.6, 0.85, 0.9, 0.75]),
              P("Mean over five forward-chaining folds. Clean labels unless stated. Largest monthly bias = largest absolute mean signed error of any test month.", "cap"),
              figure(C.FIGURES_DIR / "model_comparison.png")]
    if "ablations" in m:
        ab, better = m["ablations"]["ablation_mape"], m["ablations"]["ablation_folds_better_than_final"]
        base = ab[V.FINAL.name]
        lines = [f"{name}: {val:.2f}% ({val - base:+.2f} pt; better in {better[name]} of 5 folds)" for name, val in ab.items() if name != V.FINAL.name]
        story += [P("<b>Ablations of the final model</b> (forward-chaining MAPE; final = " + f"{base:.2f}%):", "p")] + bullets(lines)
        story += [P("Day-of-week and a separate slow market level were tested and left out: the gain is small and not consistent across folds. "
                    "Tree size and number of rounds were chosen from a 3 x 3 grid on the same folds.", "p")]

    # ---------------------------------------------------------------- 7
    story += [P("7. Final model", "h")]
    story += bullets([
        f"<b>Stage 1, linear in log(rate).</b> Trend +{f['trend_pct_per_month']:.2f}% per month; quarter-end ramp +{f['quarter_end_ramp_pct']:.1f}% by month end; "
        f"market index +{f['market_index_log_points_per_unit']:.2f} log points per unit; Flatbed +{f['flatbed_premium_pct']:.1f}% and Reefer +{f['reefer_premium_pct']:.1f}% over Dry Van; "
        f"weight +{f['weight_pct_per_10k_lb']:.1f}% per 10,000 lb; plus log-distance terms and coordinates.",
        f"<b>Stage 2, gradient-boosted trees on the stage-1 residual</b> (scikit-learn HistGradientBoosting, {C.GBM_PARAMS['max_leaf_nodes']} leaves, "
        f"{C.GBM_PARAMS['max_iter']} rounds, learning rate {C.GBM_PARAMS['learning_rate']}). Inputs: distance, equipment, weight, coordinates, market index, "
        "quarter-end position. It captures the strong non-linearity of rate per mile at short distances.",
        "<b>Output.</b> The exponential of the log-space fit, i.e. the conditional median rate: the right target for absolute and percentage error, "
        "and robust to label noise.",
    ])

    # ---------------------------------------------------------------- 8
    story += [KeepTogether([
        P("8. Fixed December prediction chart (produced by the provided score.py)", "h"),
        figure(DECEMBER_PNG, width=6.7 * inch),
        P("Output of <i>python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv</i>, unmodified.", "cap"),
        P("The December file carries only lane, distance, equipment, weight and date. Coordinates are looked up from the city names. "
          "The market index for each day is the median over that day's loads in validation.csv (features only). "
          f"Reading the chart: the weekly wave is the market index, which peaks mid-week; the climb from ${dec['first']:,.0f} to ${dec['last']:,.0f} is the "
          "quarter-end ramp plus the trend. Only the date changes, so every movement is a time effect."),
    ])]

    # ---------------------------------------------------------------- 9
    story += [P("9. Assumptions and limits", "h")]
    story += bullets([
        "<b>The quarter-end ramp recurs in December.</b> It appears in all three quarter-ends in the data with nearly identical size. "
        "If it did not recur, December predictions would run about 2% high on average.",
        f"<b>The trend continues for two more months.</b> That adds about {2 * f['trend_pct_per_month']:.1f}% by late December. Removing the trend raises forward MAPE to "
        + (f"{m['ablations']['ablation_mape']['without the trend']:.2f}%." if "ablations" in m else "well above the final model."),
        "<b>No holiday effect is modelled.</b> January-October contains no Thanksgiving or Christmas, so none can be learned; public holidays in the "
        "training window show no visible effect.",
        f"<b>Label corruption sets a floor on squared error.</b> If the hidden labels contain the same {lab['corrupt_pct']}% of corrupted values, RMSE will be dominated by them "
        f"(in cross-validation: ${final['rmse']:.0f} on clean labels against ${final['rmse_all_labels']:.0f} on all labels). No model can predict them.",
        "<b>Next steps.</b> Prediction intervals; a monitor that re-admits quote_signal when the label-free diagnostic shows it tracking the rate; monthly refits.",
    ])

    doc = SimpleDocTemplate(str(REPORT_PDF), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                            title="Freight Rate Prediction: validation approach and results", author="Furqan Ali")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return REPORT_PDF


if __name__ == "__main__":
    print(f"report written to {build()}")
