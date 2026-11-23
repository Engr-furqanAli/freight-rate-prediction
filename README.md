# Freight Rate Prediction

Solution to the Spotter Machine Learning Engineer assessment: predict `posted_rate` for 12,000
loads in November-December 2025 from labelled loads covering January-October 2025.

| Deliverable | Location |
|---|---|
| Predictions (`load_id,predicted_rate`, 12,000 rows) | [`validation_predictions.csv`](validation_predictions.csv) |
| Report: validation approach, split, December chart | [`reports/Freight_Rate_Report.pdf`](reports/Freight_Rate_Report.pdf) |
| Fixed December chart from the provided `score.py` | [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) |
| Completed December inputs | [`data/december_chart_inputs.csv`](data/december_chart_inputs.csv) |

## Result

Forward-chaining cross-validation, five folds, each scored on the **next two unseen months**
(the same shape as the real task: labels end 31 October, predictions cover November and December).

| Model | MAPE | MAE | RMSE | Worst fold |
|---|---|---|---|---|
| A. Naive: median $/mile by equipment x distance | 4.36% | $102 | $140 | 6.03% |
| B. GBM with `quote_signal` + calendar features | 3.13% | $77 | $119 | 4.75% |
| C. GBM without `quote_signal` + calendar features | 2.69% | $65 | $94 | 4.23% |
| D. Hybrid: trend only | 2.37% | $56 | $80 | 4.38% |
| **E. Hybrid: trend + quarter-end ramp (final)** | **1.49%** | **$35** | **$51** | **1.76%** |

Scored on clean labels; see the report for all-label figures. Every number is reproduced by
`python -m freight_rate.run` and stored in [`reports/metrics.json`](reports/metrics.json).

## What the data showed

1. **This is a forecast.** Training covers Jan-Oct, the loads to predict are Nov-Dec. A random
   split is the wrong tool: it ranks the `quote_signal` model best (1.05% vs 1.33% MAPE), while
   scoring forward in time reverses the ranking (3.13% vs 1.49%).
2. **`quote_signal` is unstable.** It equals the rate per mile in five months, is its mirror
   image in four, and is unrelated in one. A label-free diagnostic shows it is pure noise on
   every day of the validation period. It is excluded.
3. **Rates ramp about 4% through each quarter-end month** (March, June, September) and reset on
   the first day of the next quarter. December is a quarter-end month and holds 51% of the loads
   to predict, so the ramp is modelled as a calendar position that carries over.
4. **A steady trend of +0.6% per month.** Tree models cannot extrapolate it, which is why the
   time components live in a linear stage.
5. **Eight cities in `validation.csv` never appear in training** (12% of its loads). Geography
   enters through coordinates rather than city names; a held-out-cities test confirms this
   generalises (1.57% vs 1.44% MAPE).
6. **Data quality.** 1.41% of labels are corrupted by a 0.17-0.47x or 2.1-5.3x factor (removed
   from training); weights have sign flips and gaps (absolute value, median fill); `market_index`
   has gaps (same-day median).

## Model

`log(rate) = linear stage + gradient-boosted residual`

- **Linear stage** (extrapolates): equipment, log-distance, weight, coordinates, market index,
  trend, quarter-end ramp.
- **Boosted trees** (scikit-learn `HistGradientBoostingRegressor`) on the residual: non-linear
  effects of distance, weight and geography. The trees get no time index.
- Prediction is the exponential of the fit, i.e. the conditional median rate.

## Run it

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

python -m freight_rate.run          # audit -> validation -> final fit -> prediction files (~2 min)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

python -m freight_rate.plots        # report figures
python -m freight_rate.report       # reports/Freight_Rate_Report.pdf
python -m pytest -q                 # 9 tests
```

Options: `python -m freight_rate.run --skip-validation` only refits and rewrites the prediction
files (~10 s); `--ablations` also reruns the feature ablations and the hyper-parameter grid
(adds ~3 min).

## Layout

```
src/freight_rate/
  config.py       paths, cleaning thresholds, feature lists, model settings
  data.py         loading, cleaning rules, label-noise filter
  quality.py      data audit and quote_signal regime diagnostics
  features.py     feature engineering, including the quarter-end ramp
  model.py        HybridRateModel
  validation.py   forward-chaining CV, unseen-city hold-out, random-split contrast, ablations
  run.py          end-to-end pipeline (CLI)
  plots.py        report figures
  report.py       PDF report built from reports/metrics.json
tests/            unit tests for cleaning, features, the label filter, the model and the outputs
data/             provided CSV files (december_chart_inputs.csv is filled in by the pipeline)
reports/          report PDF, figures, metrics.json and per-fold CSVs
score.py          provided scorer, unmodified
docs/             original assessment brief
```

## Assumptions

- The quarter-end ramp recurs in December. It appears in all three quarter-ends in the data with
  nearly the same size; if it did not recur, December predictions would run about 2% high.
- The trend continues for two more months (about +1.2% by late December).
- No holiday effect is modelled: January-October contains no Thanksgiving or Christmas.
- The December chart file has no `market_index`, so each day uses the median index of that day's
  loads in `validation.csv` (features only, no labels).
