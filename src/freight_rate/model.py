"""Hybrid rate model: a linear model that extrapolates + boosted trees for non-linear structure."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from . import config as C


class HybridRateModel:
    """log(rate) = stage-1 linear model + stage-2 gradient-boosted residual.

    Why two stages
    --------------
    The scoring period (Nov-Dec) lies entirely beyond the training window, and rates carry a
    steady upward trend plus a quarter-end ramp. Tree ensembles cannot extrapolate either: past
    the last training date they hold the last level. So the components that must extrapolate
    (trend, quarter-end ramp, market index) live in a linear model in log space, and the trees
    only learn what is left: the non-linear effect of distance, weight and geography. The
    trees deliberately get no time index.

    Predictions are ``exp`` of the log-space fit, i.e. the conditional median of the rate, which
    is the right target for absolute / percentage error and robust to residual label noise.

    Passing ``linear_features=None`` gives a plain GBM on log rate-per-mile, used for ablations.
    """

    def __init__(
        self,
        linear_features: Sequence[str] | None = C.LINEAR_FEATURES,
        gbm_features: Sequence[str] | None = C.GBM_FEATURES,
        gbm_params: dict | None = None,
    ) -> None:
        self.linear_features = list(linear_features) if linear_features else None
        self.gbm_features = list(gbm_features) if gbm_features else None
        self.gbm_params = {**C.GBM_PARAMS, **(gbm_params or {})}

    # ------------------------------------------------------------------------------------
    def _stage1(self, X: pd.DataFrame) -> np.ndarray:
        if self.linear_features:
            return self.linear_.predict(X[self.linear_features])
        return X["log_distance"].to_numpy() + self.offset_

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "HybridRateModel":
        log_y = np.log(np.asarray(y, dtype=float))
        if self.linear_features:
            self.linear_ = LinearRegression().fit(X[self.linear_features], log_y)
        else:
            self.offset_ = float(np.mean(log_y - X["log_distance"].to_numpy()))
        if self.gbm_features:
            residual = log_y - self._stage1(X)
            self.gbm_ = HistGradientBoostingRegressor(**self.gbm_params).fit(X[self.gbm_features], residual)
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        out = self._stage1(X)
        if self.gbm_features:
            out = out + self.gbm_.predict(X[self.gbm_features])
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X))

    def linear_effects(self) -> dict[str, float]:
        """Stage-1 coefficients (log points per unit of each feature), for interpretation."""
        if not self.linear_features:
            return {}
        return {name: float(c) for name, c in zip(self.linear_features, self.linear_.coef_)}
