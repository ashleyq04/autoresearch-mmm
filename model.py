"""
Current working/champion model for the AutoResearch MMM workflow.

This file holds the latest accepted model inside the allowed
interpretable MMM search space. It is the file the agent edits
during the AutoResearch loop to improve validation RMSE.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class BoundedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, nonnegative_start_idx, nonnegative_feature_count):
        self.nonnegative_start_idx = nonnegative_start_idx
        self.nonnegative_feature_count = nonnegative_feature_count

    def _dense(self, X):
        return X.toarray() if sparse.issparse(X) else np.asarray(X)

    def fit(self, X, y):
        X = self._dense(X)
        y = np.asarray(y)

        design = np.column_stack([np.ones(X.shape[0]), X])
        lower_bounds = np.full(design.shape[1], -np.inf)
        upper_bounds = np.full(design.shape[1], np.inf)

        start = self.nonnegative_start_idx
        stop = start + self.nonnegative_feature_count
        lower_bounds[start + 1:stop + 1] = 0.0

        solution = lsq_linear(design, y, bounds=(lower_bounds, upper_bounds), lsmr_tol="auto")
        self.intercept_ = solution.x[0]
        self.coef_ = solution.x[1:]
        return self

    def predict(self, X):
        X = self._dense(X)
        return self.intercept_ + X @ self.coef_


def build_model():
    spend_features = [
        "Channel3_spend",
    ]

    adstock_features = [
        "Channel0_spend_adstock_07",
        "Channel1_spend_adstock_07",
        "Channel2_spend_adstock_07",
        "Channel3_spend_adstock_03",
        "Channel4_spend_adstock_03",
    ]

    control_features = [
        "competitor_sales_control",
        "sentiment_score_control",
        "Promo",
        "week_sin",
        "week_cos",
    ]

    numeric_features = spend_features + adstock_features + control_features

    preprocessor = ColumnTransformer(
        transformers=[
            ("geo", OneHotEncoder(drop="first", handle_unknown="ignore"), ["geo"]),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    # One-hot geo columns come first, followed by numeric_features in order.
    geo_feature_count = 39

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            BoundedLinearRegression(
                nonnegative_start_idx=geo_feature_count,
                nonnegative_feature_count=len(spend_features) + len(adstock_features),
            ),
        ),
    ])
