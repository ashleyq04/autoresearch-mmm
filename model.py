"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Marketing Model Mix.
The function build_model() must return an sklearn-compatible estimator.
"""


import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline


class BoundedLinearRegression(BaseEstimator, RegressorMixin):
    def _dense(self, X):
        return X.toarray() if sparse.issparse(X) else np.asarray(X)

    def fit(self, X, y):
        X = self._dense(X)
        y = np.asarray(y)

        design = np.column_stack([np.ones(X.shape[0]), X])
        lower_bounds = np.full(design.shape[1], -np.inf)
        upper_bounds = np.full(design.shape[1], np.inf)

        # Preprocessor output is: geo dummies, 10 transformed spend columns, 3 controls.
        geo_feature_count = X.shape[1] - 13
        for idx in range(geo_feature_count, geo_feature_count + 10):
            lower_bounds[idx + 1] = 0.0

        solution = lsq_linear(design, y, bounds=(lower_bounds, upper_bounds), lsmr_tol="auto")
        self.intercept_ = solution.x[0]
        self.coef_ = solution.x[1:]
        return self

    def predict(self, X):
        X = self._dense(X)
        return self.intercept_ + X @ self.coef_


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("geo", OneHotEncoder(drop="first", handle_unknown="ignore"), ["geo"]),
            (
                "log_spend",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                [
                    "Channel0_spend",
                    "Channel1_spend",
                    "Channel2_spend",
                    "Channel3_spend",
                    "Channel4_spend",
                    "Channel0_spend_lag1",
                    "Channel1_spend_lag1",
                    "Channel2_spend_lag1",
                    "Channel3_spend_lag1",
                    "Channel4_spend_lag1",
                ],
            ),
        ],
        remainder="passthrough",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", BoundedLinearRegression()),
    ])
