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


class BoundedRidgeRegression(BoundedLinearRegression):
    def __init__(self, nonnegative_start_idx, nonnegative_feature_count, alpha=1.0):
        super().__init__(nonnegative_start_idx, nonnegative_feature_count)
        self.alpha = alpha

    def fit(self, X, y):
        X = self._dense(X)
        y = np.asarray(y)

        design = np.column_stack([np.ones(X.shape[0]), X])
        lower_bounds = np.full(design.shape[1], -np.inf)
        upper_bounds = np.full(design.shape[1], np.inf)

        start = self.nonnegative_start_idx
        stop = start + self.nonnegative_feature_count
        lower_bounds[start + 1:stop + 1] = 0.0

        penalty = np.sqrt(self.alpha) * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        augmented_design = np.vstack([design, penalty])
        augmented_target = np.concatenate([y, np.zeros(design.shape[1])])

        solution = lsq_linear(
            augmented_design,
            augmented_target,
            bounds=(lower_bounds, upper_bounds),
            lsmr_tol="auto",
        )
        self.intercept_ = solution.x[0]
        self.coef_ = solution.x[1:]
        return self


class AddInteractionFeatures(BaseEstimator):
    def __init__(self, interactions, insert_after_idx):
        self.interactions = list(interactions)
        self.insert_after_idx = insert_after_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        interaction_columns = []
        for left_idx, right_idx in self.interactions:
            interaction_columns.append((X[:, left_idx] * X[:, right_idx]).reshape(-1, 1))

        left_block = X[:, :self.insert_after_idx]
        right_block = X[:, self.insert_after_idx:]
        return np.hstack([left_block, *interaction_columns, right_block])

    def get_feature_names_out(self, input_features=None):
        input_features = list(input_features)
        interaction_names = [
            f"{input_features[left_idx]}_x_{input_features[right_idx]}"
            for left_idx, right_idx in self.interactions
        ]
        return np.array(
            input_features[:self.insert_after_idx] + interaction_names + input_features[self.insert_after_idx:],
            dtype=object,
        )


def build_model():
    spend_features = []

    adstock_features = [
        "Channel0_spend_adstock_07",
        "Channel1_spend_adstock_07",
        "Channel2_spend_adstock_07",
        "Channel4_spend_adstock_03",
    ]

    control_features = [
        "competitor_sales_control",
        "Promo",
        "week_sin",
        "week_cos",
    ]

    numeric_features = spend_features + adstock_features + control_features
    # One-hot geo columns come first, followed by numeric_features in order.
    geo_feature_count = 39
    media_feature_count = len(spend_features) + len(adstock_features)
    interaction_pairs = [
        (
            geo_feature_count + numeric_features.index("Channel4_spend_adstock_03"),
            geo_feature_count + numeric_features.index("Promo"),
        ),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("geo", OneHotEncoder(drop="first", handle_unknown="ignore"), ["geo"]),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "interactions",
            AddInteractionFeatures(
                interactions=interaction_pairs,
                insert_after_idx=geo_feature_count + media_feature_count,
            ),
        ),
        (
            "model",
            BoundedLinearRegression(
                nonnegative_start_idx=geo_feature_count,
                nonnegative_feature_count=media_feature_count + len(interaction_pairs),
            ),
        ),
    ])
