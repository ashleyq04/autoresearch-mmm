"""
Current working/champion model for the AutoResearch MMM workflow.

This file holds the latest accepted model inside the allowed
interpretable MMM search space. It is the file the agent edits
during the AutoResearch loop to improve validation RMSE.
"""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_model():
    baseline_features = [
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
        "competitor_sales_control",
        "sentiment_score_control",
        "Promo",
        "week_sin",
        "week_cos",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("geo", OneHotEncoder(drop="first", handle_unknown="ignore"), ["geo"]),
            ("num", "passthrough", baseline_features),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
