"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Marketing Model Mix.
The function build_model() must return an sklearn-compatible estimator.
"""


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("geo", OneHotEncoder(handle_unknown="ignore"), ["geo"]),
        ],
        remainder="passthrough"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0)),
    ])



'''
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    return Pipeline([
        # ("scaler", StandardScaler()), removing scaling to keep in og, interpretable units 
        ("model", Ridge(alpha=1.0)),
    ])
'''