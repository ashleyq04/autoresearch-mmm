# AutoResearch Agent Instructions

## Objective

Minimize **validation RMSE** to optimize budget allocation insights while maintaining a linear model to preserve interpretability.

## Rules

1. You may **ONLY** modify `model.py`
2. `prepare.py`, `run.py` , and the data/ folder are **FROZEN** — do not touch them
3. Interpretability: Maintain a linear functional form to ensure coefficients represent direct ROI.
4. `build_model()` must return an sklearn-compatible estimator (Pipeline preferred)
5. Training + evaluation must complete in **under 60 seconds** on CPU
6. No additional data sources or external downloads

## Workflow

```
1. Read current model.py and the results.tsv history.
2. Propose ONE "Marketing-Safe" transformation from the library below.
3. Edit model.py to apply the transformation within the build_model() pipeline.
4. Run: python run.py "description of change"
5. Check the logged status in results.tsv.
6. If status=keep: keep the model.py change.
7. If status=discard: revert model.py to the previous version.
8. Repeat from step 1.
```

Use `python run.py --baseline "baseline description"` for the first baseline run.
For later experiments, `run.py` automatically marks the result as `keep` or `discard`
by comparing the new RMSE with the best prior non-discard result.

## Library of Marketing-Safe Transformations

The agent should prioritize transformations that preserve a linear model form in the final estimator:

- Pairwise interaction terms between existing predictors
- Log transforms of spend variables
- Simple feature selection among the existing predictors
- Robust handling of outliers through fixed preprocessing transforms

Do not use transformations that require direct access to raw time or geo columns unless those features are already provided by prepare.py.

## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not try non-linear models
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- No Signature Changes: Do not change the function signature of `build_model()`; it must return an unfitted sklearn-compatible estimator.
- No Negative ROI: Never keep a model where spending money appears to decrease revenue
- No GPU/External Data: Keep the model CPU-compatible and strictly within the `geo_all_channels.csv` context.
