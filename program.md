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
7. A model is only eligible to be kept if all marketing spend effects remain nonnegative.

## Acceptance Criteria

A model change can be kept only if **both** conditions hold:

1. Validation RMSE improves over the best prior non-discard result.
2. All marketing spend and lagged spend coefficients are nonnegative.

If either condition fails, discard the change and revert `model.py`.

Interpretation rule:
- Negative coefficients on `Channel*_spend` or `Channel*_spend_lag1` are not allowed.
- If a transformation changes the feature names, the corresponding transformed spend features must still remain nonnegative.
- Prefer model specifications that enforce this structurally rather than checking it after fitting.

## Workflow

```
1. Start each new AutoResearch session with `python run.py --baseline "baseline description"`.
   This creates a new numbered session log such as `results_1.tsv`, `results_2.tsv`, etc.
2. Read current model.py and the active session results history.
3. Propose ONE "Marketing-Safe" transformation from the library below.
4. Edit model.py to apply the transformation within the build_model() pipeline.
5. Run: python run.py "description of change"
6. Check the logged status in the active `results_<n>.tsv` file and inspect the printed coefficients.
7. If RMSE improved but any spend-related coefficient is negative, treat the run as `discard` and revert `model.py`.
8. Only if both RMSE improves and spend-related coefficients stay nonnegative, keep the change.
9. Repeat from step 2 for the next idea.
10. Run `python prepare.py` at the end of the session to generate the matching `performance_<n>.png`.
```

Use `python run.py --baseline "baseline description"` for the first baseline run of each session.
For later experiments, `run.py` automatically marks the result as `keep` or `discard`
by comparing the new RMSE with the best prior non-discard result.
Each logged row also includes total runtime and training runtime for that iteration.

## Library of Marketing-Safe Transformations

The agent should prioritize transformations that preserve a linear model form in the final estimator:

- Feature selection over the frozen MMM feature library exposed by `prepare.py`
- Pairwise interaction terms between existing predictors
- Log transforms of spend variables
- Simple feature selection among the existing predictors
- Robust handling of outliers through fixed preprocessing transforms

Baseline-vs-search-space rule:
- Treat the current `model.py` as a simple baseline specification.
- `prepare.py` may expose richer frozen features such as seasonality, adstock, and transformed spend.
- The agent may use those frozen features by editing `model.py`, but must not modify `prepare.py` during the loop.

Do not use transformations that require direct access to raw time or geo columns unless those features are already provided by prepare.py.

## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not try non-linear models
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- No Signature Changes: Do not change the function signature of `build_model()`; it must return an unfitted sklearn-compatible estimator.
- No Negative ROI: Never keep a model where spending money appears to decrease revenue
- No GPU/External Data: Keep the model CPU-compatible and strictly within the `geo_all_channels.csv` context.
