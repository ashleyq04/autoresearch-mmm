# AutoResearch Agent Instructions

## Objective

Minimize **validation AIC (Akaike Information Criterion)** to optimize budget allocation insights. AIC is our North Star because it rewards model fit (RMSE) while strictly penalizing feature bloat (k)—essential for our 156-week sample.


## Rules

1. You may **ONLY** modify `model.py`
2. `prepare.py`, `run.py` , and the data/ folder are **FROZEN** — do not touch them
3. The Logic Check: Marketing spend and impression coefficients MUST be non-negative. If run.py reports Logic Check Failed, you must DISCARD and revert immediately.
4. Interpretability: Maintain a linear functional form (Non-Negative Least Squares preferred) to ensure coefficients represent direct ROI.
5. `build_model()` must return an sklearn-compatible estimator (Pipeline preferred)
6. Training + evaluation must complete in **under 60 seconds** on CPU
7. No additional data sources or external downloads

## Workflow

```
1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run:  python run.py "description of change"
5. Check val_rmse in output
6. If improved:  git add model.py && git commit -m "feat: <description>"
7. If worse:     git checkout model.py   (revert)
8. Repeat from step 1
```

```
1. Read current model.py and the results.tsv history.
2. Propose ONE "Marketing-Safe" transformation from the library below.
3. Edit model.py to apply the transformation within the build_model() pipeline.
4. Run: python run.py "applied 2-week lag to TV spend".
5. Evaluate Output:
 - If logic_passed: True AND val_aic is lower than the best-so-far: KEEP (via git commit).
 - If logic_passed: False OR val_aic increased: DISCARD (via git checkout model.py).
6. Repeat to find the next incremental improvement.
```

## Library of "Marketing-Safe" Transformations

The agent should prioritize these statistically sound MMM transformations:
- Time Lags: df['column'].shift(1) to capture delayed effects (1–4 weeks).
- Geometric Decay (Adstock): Capturing the "carryover" effect of past advertising spend.
- Hill Functions: Modeling diminishing returns (saturation) as spend increases.
- Log-Log Transforms: To estimate "Elasticity" of demand.
- Seasonality Controls: Interaction terms with month or "Promo" flags.

## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- No Signature Changes: Do not change the function signature of    `build_model()`; it must always return a fitted estimator.
- No Negative ROI: Never keep a model where spending money appears to decrease revenue
- No Feature Bloat: Avoid adding features that lower $RMSE$ but increase AIC (this is noise-chasing).
- No GPU/External Data: Keep the model CPU-compatible and strictly within the `national_all_channels.csv` context.