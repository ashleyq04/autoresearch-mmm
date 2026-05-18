# AutoResearch Project: Marketing Mix Modeling
Project Owner: Ashley Qiu

A minimal, CPU-only AutoResearch project for **STAT 390**.

---

## Project Goal

This repository is designed as an **interpretable AutoResearch marketing mix modeling (MMM) workflow** that other marketers can adapt to their own data.

The project has two levels of purpose:

- **Repo-level purpose:** provide a reusable, interpretable modeling workflow for understanding how marketing spend relates to revenue.
- **Agent-level objective:** improve validation RMSE while staying inside a constrained, interpretable MMM model class.

In other words, RMSE is the optimization target for the agent, while interpretability and business usability are the design goals of the repository.

## Problem

Predict geo-level revenue from marketing spend and control variables.
**Metric**: validation RMSE (lower is better).
**Data**: `data/geo_all_channels.csv`.

## Project Structure

```
autoresearch-mmm/
├── prepare.py          # FROZEN — data loading, evaluation metric, plotting
├── baseline_model.py   # Simple interpretable reference specification
├── model.py            # Current best editable/recommended interpretable model
├── reset_model.py      # Restore model.py back to the canonical baseline
├── run.py              # Run a single experiment and auto-label keep/discard
├── program.md          # Agent instructions and search constraints
├── error_log.md        # Running taxonomy of recurring model failures
├── results_<n>.tsv     # Session experiment log (auto-generated)
├── performance_<n>.png # Session performance plot (auto-generated)
├── results_cumulative.tsv    # Curated combined results across selected sessions
└── performance_cumulative.png # Curated combined performance plot
```

**Key loop rule**: during AutoResearch, the agent may only modify `model.py`. Everything else is frozen.

**Key distinction**:
- `baseline_model.py` is the stable reference model for comparison and reuse.
- `model.py` is the current best model found within the allowed interpretable MMM search space.

`error_log.md` stores a concise running taxonomy of recurring failure modes observed across AutoResearch sessions. It summarizes patterns and lessons rather than duplicating the full experiment log in `results_<n>.tsv`.

## Agent Rule Update

The agent search rules in [program.md](/Users/ashley/Documents/NU/Fourth_Year/STAT390/autoresearch-mmm/program.md) were expanded **after Session 6**.

Starting in **Session 7**, the agent is allowed to search a slightly broader but still interpretable MMM space inside `model.py`, including:
- more systematic feature subset search over the frozen feature library
- constrained regularized linear specifications
- a small number of business-safe promotion-media interactions
- fixed outlier-robust preprocessing

This update does **not** reset the project baseline:
- `baseline_model.py` remains the historical simple reference model
- `model.py` remains the current working champion
- `prepare.py`, the dataset, the split, and the evaluation metric remain frozen

As a result, Sessions 1-6 should be interpreted as runs under the original narrower agent rules, while Session 7 onward uses the revised search guidance.

Session 7 is also a transition point in a second sense: it was the first session allowed to try custom promotion-media interaction features inside `model.py`. That session found a better-RMSE model, but it also showed that a custom post-preprocessor feature can make coefficient reporting harder to audit with the existing frozen `run.py` output. For that reason, the agent rules were tightened again after Session 7 so future sessions must preserve a clear, recoverable mapping between each final engineered feature and its coefficient before a model can be kept.

## Reporting Note

On May 18, 2026, `run.py` was updated to improve coefficient reporting for models that add engineered features after preprocessing (such as sparse promotion-media interaction terms). This change was a reporting and interpretability fix only: it did **not** modify the dataset, train/validation split, evaluation metric, or model search space, so previously logged validation RMSE and keep/discard decisions remain comparable across sessions.

Earlier sessions should therefore be interpreted as having fully valid performance results, but potentially less reliable printed coefficient-name alignment for models with post-preprocessor engineered features.

---

## How to Run the Agent Loop

### Quick start (copy-paste this prompt into your agent)

```
Read program.md for your instructions, then read model.py.
Run `python run.py --baseline "session start"` to start a new session and establish the current model's session-start RMSE.
Then enter the AutoResearch loop:

1. Propose one modification to model.py (e.g., different estimator,
   feature engineering, hyperparameter change).
2. Edit model.py with your change.
3. Run: python run.py "<short description of what you changed>"
4. Check the logged status in the active `results_<n>.tsv`.
   - If status = keep: KEEP the change.
   - If status = discard: REVERT model.py to the previous version.
5. Repeat from step 1. Try at least 6 different ideas.

After all iterations, run `python prepare.py` to generate `performance_<n>.png`.
Print a summary table of all experiments and which were kept vs discarded.
```

---

## Plotting Results

After running experiments:

```bash
python prepare.py
# Generates performance_<n>.png from results_<n>.tsv
```

To build the curated cumulative view across the selected meaningful sessions:

```bash
python -c "from prepare import build_cumulative_artifacts; build_cumulative_artifacts()"
# Generates results_cumulative.tsv and performance_cumulative.png
```

This produces a two-panel chart:
- **Top**: validation RMSE over iterations (green = keep, red = discard, blue = baseline)
- **Bottom**: validation R² over iterations
- **Green line**: best-so-far envelope

## Baseline vs Current Model

`prepare.py` now exposes a richer frozen MMM-style feature library, including:
- raw spend
- one-period spend lags
- fixed-decay adstock features
- `log1p` spend features
- simple weekly seasonality controls

The reference baseline model in `baseline_model.py` intentionally stays simple and uses only:
- geo fixed effects via one-hot encoding
- current-period spend features
- one-period lagged spend features
- control variables for competitor sales, sentiment, and promotion
- simple weekly seasonality controls

The current working model in `model.py` may differ from this baseline because it stores the latest accepted champion from the AutoResearch loop.

---
