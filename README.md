# AutoResearch Project: Marketing Modelling Mix
Project Owner: Ashley Qiu

A minimal, CPU-only AutoResearch project for **STAT 390** class demonstration.
Shows the full agent loop: modify code → evaluate → keep or discard → repeat.

---

## Problem

Predict geo-level revenue from marketing spend and control variables.
**Metric**: validation RMSE (lower is better).
**Data**: `data/geo_all_channels.csv`.

## Project Structure

```
autoresearch-mmm/
├── prepare.py      # FROZEN — data loading, evaluation metric, plotting
├── model.py        # EDITABLE — agent modifies only this file
├── run.py          # Run a single experiment and auto-label keep/discard
├── program.md      # Agent instructions (the agent reads this)
├── results.tsv     # Experiment log (auto-generated)
└── performance.png # Performance plot (auto-generated)
```

**Key rule**: the agent may only modify `model.py`. Everything else is frozen.

---

## How to Run the Agent Loop

### Quick start (copy-paste this prompt into your agent)

```
Read program.md for your instructions, then read model.py.
Run `python run.py --baseline "baseline linear model"` to establish the baseline RMSE.
Then enter the AutoResearch loop:

1. Propose one modification to model.py (e.g., different estimator,
   feature engineering, hyperparameter change).
2. Edit model.py with your change.
3. Run: python run.py "<short description of what you changed>"
4. Check the logged status in results.tsv.
   - If status = keep: KEEP the change.
   - If status = discard: REVERT model.py to the previous version.
5. Repeat from step 1. Try at least 6 different ideas.

After all iterations, run `python prepare.py` to generate performance.png.
Print a summary table of all experiments and which were kept vs discarded.
```

### More specific prompt (if you want to control the search)

```
You are an AutoResearch agent. Read program.md for rules.

Your job: minimize val_rmse on marketing-generated revenue predictions by modifying model.py.

Constraints:
- model.py must define build_model() returning an sklearn estimator
- Do NOT modify prepare.py or run.py
- Each experiment must finish in < 60 seconds

Search strategy:
1. Start with baseline (LinearRegression)
2. Try regularized linear models (Ridge, Lasso, ElasticNet)
3. Try transformations defined in program.md, including lag-feature selection

For each experiment:
- Run: python run.py "<description>"
- Read the logged status from results.tsv
- If status=keep → keep the change
- If status=discard → revert model.py to previous version
- Log your reasoning for each decision

After finishing, run: python prepare.py
```

---

## Plotting Results

After running experiments:

```bash
python prepare.py
# Generates performance.png from results.tsv
```

This produces a two-panel chart:
- **Top**: validation RMSE over iterations (green = keep, red = discard, blue = baseline)
- **Bottom**: validation R² over iterations
- **Green line**: best-so-far envelope

## Current Baseline

The baseline model in `model.py` is an ordinary least squares linear regression with:
- geo fixed effects via one-hot encoding
- current-period spend features
- one-period lagged spend features
- control variables for competitor sales, sentiment, and promotion

---
