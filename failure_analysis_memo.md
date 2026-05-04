# Failure Analysis Memo

## Project Goal

This project uses an AutoResearch loop to improve an interpretable marketing revenue model. The model is meant to stay simple enough for marketers to understand, while still making reasonably accurate revenue predictions.

Within that broader goal, the AutoResearch loop itself uses **validation RMSE as the optimization target**. In each experiment, the agent proposes a change to `model.py`, trains the model, evaluates it on the fixed validation set, and keeps the change only if validation RMSE improves while the model stays inside the allowed interpretable model class.

In other words, interpretability defines the search boundary, and validation RMSE is the metric being minimized inside that boundary.

## Controlled Experiment Setup

The experiment was controlled by keeping the following parts fixed across runs:

- the dataset
- the train/validation split
- the feature-generation code in `prepare.py`
- the evaluation metric: validation RMSE
- the runtime environment

The only file changed during the loop was `model.py`, so each experiment tested a model specification change while holding the rest of the pipeline constant.

## Summary of Progress

The project started from a simple linear baseline with validation RMSE of `73397.180606` in Session 3.

The largest gains came from two changes:

1. enforcing nonnegative marketing effects in a bounded linear model
2. replacing simple lag terms with adstock-style carryover terms

That reduced RMSE to `69643.239067`.

In Session 5, the model improved further by simplifying raw spend terms and tuning carryover separately for some channels. The best model in the current repo reached validation RMSE `69577.322385`.

## What Failed and What We Learned

### 1. Extra transformed features did not help by default

Some experiments added extra transformed spend terms, such as log-transformed spend, on top of the existing model. These changes made the model more complicated but did not improve validation RMSE.

This suggests that adding more versions of the same spend signal does not automatically help. In this project, simpler feature sets were often better than broader feature expansion.

### 2. Some delayed-effect features were useful, but not all of them

Replacing simple one-period lag terms with adstock features improved the model. This means the data benefited from a better way of representing delayed marketing effects.

However, adding extra lag terms on top of the adstock structure made performance worse. For example, adding `Channel0_spend_lag1` to the accepted adstock model increased RMSE.

This suggests that once delayed effects were already captured well, extra lag terms became redundant rather than helpful.

### 3. Removing too much structure also hurt

One experiment removed raw spend terms entirely from the adstock model and performed worse. Another removed the last remaining raw `Channel3_spend` term after shortening its carryover and also failed to improve RMSE.

This suggests that some immediate-response information still matters, even when delayed effects are already modeled.

### 4. Carryover behavior was not the same for every channel

A useful result from Session 5 was that not all channels preferred the same carryover setting.

Using a shorter carryover setting for `Channel3` improved performance. Shortening `Channel4` as well improved it further. But applying the same shorter setting to `Channel2` or `Channel0` made performance worse.

This suggests that different channels may affect revenue on different time scales, so treating every channel the same can miss useful structure.

## Error Taxonomy

The main recurring failure types were:

- **Performance regression**: a model change increased validation RMSE relative to the current best model
- **Weak carryover specification**: a lag or carryover choice did not match the way a channel actually affected revenue
- **Redundant feature expansion**: extra transformed features increased complexity without improving fit
- **Feature removal oversimplification**: removing useful spend terms weakened performance

These categories are recorded in `error_log.md` and were used to summarize repeated failure patterns across sessions.

## Final Takeaway

The main lesson from the failed experiments is that improvement came from better structure, not from adding many more features.

The most successful direction was:

- keep the model linear and interpretable
- enforce nonnegative marketing effects
- represent delayed effects with adstock rather than simple lags
- allow some channels to use different carryover settings
- avoid adding or removing features unless the change clearly improves validation RMSE

Overall, the failed experiments helped narrow the search toward a smaller, more interpretable model that fit the data better than the original baseline.
