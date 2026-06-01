# What Actually Worked Memo

## Purpose

This memo summarizes the model changes that genuinely improved validation RMSE in the AutoResearch MMM workflow, rather than the ideas that were only tested or discussed.

## Best Result vs. Baseline

- Baseline reference RMSE: `73397.180606`
- Current best RMSE: `69367.065355`

## What Actually Worked

### 1. Enforcing nonnegative media effects

One of the biggest practical improvements came from replacing an unconstrained linear model with a bounded linear model that forces media-related coefficients to stay nonnegative. This improved fit while also making the model more business-safe, since the final model no longer suggests that spending more on a channel directly reduces revenue.

### 2. Using adstock features instead of simple lag structure

Replacing the original lag-based media representation with adstock-style carryover features improved RMSE substantially. This suggests that delayed media effects were better captured by carryover-style inputs than by a simple one-period lag.

### 3. Allowing different carryover behavior by channel

The strongest accepted pre-interaction model did not use the same carryover setting for every channel. In particular, shorter carryover worked better for some channels while slower carryover remained better for others. This helped the model better reflect the idea that different marketing channels can influence revenue over different time horizons.

### 4. Simplifying away a useless raw spend term

By Session 8, the raw `Channel3_spend` term was consistently being driven to a coefficient of zero. Removing that term did not worsen RMSE and produced a simpler champion model, which is useful evidence that the remaining adstock-based terms were already capturing the relevant signal.

### 5. Adding one sparse, business-readable promotion-media interaction

One of the last meaningful structural improvements came from a single bounded interaction between `Promo` and `Channel4_spend_adstock_03`. This means the model fit improved when it was allowed to represent the idea that Channel 4 media may work differently during promotional periods. Importantly, this only worked when the interaction stayed sparse and clearly tied to one channel.

### 6. Adding mild bounded ridge regularization to the final interaction model

The final improvement in the repo came from applying light bounded ridge regularization to the no-sentiment interaction champion. This preserved the compact interpretable structure while slightly improving validation RMSE from the earlier interaction-based best of `69452.274146` to the final best of `69367.065355`.

## What This Suggests

The useful improvements came from better structure, not from adding many more features. In this project, the most successful path was:

- keep the model linear
- enforce nonnegative media effects
- represent delayed media impact with adstock
- allow some channel-level carryover differences
- add at most a very small number of business-justified interaction terms

## Practical Takeaway

At this stage, RMSE appears to be approaching the lower end of what this frozen, interpretable search space can support. That makes the current best model valuable not only because it predicts better than the original baseline, but also because it shows which kinds of MMM-safe complexity were worth adding and which were not.
