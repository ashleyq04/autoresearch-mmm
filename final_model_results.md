# Final Model Results

This document is the working home for the project's final model results and interpretation.

Status note:
As of May 18, 2026, this file records the current best model in the repo, not necessarily the absolute best model that will appear by the end of the project. If additional AutoResearch loops improve the champion model, update this file rather than treating this snapshot as final.

## Executive Summary

As of May 18, 2026, the current best model in this repo is a compact interpretable marketing mix model that performs best with a small number of carryover-adjusted media terms, one promotion-sensitive interaction, geo fixed effects, and simple controls. The model suggests that some marketing effects persist beyond the same week, and that at least one channel appears more effective when promotions are active.

For a marketing director, the main value of this report is decision support. It can be used to prioritize which channels deserve protection, closer review, or controlled budget testing. It should not be treated as a fully automatic budget optimizer or as final proof of causal ROI by channel.

## 1. Current Model Status

- Snapshot date: May 18, 2026
- Snapshot role: current best model as of today
- Current champion file: [model.py](/Users/ashley/Documents/NU/Fourth_Year/STAT390/autoresearch-mmm/model.py)
- Most relevant session log: [results_10.tsv](/Users/ashley/Documents/NU/Fourth_Year/STAT390/autoresearch-mmm/results_10.tsv)
- Current session marker: `results_10.tsv`
- Current champion description in session 10: `increase bounded ridge strength on no-sentiment interaction champion`

This section should eventually be updated to say whether the model is fully locked as the final model. For now, it should be read as a current-best snapshot.

## 2. Model Snapshot

This snapshot was produced by fitting the current [model.py](/Users/ashley/Documents/NU/Fourth_Year/STAT390/autoresearch-mmm/model.py) once on the fixed training split from [prepare.py](/Users/ashley/Documents/NU/Fourth_Year/STAT390/autoresearch-mmm/prepare.py) and evaluating on the fixed validation split.

| Item | Value |
| --- | ---: |
| Train rows | 4960 |
| Validation rows | 1240 |
| Validation RMSE | 69367.065355 |
| Validation R^2 | 0.693530 |
| Intercept | 43204.623677 |
| Final feature count | 48 |

Context:
- In `results_10.tsv`, the session-start baseline is `69369.566914`.
- The current champion improves on that session-start baseline by about `2.50` RMSE points.
- This is a small improvement, so the result should be treated as incremental rather than transformational.

## 3. What The Current Model Is Saying

Current best model structure as of May 18, 2026:
- Linear MMM pipeline with one-hot geo effects and selected numeric features.
- Bounded ridge regression final estimator.
- Nonnegative constraints on media-derived coefficients and the approved media-promotion interaction.
- Four adstock media terms:
  - `Channel0_spend_adstock_07`
  - `Channel1_spend_adstock_07`
  - `Channel2_spend_adstock_07`
  - `Channel4_spend_adstock_03`
- One sparse interaction term:
  - `Channel4_spend_adstock_03 x Promo`
- Control variables:
  - `competitor_sales_control`
  - `Promo`
  - `week_sin`
  - `week_cos`
- Excluded from the current champion:
  - raw spend terms
  - lag terms
  - log1p spend terms
  - `sentiment_score_control`

Short description:
This current-best model is a bounded ridge regression MMM that uses a small set of carryover-adjusted marketing features, one promotion interaction, geo fixed effects, and simple seasonality and control terms.

## 4. Actionable Marketing Implications

### 4.1 Channel Prioritization Signal

The strongest current positive media association comes from the selected carryover-adjusted Channel 2 term, followed by the Channel 1 and Channel 0 adstock terms. In practical terms, that means these channels currently look more promising to protect, monitor closely, and consider for disciplined budget support than channels that dropped out of the final compact specification.

### 4.2 Promotion Strategy Signal

The `Channel4_spend_adstock_03 x Promo` term is positive and much larger than the standalone `Channel4_spend_adstock_03` main effect. The practical implication is that this channel may be more valuable when paired with promotions than when run in a more always-on way without promotional support.

### 4.3 Budget Review Signal

This model does not support reading the coefficients as exact budget allocation rules, but it does support directional prioritization:
- channels with stronger stable positive modeled contribution are candidates to protect or test upward
- channels with weak or near-zero standalone contribution should be reviewed before receiving additional budget
- promotion-supported channels may deserve scenario testing rather than simple flat budget increases

### 4.4 Portfolio Design Signal

The current champion favors a compact feature set over a sprawling one. That suggests a simpler marketing story may fit this dataset better than a highly complex allocation narrative. For a director, that is useful because it points toward a smaller number of important spending relationships rather than a broad claim that every channel is equally material.

## 5. Recommended Uses For A Marketing Director

This report can be used to:
- prioritize which channels deserve budget protection
- identify channels that may be stronger during promotional periods
- flag weak-signal channels for review, creative changes, or tighter testing
- support internal discussion about where to run budget reallocation experiments
- create a structured basis for follow-up measurement and controlled business testing

This report should not be used by itself to:
- set exact dollar budgets automatically
- claim causal ROI with certainty
- make major spend shifts without business review or follow-up validation

## 6. Suggested Director Actions

Based on the current model snapshot, the most reasonable actions are:
- maintain or prioritize attention on the channels with the strongest positive modeled association
- treat the promotion-sensitive channel as a candidate for promo-linked planning rather than generic always-on budget growth
- review channels that are absent from the current compact champion before increasing spend there
- use these findings to design the next round of controlled budget tests, holdouts, or business experiments

## 7. Supporting Technical Evidence

### 7.1 Marketing Terms

| Feature | Coefficient |
| --- | ---: |
| `num__Channel0_spend_adstock_07` | 0.1674429428 |
| `num__Channel1_spend_adstock_07` | 0.1830190639 |
| `num__Channel2_spend_adstock_07` | 0.3046544644 |
| `num__Channel4_spend_adstock_03` | 0.0000272555 |

Notes:
- All selected media coefficients are nonnegative, which satisfies the business-safe sign rule.
- `Channel2_spend_adstock_07` is the largest positive media coefficient in the current specification.
- `Channel4_spend_adstock_03` has almost no standalone effect in the base term, so most of its useful signal may be coming through the promotion interaction instead of the main effect alone.

### 7.2 Interaction Terms

| Feature | Coefficient |
| --- | ---: |
| `num__Channel4_spend_adstock_03_x_num__Promo` | 0.3310740512 |

Notes:
- The positive interaction suggests that this channel's carryover-adjusted effect is stronger during promotional periods than during non-promo periods.

### 7.3 Control Variables

| Feature | Coefficient |
| --- | ---: |
| `num__competitor_sales_control` | -13380.5896608700 |
| `num__Promo` | 7074.1142311687 |
| `num__week_sin` | -2398.0054112691 |
| `num__week_cos` | 1373.2894940313 |

Notes:
- The competitor control is negative, which is directionally plausible if stronger competitor activity corresponds to lower revenue.
- The direct `Promo` coefficient is positive, which suggests promotions are associated with higher revenue after controlling for the other included terms.
- The seasonality controls indicate that recurring calendar structure still matters in the current model.

## 8. Limitations And Guardrails

- These coefficients are associative, not automatically causal.
- The adstock coefficients describe carryover-adjusted spend features, not raw weekly spend in isolation.
- Coefficients are conditional on the included features, the fixed time split, and the current modeling constraints.
- Geo coefficients should be read as baseline location adjustments, not marketing effectiveness estimates.
- The current improvement over the session-start baseline is small, so this should be presented as a careful incremental refinement rather than a major performance leap.
- Because this file is only a May 18, 2026 snapshot, later experiments may replace this specification or change the interpretation.

Practical guardrail:
Use this report for prioritization, scenario planning, and experiment design, not as an automatic final budget allocation engine.

## 9. Appendix: Geo Effects

These coefficients are geo fixed effects from one-hot encoding with one omitted reference geography. They are included for completeness, but they are not the main decision variables for a marketing director.

| Feature | Coefficient |
| --- | ---: |
| `geo__geo_Geo1` | 17377.5568176020 |
| `geo__geo_Geo10` | 329616.6202014536 |
| `geo__geo_Geo11` | 169535.6016754089 |
| `geo__geo_Geo12` | 154229.4379075663 |
| `geo__geo_Geo13` | 114090.6536125352 |
| `geo__geo_Geo14` | 211018.7133776836 |
| `geo__geo_Geo15` | 217101.9185635334 |
| `geo__geo_Geo16` | 87049.1523606368 |
| `geo__geo_Geo17` | 217572.5799904338 |
| `geo__geo_Geo18` | 147909.5871639625 |
| `geo__geo_Geo19` | 93676.7309736732 |
| `geo__geo_Geo2` | 249616.1704397379 |
| `geo__geo_Geo20` | 150581.5031021846 |
| `geo__geo_Geo21` | 127054.7005240254 |
| `geo__geo_Geo22` | 120433.7759264442 |
| `geo__geo_Geo23` | 298875.1919267042 |
| `geo__geo_Geo24` | 2610.5980454273 |
| `geo__geo_Geo25` | 248184.0515180570 |
| `geo__geo_Geo26` | 44771.5500428544 |
| `geo__geo_Geo27` | 148898.1305007030 |
| `geo__geo_Geo28` | 206162.3544912108 |
| `geo__geo_Geo29` | 123554.1182133531 |
| `geo__geo_Geo3` | 33635.4526130621 |
| `geo__geo_Geo30` | 62312.6043780592 |
| `geo__geo_Geo31` | 272190.6670475329 |
| `geo__geo_Geo32` | 189928.8882817059 |
| `geo__geo_Geo33` | 166949.3351819415 |
| `geo__geo_Geo34` | 174453.2305033839 |
| `geo__geo_Geo35` | 200078.2038310757 |
| `geo__geo_Geo36` | 360365.6009446868 |
| `geo__geo_Geo37` | 79335.4454442227 |
| `geo__geo_Geo38` | 45644.7742152327 |
| `geo__geo_Geo39` | 347379.6626256891 |
| `geo__geo_Geo4` | 84919.0892921645 |
| `geo__geo_Geo5` | 65318.2826586009 |
| `geo__geo_Geo6` | 33269.1309958271 |
| `geo__geo_Geo7` | 271897.7328925089 |
| `geo__geo_Geo8` | 70665.9501421536 |
| `geo__geo_Geo9` | 183772.5122888093 |

## 10. Update Checklist For Later

When the champion model changes, update the following:
- snapshot date
- session file reference
- champion description
- RMSE and R^2
- model specification
- actionable implications
- recommended actions
- supporting coefficient tables

When the project is fully done, add:
- final locked-model statement
- final ablation/comparison summary
- final narrative interpretation for the report
- any repeated validation or stability-check summary
