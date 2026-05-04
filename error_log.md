# Error Log

This file records recurring model failure patterns observed during the AutoResearch MMM loop.

## Taxonomy

### 1. Performance Regression
Definition: Validation RMSE worsens relative to the best accepted model.

### 2. Sign Inconsistency
Definition: A spend-related feature receives a negative coefficient, making the model business-inadmissible.

### 3. Weak Carryover Specification
Definition: A lag or adstock choice fails to capture delayed media effects as effectively as competing specifications.

### 4. Redundant Feature Expansion
Definition: Added transformed features increase complexity without improving validation RMSE.

### 5. Feature Removal Oversimplification
Definition: Removing important spend terms or baseline structure weakens model performance.

## Observations

### Session 3

- `log current spend with linear lag terms`
  Status: `discard`
  Category: `Redundant Feature Expansion`
  Note: Added transformed spend features but RMSE worsened relative to the current best model.

- `use adstock decay 0.7 without raw spend terms`
  Status: `discard`
  Category: `Feature Removal Oversimplification`
  Note: Removing raw spend terms reduced predictive performance.

- `combine log current spend with adstock decay 0.7`
  Status: `discard`
  Category: `Redundant Feature Expansion`
  Note: Added complexity without improving over the current champion model.

### Session 5

- `add Channel0 lag to pruned raw-plus-adstock model`
  Status: `discard`
  Category: `Weak Carryover Specification`
  Note: Adding an explicit lag on top of the accepted adstock structure worsened RMSE.

- `drop raw Channel3 after switching Channel3 adstock to 0.3`
  Status: `discard`
  Category: `Feature Removal Oversimplification`
  Note: Removing the remaining raw spend term did not outperform the current champion.

- `use faster adstock decay 0.3 for Channel2 Channel3 and Channel4`
  Status: `discard`
  Category: `Weak Carryover Specification`
  Note: Extending the shorter carryover assumption to Channel2 reduced validation performance.

- `use faster adstock decay 0.3 for Channel0 Channel3 and Channel4`
  Status: `discard`
  Category: `Weak Carryover Specification`
  Note: Shortening Channel0 carryover weakened the fit relative to the accepted mixed-decay model.
