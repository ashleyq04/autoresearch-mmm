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
