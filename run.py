"""
Run one experiment: build model, train, evaluate, log result.
"""
import csv
import os
import sys
import time
import subprocess
from prepare import load_data, evaluate, log_result


RESULTS_FILE = "results.tsv"


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def get_best_previous_rmse():
    if not os.path.exists(RESULTS_FILE):
        return None

    best_rmse = None
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] == "discard":
                continue
            rmse = float(row["val_rmse"])
            best_rmse = rmse if best_rmse is None else min(best_rmse, rmse)

    return best_rmse


def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []

    manual_override = False
    for a in args:
        if a == "--baseline":
            status = "baseline"
            manual_override = True
        elif a == "--discard":
            status = "discard"
            manual_override = True
        else:
            description_parts.append(a)
    
    description = " ".join(description_parts) if description_parts else "experiment"

    previous_best_rmse = None if manual_override else get_best_previous_rmse()

    # 1. Load data
    X_train, y_train, X_val, y_val, feature_names = load_data()
    print(f"Data: {X_train.shape[0]} train, {X_val.shape[0]} val, {len(feature_names)} features")

    # 2. Build model (editable)
    from model import build_model
    model = build_model()
    print(f"Model: {model}")

    # 3. Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    # 4. Evaluate
    val_rmse, val_r2 = evaluate(model, X_val, y_val)

    # Get expanded feature names after preprocessing
    preprocessor = model.named_steps["preprocessor"]
    expanded_feature_names = preprocessor.get_feature_names_out()

    coefs = model.named_steps["model"].coef_
    intercept = model.named_steps["model"].intercept_

    print("\nModel Coefficients:")
    print(f"Intercept: {intercept:.2f}")

    for name, coef in zip(expanded_feature_names, coefs):
        print(f"{name}: {coef:.4f}")




    print(f"val_rmse: {val_rmse:.6f}")
    print(f"val_r2:   {val_r2:.6f}")

    if not manual_override:
        if previous_best_rmse is None:
            status = "keep"
            print("Status: keep (first non-baseline run)")
        elif val_rmse < previous_best_rmse:
            status = "keep"
            print(f"Status: keep (improved over best previous RMSE {previous_best_rmse:.6f})")
        else:
            status = "discard"
            print(f"Status: discard (did not beat best previous RMSE {previous_best_rmse:.6f})")

    # 6. Log
    commit = get_git_hash()
    log_result(commit, val_rmse, val_r2, status, description)
    print(f"Result logged to results.tsv (status={status})")


if __name__ == "__main__":
    main()
    
