"""
Run one experiment: build model, train, evaluate, log result.
Now updated with automated Logic Check and AIC tracking.
"""
import sys
import time
import subprocess
import pandas as pd
from prepare import load_data, evaluate, log_result


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def main():
    args = sys.argv[1:]
    # Default status is "keep", but the script will now auto-detect "discard"
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

    # 1. Load data (frozen)
    # Uses national_all_channels.csv from the /data folder [cite: 1270, 1278]
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

    # 4. Evaluate (frozen metric with Logic Check)
    # Now returns 4 values: rmse, r2, aic, logic_passed 
    val_rmse, val_r2, val_aic, logic_passed = evaluate(model, X_val, y_val, feature_names)
    
    # 5. Automated Decision Logic
    if not manual_override:
        # Step A: Prioritize the Logic Check [cite: 1447]
        if not logic_passed:
            status = "discard"
            print("ALERT: Logic Check Failed (Negative Marketing Coefficient). Forced Discard.") [cite: 1448]
        # Step B: In future steps, you will compare val_aic to your best_aic here [cite: 1449]

    print(f"val_rmse: {val_rmse:.6f}")
    print(f"val_r2:   {val_r2:.6f}")
    print(f"val_aic:  {val_aic:.4f}")

    # 6. Log
    commit = get_git_hash()
    # Updated to include val_aic in the log [cite: 1408, 1451]
    log_result(commit, val_rmse, val_r2, val_aic, status, description)
    print(f"Result logged to results.tsv (status={status})")


if __name__ == "__main__":
    main()