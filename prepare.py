"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metric, and plotting.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
import os


# ── Constants ──────────────────────────────────────────────
RESULTS_FILE = "results.tsv"

# ── Data ───────────────────────────────────────────────────
def load_data():
    """
    Load and split the Google Meridian Marketing dataset.
    Target: calculated revenue (conversions * revenue_per_conversion)
    Features: marketing channel spend and controls from geo_all_channels.csv
    """
    # Load the dataset from local folder
    df = pd.read_csv("data/geo_all_channels.csv")
    
    # Change time column to time-series
    df["time"] = pd.to_datetime(df["time"])

    # Calculate the Target Variable (Revenue) 
    # Since 'revenue' isn't a direct column, we create it manually
    df['revenue'] = df['conversions'] * df['revenue_per_conversion']
    
    # Define features (Marketing channels + any control variables)
    # baseline: spend + controls ONLY
    feature_cols = ['geo',
                    # 'Channel0_impression', 'Channel1_impression', 
                    # 'Channel2_impression','Channel3_impression',
                    # 'Channel4_impression', 
                    'Channel0_spend',
                    'Channel1_spend','Channel2_spend','Channel3_spend',
                    'Channel4_spend',
                    #'Organic_channel0_impression',
                    'competitor_sales_control','sentiment_score_control','Promo'] 
    
    # Time-based split 
    # perform a time-based holdout, reserving the final 20% of periods for validation across all geographies

    times = sorted(df["time"].unique())
    split_idx = int(len(times) * 0.8)

    train_times = times[:split_idx]
    val_times = times[split_idx:]

    train_df = df[df["time"].isin(train_times)]
    val_df = df[df["time"].isin(val_times)]

    # Create training and validation sets
    X_train = train_df[feature_cols]
    y_train = train_df["revenue"]

    X_val = val_df[feature_cols] # validation set 
    y_val = val_df["revenue"]

    return X_train, y_train, X_val, y_val, feature_cols


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, X_val, y_val):
    """
    Compute validation metrics for the current model.

    North star metric:
        - RMSE (lower is better)

    Secondary metric:
        - R² (higher is better)

    Returns:
        rmse, r2
    """

    # 1. Generate predictions on the validation set
    y_pred = model.predict(X_val)

    # 2. Compute mean squared error
    mse = mean_squared_error(y_val, y_pred)

    # 3. Convert MSE to RMSE
    # RMSE is the main metric because it measures prediction error
    # on held-out data in the same units as the target variable.
    rmse = float(np.sqrt(mse))

    # 4. Compute R² as additional context
    # R² is not the optimization target, but it is helpful for interpretation.
    r2 = float(r2_score(y_val, y_pred))

    # 5. Return the fixed evaluation outputs
    return rmse, r2


# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, val_rmse, val_r2, status, description):
    """
    Append one experiment result to results.tsv.

    Each row represents a single model run, including:
        - experiment_id: git commit or unique identifier
        - val_rmse: validation RMSE (north star metric)
        - val_r2: validation R² (context metric)
        - status: baseline / keep / discard
        - description: short description of the experiment
    """

    # 1. Check if results file already exists
    # If not, we will write the header first
    file_exists = os.path.exists(RESULTS_FILE)

    # 2. Open file in append mode
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # 3. Write header if file is new
        if not file_exists:
            writer.writerow([
                "experiment",
                "val_rmse",
                "val_r2",
                "status",
                "description"
            ])

        # 4. Write experiment result row
        writer.writerow([
            experiment_id,
            f"{val_rmse:.6f}",
            f"{val_r2:.6f}",
            status,
            description
        ])


# ── Plotting ───────────────────────────────────────────────
'''
def plot_results(save_path="performance.png"):
    """Plot validation RMSE over experiments from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, rmses, r2s, statuses, descriptions = [], [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            rmses.append(float(row["val_rmse"]))
            r2s.append(float(row["val_r2"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    # Colors: green=keep, red=discard, blue=baseline
    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # ── Outlier handling: clip axes to reasonable range ──
    rmse_sorted = sorted(rmses)
    q75 = np.percentile(rmses, 75)
    iqr = np.percentile(rmses, 75) - np.percentile(rmses, 25)
    rmse_upper = q75 + 2.5 * max(iqr, 0.1)  # generous fence

    r2_sorted = sorted(r2s)
    r2_lower = max(min(r2s), np.percentile(r2s, 25) - 2.5 * max(
        np.percentile(r2s, 75) - np.percentile(r2s, 25), 0.1))

    # ── Top: RMSE ──
    ax1.scatter(range(len(rmses)), rmses, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax1.plot(range(len(rmses)), rmses, "k--", alpha=0.2, zorder=2)

    # Best-so-far envelope
    best_so_far = []
    current_best = float("inf")
    for r in rmses:
        current_best = min(current_best, r)
        best_so_far.append(current_best)
    ax1.plot(range(len(rmses)), best_so_far, color="#2ecc71", linewidth=2.5, label="Best so far")

    # Clip y-axis: show main range, mark outliers with arrow
    reasonable_max = min(max(rmses), rmse_upper)
    ax1.set_ylim(min(rmses) * 0.9, reasonable_max * 1.1)
    for i, r in enumerate(rmses):
        if r > reasonable_max:
            ax1.annotate(
                f"{r:.2f}", xy=(i, reasonable_max * 1.05),
                fontsize=8, ha="center", color="#e74c3c", fontweight="bold",
            )
            ax1.annotate(
                "", xy=(i, reasonable_max * 1.08), xytext=(i, reasonable_max * 1.02),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
            )

    ax1.set_ylabel("Validation RMSE (lower is better)", fontsize=12)
    ax1.set_title("AutoResearch Demo: California Housing Regression", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # ── Bottom: R² ──
    ax2.scatter(range(len(r2s)), r2s, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax2.plot(range(len(r2s)), r2s, "k--", alpha=0.2, zorder=2)

    best_r2 = []
    current_best_r2 = -float("inf")
    for r in r2s:
        current_best_r2 = max(current_best_r2, r)
        best_r2.append(current_best_r2)
    ax2.plot(range(len(r2s)), best_r2, color="#2ecc71", linewidth=2.5, label="Best so far")

    # Clip y-axis for R²
    reasonable_r2_min = max(min(r2s), r2_lower)
    ax2.set_ylim(min(reasonable_r2_min * 1.1 if reasonable_r2_min < 0 else reasonable_r2_min * 0.9, -0.1),
                 max(r2s) * 1.05 if max(r2s) > 0 else 0.1)
    for i, r in enumerate(r2s):
        if r < reasonable_r2_min:
            ypos = ax2.get_ylim()[0] * 0.95
            ax2.annotate(
                f"{r:.1f}", xy=(i, ypos),
                fontsize=8, ha="center", color="#e74c3c", fontweight="bold",
            )

    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.set_ylabel("Validation R² (higher is better)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # x-tick labels = short descriptions
    short_labels = [d[:22] + ".." if len(d) > 24 else d for d in descriptions]
    ax2.set_xticks(range(len(rmses)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)

    # Legend for status colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="keep (improved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="discard (regressed)"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_results()
'''