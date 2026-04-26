"""
Core data loading, train/val split, evaluation metric, and plotting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
import os
import re


# ── Constants ──────────────────────────────────────────────
SESSION_MARKER_FILE = ".current_session"
RESULTS_FILE_TEMPLATE = "results_{session_id}.tsv"
PERFORMANCE_FILE_TEMPLATE = "performance_{session_id}.png"


def _list_session_ids():
    pattern = re.compile(r"^results_(\d+)\.tsv$")
    session_ids = []
    for filename in os.listdir("."):
        match = pattern.match(filename)
        if match:
            session_ids.append(int(match.group(1)))
    return sorted(session_ids)


def get_current_session_id():
    if not os.path.exists(SESSION_MARKER_FILE):
        return None

    with open(SESSION_MARKER_FILE) as f:
        session_id = f.read().strip()

    return int(session_id) if session_id.isdigit() else None


def set_current_session_id(session_id):
    with open(SESSION_MARKER_FILE, "w") as f:
        f.write(str(session_id))


def start_new_session():
    existing_ids = _list_session_ids()
    session_id = 1 if not existing_ids else max(existing_ids) + 1
    set_current_session_id(session_id)
    return session_id


def resolve_session_id(create=False):
    session_id = get_current_session_id()
    if session_id is not None:
        return session_id

    existing_ids = _list_session_ids()
    if existing_ids and not create:
        return max(existing_ids)

    if create:
        return start_new_session()

    return None


def get_results_file(session_id=None, create=False):
    session_id = resolve_session_id(create=create) if session_id is None else session_id
    if session_id is None:
        return None
    return RESULTS_FILE_TEMPLATE.format(session_id=session_id)


def get_performance_file(session_id=None, create=False):
    session_id = resolve_session_id(create=create) if session_id is None else session_id
    if session_id is None:
        return None
    return PERFORMANCE_FILE_TEMPLATE.format(session_id=session_id)


def _adstock(series, decay):
    values = series.to_numpy(dtype=float)
    out = np.zeros_like(values)
    for i, value in enumerate(values):
        out[i] = value if i == 0 else value + decay * out[i - 1]
    return out

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

    # Sort within each geography so lagged features use the prior period.
    df = df.sort_values(["geo", "time"]).copy()

    spend_cols = [
        "Channel0_spend",
        "Channel1_spend",
        "Channel2_spend",
        "Channel3_spend",
        "Channel4_spend",
    ]

    # Frozen seasonality controls: simple weekly cycle.
    week_of_year = df["time"].dt.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week_of_year / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week_of_year / 52.0)

    lagged_spend_cols = []
    for col in spend_cols:
        lag_col = f"{col}_lag1"
        df[lag_col] = df.groupby("geo")[col].shift(1)
        lagged_spend_cols.append(lag_col)

    # Frozen MMM-style candidate library for model.py to choose from.
    log_spend_cols = []
    adstock_cols = []
    for col in spend_cols:
        log_col = f"{col}_log1p"
        df[log_col] = np.log1p(df[col])
        log_spend_cols.append(log_col)

        for decay in (0.3, 0.7):
            adstock_col = f"{col}_adstock_{str(decay).replace('.', '')}"
            df[adstock_col] = df.groupby("geo")[col].transform(lambda s, d=decay: _adstock(s, d))
            adstock_cols.append(adstock_col)
    
    # Define features (Marketing channels + any control variables)
    # baseline: current spend, 1-period spend lags, and controls
    feature_cols = [
        "geo",
        *spend_cols,
        *lagged_spend_cols,
        *log_spend_cols,
        *adstock_cols,
        "competitor_sales_control",
        "sentiment_score_control",
        "Promo",
        "week_sin",
        "week_cos",
    ]

    # The first row in each geography has no lagged spend values.
    df = df.dropna(subset=lagged_spend_cols)
    
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
def log_result(
    experiment_id,
    val_rmse,
    val_r2,
    status,
    description,
    runtime_sec,
    train_time_sec,
    results_file=None,
):
    """
    Append one experiment result to the active session results file.

    Each row represents a single model run, including:
        - experiment_id: git commit or unique identifier
        - val_rmse: validation RMSE (north star metric)
        - val_r2: validation R² (context metric)
        - status: baseline / keep / discard
        - description: short description of the experiment
        - runtime_sec: total runtime for the iteration
        - train_time_sec: runtime for model.fit()
    """
    results_file = results_file or get_results_file(create=True)

    # 1. Check if results file already exists
    # If not, we will write the header first
    file_exists = os.path.exists(results_file)

    # 2. Open file in append mode
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # 3. Write header if file is new
        if not file_exists:
            writer.writerow([
                "experiment",
                "val_rmse",
                "val_r2",
                "status",
                "description",
                "runtime_sec",
                "train_time_sec",
            ])

        # 4. Write experiment result row
        writer.writerow([
            experiment_id,
            f"{val_rmse:.6f}",
            f"{val_r2:.6f}",
            status,
            description,
            f"{runtime_sec:.6f}",
            f"{train_time_sec:.6f}",
        ])


# ── Plotting ───────────────────────────────────────────────
def plot_results(results_file=None, save_path=None):
    """Plot validation RMSE over experiments from the active session results file."""
    results_file = results_file or get_results_file()
    if results_file is None or not os.path.exists(results_file):
        print("No session results file found. Run a baseline first.")
        return

    save_path = save_path or get_performance_file()
    experiments, rmses, r2s, statuses, descriptions = [], [], [], [], []
    with open(results_file) as f:
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
    ax1.set_title("AutoResearch Demo: Marketing Mix Modeling", fontsize=14, fontweight="bold")
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
