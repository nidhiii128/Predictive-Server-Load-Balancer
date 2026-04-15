"""
==============================================================
 Predictive Server Load Balancer
 PART 3 – Integration: ML Prediction → Fuzzy Logic Decision
==============================================================
This is the MAIN entry point that ties everything together:
  1. Loads the trained ML model
  2. Takes new server metrics as input
  3. ML model predicts load category
  4. Fuzzy logic converts predicted load + current CPU → scaling decision
  5. Prints and plots final outputs
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings("ignore")

from fuzzy_system import get_scaling_decision, plot_membership_functions

os.makedirs("results", exist_ok=True)

# ─── Load artefacts ──────────────────────────────────────────────────────────
rf      = joblib.load("models/random_forest.pkl")
scaler  = joblib.load("models/scaler.pkl")
X_COLS  = list(np.load("models/x_cols.npy", allow_pickle=True))
LABEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# ─── Load dataset (for integration demo) ─────────────────────────────────────
CSV_PATH = "data/final-complete-data-set.csv"
df_raw = pd.read_csv(CSV_PATH)
COLS = ["timestamp", "cpu_total", "cpu_idle", "load_min1", "load_min5",
        "mem_used", "mem_percent", "network_lo_rx", "network_lo_tx"]
df = df_raw[COLS].copy()

# ── Reproduce the same preprocessing as ml_model.py ─────────────────────────
df.fillna(df.median(numeric_only=True), inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]   = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["second"] = df["timestamp"].dt.second
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURE_COLS = ["cpu_total", "cpu_idle", "load_min1", "load_min5",
                "mem_used", "mem_percent", "network_lo_rx", "network_lo_tx"]
for col in FEATURE_COLS:
    for lag in [1, 2, 3]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

exclude = {"load_category", "load_category_label", "timestamp", "hour", "minute", "second"}
scale_cols = [c for c in df.columns if c not in exclude]
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.transform(df[scale_cols])

# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION: Predict and decide for a single row
# ─────────────────────────────────────────────────────────────────────────────
def predict_and_decide(row_index: int, verbose=True) -> dict:
    """
    Given an index into the dataset:
      - Extract feature vector
      - ML model predicts load category
      - Map category back to a numeric cpu % estimate
      - Fuzzy system decides scaling action

    Returns a dict with all relevant info.
    """
    row = df_scaled.iloc[row_index]
    X   = row[X_COLS].values.reshape(1, -1)

    # Step A: ML Prediction
    predicted_label = int(rf.predict(X)[0])
    predicted_proba = rf.predict_proba(X)[0]
    load_category   = LABEL_MAP[predicted_label]

    # Map label → representative CPU % for fuzzy system input
    load_to_cpu = {"LOW": 25, "MEDIUM": 65, "HIGH": 88}
    predicted_cpu_pct = load_to_cpu[load_category]

    # Step B: Current actual CPU
    actual_cpu = float(df.iloc[row_index]["cpu_total"])

    # Step C: Fuzzy Logic Decision
    fuzzy_result = get_scaling_decision(predicted_cpu_pct, actual_cpu)

    result = {
        "row_index"        : row_index,
        "actual_cpu"       : round(actual_cpu, 2),
        "predicted_label"  : predicted_label,
        "load_category"    : load_category,
        "proba_LOW"        : round(predicted_proba[0], 3),
        "proba_MEDIUM"     : round(predicted_proba[1], 3),
        "proba_HIGH"       : round(predicted_proba[2], 3),
        "fuzzy_crisp"      : fuzzy_result["scaling_crisp"],
        "scaling_decision" : fuzzy_result["scaling_decision"],
    }

    if verbose:
        print("─" * 55)
        print(f"  Row          : {row_index}")
        print(f"  Actual CPU   : {actual_cpu:.2f}%")
        print(f"  Predicted    : {load_category}  "
              f"(LOW={result['proba_LOW']:.2f} | "
              f"MED={result['proba_MEDIUM']:.2f} | "
              f"HIGH={result['proba_HIGH']:.2f})")
        print(f"  Fuzzy Crisp  : {fuzzy_result['scaling_crisp']:.3f}")
        print(f"  ➤  SCALING DECISION: {fuzzy_result['scaling_decision']}")
        print("─" * 55)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH DEMO: Run on a sample of the dataset
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_demo(n_samples=300):
    print("=" * 55)
    print("  Integration Demo – Batch Run")
    print("=" * 55)

    indices  = np.linspace(0, len(df) - 1, n_samples, dtype=int)
    results  = [predict_and_decide(i, verbose=False) for i in indices]
    results_df = pd.DataFrame(results)

    # ── Decision distribution ────────────────────────────────────────────────
    print("\nScaling Decision Distribution:")
    print(results_df["scaling_decision"].value_counts())

    # ── Plot 1: CPU trend + scaling decisions ─────────────────────────────────
    colour_map = {"NO SCALE": "green", "SCALE SLIGHTLY": "orange", "SCALE HIGH": "red"}

    fig, ax1 = plt.subplots(figsize=(15, 5))
    cpu_vals = results_df["actual_cpu"].values
    ax1.plot(range(len(cpu_vals)), cpu_vals, color="steelblue", linewidth=1, label="Actual CPU %")
    ax1.axhline(80, color="red",    linestyle="--", linewidth=0.8, label="HIGH threshold")
    ax1.axhline(50, color="orange", linestyle="--", linewidth=0.8, label="MEDIUM threshold")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("CPU Total (%)")
    ax1.set_title("CPU Usage Trend with ML-predicted Load Category")

    # Scatter scaling decisions on top
    for decision, colour in colour_map.items():
        mask = results_df["scaling_decision"] == decision
        ax1.scatter(results_df.index[mask],
                    results_df["actual_cpu"][mask],
                    c=colour, label=decision, s=18, zorder=5, alpha=0.7)

    ax1.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/integration_cpu_decisions.png", dpi=150)
    print("\nSaved → results/integration_cpu_decisions.png")

    # ── Plot 2: Fuzzy crisp output over time ─────────────────────────────────
    plt.figure(figsize=(14, 3))
    plt.plot(results_df["fuzzy_crisp"].values, color="purple", linewidth=1)
    plt.axhline(6.5, color="red",    linestyle="--", linewidth=0.8, label="SCALE HIGH boundary")
    plt.axhline(3.5, color="orange", linestyle="--", linewidth=0.8, label="SCALE SLIGHTLY boundary")
    plt.xlabel("Sample Index")
    plt.ylabel("Fuzzy Output (0-10)")
    plt.title("Fuzzy Defuzzified Scaling Output Over Time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/fuzzy_output_trend.png", dpi=150)
    print("Saved → results/fuzzy_output_trend.png")

    # ── Plot 3: Predicted category distribution ───────────────────────────────
    plt.figure(figsize=(7, 4))
    results_df["load_category"].value_counts().plot(kind="bar",
        color=["green","orange","red"], edgecolor="black")
    plt.title("ML-Predicted Load Category Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/predicted_category_dist.png", dpi=150)
    print("Saved → results/predicted_category_dist.png")

    results_df.to_csv("results/batch_results.csv", index=False)
    print("Saved → results/batch_results.csv")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE PREDICTION: Simulate real-time input
# ─────────────────────────────────────────────────────────────────────────────
def live_predict(cpu_total, cpu_idle, load_min1, load_min5,
                 mem_used, mem_percent, network_lo_rx, network_lo_tx):
    """
    Simulate a live single-row prediction.
    Pass current server metrics; get back the scaling decision.
    Lag features are set to the same values (first observation assumption).
    """
    print("\n" + "=" * 55)
    print("  LIVE PREDICTION")
    print("=" * 55)

    feature_vals = {
        "cpu_total": cpu_total, "cpu_idle": cpu_idle,
        "load_min1": load_min1, "load_min5": load_min5,
        "mem_used": mem_used,   "mem_percent": mem_percent,
        "network_lo_rx": network_lo_rx, "network_lo_tx": network_lo_tx,
        "hour": 0, "minute": 0, "second": 0,
    }
    # Add lag features (copy current values for lag1/2/3 since no history)
    base_cols = ["cpu_total","cpu_idle","load_min1","load_min5",
                 "mem_used","mem_percent","network_lo_rx","network_lo_tx"]
    for col in base_cols:
        for lag in [1, 2, 3]:
            feature_vals[f"{col}_lag{lag}"] = feature_vals[col]

    row_df = pd.DataFrame([feature_vals])
    # Use the exact columns the scaler was fitted on (in the same order)
    scaler_cols = scaler.feature_names_in_.tolist()
    for c in scaler_cols:
        if c not in row_df.columns:
            row_df[c] = 0
    row_scaled = scaler.transform(row_df[scaler_cols])

    # Build the correct column order for the model
    row_scaled_df = pd.DataFrame(row_scaled, columns=scaler_cols)
    for c in X_COLS:
        if c not in row_scaled_df.columns:
            row_scaled_df[c] = 0
    row_final = row_scaled_df[X_COLS]

    predicted_label = int(rf.predict(row_final.values)[0])
    predicted_proba = rf.predict_proba(row_final.values)[0]
    load_category   = LABEL_MAP[predicted_label]

    load_to_cpu = {"LOW": 25, "MEDIUM": 65, "HIGH": 88}
    fuzzy_result = get_scaling_decision(load_to_cpu[load_category], cpu_total)

    print(f"\n  Input Metrics:")
    print(f"    cpu_total   = {cpu_total}%")
    print(f"    cpu_idle    = {cpu_idle}%")
    print(f"    load_min1   = {load_min1}")
    print(f"    mem_percent = {mem_percent}%")
    print(f"\n  ➤  ML Predicted Load  : {load_category}")
    print(f"     Probabilities       : LOW={predicted_proba[0]:.2f} | "
          f"MED={predicted_proba[1]:.2f} | HIGH={predicted_proba[2]:.2f}")
    print(f"\n  ➤  Fuzzy Crisp Output : {fuzzy_result['scaling_crisp']:.3f}")
    print(f"  ➤  SCALING DECISION   : {fuzzy_result['scaling_decision']}")
    print("=" * 55)

    return load_category, fuzzy_result["scaling_decision"]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Show membership functions
    plot_membership_functions()

    # 2. Print a few individual predictions
    print("\n" + "=" * 55)
    print("  Sample Individual Predictions")
    print("=" * 55)
    for idx in [10, 500, 1200, 2500, 3500]:
        predict_and_decide(idx)

    # 3. Batch run for plots
    results = run_batch_demo(n_samples=300)

    # 4. Live demo with custom values
    live_predict(
        cpu_total=87, cpu_idle=10, load_min1=3.5, load_min5=2.8,
        mem_used=2_500_000_000, mem_percent=62,
        network_lo_rx=510_000, network_lo_tx=490_000
    )

    live_predict(
        cpu_total=28, cpu_idle=70, load_min1=0.8, load_min5=0.5,
        mem_used=900_000_000, mem_percent=22,
        network_lo_rx=300_000, network_lo_tx=295_000
    )

    print("\n✅  Integration complete. All results saved in /results/")
