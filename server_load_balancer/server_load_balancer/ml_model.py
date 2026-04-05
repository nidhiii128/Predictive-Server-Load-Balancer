"""
==============================================================
 Predictive Server Load Balancer
 PART 1 – Machine Learning (Preprocessing + Model Training)
==============================================================
Author  : 3rd Year Computer Engineering Student
Subject : Mini Project – ML + Soft Computing
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # headless matplotlib (no GUI needed)
import matplotlib.pyplot as plt

from sklearn.preprocessing   import MinMaxScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay)
import joblib
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ─── Step 1: Load Dataset ─────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — Loading Dataset")
print("=" * 60)

CSV_PATH = "data/final-complete-data-set.csv"
df = pd.read_csv(CSV_PATH)

# Keep ONLY the relevant columns
COLS = ["timestamp", "cpu_total", "cpu_idle", "load_min1", "load_min5",
        "mem_used", "mem_percent", "network_lo_rx", "network_lo_tx"]
df = df[COLS].copy()

print(f"Shape  : {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))

# ─── Step 2: Preprocessing ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — Preprocessing")
print("=" * 60)

# 2a. Handle missing values
missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0] if missing.sum() > 0 else "None")
df.fillna(df.median(numeric_only=True), inplace=True)

# 2b. Convert timestamp to datetime; extract time features
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]      = df["timestamp"].dt.hour
df["minute"]    = df["timestamp"].dt.minute
df["second"]    = df["timestamp"].dt.second
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)
print("Timestamp converted; time features extracted.")

# ─── Step 3: Target Variable ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — Creating Target Variable (Load Category)")
print("=" * 60)
"""
Rule:
  cpu_total > 80   → HIGH   (label 2)
  cpu_total 50–80  → MEDIUM (label 1)
  cpu_total < 50   → LOW    (label 0)
"""
def categorise(val):
    if val > 80:   return "HIGH"
    elif val > 50: return "MEDIUM"
    else:          return "LOW"

df["load_category"]       = df["cpu_total"].apply(categorise)
df["load_category_label"] = df["load_category"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})

print(df["load_category"].value_counts())

# ─── Step 4: Feature Engineering (Lag Features) ───────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — Feature Engineering (Lag Features)")
print("=" * 60)
"""
Lag features capture 'what happened 1, 2, 3 steps ago'.
This turns the problem into a time-series-aware one for tabular models.
"""
FEATURE_COLS = ["cpu_total", "cpu_idle", "load_min1", "load_min5",
                "mem_used", "mem_percent", "network_lo_rx", "network_lo_tx"]

for col in FEATURE_COLS:
    for lag in [1, 2, 3]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"After lag features, shape: {df.shape}")

# ─── Step 5: Normalisation ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — Normalisation (MinMaxScaler)")
print("=" * 60)

# All numeric columns except target and time-derived columns
exclude = {"load_category", "load_category_label", "timestamp", "hour", "minute", "second"}
scale_cols = [c for c in df.columns if c not in exclude]

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved → models/scaler.pkl")

# ─── Step 6: Train / Test Split ───────────────────────────────────────────────
X_COLS = [c for c in df_scaled.columns
          if c not in {"load_category", "load_category_label", "timestamp"}]

X = df_scaled[X_COLS].values
y = df_scaled["load_category_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ─── Step 7: Model Training ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7 — Model Training")
print("=" * 60)

# ── 7a. Random Forest ────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
print(f"\n[Random Forest] Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred, target_names=["LOW","MEDIUM","HIGH"]))
joblib.dump(rf, "models/random_forest.pkl")

# ── 7b. Logistic Regression ──────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
print(f"\n[Logistic Regression] Accuracy: {lr_acc:.4f}")
print(classification_report(y_test, lr_pred, target_names=["LOW","MEDIUM","HIGH"]))
joblib.dump(lr, "models/logistic_regression.pkl")

# ─── Step 8: Evaluation Plots ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8 — Generating Evaluation Plots")
print("=" * 60)

# ── Plot 1: Confusion Matrices ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, title in zip(axes,
                            [rf_pred, lr_pred],
                            ["Random Forest", "Logistic Regression"]):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["LOW","MEDIUM","HIGH"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{title}\nAccuracy: {accuracy_score(y_test, pred):.4f}")

plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=150)
print("Saved → results/confusion_matrices.png")

# ── Plot 2: CPU Usage Trend ──────────────────────────────────────────────────
sample = df.iloc[:500]
plt.figure(figsize=(14, 4))
plt.plot(sample.index, sample["cpu_total"], color="steelblue", label="cpu_total", linewidth=1)
plt.axhline(80, color="red",    linestyle="--", label="HIGH threshold (80%)")
plt.axhline(50, color="orange", linestyle="--", label="MEDIUM threshold (50%)")
plt.fill_between(sample.index, 80, 100, alpha=0.1, color="red")
plt.fill_between(sample.index, 50, 80,  alpha=0.1, color="orange")
plt.fill_between(sample.index, 0,  50,  alpha=0.1, color="green")
plt.xlabel("Time (row index)")
plt.ylabel("CPU Total (%)")
plt.title("CPU Usage Trend with Load Thresholds (first 500 samples)")
plt.legend()
plt.tight_layout()
plt.savefig("results/cpu_trend.png", dpi=150)
print("Saved → results/cpu_trend.png")

# ── Plot 3: Actual vs Predicted (Random Forest) ──────────────────────────────
idx = np.arange(200)
plt.figure(figsize=(14, 4))
plt.plot(idx, y_test[:200],    "b-",  label="Actual",             linewidth=1.2)
plt.plot(idx, rf_pred[:200],   "r--", label="RF Predicted",       linewidth=1.2)
plt.yticks([0, 1, 2], ["LOW", "MEDIUM", "HIGH"])
plt.xlabel("Sample Index")
plt.ylabel("Load Category")
plt.title("Actual vs Predicted Load Category (Random Forest – first 200 test samples)")
plt.legend()
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=150)
print("Saved → results/actual_vs_predicted.png")

# ── Plot 4: Feature Importance (Random Forest) ────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X_COLS).sort_values(ascending=False)
plt.figure(figsize=(14, 5))
importances[:15].plot(kind="bar", color="steelblue")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
print("Saved → results/feature_importance.png")

# ─── Save model metadata ──────────────────────────────────────────────────────
np.save("models/x_cols.npy", np.array(X_COLS))
print("\n✅  ML Part complete. Best model: Random Forest")
print(f"   RF Accuracy  = {rf_acc:.4f}")
print(f"   LR Accuracy  = {lr_acc:.4f}")
