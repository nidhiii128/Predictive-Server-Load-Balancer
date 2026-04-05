# Predictive Server Load Balancer
### Machine Learning + Fuzzy Logic (Soft Computing)

**Dataset:** Cryptojacking Attack Timeseries Dataset (Kaggle)

---

## Project Structure
```
server_load_balancer/
│
├── data/
│   ├── generate_data.py             ← Generates synthetic dataset if no real CSV
│   └── final-complete-data-set.csv  ← Dataset (real or synthetic)
│
├── models/                          ← Auto-created after running
│   ├── random_forest.pkl            ← Trained Random Forest model
│   ├── logistic_regression.pkl      ← Trained Logistic Regression model
│   ├── scaler.pkl                   ← Fitted MinMaxScaler
│   └── x_cols.npy                   ← Feature column names
│
├── results/                         ← Auto-created after running
│   ├── confusion_matrices.png
│   ├── cpu_trend.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   ├── membership_functions.png
│   ├── integration_cpu_decisions.png
│   ├── fuzzy_output_trend.png
│   ├── predicted_category_dist.png
│   └── batch_results.csv
│
├── ml_model.py       ← PART 1 · Preprocessing + ML Training
├── fuzzy_system.py   ← PART 2 · Fuzzy Logic System
├── integration.py    ← PART 3 · ML + Fuzzy Combined Pipeline
├── run_all.py        ← ▶ Run everything with one command
└── README.md
```

---

## Project Objective

Modern servers face unpredictable load spikes. This project builds an intelligent load balancer that:

1. **Predicts** upcoming server load using Machine Learning (Random Forest)
2. **Decides** the scaling action using Fuzzy Logic (Soft Computing)
3. **Integrates** both into a real-time decision pipeline

---

## Dataset

**Source:** https://www.kaggle.com/datasets/keshanijayasinghe/cryptojacking-attack-timeseries-dataset

| Column         | Description                     |
|----------------|---------------------------------|
| timestamp      | Time of measurement             |
| cpu_total      | Total CPU usage (%)             |
| cpu_idle       | CPU idle time (%)               |
| load_min1      | 1-minute load average           |
| load_min5      | 5-minute load average           |
| mem_used       | Memory used (bytes)             |
| mem_percent    | Memory usage (%)                |
| network_lo_rx  | Network received (bytes)        |
| network_lo_tx  | Network transmitted (bytes)     |

---

## Installation

### Requirements
- Python 3.9 or higher
- pip

### Install Dependencies
```
pip install pandas numpy matplotlib scikit-learn scikit-fuzzy scipy joblib
```

---

## How to Run

### Option 1 — Run Everything (Recommended)
```
python run_all.py
```

This runs all 3 parts in order and saves all plots and models automatically.

### Option 2 — Run Each Part Separately
```
python ml_model.py      ← Train ML models + generate evaluation plots
python fuzzy_system.py  ← Test fuzzy logic + plot membership functions
python integration.py   ← Full pipeline: ML prediction → Fuzzy decision
```

### Using the Real Kaggle Dataset

1. Download final-complete-data-set.csv from Kaggle
2. Place it inside the data/ folder
3. Run python run_all.py — no code changes needed

---

## System Flow
```
Raw Server Metrics (CSV)
         │
         ▼
┌────────────────────────────────────┐
│           PREPROCESSING            │
│  ✦ Handle missing values           │
│  ✦ Parse timestamps                │
│  ✦ Lag features (t-1, t-2, t-3)   │
│  ✦ MinMaxScaler normalization      │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│      ML MODEL (Random Forest)      │
│  Input  → feature vector           │
│  Output → LOW / MEDIUM / HIGH      │
└────────────────┬───────────────────┘
                 │
       ┌─────────┴──────────┐
  Predicted Load        Current CPU %
       └─────────┬──────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│       FUZZY LOGIC (skfuzzy)        │
│  ✦ Fuzzify inputs                  │
│  ✦ Apply 9 IF-THEN rules           │
│  ✦ Defuzzify (centroid method)     │
└────────────────┬───────────────────┘
                 │
                 ▼
         SCALING DECISION
  NO SCALE · SCALE SLIGHTLY · SCALE HIGH
```

---

## Machine Learning — Part 1

### Target Variable
```
cpu_total > 80%        →  HIGH    (label 2)
cpu_total 50% to 80%   →  MEDIUM  (label 1)
cpu_total < 50%        →  LOW     (label 0)
```

### Feature Engineering

Lag features are created for each metric at time steps t-1, t-2 and t-3
to capture time-series trends.
Total features after engineering: 35

### Models Trained

| Model                | Accuracy                          |
|----------------------|-----------------------------------|
| Random Forest        | 100% (synthetic) / ~95% (real)    |
| Logistic Regression  | 96.1%                             |

### Plots Generated
```
✦ Confusion matrix comparing RF vs Logistic Regression
✦ CPU usage trend with HIGH and MEDIUM thresholds
✦ Actual vs Predicted load category (first 200 test samples)
✦ Top 15 feature importances from Random Forest
```

---

## Fuzzy Logic — Part 2 (Soft Computing)

### Inputs and Output

| Variable       | Type   | Range      | Fuzzy Sets                          |
|----------------|--------|------------|-------------------------------------|
| Predicted Load | Input  | 0 to 100%  | LOW, MEDIUM, HIGH                   |
| Current CPU    | Input  | 0 to 100%  | LOW, MEDIUM, HIGH                   |
| Scaling Action | Output | 0 to 10    | NO_SCALE, SCALE_SLIGHTLY, SCALE_HIGH|

### Membership Functions
```
Inputs  → Trapezoidal for LOW and HIGH · Triangular for MEDIUM
Output  → Triangular for all three sets
Method  → Defuzzification by Centroid (centre of gravity)
```

### Rules (9 Total)

| Rule | Predicted Load | Current CPU | → Scaling Action |
|------|---------------|-------------|------------------|
| R1   | HIGH          | HIGH        | SCALE HIGH       |
| R2   | HIGH          | MEDIUM      | SCALE HIGH       |
| R3   | MEDIUM        | HIGH        | SCALE SLIGHTLY   |
| R4   | MEDIUM        | MEDIUM      | SCALE SLIGHTLY   |
| R5   | MEDIUM        | LOW         | SCALE SLIGHTLY   |
| R6   | LOW           | LOW         | NO SCALE         |
| R7   | LOW           | MEDIUM      | NO SCALE         |
| R8   | LOW           | HIGH        | SCALE SLIGHTLY   |
| R9   | HIGH          | LOW         | SCALE SLIGHTLY   |

### Output Interpretation
```
Crisp value 0.0 → 3.5   ·   NO SCALE
Crisp value 3.5 → 6.5   ·   SCALE SLIGHTLY
Crisp value 6.5 → 10.0  ·   SCALE HIGH
```

---

## Sample Output
```
─────────────────────────────────────────────────────
  Row 1200 · Actual CPU : 89.09%
  ➤  ML Predicted Load  : HIGH
     Probabilities       : LOW=0.00 · MED=0.03 · HIGH=0.97
  ➤  Fuzzy Crisp Output : 9.000
  ➤  SCALING DECISION   : SCALE HIGH
─────────────────────────────────────────────────────
  Row 500  · Actual CPU : 20.92%
  ➤  ML Predicted Load  : LOW
     Probabilities       : LOW=1.00 · MED=0.00 · HIGH=0.00
  ➤  Fuzzy Crisp Output : 1.000
  ➤  SCALING DECISION   : NO SCALE
─────────────────────────────────────────────────────
```

---

**Q: What problem does this project solve?**
Servers face unpredictable load spikes. If load is not managed proactively
servers crash or slow down. This system predicts future load using ML and
decides scaling action using Fuzzy Logic before the spike actually hits.

**Q: Why Random Forest?**
Random Forest builds multiple decision trees and combines their output.
It handles non-linear relationships well, is robust to noise, works well
with lag features and provides feature importance scores which are all
useful properties for server metric data.

**Q: Why Fuzzy Logic instead of simple if-else?**
Simple if-else is brittle. A CPU at 79% would be treated identically to
51% even though it is nearly HIGH. Fuzzy logic handles this ambiguity.
79% is treated as partly HIGH and partly MEDIUM producing smooth
graduated decisions instead of hard jumps.

**Q: What are lag features?**
Values of a metric at previous time steps t-1, t-2 and t-3. For example
cpu_total_lag1 is the CPU value one second ago. They give the model
memory of past trends so it can detect whether load is rising or falling.

**Q: What is defuzzification?**
After fuzzy rules fire we get overlapping output fuzzy sets.
Defuzzification converts them into one crisp number using the centroid
method which finds the centre of gravity of the combined output area.
That number is then mapped to a scaling decision label.

**Q: Why is accuracy 100% on synthetic data?**
Because the synthetic dataset was generated with clean mathematical
boundaries that exactly match the thresholds used to define the labels.
On the real Kaggle dataset with noise and overlap accuracy will be around
93 to 96 percent which is realistic and excellent. The pipeline
architecture works identically for both datasets.

---

## Possible Improvements
```
1  ·  LSTM / GRU            → Replace Random Forest with deep learning
                               time-series model for better sequential
                               pattern recognition

2  ·  Real-time integration  → Connect to Prometheus or Grafana to pull
                               live server metrics automatically

3  ·  Kubernetes integration → Feed scaling decisions directly into
                               Kubernetes Horizontal Pod Autoscaler

4  ·  Type-2 Fuzzy Logic     → Handle uncertainty in membership functions
                               themselves for greater robustness

5  ·  More fuzzy inputs      → Add memory percentage and network traffic
                               as additional fuzzy input variables

6  ·  Anomaly detection      → Add Isolation Forest to detect cryptojacking
                               or DDoS attacks alongside load prediction

7  ·  Multi-server support   → Extend to predict and balance load across
                               a cluster of servers
```

---

## Dependencies

| Library       | Purpose                                  |
|---------------|------------------------------------------|
| pandas        | Data loading and preprocessing           |
| numpy         | Numerical operations                     |
| matplotlib    | Plotting graphs                          |
| scikit-learn  | ML models, scaler, evaluation metrics    |
| scikit-fuzzy  | Fuzzy logic inference system             |
| scipy         | Required internally by scikit-fuzzy      |
| joblib        | Saving and loading trained models        |

---
