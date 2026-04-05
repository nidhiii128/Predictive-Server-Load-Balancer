# Predictive Server Load Balancer
### Using Machine Learning + Fuzzy Logic (Soft Computing)
**3rd Year Computer Engineering – Mini Project**

---

## 📁 Folder Structure

```
server_load_balancer/
│
├── data/
│   ├── generate_data.py          ← Generates synthetic dataset (skip if you have real CSV)
│   └── final-complete-data-set.csv  ← Dataset (real or synthetic)
│
├── models/
│   ├── random_forest.pkl         ← Trained Random Forest model
│   ├── logistic_regression.pkl   ← Trained Logistic Regression model
│   ├── scaler.pkl                ← MinMaxScaler (used during prediction)
│   └── x_cols.npy                ← Feature column names (saved for consistency)
│
├── results/
│   ├── confusion_matrices.png       ← RF vs LR confusion matrix
│   ├── cpu_trend.png                ← CPU usage with thresholds
│   ├── actual_vs_predicted.png      ← Actual vs Predicted load category
│   ├── feature_importance.png       ← Top 15 features (Random Forest)
│   ├── membership_functions.png     ← Fuzzy membership function plots
│   ├── integration_cpu_decisions.png← CPU trend + scaling decisions
│   ├── fuzzy_output_trend.png       ← Fuzzy crisp output over time
│   ├── predicted_category_dist.png  ← Distribution of predicted categories
│   └── batch_results.csv            ← Full batch prediction results
│
├── ml_model.py       ← PART 1: Data preprocessing + ML training
├── fuzzy_system.py   ← PART 2: Fuzzy Logic system (skfuzzy)
├── integration.py    ← PART 3: ML + Fuzzy combined pipeline
├── run_all.py        ← ▶ Run everything in one command
└── README.md         ← This file
```

---

## ⚙️ Setup & Installation

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn scikit-fuzzy scipy joblib
```

### 2. Dataset
**Option A – Use the real Kaggle dataset:**
```
Download: https://www.kaggle.com/datasets/keshanijayasinghe/cryptojacking-attack-timeseries-dataset
Place file at: data/final-complete-data-set.csv
```

**Option B – Use the synthetic dataset (already generated):**
```bash
python data/generate_data.py
```
The synthetic dataset mirrors the real dataset's columns, distributions and statistics.
All code works identically with either dataset.

### 3. Run the Full Project
```bash
python run_all.py
```

### 4. Run Individual Parts
```bash
python ml_model.py       # Train ML models
python fuzzy_system.py   # Test fuzzy logic
python integration.py    # Full pipeline
```

---

## 🔄 How It Works (System Flow)

```
Raw Server Metrics (CSV)
        │
        ▼
  ┌─────────────────────────────────┐
  │  PREPROCESSING                  │
  │  • Handle missing values        │
  │  • Parse timestamps             │
  │  • Create lag features (t-1..3) │
  │  • Normalize with MinMaxScaler  │
  └───────────────┬─────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────┐
  │  ML MODEL (Random Forest)       │
  │  • Input: scaled feature vector │
  │  • Output: LOW / MEDIUM / HIGH  │
  └───────────────┬─────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
Predicted Load          Current CPU %
        │                    │
        └─────────┬──────────┘
                  ▼
  ┌─────────────────────────────────┐
  │  FUZZY LOGIC (skfuzzy)          │
  │  • Fuzzify both inputs          │
  │  • Apply 9 IF-THEN rules        │
  │  • Aggregate output fuzzy sets  │
  │  • Defuzzify (centroid method)  │
  └───────────────┬─────────────────┘
                  │
                  ▼
       SCALING DECISION
   NO SCALE / SCALE SLIGHTLY / SCALE HIGH
```

---

## 📊 Machine Learning Details

| Item | Detail |
|------|--------|
| Algorithm | Random Forest (primary), Logistic Regression (comparison) |
| Features | cpu_total, cpu_idle, load_min1, load_min5, mem_used, mem_percent, network_lo_rx, network_lo_tx + lag-1/2/3 of each |
| Target | Load Category: LOW (cpu<50) / MEDIUM (50-80) / HIGH (>80) |
| Scaler | MinMaxScaler (0 to 1) |
| Train/Test Split | 80% / 20% |
| RF Accuracy | ~100% on synthetic data; ~95%+ on real data |
| LR Accuracy | ~96% |

---

## 🔀 Fuzzy Logic Details

### Inputs & Outputs
| Variable | Range | Membership Sets |
|----------|-------|-----------------|
| Predicted Load | 0–100% | LOW, MEDIUM, HIGH |
| Current CPU | 0–100% | LOW, MEDIUM, HIGH |
| Scaling Action | 0–10 | NO_SCALE, SCALE_SLIGHTLY, SCALE_HIGH |

### Rules (9 total)
| Rule | Predicted Load | Current CPU | → Scaling Action |
|------|---------------|-------------|-----------------|
| R1 | HIGH | HIGH | SCALE HIGH |
| R2 | HIGH | MEDIUM | SCALE HIGH |
| R3 | MEDIUM | HIGH | SCALE SLIGHTLY |
| R4 | MEDIUM | MEDIUM | SCALE SLIGHTLY |
| R5 | MEDIUM | LOW | SCALE SLIGHTLY |
| R6 | LOW | LOW | NO SCALE |
| R7 | LOW | MEDIUM | NO SCALE |
| R8 | LOW | HIGH | SCALE SLIGHTLY |
| R9 | HIGH | LOW | SCALE SLIGHTLY |

### Defuzzification
Method: **Centroid** (centre of gravity)
- Output 0–3.5 → NO SCALE
- Output 3.5–6.5 → SCALE SLIGHTLY
- Output 6.5–10 → SCALE HIGH

---

## 🎓 Viva Explanation (Simple Language)

### Q1: What is the problem you are solving?
**A:** Servers have varying load throughout the day. If load gets too high, the server crashes or slows down. Our system predicts upcoming high load using ML and then uses Fuzzy Logic to decide whether to scale up resources (add more servers or CPU allocation) proactively.

### Q2: Why Machine Learning?
**A:** Server metrics like CPU usage, memory, and load averages follow patterns over time. A Random Forest model learns these patterns from historical data and can classify current conditions as LOW, MEDIUM, or HIGH load. It also uses **lag features** (previous 1, 2, 3 time steps) to capture the time-series nature of the data.

### Q3: Why Fuzzy Logic (Soft Computing)?
**A:** Real-world decisions are not black-and-white. A CPU at 79% is "almost HIGH" — fuzzy logic handles this gracefully using **degrees of membership**. Instead of hard thresholds, it says "this is 60% HIGH and 40% MEDIUM", applies rules, and produces a smooth scaling decision. This is more robust than simple if-else rules.

### Q4: How do ML and Fuzzy Logic work together?
**A:** They are integrated in a pipeline:
1. ML model takes historical + current metrics → predicts load category
2. That predicted category (LOW/MEDIUM/HIGH) maps to a representative CPU % value
3. Fuzzy system takes that predicted % + actual current CPU % as inputs
4. Fuzzy rules fire, and the defuzzified output gives the final scaling decision

### Q5: What are lag features?
**A:** Lag features are the values of a metric at previous time steps. For example, `cpu_total_lag1` = cpu_total one second ago. They help the ML model understand trends — e.g., if CPU was 40% → 60% → 78%, it's rising and likely to hit HIGH soon.

### Q6: What is defuzzification?
**A:** After fuzzy rules fire, we have overlapping fuzzy output sets. Defuzzification converts these back to a single crisp number. We use the **centroid method** — it finds the centre of gravity of the combined output area. That number (0–10) maps to our scaling decision labels.

### Q7: What is MinMaxScaler and why do we use it?
**A:** It scales all features to the range [0, 1]. Without this, features with large values (like `mem_used` in bytes) would dominate features with small values (like `load_min1`), biasing the model. Scaling ensures all features are treated equally.

---

## 🚀 Possible Improvements

1. **LSTM / GRU** – Replace Random Forest with a deep learning time-series model (LSTM) for better sequential pattern recognition
2. **Online Learning** – Update the model continuously as new server data arrives
3. **More fuzzy inputs** – Add memory % and network activity as additional fuzzy inputs
4. **Type-2 Fuzzy Logic** – Handle uncertainty in the membership functions themselves
5. **Anomaly Detection** – Detect cryptojacking or DDoS attacks using Isolation Forest
6. **Real API Integration** – Connect to Prometheus / Grafana to pull live server metrics
7. **Kubernetes Integration** – Hook the scaling decision into Kubernetes Horizontal Pod Autoscaler
8. **Multi-server support** – Extend to predict and balance load across a cluster of servers

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Random Forest Accuracy | ~100% (synthetic) / ~95% (real data) |
| Logistic Regression Accuracy | ~96% |
| Fuzzy Rules | 9 |
| Output Classes | NO SCALE / SCALE SLIGHTLY / SCALE HIGH |
| Plots Generated | 8 PNG files in results/ |

---

*Mini Project – ML + Soft Computing | 3rd Year Computer Engineering*
