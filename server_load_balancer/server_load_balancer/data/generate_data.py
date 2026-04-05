"""
generate_data.py
----------------
Generates a synthetic server metrics dataset that mirrors the structure and
statistical properties of final-complete-data-set.csv from Kaggle.

Run this ONLY if you don't have the real CSV.
If you have the real CSV, place it at:
    data/final-complete-data-set.csv
and skip this script entirely.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000  # number of rows

# ── Timestamps: 1-second intervals ──────────────────────────────────────────
timestamps = pd.date_range(start="2021-01-01 00:00:00", periods=N, freq="1s")

# ── Simulate realistic CPU cycles (idle/attack patterns) ────────────────────
# Normal baseline + occasional spikes simulating cryptojacking / high load
cpu_idle  = np.clip(np.random.normal(60, 15, N) + 5 * np.sin(np.linspace(0, 20, N)), 5, 95)
cpu_user  = np.clip(100 - cpu_idle - np.random.uniform(2, 8, N), 0, 80)
cpu_total = np.clip(cpu_user + np.random.uniform(1, 5, N), 0, 100)

# Inject high-load windows (simulating attack or heavy workload)
attack_start = [1200, 2500, 3800]
for s in attack_start:
    cpu_total[s:s+300] = np.clip(np.random.normal(85, 5, 300), 70, 100)
    cpu_idle[s:s+300]  = np.clip(100 - cpu_total[s:s+300], 2, 20)

# ── Load averages (correlated with CPU) ─────────────────────────────────────
load_min1  = np.clip(cpu_total / 25 + np.random.normal(0, 0.3, N), 0, 10)
load_min5  = np.clip(load_min1 * 0.8 + np.random.normal(0, 0.2, N), 0, 10)
load_min15 = np.clip(load_min5 * 0.7 + np.random.normal(0, 0.1, N), 0, 10)

# ── Memory ───────────────────────────────────────────────────────────────────
mem_used    = np.clip(np.random.normal(2e9, 3e8, N), 5e8, 4e9).astype(int)
mem_percent = np.clip(mem_used / 4e9 * 100, 10, 99)

# ── Network (loopback) ───────────────────────────────────────────────────────
network_lo_rx = np.abs(np.random.normal(5e5, 1e5, N)).astype(int)
network_lo_tx = np.abs(np.random.normal(5e5, 1e5, N)).astype(int)

df = pd.DataFrame({
    "timestamp"     : timestamps,
    "cpu_total"     : np.round(cpu_total, 2),
    "cpu_idle"      : np.round(cpu_idle,  2),
    "cpu_user"      : np.round(cpu_user,  2),
    "load_min1"     : np.round(load_min1,  3),
    "load_min5"     : np.round(load_min5,  3),
    "load_min15"    : np.round(load_min15, 3),
    "mem_used"      : mem_used,
    "mem_percent"   : np.round(mem_percent, 2),
    "network_lo_rx" : network_lo_rx,
    "network_lo_tx" : network_lo_tx,
})

df.to_csv("data/final-complete-data-set.csv", index=False)
print(f"✅  Synthetic dataset saved → data/final-complete-data-set.csv  ({N} rows)")
print(df.describe())
