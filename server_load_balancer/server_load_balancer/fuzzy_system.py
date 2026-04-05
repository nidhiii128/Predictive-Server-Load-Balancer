"""
==============================================================
 Predictive Server Load Balancer
 PART 2 – Fuzzy Logic (Soft Computing) Scaling Decision
==============================================================
Uses skfuzzy (scikit-fuzzy) to build a Mamdani fuzzy inference
system that maps (Predicted Load, CPU Usage) → Scaling Decision.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os

os.makedirs("results", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  STEP 1: Define Fuzzy Universe of Discourse (ranges for variables)
# ═══════════════════════════════════════════════════════════════════
# Input 1: predicted_load  – 0 to 100 (represents predicted cpu_total %)
# Input 2: current_cpu     – 0 to 100 (current cpu_total %)
# Output : scaling_action  – 0 to 10  (0=No Scale, 5=Scale Slightly, 10=Scale High)

predicted_load  = ctrl.Antecedent(np.arange(0, 101, 1), "predicted_load")
current_cpu     = ctrl.Antecedent(np.arange(0, 101, 1), "current_cpu")
scaling_action  = ctrl.Consequent(np.arange(0,  11, 1), "scaling_action")

# ═══════════════════════════════════════════════════════════════════
#  STEP 2: Define Membership Functions
# ═══════════════════════════════════════════════════════════════════
"""
Triangular (trimf) and Trapezoidal (trapmf) membership functions.

For predicted_load and current_cpu:
  LOW    → [0, 0, 40, 55]   trapezoidal (flat near 0, ramps up)
  MEDIUM → [45, 60, 70]     triangular  (peaks at 60)
  HIGH   → [65, 80, 100, 100] trapezoidal (flat near 100)

For scaling_action:
  NO_SCALE      → [0, 0, 3]   triangular
  SCALE_SLIGHTLY→ [2, 5, 8]   triangular
  SCALE_HIGH    → [7, 10, 10] triangular
"""

# -- predicted_load membership functions --
predicted_load["LOW"]    = fuzz.trapmf(predicted_load.universe, [0,  0, 40, 55])
predicted_load["MEDIUM"] = fuzz.trimf (predicted_load.universe, [45, 60, 75])
predicted_load["HIGH"]   = fuzz.trapmf(predicted_load.universe, [65, 80, 100, 100])

# -- current_cpu membership functions --
current_cpu["LOW"]    = fuzz.trapmf(current_cpu.universe, [0,  0, 35, 50])
current_cpu["MEDIUM"] = fuzz.trimf (current_cpu.universe, [40, 57, 72])
current_cpu["HIGH"]   = fuzz.trapmf(current_cpu.universe, [62, 78, 100, 100])

# -- scaling_action membership functions --
scaling_action["NO_SCALE"]       = fuzz.trimf(scaling_action.universe, [0,  0,  3])
scaling_action["SCALE_SLIGHTLY"] = fuzz.trimf(scaling_action.universe, [2,  5,  8])
scaling_action["SCALE_HIGH"]     = fuzz.trimf(scaling_action.universe, [7, 10, 10])

# ═══════════════════════════════════════════════════════════════════
#  STEP 3: Define Fuzzy Rules
# ═══════════════════════════════════════════════════════════════════
"""
Fuzzy Rules:
  R1: IF load is HIGH   AND cpu is HIGH   → SCALE HIGH
  R2: IF load is HIGH   AND cpu is MEDIUM → SCALE HIGH
  R3: IF load is MEDIUM AND cpu is HIGH   → SCALE SLIGHTLY
  R4: IF load is MEDIUM AND cpu is MEDIUM → SCALE SLIGHTLY
  R5: IF load is MEDIUM AND cpu is LOW    → SCALE SLIGHTLY
  R6: IF load is LOW    AND cpu is LOW    → NO SCALE
  R7: IF load is LOW    AND cpu is MEDIUM → NO SCALE
  R8: IF load is LOW    AND cpu is HIGH   → SCALE SLIGHTLY  (CPU spike but low predicted load)
  R9: IF load is HIGH   AND cpu is LOW    → SCALE SLIGHTLY  (Predicted spike, CPU still low)
"""
rule1 = ctrl.Rule(predicted_load["HIGH"]   & current_cpu["HIGH"],   scaling_action["SCALE_HIGH"])
rule2 = ctrl.Rule(predicted_load["HIGH"]   & current_cpu["MEDIUM"], scaling_action["SCALE_HIGH"])
rule3 = ctrl.Rule(predicted_load["MEDIUM"] & current_cpu["HIGH"],   scaling_action["SCALE_SLIGHTLY"])
rule4 = ctrl.Rule(predicted_load["MEDIUM"] & current_cpu["MEDIUM"], scaling_action["SCALE_SLIGHTLY"])
rule5 = ctrl.Rule(predicted_load["MEDIUM"] & current_cpu["LOW"],    scaling_action["SCALE_SLIGHTLY"])
rule6 = ctrl.Rule(predicted_load["LOW"]    & current_cpu["LOW"],    scaling_action["NO_SCALE"])
rule7 = ctrl.Rule(predicted_load["LOW"]    & current_cpu["MEDIUM"], scaling_action["NO_SCALE"])
rule8 = ctrl.Rule(predicted_load["LOW"]    & current_cpu["HIGH"],   scaling_action["SCALE_SLIGHTLY"])
rule9 = ctrl.Rule(predicted_load["HIGH"]   & current_cpu["LOW"],    scaling_action["SCALE_SLIGHTLY"])

# ═══════════════════════════════════════════════════════════════════
#  STEP 4: Build the Fuzzy Control System
# ═══════════════════════════════════════════════════════════════════
scaling_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4,
                                    rule5, rule6, rule7, rule8, rule9])
scaling_sim  = ctrl.ControlSystemSimulation(scaling_ctrl)

# ═══════════════════════════════════════════════════════════════════
#  STEP 5: Helper – interpret numeric output as label
# ═══════════════════════════════════════════════════════════════════
def interpret_scaling(value):
    """Map crisp defuzzified value to human-readable scaling label."""
    if value < 3.5:   return "NO SCALE"
    elif value < 6.5: return "SCALE SLIGHTLY"
    else:             return "SCALE HIGH"

# ═══════════════════════════════════════════════════════════════════
#  STEP 6: Core function – get scaling decision
# ═══════════════════════════════════════════════════════════════════
def get_scaling_decision(predicted_load_val: float, current_cpu_val: float) -> dict:
    """
    Inputs:
      predicted_load_val – ML-predicted cpu_total (0-100)
      current_cpu_val    – current actual cpu_total (0-100)
    Returns:
      dict with crisp output value and label
    """
    # Clip to universe bounds
    pl = float(np.clip(predicted_load_val, 0, 100))
    cc = float(np.clip(current_cpu_val,    0, 100))

    scaling_sim.input["predicted_load"] = pl
    scaling_sim.input["current_cpu"]    = cc
    scaling_sim.compute()

    crisp_val = scaling_sim.output["scaling_action"]
    label     = interpret_scaling(crisp_val)

    return {
        "predicted_load"  : pl,
        "current_cpu"     : cc,
        "scaling_crisp"   : round(crisp_val, 3),
        "scaling_decision": label,
    }

# ═══════════════════════════════════════════════════════════════════
#  STEP 7: Plot Membership Functions
# ═══════════════════════════════════════════════════════════════════
def plot_membership_functions():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Predicted Load
    axes[0].plot(predicted_load.universe,
                 fuzz.trapmf(predicted_load.universe, [0, 0, 40, 55]),   "b",  label="LOW")
    axes[0].plot(predicted_load.universe,
                 fuzz.trimf (predicted_load.universe, [45, 60, 75]),     "g",  label="MEDIUM")
    axes[0].plot(predicted_load.universe,
                 fuzz.trapmf(predicted_load.universe, [65, 80, 100, 100]),"r",  label="HIGH")
    axes[0].set_title("Predicted Load (Input 1)")
    axes[0].set_xlabel("CPU %"); axes[0].legend(); axes[0].set_ylim(-0.05, 1.1)

    # Current CPU
    axes[1].plot(current_cpu.universe,
                 fuzz.trapmf(current_cpu.universe, [0, 0, 35, 50]),      "b",  label="LOW")
    axes[1].plot(current_cpu.universe,
                 fuzz.trimf (current_cpu.universe, [40, 57, 72]),        "g",  label="MEDIUM")
    axes[1].plot(current_cpu.universe,
                 fuzz.trapmf(current_cpu.universe, [62, 78, 100, 100]),   "r",  label="HIGH")
    axes[1].set_title("Current CPU (Input 2)")
    axes[1].set_xlabel("CPU %"); axes[1].legend(); axes[1].set_ylim(-0.05, 1.1)

    # Scaling Action
    axes[2].plot(scaling_action.universe,
                 fuzz.trimf(scaling_action.universe, [0, 0, 3]),         "b",  label="NO SCALE")
    axes[2].plot(scaling_action.universe,
                 fuzz.trimf(scaling_action.universe, [2, 5, 8]),         "g",  label="SCALE SLIGHTLY")
    axes[2].plot(scaling_action.universe,
                 fuzz.trimf(scaling_action.universe, [7, 10, 10]),       "r",  label="SCALE HIGH")
    axes[2].set_title("Scaling Action (Output)")
    axes[2].set_xlabel("Scale Level"); axes[2].legend(); axes[2].set_ylim(-0.05, 1.1)

    plt.suptitle("Fuzzy Membership Functions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/membership_functions.png", dpi=150, bbox_inches="tight")
    print("Saved → results/membership_functions.png")

# ═══════════════════════════════════════════════════════════════════
#  Quick standalone test of the fuzzy system
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_membership_functions()

    print("\n" + "=" * 55)
    print("  Fuzzy System – Quick Test Cases")
    print("=" * 55)
    test_cases = [
        (85, 88, "Should → SCALE HIGH"),
        (65, 55, "Should → SCALE HIGH / SLIGHTLY"),
        (55, 60, "Should → SCALE SLIGHTLY"),
        (30, 25, "Should → NO SCALE"),
        (20, 85, "Should → SCALE SLIGHTLY (CPU spike)"),
        (90, 15, "Should → SCALE SLIGHTLY (predicted spike)"),
    ]
    for pl, cc, note in test_cases:
        res = get_scaling_decision(pl, cc)
        print(f"  Load={pl:3d}%  CPU={cc:3d}%  →  Crisp={res['scaling_crisp']:.2f}"
              f"  Decision: {res['scaling_decision']:<16s}  ({note})")
