"""
==============================================================
 Predictive Server Load Balancer
 run_all.py – Run the entire project in one command
==============================================================
Steps:
  1. Generate synthetic dataset (skip if you have the real CSV)
  2. Train ML models
  3. Run fuzzy system tests
  4. Run integration (ML + Fuzzy combined)
"""

import subprocess, sys, os

def run(script, desc):
    print(f"\n{'='*60}")
    print(f"  ▶  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed with code {result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Step 1: Generate dataset only if real CSV not present
    csv_path = "data/final-complete-data-set.csv"
    if not os.path.exists(csv_path):
        run("data/generate_data.py", "Generating synthetic dataset")
    else:
        print(f"\n✅  Using existing dataset: {csv_path}")

    # Step 2: Train ML models
    run("ml_model.py", "Machine Learning – Preprocessing & Training")

    # Step 3: Test fuzzy system standalone
    run("fuzzy_system.py", "Fuzzy Logic – System Test & Membership Plots")

    # Step 4: Full integration
    run("integration.py", "Integration – ML + Fuzzy Logic Combined")

    print("\n" + "="*60)
    print("  ✅  ALL STEPS COMPLETE")
    print("  📁  Results saved in: results/")
    print("  📁  Models  saved in: models/")
    print("="*60)
    print("\nGenerated files:")
    for f in sorted(os.listdir("results/")):
        print(f"  results/{f}")
