#!/usr/bin/env python3
"""
Hyperparameter Tuning for PI-NSGA-II (Random Search)
Runs indefinitely until a time limit is reached.
Modifies `include/common.hpp` and recompiles automatically.
"""

import os
import sys
import time
import re
import random
import subprocess
import argparse
import pandas as pd
import datetime

COMMON_HPP_PATH = "include/common.hpp"
RESULTS_CSV_PATH = "results/all_runs_summary.csv"
TUNING_LOG_PATH = "tuning_results.csv"

# Search Space
SPACE = {
    "POP_SIZE":       [60, 100, 120, 150, 200, 250, 300],
    "MAX_GEN":        [100, 150, 200, 300, 400, 500],
    "CROSSOVER_PROB": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    "MUTATION_PROB":  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "MAX_TREE_DEPTH": [2, 3, 4, 5],
    "ALPHA_WEIGHT":   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}

def sample_params():
    return {k: random.choice(v) for k, v in SPACE.items()}

def modify_common_hpp(params):
    with open(COMMON_HPP_PATH, "r") as f:
        content = f.read()
    
    # Regex replacements
    content = re.sub(r'(constexpr\s+int\s+POP_SIZE\s*=\s*)\d+;', rf'\g<1>{params["POP_SIZE"]};', content)
    content = re.sub(r'(constexpr\s+int\s+MAX_GEN\s*=\s*)\d+;', rf'\g<1>{params["MAX_GEN"]};', content)
    content = re.sub(r'(constexpr\s+int\s+MAX_TREE_DEPTH\s*=\s*)\d+;', rf'\g<1>{params["MAX_TREE_DEPTH"]};', content)
    content = re.sub(r'(constexpr\s+double\s+CROSSOVER_PROB\s*=\s*)[\d\.]+;', rf'\g<1>{params["CROSSOVER_PROB"]};', content)
    content = re.sub(r'(constexpr\s+double\s+MUTATION_PROB\s*=\s*)[\d\.]+;', rf'\g<1>{params["MUTATION_PROB"]};', content)
    content = re.sub(r'(constexpr\s+double\s+ALPHA_WEIGHT\s*=\s*)[\d\.]+;', rf'\g<1>{params["ALPHA_WEIGHT"]};', content)

    with open(COMMON_HPP_PATH, "w") as f:
        f.write(content)

def recompile():
    print("  [Tuner] Recompiling project...")
    res = subprocess.run("cd build && make -j4", shell=True, capture_output=True)
    if res.returncode != 0:
        print(f"  [ERROR] Compilation failed:\n{res.stderr.decode('utf-8')}")
        sys.exit(1)

def evaluate(runs):
    print(f"  [Tuner] Running PI-NSGA-II (Runs={runs})...")
    res = subprocess.run(f"./build/pi_nsga2 --runs {runs}", shell=True, capture_output=True)
    if res.returncode != 0:
        print(f"  [ERROR] Execution failed:\n{res.stderr.decode('utf-8')}")
        return None
    
    # Parse results
    if not os.path.exists(RESULTS_CSV_PATH):
        return None
        
    df = pd.read_csv(RESULTS_CSV_PATH)
    pi = df[df["method"] == "PI-NSGA-II"]
    if pi.empty: return None

    # Score: minimize sum of best_mse_domain + best_mse_boundary across all equations
    score = pi["best_mse_domain"].mean() + pi["best_mse_boundary"].mean()
    hv = pi["hypervolume"].mean()
    runtime = pi["runtime_s"].mean()
    
    return score, hv, runtime

def main():
    parser = argparse.ArgumentParser(description="Overnight Hyperparameter Tuner for PI-NSGA-II")
    parser.add_argument("--minutes", type=float, default=480, help="Total minutes to run (default: 8 hours = 480)")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs per configuration (default: 10)")
    args = parser.parse_args()

    max_seconds = args.minutes * 60
    print("="*60)
    print(f"  PI-NSGA-II HYPERPARAMETER TUNING (Random Search)")
    print(f"  Time Limit: {args.minutes} minutes ({max_seconds} s)")
    print(f"  Runs per config: {args.runs}")
    print(f"  Optimizing for: Min (MSE Domain + MSE BC)")
    print("="*60)

    start_time = time.time()
    best_score = float('inf')
    best_params = None
    
    trial = 0
    history = []

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\n[Tuner] Time limit reached! ({elapsed/60:.1f} minutes elapsed).")
            break
            
        trial += 1
        print(f"\n--- Trial {trial} --- [{datetime.timedelta(seconds=int(elapsed))} / {datetime.timedelta(seconds=int(max_seconds))}]")
        params = sample_params()
        print(f"  Params: {params}")
        
        modify_common_hpp(params)
        recompile()
        
        t0 = time.time()
        res = evaluate(args.runs)
        rt = time.time() - t0
        
        if res is None:
            print("  [ERROR] Evaluation returned None. Skipping.")
            continue
            
        score, hv, runtime_per_pde = res
        print(f"  -> Score (MSE): {score:.4f}  |  HV: {hv:.1e}  |  Avg Runtime/PDE: {runtime_per_pde:.2f}s  |  Eval Time: {rt:.1f}s")
        
        is_best = False
        if score < best_score:
            best_score = score
            best_params = params
            is_best = True
            print("  *** NEW BEST CONFIGURATION! ***")
            
        # Log
        row = {"trial": trial, "score_mse": score, "hypervolume": hv, "eval_time_s": rt}
        row.update(params)
        history.append(row)
        
        df_hist = pd.DataFrame(history)
        df_hist.to_csv(TUNING_LOG_PATH, index=False)

    print("="*60)
    print("  TUNING FINISHED")
    print("="*60)
    if best_params:
        print(f"  Best Score (MSE): {best_score:.4f}")
        print(f"  Best Params: {best_params}")
        print("\n  -> Restoring best parameters into common.hpp and compiling...")
        modify_common_hpp(best_params)
        recompile()
        print("  -> Done! You can now run `bash run_pipeline.sh`.")
    else:
        print("  No valid configurations were evaluated.")

if __name__ == "__main__":
    main()
