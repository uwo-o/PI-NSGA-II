#!/usr/bin/env python3
"""
stats_analysis.py — Análisis estadístico completo con Media ± Std.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def wilcoxon_test(a, b):
    if len(a) < 2 or len(b) < 2: return None
    try:
        _, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        return p
    except: return None

def main():
    path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(path): return
    
    df = pd.read_csv(path)
    df["mse_total"] = pd.to_numeric(df["best_mse_domain"], errors='coerce') + pd.to_numeric(df["best_mse_boundary"], errors='coerce')
    df = df.dropna(subset=["mse_total"])
    
    pdes = sorted(df["pde"].unique())

    print("\n" + "="*110)
    print(f"{'PDE Case':<25} {'Tsoulos (Mean ± Std)':<30} {'PI-NSGA-II (Mean ± Std)':<35} {'p-val':<10} {'Winner':<10}")
    print("-" * 110)

    for pde in pdes:
        sub = df[df["pde"] == pde]
        
        a = sub[sub["method"] == "Tsoulos"]["mse_total"].values
        b = sub[sub["method"] == "PI-NSGA-II"]["mse_total"].values
        
        mean_a, std_a = (np.mean(a), np.std(a)) if len(a)>0 else (0, 0)
        mean_b, std_b = (np.mean(b), np.std(b)) if len(b)>0 else (0, 0)
        
        p = wilcoxon_test(a, b)
        p_str = f"{p:.2e}" if p is not None else "N/A"
        
        winner = "Tsoulos" if mean_a < mean_b else "PI-NSGA-II"
        
        str_a = f"{mean_a:.1e} ± {std_a:.1e}"
        str_b = f"{mean_b:.1e} ± {std_b:.1e}"
        
        print(f"{pde:<25} {str_a:<30} {str_b:<35} {p_str:<10} {winner:<10}")

    print("-" * 110)

if __name__ == "__main__":
    main()
