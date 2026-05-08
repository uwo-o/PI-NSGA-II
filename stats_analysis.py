#!/usr/bin/env python3
"""
stats_analysis.py — Análisis estadístico completo: PI-NSGA-II vs Tsoulos (GE)
Soporta problemas 1D y 2D dinámicamente.
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})
warnings.filterwarnings("ignore")

# Rutas dinámicas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
COLORS = {"Tsoulos": "#E07535", "PI-NSGA-II": "#2E86C1", "PINN": "#27AE60"}

def load_data(path):
    df = pd.read_csv(path)
    if "hypervolume" in df.columns and df["hypervolume"].max() > 1.5:
        df["hypervolume"] /= 1e8
    return df

def wilcoxon_test(a, b):
    if len(a) < 2 or len(b) < 2: return None, None, None
    try:
        U, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        mu_U, sig_U = n1*n2/2.0, np.sqrt(n1*n2*(n1+n2+1)/12.0)
        Z = (U - mu_U) / (sig_U + 1e-12)
        r = abs(Z) / np.sqrt(n1 + n2)
        return U, p, r
    except: return None, None, None

def main():
    path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    df = load_data(path)
    pdes = sorted(df["pde"].unique())
    methods = ["Tsoulos", "PI-NSGA-II"]

    print("\n" + "="*80)
    print(f"{'PDE Case':<20} {'Metric':<15} {'Tsoulos (mean)':<15} {'PI-NSGA-II':<15} {'p-val':<10}")
    print("-" * 80)

    stats_rows = []
    for pde in pdes:
        sub = df[df["pde"] == pde]
        for metric, label in [("best_mse_domain", "Dom MSE"), ("best_mse_boundary", "BC MSE")]:
            a = sub[sub["method"] == "Tsoulos"][metric].values
            b = sub[sub["method"] == "PI-NSGA-II"][metric].values
            
            mean_a, mean_b = np.mean(a) if len(a)>0 else 0, np.mean(b) if len(b)>0 else 0
            _, p, _ = wilcoxon_test(a, b)
            p_str = f"{p:.3e}" if p is not None else "N/A"
            
            print(f"{pde:<20} {label:<15} {mean_a:>15.2e} {mean_b:>15.2e} {p_str:>10}")
            stats_rows.append({"pde": pde, "metric": label, "mean_ts": mean_a, "mean_pi": mean_b, "p_val": p})

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "stats_summary.csv")
    pd.DataFrame(stats_rows).to_csv(summary_path, index=False)
    
    # Plotting (Boxplots)
    n_pdes = len(pdes)
    fig, axes = plt.subplots(1, n_pdes, figsize=(4*n_pdes, 5))
    if n_pdes == 1: axes = [axes]
    for i, pde in enumerate(pdes):
        sub = df[df["pde"] == pde]
        data = [sub[sub["method"] == m]["best_mse_domain"].values for m in methods]
        axes[i].boxplot(data, labels=methods, patch_artist=True)
        axes[i].set_title(pde)
        axes[i].set_yscale("log")
    
    plt.tight_layout()
    boxplot_path = os.path.join(RESULTS_DIR, "boxplot_mse_dynamic.png")
    fig.savefig(boxplot_path, dpi=160)
    print(f"\nStats analysis completed. Results in {summary_path}")

if __name__ == "__main__":
    main()
