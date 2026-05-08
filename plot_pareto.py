#!/usr/bin/env python3
"""
plot_pareto.py — Publication-quality Pareto analysis
Soporta problemas 1D y 2D.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
})
warnings.filterwarnings("ignore")

# Rutas dinámicas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PDE_ORDER   = ["Laplace", "Poisson", "Helmholtz", "Schrodinger"]
DIMS        = [1, 2]
LOG_EPS     = 1e-10

STYLE = {
    "Tsoulos":    dict(color="#E07535", marker="s", zorder=3, lw=1.8),
    "PI-NSGA-II": dict(color="#2E86C1", marker="o", zorder=4, lw=1.8),
    "PINN":       dict(color="#27AE60", marker="^", zorder=5, lw=1.8),
}

def load_all():
    files = glob.glob(os.path.join(RESULTS_DIR, "**", "*_pareto.csv"), recursive=True)
    if not files: raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df

def plot_convergence(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Convergence History — Best Total MSE per Generation", fontsize=14, fontweight="bold")
    
    for row, d in enumerate(DIMS):
        for col, pde in enumerate(PDE_ORDER):
            ax = axes[row, col]
            label_suffix = "_1D" if d == 1 else "_2D"
            for method in ["Tsoulos", "PI-NSGA-II"]:
                suffix_file = "pi" if method == "PI-NSGA-II" else "tsoulos"
                fname = os.path.join(RESULTS_DIR, f"{pde}{label_suffix}_{suffix_file}_convergence.csv")
                if not os.path.exists(fname): continue
                
                h = pd.read_csv(fname)
                y = h["best_total_mse"].clip(lower=LOG_EPS)
                ax.plot(h["gen"], y, label=method, color=STYLE[method]["color"], lw=1.5)

            # PINN baseline
            pinn_sub = df[(df["pde"] == pde) & (df["dim"] == d) & (df["method"] == "PINN")]
            if not pinn_sub.empty:
                p_mse = pinn_sub["mse_domain"].min() + pinn_sub["mse_boundary"].min()
                ax.axhline(y=max(p_mse, LOG_EPS), color=STYLE["PINN"]["color"], ls="--", label="PINN")

            ax.set_title(f"{pde} {d}D", fontweight="bold")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if col == 0: ax.set_ylabel("Min Total MSE")
            if row == 1: ax.set_xlabel("Generation")
            if col == 3 and row == 0: ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "convergence_history.png"), dpi=160)
    plt.close(fig)

def print_analysis(df):
    print("\n" + "="*90)
    print(f"{'Equation':<15} {'Dim':<5} {'Metric':<15} {'Tsoulos':>12} {'PI-NSGA-II':>12} {'PINN':>12} {'Winner':>10}")
    print("-" * 90)

    for pde in PDE_ORDER:
        for d in DIMS:
            sub = df[(df["pde"] == pde) & (df["dim"] == d)]
            if sub.empty: continue

            for metric, label in [("mse_domain", "Dom MSE"), ("mse_boundary", "BC MSE")]:
                vals = {}
                for m in ["Tsoulos", "PI-NSGA-II", "PINN"]:
                    m_sub = sub[sub["method"] == m]
                    vals[m] = m_sub[metric].min() if not m_sub.empty else float('inf')
                
                winner = min(vals, key=vals.get)
                pde_str = pde if label == "Dom MSE" else ""
                dim_str = str(d) if label == "Dom MSE" else ""
                print(f"{pde_str:<15} {dim_str:<5} {label:<15} {vals['Tsoulos']:>12.2e} {vals['PI-NSGA-II']:>12.2e} {vals['PINN']:>12.2e} {winner:>10}")
            print("-" * 90)

def main():
    df = load_all()
    print(f"Loaded {len(df)} points.")
    print_analysis(df)
    plot_convergence(df)
    print(f"Plots saved in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
