#!/usr/bin/env python3
"""
plot_pareto.py — Pairwise Pareto front analysis for PISR-NSGA-II
Generates 3-way trade-off plots: Domain vs Boundary, Domain vs Complexity, Boundary vs Complexity.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        200,
})
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_all():
    files = glob.glob(os.path.join(RESULTS_DIR, "**", "*_pareto.csv"), recursive=True)
    if not files: return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            # Ensure numeric types
            for col in ["mse_domain", "mse_boundary", "tree_size"]:
                if col in tmp.columns:
                    tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
            dfs.append(tmp)
        except Exception as e:
            print(f"Warning: Skipping {f} due to error: {e}")
            
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["mse_domain", "mse_boundary"])
    return df

def plot_pairwise_pareto(df):
    unique_pdes = df[["pde", "dim"]].drop_duplicates()
    
    for _, row in unique_pdes.iterrows():
        pde_name = row["pde"]
        dim = row["dim"]
        pde_label = f"{pde_name}_{dim}D"
        
        sub = df[(df["pde"] == pde_name) & (df["dim"] == dim)]
        # Filter Rank 1 only for the plot
        pareto = sub[sub["rank"] == 1]
        if pareto.empty: continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plt.subplots_adjust(wspace=0.3)
        
        # 1. Domain MSE vs Boundary MSE (Log-Log)
        ax = axes[0]
        ax.scatter(pareto["mse_domain"], pareto["mse_boundary"], color="#2E86C1", alpha=0.7, edgecolors='k', s=40)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Domain MSE (Physics)")
        ax.set_ylabel("Boundary MSE")
        ax.set_title("Physics vs. Boundary")
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # 2. Domain MSE vs Complexity (Log-Linear)
        ax = axes[1]
        ax.scatter(pareto["mse_domain"], pareto["tree_size"], color="#E67E22", alpha=0.7, edgecolors='k', s=40)
        ax.set_xscale("log")
        ax.set_xlabel("Domain MSE (Physics)")
        ax.set_ylabel("Complexity (Nodes)")
        ax.set_title("Physics vs. Complexity")
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # 3. Boundary MSE vs Complexity (Log-Linear)
        ax = axes[2]
        ax.scatter(pareto["mse_boundary"], pareto["tree_size"], color="#27AE60", alpha=0.7, edgecolors='k', s=40)
        ax.set_xscale("log")
        ax.set_xlabel("Boundary MSE")
        ax.set_ylabel("Complexity (Nodes)")
        ax.set_title("Boundary vs. Complexity")
        ax.grid(True, which="both", ls="-", alpha=0.2)

        fig.suptitle(f"Pareto Trade-offs: {pde_label}", fontsize=12, fontweight='bold', y=1.05)
        
        save_path = os.path.join(RESULTS_DIR, f"{pde_label}_pareto_pairs.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] Generated: {save_path}")

def main():
    print("Generating Pairwise Pareto Plots...")
    df = load_all()
    if df.empty:
        print("No Pareto data found in results/ directory.")
        return
    
    plot_pairwise_pareto(df)
    print("Done.")

if __name__ == "__main__":
    main()
