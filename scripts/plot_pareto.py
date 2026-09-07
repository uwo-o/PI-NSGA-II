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
        if sub.empty: continue
        # 3 objetivos (mse_domain, mse_boundary, tree_size) -> los 3 pares
        # posibles, uno por columna. No dominados (rank==1, el frente de
        # Pareto real) en verde lima; dominados (rank>1) en gris de fondo
        # para dar contexto de que tan lejos quedaron del frente.
        dominated = sub[sub["rank"] != 1]
        pareto = sub[sub["rank"] == 1]
        if pareto.empty: continue

        LIME = "#9ACD32"
        GRAY = "#B0B0B0"

        # Muchas EDPs (la mayoria de las 1D, y ahora tambien Thomas-Fermi/
        # Lane-Emden en 2D/1D) usan el ansatz de frontera EXACTO — ver
        # apply_boundary_ansatz en pi_solver.cpp — que hace mse_boundary
        # IDENTICAMENTE 0 por construccion para (casi) todo individuo. Antes
        # esta funcion siempre graficaba esa columna en escala log en los dos
        # paneles que la usan ("Physics vs. Boundary", "Boundary vs.
        # Complexity") — log(0) no es graficable, asi que matplotlib
        # simplemente OMITIA todos los puntos, dejando esos paneles casi
        # vacios y dando la falsa impresion de "muy pocos puntos dominados/no
        # dominados" cuando en realidad el problema era la escala del eje, no
        # la cantidad de datos (confirmado comparando contra Navier-Stokes,
        # que usa penalizacion blanda — mse_boundary nunca es exactamente 0
        # ahi, y esos mismos paneles salen densamente poblados). Se detecta el
        # caso degenerado y se cambia a escala lineal (el cluster en 0 queda
        # visible como una linea real, en vez de desaparecer) mas una nota
        # explicita de por que.
        bnd_degenerate = (sub["mse_boundary"].abs() < 1e-9).mean() > 0.9

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plt.subplots_adjust(wspace=0.3)

        pairs = [
            ("mse_domain", "mse_boundary", "Domain MSE (Physics)", "Boundary MSE", "Physics vs. Boundary", True, not bnd_degenerate),
            ("mse_domain", "tree_size", "Domain MSE (Physics)", "Complexity (Nodes)", "Physics vs. Complexity", True, False),
            ("mse_boundary", "tree_size", "Boundary MSE", "Complexity (Nodes)", "Boundary vs. Complexity", not bnd_degenerate, False),
        ]
        for ax, (xcol, ycol, xlabel, ylabel, title, xlog, ylog) in zip(axes, pairs):
            if not dominated.empty:
                ax.scatter(dominated[xcol], dominated[ycol], color=GRAY, alpha=0.5, s=25,
                           label="Dominada", zorder=1)
            ax.scatter(pareto[xcol], pareto[ycol], color=LIME, edgecolors='k', linewidths=0.5, s=45,
                       label="No dominada (Frente)", zorder=2)
            if xlog: ax.set_xscale("log")
            if ylog: ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            if bnd_degenerate and "mse_boundary" in (xcol, ycol):
                ax.text(0.5, 0.5, "Ansatz exacto: MSE Frontera ≡ 0\n(no es eje informativo para esta EDP)",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8,
                        color="#666666", style="italic",
                        bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
        axes[0].legend(loc="best", framealpha=0.9)

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
