#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGS_DIR = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# Estilo premium
import matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 22,
})

PDES_TO_PLOT = [
    "Laplace_1D", "Poisson_1D", "HarmonicOscillator_1D",
    "Airy_1D", "Airy_2D",
    "Fisher_1D", "Fisher_2D",
    "Duffing_1D", "Duffing_2D",
    "Thomas-Fermi_1D", "Thomas-Fermi_2D",
    "Navier-Stokes_2D", "Navier-Stokes-Unsteady_2D",
    "Lane-Emden_1D", "Troesch_1D", "Ginzburg-Landau_1D", "Painleve-I_1D"
]

def plot_convergence():
    # Calcular filas necesarias para 3 columnas
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 4.5 * n_rows), sharey=False, layout="constrained")
    axes = axes.flatten()
    
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = axes[idx]
        # Cargar PISR-NSGA-II
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_convergence.csv")
        if os.path.exists(gn_file):
            df_gn = pd.read_csv(gn_file)
            ax.plot(df_gn["gen"], df_gn["best_total_mse"], label="PISR-NSGA-II", color="#2E86C1", lw=3)
            
        ax.set_title(pde_name.replace("_", " "), pad=15, fontweight="bold")
        ax.set_xlabel("Generations", labelpad=8)
        ax.set_ylabel("Best Total MSE", labelpad=8)
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    # Ocultar ejes vacíos si hay un número impar de PDEs
    for i in range(n_pdes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Convergence History: PISR-NSGA-II Constants Optimization", fontweight="bold")
    out_path = os.path.join(FIGS_DIR, "convergence_history.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_parsimony():
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 4.5 * n_rows), sharey=False, layout="constrained")
    axes = axes.flatten()
    
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = axes[idx]
        
        # Cargar PISR-NSGA-II pareto
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_pareto.csv")
        if os.path.exists(gn_file):
            df = pd.read_csv(gn_file)
            df = df[df["mse_domain"] < 1e5] # Filtrar outliers iniciales
            df["total_mse"] = df["mse_domain"] + df["mse_boundary"]
            # Agrupar por tree_size y tomar el mejor total_mse
            best_by_size = df.groupby("tree_size")["total_mse"].min().reset_index()
            best_by_size = best_by_size.sort_values("tree_size")
            ax.step(best_by_size["tree_size"], best_by_size["total_mse"], where="post",
                    label="PISR-NSGA-II", color="#2E86C1", marker="o", lw=2.5)
            
        ax.set_title(pde_name.replace("_", " "), pad=15, fontweight="bold")
        ax.set_xlabel("Complexity (Tree Nodes)", labelpad=8)
        ax.set_ylabel("Total MSE", labelpad=8)
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    for i in range(n_pdes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Parsimony Pressure: Accuracy vs. Expression Complexity", fontweight="bold")
    out_path = os.path.join(FIGS_DIR, "parsimony_pressure.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_pareto_fronts():
    """
    Genera una grilla limpia de frentes de Pareto 3D sin barras individuales.
    """
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(24, 6 * n_rows), layout="constrained")
    
    last_scatter = None
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_pareto.csv")
        if os.path.exists(gn_file):
            df = pd.read_csv(gn_file)
            df = df[(df["mse_domain"] < 1e5) & (df["mse_boundary"] < 1e5)]
            if len(df) == 0: continue
                
            x, y, z, r = df["mse_domain"].clip(lower=1e-20), df["mse_boundary"].clip(lower=1e-20), df["tree_size"], df["rank"]
            mask_rank1, mask_dominated = (r == 1), (r > 1)
            
            if mask_dominated.any():
                ax.scatter(np.log10(x[mask_dominated]), np.log10(y[mask_dominated]), z[mask_dominated], 
                           c='lightgray', s=15, alpha=0.2)
            
            if mask_rank1.any():
                last_scatter = ax.scatter(np.log10(x[mask_rank1]), np.log10(y[mask_rank1]), z[mask_rank1], 
                                     c=z[mask_rank1], cmap='viridis', s=60, alpha=1.0, 
                                     edgecolors="white", linewidth=0.3)
            
        ax.set_title(pde_name.replace("_", " "), pad=5, fontsize=20, fontweight="bold")
        ax.set_xlabel("log(Dom)", labelpad=10, fontsize=14)
        ax.set_ylabel("log(Bnd)", labelpad=10, fontsize=14)
        ax.set_zlabel("Nodes", labelpad=10, fontsize=14)
        ax.view_init(elev=20, azim=-120)

    # Una sola barra de color global para toda la grilla
    if last_scatter:
        cbar = fig.colorbar(last_scatter, ax=fig.get_axes(), shrink=0.4, aspect=40, pad=0.02, location='right')
        cbar.set_label("Complexity (Total Nodes)", rotation=270, labelpad=25, fontsize=18)

    plt.suptitle("3D Pareto Landscape: Domain vs. Boundary vs. Complexity", fontsize=32, fontweight="bold", y=1.02)
    out_path = os.path.join(FIGS_DIR, "pareto_fronts_3d.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_pairwise_pareto_fronts():
    """
    Genera una matriz de plots por pares (2D) para cada PDE,
    incluyendo soluciones dominadas en gris claro.
    """
    for pde_name in PDES_TO_PLOT:
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_pareto.csv")
        if not os.path.exists(gn_file): continue
        
        df = pd.read_csv(gn_file)
        df = df[(df["mse_domain"] < 1e5) & (df["mse_boundary"] < 1e5)]
        if len(df) == 0: continue
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), layout="constrained")

        mask_rank1 = (df["rank"] == 1)
        mask_dominated = (df["rank"] > 1)
        LIME = "#9ACD32"
        GRAY = "#B0B0B0"

        # 1. Domain vs Boundary (Log-Log)
        ax = axes[0]
        if mask_dominated.any():
            ax.scatter(df[mask_dominated]["mse_domain"], df[mask_dominated]["mse_boundary"],
                       c=GRAY, s=25, alpha=0.5, label='Dominada', zorder=1)
        ax.scatter(df[mask_rank1]["mse_domain"], df[mask_rank1]["mse_boundary"],
                   c=LIME, s=60, edgecolors='k', linewidth=0.5, label='No dominada (Frente)', zorder=10)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Domain MSE", fontsize=16); ax.set_ylabel("Boundary MSE", fontsize=16)
        ax.set_title("Physics vs. Boundary", fontsize=18, fontweight="bold")
        ax.grid(True, which="both", ls="-", alpha=0.1)
        ax.legend(fontsize=14)

        # 2. Domain vs Complexity (Log-Linear)
        ax = axes[1]
        if mask_dominated.any():
            ax.scatter(df[mask_dominated]["mse_domain"], df[mask_dominated]["tree_size"],
                       c=GRAY, s=25, alpha=0.5, zorder=1)
        ax.scatter(df[mask_rank1]["mse_domain"], df[mask_rank1]["tree_size"],
                   c=LIME, s=60, edgecolors='k', linewidth=0.5, zorder=10)
        ax.set_xscale("log")
        ax.set_xlabel("Domain MSE", fontsize=16); ax.set_ylabel("Complexity (Nodes)", fontsize=16)
        ax.set_title("Physics vs. Complexity", fontsize=18, fontweight="bold")
        ax.grid(True, which="both", ls="-", alpha=0.1)

        # 3. Boundary vs Complexity (Log-Linear)
        ax = axes[2]
        if mask_dominated.any():
            ax.scatter(df[mask_dominated]["mse_boundary"], df[mask_dominated]["tree_size"],
                       c=GRAY, s=25, alpha=0.5, zorder=1)
        ax.scatter(df[mask_rank1]["mse_boundary"], df[mask_rank1]["tree_size"],
                   c=LIME, s=60, edgecolors='k', linewidth=0.5, zorder=10)
        ax.set_xscale("log")
        ax.set_xlabel("Boundary MSE", fontsize=16); ax.set_ylabel("Complexity (Nodes)", fontsize=16)
        ax.set_title("Boundary vs. Complexity", fontsize=18, fontweight="bold")
        ax.grid(True, which="both", ls="-", alpha=0.1)

        plt.suptitle(f"Pairwise Pareto Trade-offs: {pde_name.replace('_', ' ')}", fontsize=22, fontweight='bold')
        out_path = os.path.join(FIGS_DIR, f"pareto_pairs_{pde_name}.pdf")
        plt.savefig(out_path, bbox_inches='tight', format='pdf')
        plt.close()
        print(f"Generated: {out_path}")

if __name__ == "__main__":
    plot_parsimony()
    plot_pareto_fronts()
    # plot_pairwise_pareto_fronts() ya no se llama: redundante con
    # scripts/plot_pareto.py (mismo plot pareado por EDP, con el estilo
    # gris/verde-lima correcto sin interpolacion de lineas) — esa es ahora la
    # unica fuente para esas figuras, copiadas a report/figures/ por
    # generate_report.sh.
