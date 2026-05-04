#!/usr/bin/env python3
"""
plot_solutions.py — Genera gráficas 3D de las superficies de las soluciones
simbólicas extraídas (Exacta vs Koza vs PI-NSGA-II) y mapas de error absoluto.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
})

RESULTS_DIR = "results"
PDE_ORDER = ["Laplace", "Poisson", "Helmholtz", "Schrodinger"]

def plot_solution_3d(pde):
    path_koza = os.path.join(RESULTS_DIR, f"grid_{pde}_Koza.csv")
    path_pi   = os.path.join(RESULTS_DIR, f"grid_{pde}_PI-NSGA-II.csv")

    if not os.path.exists(path_koza) or not os.path.exists(path_pi):
        print(f"[SKIP] No data for {pde}. Ensure verbose=True (or --runs 1) was used in pi_nsga2.")
        return

    df_koza = pd.read_csv(path_koza)
    df_pi   = pd.read_csv(path_pi)

    # Convertir a mallas 2D
    # Asumimos que x, y son un grid regular
    N = int(np.sqrt(len(df_koza)))
    
    # Extraer X, Y y reshape
    X = df_koza["x"].values.reshape(N, N)
    Y = df_koza["y"].values.reshape(N, N)
    Z_exact = df_koza["u_exact"].values.reshape(N, N)
    
    Z_koza = df_koza["u_approx"].values.reshape(N, N)
    Z_pi   = df_pi["u_approx"].values.reshape(N, N)

    err_koza = np.abs(Z_exact - Z_koza)
    err_pi   = np.abs(Z_exact - Z_pi)

    fig = plt.figure(figsize=(22, 5))
    fig.suptitle(f"{pde}'s Equation — Symbolic Solution Surfaces", fontsize=14, fontweight="bold", y=0.95)

    # kwargs para el plot
    surf_kwargs = dict(cmap="viridis", edgecolor="none", alpha=0.9, antialiased=True)
    err_kwargs  = dict(cmap="inferno", edgecolor="none", alpha=0.9, antialiased=True)

    # 1. Exact
    ax1 = fig.add_subplot(1, 5, 1, projection="3d")
    ax1.plot_surface(X, Y, Z_exact, **surf_kwargs)
    ax1.set_title("Exact Solution ($u^*$)")
    
    # 2. Koza
    ax2 = fig.add_subplot(1, 5, 2, projection="3d")
    ax2.plot_surface(X, Y, Z_koza, **surf_kwargs)
    ax2.set_title("Koza (BNF)")

    # 3. PI-NSGA-II
    ax3 = fig.add_subplot(1, 5, 3, projection="3d")
    ax3.plot_surface(X, Y, Z_pi, **surf_kwargs)
    ax3.set_title("PI-NSGA-II")

    # 4. Error Koza
    ax4 = fig.add_subplot(1, 5, 4, projection="3d")
    surf4 = ax4.plot_surface(X, Y, err_koza, **err_kwargs)
    ax4.set_title(f"Error Koza\n(max: {err_koza.max():.2e})")

    # 5. Error PI
    ax5 = fig.add_subplot(1, 5, 5, projection="3d")
    surf5 = ax5.plot_surface(X, Y, err_pi, **err_kwargs)
    ax5.set_title(f"Error PI-NSGA-II\n(max: {err_pi.max():.2e})")

    # Ajustes comunes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u" if ax in [ax1,ax2,ax3] else "|u - u*|")
        ax.view_init(elev=25, azim=-45)
        ax.dist = 11  # Alejar un poco la cámara dentro de cada subplot

    # Ajustar espacio manual para evitar solapamiento de ejes Z
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.35)
    outpath = os.path.join(RESULTS_DIR, f"solution_3d_{pde}.png")
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close(fig)

def compile_expressions():
    outpath = os.path.join(RESULTS_DIR, "best_expressions.txt")
    with open(outpath, "w") as fout:
        fout.write("==================================================\n")
        fout.write("  MEJORES EXPRESIONES SIMBÓLICAS (Formato LaTeX)\n")
        fout.write("==================================================\n\n")

        for pde in PDE_ORDER:
            fout.write(f"--- {pde.upper()} ---\n")
            for method in ["Koza", "PI-NSGA-II"]:
                f_path = os.path.join(RESULTS_DIR, f"expr_{pde}_{method}.tex")
                if os.path.exists(f_path):
                    with open(f_path, "r") as fin:
                        fout.write(f"[{method}]:\n")
                        # Read line by line, keep lines starting with $$
                        for line in fin:
                            if line.startswith("$$"):
                                fout.write("  " + line.strip() + "\n")
                else:
                    fout.write(f"[{method}]: (No data)\n")
            fout.write("\n")
    print(f"  Saved: {outpath}")

def main():
    print("Generando superficies 3D...")
    for pde in PDE_ORDER:
        plot_solution_3d(pde)
    compile_expressions()
    print("Hecho.")

if __name__ == "__main__":
    main()
