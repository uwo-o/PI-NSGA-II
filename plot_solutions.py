#!/usr/bin/env python3
"""
plot_solutions.py — Visualización de soluciones 1D y 2D.
"""
import os, glob, numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
})

# Rutas dinámicas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PDE_ORDER = ["Laplace", "Poisson", "Helmholtz", "Schrodinger", "NonlinearPoisson", "Liouville", "Sine-Gordon"]

def find_file(pattern):
    matches = glob.glob(os.path.join(RESULTS_DIR, "**", pattern), recursive=True)
    return matches[0] if matches else None

def plot_solution(pde, dim):
    suffix = f"_{dim}D"
    path_ts = find_file(f"grid_{pde}{suffix}_Tsoulos.csv")
    path_pi = find_file(f"grid_{pde}{suffix}_PI-NSGA-II.csv")
    path_pn = find_file(f"grid_{pde}{suffix}_PINN.csv")

    if not path_pi:
        print(f"[SKIP] No data for {pde} {dim}D.")
        return

    df_pi = pd.read_csv(path_pi)
    df_ts = pd.read_csv(path_ts) if path_ts else None
    df_pn = pd.read_csv(path_pn) if path_pn else None

    if dim == 1:
        # --- 1D PLOTS ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(f"{pde} 1D — Methods Comparison", fontweight="bold")
        
        # 1. Solutions
        ax1.plot(df_pi["x"], df_pi["u_exact"], 'k-', lw=2.5, label="Exact", alpha=0.6)
        ax1.plot(df_pi["x"], df_pi["u_approx"], 'b--', lw=1.5, label="PI-NSGA-II")
        if df_ts is not None: ax1.plot(df_ts["x"], df_ts["u_approx"], 'r:', lw=1.5, label="Tsoulos")
        if df_pn is not None: ax1.plot(df_pn["x"], df_pn["u_approx"], 'g-.', lw=1.5, label="PINN")
        ax1.set_ylabel("u(x)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        # 2. Errors (Log scale)
        err_pi = np.abs(df_pi["u_exact"] - df_pi["u_approx"])
        ax2.plot(df_pi["x"], err_pi, 'b-', label="Error PI")
        if df_ts is not None:
            err_ts = np.abs(df_ts["u_exact"] - df_ts["u_approx"])
            ax2.plot(df_ts["x"], err_ts, 'r-', label="Error Tsoulos")
        if df_pn is not None:
            err_pn = np.abs(df_pn["u_exact"] - df_pn["u_approx"])
            ax2.plot(df_pn["x"], err_pn, 'g-', label="Error PINN")
        
        ax2.set_yscale("log"); ax2.set_ylabel("Abs Error |u - u*|"); ax2.set_xlabel("x")
        ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
        
        outpath = os.path.join(RESULTS_DIR, f"solution_1d_{pde}.png")
        fig.savefig(outpath, dpi=160, bbox_inches="tight")
        print(f"  Saved 1D: {outpath}")
        plt.close(fig)
        
    else:
        # --- 2D PLOTS ---
        N = int(np.sqrt(len(df_pi)))
        X = df_pi["x"].values.reshape(N, N); Y = df_pi["y"].values.reshape(N, N)
        Z_ex = df_pi["u_exact"].values.reshape(N, N)
        Z_pi = df_pi["u_approx"].values.reshape(N, N)

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"{pde} 2D — Solution and Error Comparison", fontsize=14, fontweight="bold", y=0.98)
        
        def add_surf(pos, Z, title, cmap='viridis', is_error=False):
            ax = fig.add_subplot(2, 4, pos, projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, antialiased=True)
            t = f"{title}\n(max={Z.max():.2e})" if is_error else title
            ax.set_title(t); ax.view_init(elev=30, azim=-45)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            return ax

        # Row 1: Solutions
        add_surf(1, Z_ex, "Exact Solution")
        add_surf(2, Z_pi, "PI-NSGA-II")
        if df_ts is not None:
            Z_ts = df_ts["u_approx"].values.reshape(N, N)
            add_surf(3, Z_ts, "Tsoulos (GE)", 'copper')
        if df_pn is not None:
            Z_pn = df_pn["u_approx"].values.reshape(N, N)
            add_surf(4, Z_pn, "PINN (Neural Net)", 'summer')
            
        # Row 2: Errors
        err_pi = np.abs(Z_ex - Z_pi)
        add_surf(6, err_pi, "Error PI-NSGA-II", 'inferno', True)
        if df_ts is not None:
            Z_ts = df_ts["u_approx"].values.reshape(N, N)
            err_ts = np.abs(Z_ex - Z_ts)
            add_surf(7, err_ts, "Error Tsoulos", 'pink', True)
        if df_pn is not None:
            Z_pn = df_pn["u_approx"].values.reshape(N, N)
            err_pn = np.abs(Z_ex - Z_pn)
            add_surf(8, err_pn, "Error PINN", 'magma', True)

        fig.tight_layout()
        outpath = os.path.join(RESULTS_DIR, f"solution_2d_{pde}.png")
        fig.savefig(outpath, dpi=160, bbox_inches="tight")
        print(f"  Saved 2D: {outpath}")
        plt.close(fig)

def main():
    for pde in ["Laplace", "Poisson", "Helmholtz", "Schrodinger", "NonlinearPoisson", "Liouville", "Sine-Gordon"]:
        for d in [1, 2]:
            if pde == "NonlinearPoisson" and d == 1: continue
            plot_solution(pde, d)

if __name__ == "__main__":
    main()
