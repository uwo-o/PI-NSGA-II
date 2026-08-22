#!/usr/bin/env python3
"""
plot_solutions.py — Visualización de soluciones simbólicas vs verdad numérica (RK4/FD).
Rediseñado para máxima legibilidad y eliminación de solapamientos.
"""
import os, glob, numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

matplotlib.use("Agg")
# Ajuste de fuentes globales para alta resolución
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 22,
})

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGS_DIR    = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

PDE_ORDER = [
    "Laplace", "Poisson", "HarmonicOscillator",
    "Airy", "Fisher", "Duffing", "Thomas-Fermi",
    "Navier-Stokes", "Navier-Stokes-Unsteady",
    "Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I"
]

NUMERICAL_TRUTH = {
    "Airy", "HarmonicOscillator", "Liouville", "Sine-Gordon", 
    "NonlinearPoisson", "Fisher", "Duffing", "Thomas-Fermi",
    "Bratu", "Allen-Cahn", "Lane-Emden",
    "Troesch", "Ginzburg-Landau", "Painleve-I"
}

PDE_LABELS = {
    "Laplace":            r"Laplace: $\nabla^2 u = 0$",
    "Poisson":            r"Poisson: $\nabla^2 u = f$",
    "Helmholtz":          r"Helmholtz: $\nabla^2 u + k^2 u = f$",
    "Schrodinger":        r"Schrödinger: $-u'' + V u = E u$",
    "Airy":               r"Airy Equation",
    "HarmonicOscillator": r"Harmonic Oscillator",
    "Fisher":             r"Fisher-KPP Equation",
    "Duffing":            r"Duffing Oscillator",
    "Thomas-Fermi":       r"Thomas-Fermi Equation",
    "NonlinearPoisson":   r"Nonlinear Poisson",
    "Liouville":          r"Liouville Equation",
    "Sine-Gordon":        r"Sine-Gordon Equation",
    "Navier-Stokes":      r"Navier-Stokes: Viscous Steady Flow",
    "Navier-Stokes-Unsteady": r"Navier-Stokes: Taylor-Green Vortex",
    "Bratu":              r"Bratu Equation",
    "Allen-Cahn":         r"Allen-Cahn Equation",
    "Lane-Emden":         r"Lane-Emden Equation",
    "Troesch":            r"Troesch Equation",
    "Ginzburg-Landau":    r"Ginzburg-Landau Equation",
    "Painleve-I":         r"Painleve-I Equation"
}

CMAP_SOLUTION = "viridis"
CMAP_ERROR    = "inferno"

def find_file(pattern):
    hits = glob.glob(os.path.join(RESULTS_DIR, "**", pattern), recursive=True)
    return hits[0] if hits else None

def truth_label(pde):
    return "Numerical Truth" if pde in NUMERICAL_TRUTH else "Analytical"

# ─── 1D ────────────────────────────────────────────────────────────────────────
def plot_1d(pde, df_pi, df_pn=None, df_pysr=None, df_sindy=None):
    fig = plt.figure(figsize=(14, 12), layout="constrained")
    fig.suptitle(PDE_LABELS.get(pde, pde) + "  —  1D Analysis", fontweight="bold")
    
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1])
    ax_sol = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    x = df_pi["x"].values
    u_exact = df_pi["u_exact"].values

    ax_sol.plot(x, u_exact, "k-", lw=4.0, label=truth_label(pde), alpha=0.8, zorder=3)
    
    if df_pi is not None:
        u_pi = df_pi["u_approx"].values
        ax_sol.plot(x, u_pi, "b--", lw=2.5, label="PISR-NSGA-II", zorder=10)
        err_pi = np.abs(u_exact - u_pi)
        err_pi = np.clip(err_pi, 1e-32, None)
        ax_err.plot(x, err_pi, "b-", lw=2.0, label="Error PISR-NSGA-II")

    if df_pn is not None:
        pn_x = df_pn["x"] if "x" in df_pn.columns else df_pn["t"]
        ax_sol.plot(pn_x, df_pn["u_approx"], "g-.", lw=2.0, label="DeepXDE", zorder=6)
        err_pn = np.abs(df_pn["u_exact"].values - df_pn["u_approx"].values)
        err_pn = np.clip(err_pn, 1e-32, None)
        ax_err.plot(pn_x, err_pn, "g-", lw=1.5, label="Error DeepXDE")

    if df_pysr is not None:
        pysr_x = df_pysr["x"] if "x" in df_pysr.columns else df_pysr["t"]
        ax_sol.plot(pysr_x, df_pysr["u_approx"], "c:", lw=2.0, label="PySR", zorder=5)
        err_pysr = np.abs(df_pysr["u_exact"].values - df_pysr["u_approx"].values)
        ax_err.plot(pysr_x, np.clip(err_pysr, 1e-32, None), "c-", lw=1.5, label="Error PySR")

    if df_sindy is not None:
        sindy_x = df_sindy["x"] if "x" in df_sindy.columns else df_sindy["t"]
        ax_sol.plot(sindy_x, df_sindy["u_approx"], "m--", lw=2.0, label="PySINDy", zorder=4)
        err_sindy = np.abs(df_sindy["u_exact"].values - df_sindy["u_approx"].values)
        ax_err.plot(sindy_x, np.clip(err_sindy, 1e-32, None), "m-", lw=1.5, label="Error PySINDy")

    ax_sol.set_ylabel("u(x)", labelpad=15)
    ax_sol.legend(frameon=True, loc="best")
    ax_sol.grid(True, alpha=0.2)
    ax_sol.set_xlim(x.min(), x.max())

    ax_err.set_yscale("log")
    ax_err.set_xlabel("x", labelpad=10)
    ax_err.set_ylabel("Absolute Error", labelpad=15)
    ax_err.legend(frameon=True, loc="best")
    ax_err.grid(True, which="both", alpha=0.2)
    ax_err.set_xlim(x.min(), x.max())

    out = os.path.join(FIGS_DIR, f"solution_1d_{pde}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1D] {pde}: {out}")

# ─── 2D ────────────────────────────────────────────────────────────────────────
def plot_2d(pde, df_pi, df_pn=None, df_pysr=None):
    if "t" in df_pi.columns:
        t_slice = np.unique(df_pi["t"].values)[0]
        df_pi = df_pi[df_pi["t"] == t_slice]
        if df_pn is not None and "t" in df_pn.columns:
            df_pn = df_pn[df_pn["t"] == t_slice]

    N = int(np.sqrt(len(df_pi)))
    if N * N != len(df_pi): return

    X = df_pi["x"].values.reshape(N, N)
    Y = df_pi["y"].values.reshape(N, N)
    Z_ex = df_pi["u_exact"].values.reshape(N, N)
    Z_pi = df_pi["u_approx"].values.reshape(N, N)

    cols = [
        ("Ground Truth", Z_ex, None, CMAP_SOLUTION),
        ("PISR-NSGA-II", Z_pi, np.abs(Z_ex - Z_pi), CMAP_SOLUTION)
    ]
    
    if df_pn is not None and len(df_pn) == N * N:
        Z_pn = df_pn["u_approx"].values.reshape(N, N)
        cols.append(("DeepXDE", Z_pn, np.abs(Z_ex - Z_pn), CMAP_SOLUTION))

    if df_pysr is not None and len(df_pysr) == N * N:
        Z_pysr = df_pysr["u_approx"].values.reshape(N, N)
        cols.append(("PySR", Z_pysr, np.abs(Z_ex - Z_pysr), CMAP_SOLUTION))

    n_cols = len(cols)
    fig = plt.figure(figsize=(6 * n_cols, 10), layout="constrained")
    fig.suptitle(PDE_LABELS.get(pde, pde) + "  —  2D Comparison", fontweight="bold")
    
    gs = GridSpec(2, n_cols, figure=fig, height_ratios=[1.2, 1])

    for i, (name, Z, E, cmap) in enumerate(cols):
        # Row 0: 3D Surface
        ax_3d = fig.add_subplot(gs[0, i], projection='3d')
        surf = ax_3d.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.9, antialiased=True)
        ax_3d.set_title(name, fontsize=16, pad=10, fontweight="bold")
        ax_3d.set_xticks([0, 1]); ax_3d.set_yticks([0, 1])
        ax_3d.view_init(elev=25, azim=-60)
        
        # Row 1: Heatmap
        ax_2d = fig.add_subplot(gs[1, i])
        data = E if (E is not None and i == n_cols-1) else Z
        im = ax_2d.imshow(data, origin="lower", extent=[0, 1, 0, 1], cmap=(CMAP_ERROR if E is not None else cmap))
        ax_2d.set_title("Error Map" if E is not None else "Top View", fontsize=14)
        
        cbar = fig.colorbar(im, ax=ax_2d, shrink=0.6, pad=0.05)
        ax_2d.set_xticks([0, 0.5, 1]); ax_2d.set_yticks([0, 0.5, 1])

    out = os.path.join(FIGS_DIR, f"solution_2d_{pde}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2D] {pde}: {out}")

def plot_equation(pde, dim):
    suffix = f"_{dim}D"
    
    path_pi = find_file(f"grid_{pde}{suffix}_PISR-EMOAD.csv")
    path_pn = find_file(f"grid_{pde}{suffix}_DeepXDE.csv") or find_file(f"grid_{pde}{suffix}_PINN.csv")
    path_pysr = find_file(f"grid_{pde}{suffix}_PySR.csv")
    path_sindy = find_file(f"grid_{pde}{suffix}_PySINDy.csv")  # solo existe en 1D

    if not path_pi: return
    df_pi = pd.read_csv(path_pi)
    df_pn = pd.read_csv(path_pn) if path_pn else None
    df_pysr = pd.read_csv(path_pysr) if path_pysr else None
    df_sindy = pd.read_csv(path_sindy) if path_sindy else None

    if dim == 1: plot_1d(pde, df_pi, df_pn, df_pysr, df_sindy)
    else: plot_2d(pde, df_pi, df_pn, df_pysr)

def main():
    skips_1d = {"NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes", "Navier-Stokes-Unsteady", "Bratu", "Allen-Cahn"}
    skips_2d = {"Lane-Emden", "Laplace", "Poisson", "HarmonicOscillator"}
    for pde in PDE_ORDER:
        if pde not in skips_1d: plot_equation(pde, 1)
        if pde not in skips_2d: plot_equation(pde, 2)
    print("\nPlots updated in", FIGS_DIR)

if __name__ == "__main__":
    main()
