#!/usr/bin/env python3
"""
plot_solutions.py — Visualización de soluciones simbólicas vs verdad numérica (RK4/FD).

Para cada ecuación:
  - 1D: curva simbólica vs verdad (exacta o RK4) + panel de error absoluto.
  - 2D: superficie simbólica + verdad + heatmap de error.

La columna u_exact en los CSV puede ser:
  - La solución analítica exacta (Laplace, Poisson, …)
  - La solución numérica RK4/Relajación Jacobi (Airy, HO, Liouville, Sine-Gordon)
"""
import os, glob, numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi":  120,
})

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Todas las ecuaciones que el benchmark genera
PDE_ORDER = [
    "Laplace", "Poisson", "Helmholtz", "Schrodinger",
    "Airy", "HarmonicOscillator",
    "NonlinearPoisson", "Liouville", "Sine-Gordon",
]

# Ecuaciones cuya "u_exact" en el CSV proviene de RK4 (no hay fórmula cerrada)
NUMERICAL_TRUTH = {"Airy", "HarmonicOscillator", "Liouville", "Sine-Gordon", "NonlinearPoisson"}

# Etiqueta del eje / título para cada ecuación
PDE_LABELS = {
    "Laplace":            r"Laplace: $\nabla^2 u = 0$",
    "Poisson":            r"Poisson: $\nabla^2 u = f$",
    "Helmholtz":          r"Helmholtz: $\nabla^2 u + k^2 u = f$",
    "Schrodinger":        r"Schrödinger: $-u'' + V u = E u$",
    "Airy":               r"Airy: $u'' = x\,u$  [RK4 truth]",
    "HarmonicOscillator": r"Harmonic Oscillator: $u'' = (x^2-1)u$  [RK4 truth]",
    "NonlinearPoisson":   r"Nonlinear Poisson: $\nabla^2 u + u^2 = f$  [FD truth]",
    "Liouville":          r"Liouville: $\nabla^2 u = e^u$  [FD truth]",
    "Sine-Gordon":        r"Sine-Gordon: $\nabla^2 u = \sin(u)$  [FD truth]",
}

CMAP_SOLUTION = "viridis"
CMAP_ERROR    = "inferno"


def find_file(pattern):
    hits = glob.glob(os.path.join(RESULTS_DIR, "**", pattern), recursive=True)
    return hits[0] if hits else None


def truth_label(pde):
    return "RK4 / FD truth" if pde in NUMERICAL_TRUTH else "Analytical"


# ─── 1D ────────────────────────────────────────────────────────────────────────
def plot_1d(pde, df_pi, df_pn=None):
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(PDE_LABELS.get(pde, pde) + "  —  1D", fontweight="bold", fontsize=11)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.35)

    ax_sol = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    x = df_pi["x"].values
    u_exact  = df_pi["u_exact"].values
    u_approx = df_pi["u_approx"].values

    # Panel superior: soluciones
    ax_sol.plot(x, u_exact,  "k-",  lw=2.2, label=truth_label(pde), alpha=0.75, zorder=3)
    ax_sol.plot(x, u_approx, "b--", lw=1.8, label="PI-NSGA-II (simbólico)", zorder=4)
    if df_pn is not None:
        ax_sol.plot(df_pn["x"], df_pn["u_approx"], "g-.", lw=1.5, label="PINN", zorder=5)
    ax_sol.fill_between(x, u_exact, u_approx, alpha=0.10, color="blue")
    ax_sol.set_ylabel("u(x)")
    ax_sol.legend(framealpha=0.85)
    ax_sol.grid(True, alpha=0.3)

    # Panel inferior: error absoluto (log)
    err = np.abs(u_exact - u_approx)
    # Evitar log de cero
    err_pos = err[err > 0]
    ax_err.plot(x, err, "b-", lw=1.5, label="Error PI-NSGA-II")
    if df_pn is not None:
        err_pn = np.abs(df_pn["u_exact"].values - df_pn["u_approx"].values)
        ax_err.plot(df_pn["x"], err_pn, "g-", lw=1.2, label="Error PINN")
    if len(err_pos) > 0:
        ax_err.set_yscale("log")
    ax_err.set_xlabel("x")
    ax_err.set_ylabel(r"$|u - u^*|$")
    ax_err.legend(framealpha=0.85)
    ax_err.grid(True, which="both", alpha=0.3)

    # Anotación: MSE
    mse = np.mean(err**2)
    ax_err.annotate(f"MSE = {mse:.3e}", xy=(0.02, 0.92), xycoords="axes fraction",
                    fontsize=8, color="blue",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    out = os.path.join(RESULTS_DIR, f"solution_1d_{pde}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1D] {pde}: {out}")


# ─── 2D ────────────────────────────────────────────────────────────────────────
def plot_2d(pde, df_pi, df_pn=None):
    N = int(np.sqrt(len(df_pi)))
    if N * N != len(df_pi):
        print(f"  [WARN] {pde} 2D: len={len(df_pi)} no es cuadrado perfecto, saltando.")
        return

    X = df_pi["x"].values.reshape(N, N)
    Y = df_pi["y"].values.reshape(N, N)
    Z_ex  = df_pi["u_exact"].values.reshape(N, N)
    Z_pi  = df_pi["u_approx"].values.reshape(N, N)
    E_pi  = np.abs(Z_ex - Z_pi)

    has_pinn = df_pn is not None
    n_cols = 3 if not has_pinn else 4
    fig = plt.figure(figsize=(6 * n_cols, 11))
    fig.suptitle(PDE_LABELS.get(pde, pde) + "  —  2D",
                 fontweight="bold", fontsize=12, y=0.99)

    def surface(pos, Z, title, cmap=CMAP_SOLUTION):
        ax = fig.add_subplot(2, n_cols, pos, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, antialiased=True)
        ax.set_title(title + f"\nmax={Z.max():.2e}  min={Z.min():.2e}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.view_init(elev=28, azim=-50)
        fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
        return ax

    def heatmap(pos, Z, title, cmap=CMAP_SOLUTION):
        ax = fig.add_subplot(2, n_cols, pos)
        im = ax.imshow(Z, origin="lower", extent=[0,1,0,1], cmap=cmap, aspect="auto")
        ax.set_title(title + f"\nmax={Z.max():.2e}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)
        return ax

    # Fila 1: superficies 3D
    surface(1, Z_ex,  truth_label(pde))
    surface(2, Z_pi,  "PI-NSGA-II")
    surface(3, E_pi,  "Error |u - u*|", cmap=CMAP_ERROR)
    if has_pinn:
        Z_pn = df_pn["u_approx"].values.reshape(N, N)
        surface(4, Z_pn, "PINN", cmap="summer")

    # Fila 2: heatmaps
    heatmap(n_cols + 1, Z_ex,  truth_label(pde) + " (top view)")
    heatmap(n_cols + 2, Z_pi,  "PI-NSGA-II (top view)")
    heatmap(n_cols + 3, E_pi,  "Error (top view)", cmap=CMAP_ERROR)
    if has_pinn:
        E_pn = np.abs(Z_ex - df_pn["u_approx"].values.reshape(N, N))
        heatmap(n_cols + 4, E_pn, "Error PINN (top view)", cmap="magma")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(RESULTS_DIR, f"solution_2d_{pde}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2D] {pde}: {out}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def plot_equation(pde, dim):
    suffix = f"_{dim}D"
    path_pi = find_file(f"grid_{pde}{suffix}_PI-NSGA-II.csv")
    path_pn = find_file(f"grid_{pde}{suffix}_PINN.csv")

    if not path_pi:
        print(f"  [SKIP] Sin datos para {pde} {dim}D.")
        return

    try:
        df_pi = pd.read_csv(path_pi)
        df_pn = pd.read_csv(path_pn) if path_pn else None

        # Si u_exact es todo ceros o NaN → no graficamos la "verdad" (log de error no tiene sentido)
        u_ex = df_pi["u_exact"].values
        if np.all(u_ex == 0) or np.all(~np.isfinite(u_ex)):
            print(f"  [WARN] {pde} {dim}D: u_exact todo cero/NaN, graficando solo aprox.")
            df_pi["u_exact"] = df_pi["u_approx"]   # fallback: error = 0

        if dim == 1:
            plot_1d(pde, df_pi, df_pn)
        else:
            plot_2d(pde, df_pi, df_pn)

    except Exception as e:
        print(f"  [ERROR] {pde} {dim}D: {e}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    skips_1d = {"NonlinearPoisson", "Liouville", "Sine-Gordon"}   # solo 2D
    skips_2d = {"Airy"}                                            # solo 1D

    for pde in PDE_ORDER:
        if pde not in skips_1d:
            plot_equation(pde, 1)
        if pde not in skips_2d:
            plot_equation(pde, 2)

    print("\nPlots updated in", RESULTS_DIR)


if __name__ == "__main__":
    main()
