#!/usr/bin/env python3
"""
report/generate_report.py — Genera tablas LaTeX centradas en PISR-NSGA-II.

Fuentes de datos:
  - PI-NSGA-II / RK4/FDM : results/all_runs_summary.csv
  - PINN                  : results/**/*_pinn_pareto.csv  (formato: pde, dim, mse_domain, mse_boundary)
"""
import os, sys, shutil, glob
import numpy as np
import pandas as pd

REPORT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(REPORT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR  = os.path.join(REPORT_DIR, "tables")
FIGS_DIR    = os.path.join(REPORT_DIR, "figures")

# Ecuaciones base (en orden de la tabla)
PDE_BASE_NAMES = [
    "Airy", "Fisher", "Duffing", "Thomas-Fermi",
    "Navier-Stokes", "Navier-Stokes-Unsteady",
    "Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I"
]
DIMS = [1, 2]

# PDEs que solo existen en 2D
ONLY_2D = {"Navier-Stokes", "Navier-Stokes-Unsteady"}
# PDEs que solo existen en 1D
ONLY_1D = {"Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I"}

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)


# ─── Formato numérico ─────────────────────────────────────────────────────────
def fmt_sci_stat(m, s):
    """Devuelve cadena LaTeX $(a \\pm b) \\times 10^{e}$ o «—» si inválido."""
    if m is None or np.isnan(m) or not np.isfinite(m):
        return "—"
    if m == 0.0:
        return r"(0.00 \pm 0.00) \times 10^{0}"

    def get_exp(v):
        if v == 0 or not np.isfinite(v): return 0
        return int(np.floor(np.log10(abs(v))))

    exp_m  = get_exp(m)
    mant_m = m / (10 ** exp_m)
    s_safe = s if (s is not None and np.isfinite(s)) else 0.0
    mant_s = s_safe / (10 ** exp_m) if exp_m != 0 else s_safe
    return rf"({mant_m:.2f} \pm {mant_s:.2f}) \times 10^{{{exp_m}}}"


# ─── Carga de datos ───────────────────────────────────────────────────────────
def load_pi_rk4() -> pd.DataFrame:
    """Carga all_runs_summary.csv (PI-NSGA-II + RK4/FDM)."""
    path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["best_mse_domain"]   = pd.to_numeric(df["best_mse_domain"],   errors="coerce")
    df["best_mse_boundary"] = pd.to_numeric(df["best_mse_boundary"], errors="coerce")
    df["runtime_s"]         = pd.to_numeric(df["runtime_s"],         errors="coerce")
    df["mse_total"] = df["best_mse_domain"] + df["best_mse_boundary"]
    return df


def load_pinn() -> pd.DataFrame:
    pattern = os.path.join(RESULTS_DIR, "**", "*pinn_pareto.csv")
    files   = glob.glob(pattern, recursive=True)
    files  += glob.glob(os.path.join(RESULTS_DIR, "*pinn_pareto.csv"))
    files   = list(set(files))

    if not files: return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            tmp["mse_domain"]   = pd.to_numeric(tmp["mse_domain"],   errors="coerce")
            tmp["mse_boundary"] = pd.to_numeric(tmp["mse_boundary"], errors="coerce")
            # PINNs suelen no tener runtime_s en los CSVs individuales, asignamos NaN
            if "runtime_s" not in tmp.columns: tmp["runtime_s"] = np.nan
            else: tmp["runtime_s"] = pd.to_numeric(tmp["runtime_s"], errors="coerce")

            if "dim" in tmp.columns:
                tmp["pde"] = tmp["pde"].astype(str) + "_" + tmp["dim"].astype(int).astype(str) + "D"
            tmp = tmp.rename(columns={"mse_domain": "best_mse_domain", "mse_boundary": "best_mse_boundary"})
            tmp["method"] = "DeepXDE"
            tmp["mse_total"] = tmp["best_mse_domain"] + tmp["best_mse_boundary"]
            dfs.append(tmp[["method", "pde", "best_mse_domain", "best_mse_boundary", "mse_total", "runtime_s"]])
        except Exception as e: print(f"  [WARN] Skipping {f}: {e}")

    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(["method", "pde"], as_index=False).agg(
        best_mse_domain  = ("best_mse_domain",  "mean"),
        best_mse_boundary= ("best_mse_boundary","mean"),
        mse_total        = ("mse_total",         "mean"),
        mse_total_std    = ("mse_total",         "std"),
        runtime_s        = ("runtime_s",         "mean"),
    )
    return df


def load_all() -> pd.DataFrame:
    df_main = load_pi_rk4()
    df_pinn = load_pinn()
    if not df_main.empty: df_main["mse_total_std"] = 0.0
    if df_pinn.empty and df_main.empty: return pd.DataFrame()
    if df_pinn.empty: return df_main
    if df_main.empty: return df_pinn
    keep = ["method", "pde", "best_mse_domain", "best_mse_boundary", "mse_total", "mse_total_std", "runtime_s"]
    for c in keep:
        if c not in df_main.columns: df_main[c] = np.nan
        if c not in df_pinn.columns: df_pinn[c] = np.nan
    return pd.concat([df_main[keep], df_pinn[keep]], ignore_index=True)

# ─── Tabla 3: Tiempo de Ejecución ─────────────────────────────────────────────
def make_runtime_table():
    df = load_all()
    if df.empty: return

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Execution Time Comparison (seconds).}",
        r"  \label{tab:runtime_stats}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{RK4/FDM} & \textbf{DeepXDE} & \textbf{PISR-NSGA-II} \\",
        r"    \midrule"
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            if pde_base in ONLY_2D and d == 1: continue
            if pde_base in ONLY_1D and d == 2: continue
            full_name = f"{pde_base}_{d}D"
            sub = df[df["pde"] == full_name]

            t_pi  = sub[sub["method"] == "PI-NSGA-II"]["runtime_s"].mean()
            t_num = sub[sub["method"] == "RK4/FDM"]["runtime_s"].mean()
            t_pinn = sub[sub["method"] == "DeepXDE"]["runtime_s"].mean()

            def f_t(v):
                if v is None or np.isnan(v) or not np.isfinite(v): return "---"
                if v < 0.001: return f"{v:.2e}s"
                return f"{v:.3f}s"

            lines.append(rf"    {pde_base} & {d}D & {f_t(t_num)} & {f_t(t_pinn)} & {f_t(t_pi)} \\")
            lines.append(r"    \midrule")

    if len(lines) > 9: lines[-1] = r"    \bottomrule"
    else: lines.append(r"    \bottomrule")

    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]
    out = os.path.join(TABLES_DIR, "runtime_comparison.tex")
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


# ─── Tabla 1: Solo PISR-NSGA-II (métricas detalladas) ────────────────────────
def make_symbolic_table():
    summary_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(summary_path): return
    df = pd.read_csv(summary_path)
    df["best_mse_domain"]   = pd.to_numeric(df["best_mse_domain"],   errors="coerce")
    df["best_mse_boundary"] = pd.to_numeric(df["best_mse_boundary"], errors="coerce")
    df = df.dropna(subset=["best_mse_domain", "best_mse_boundary"])
    df["mse_total"] = df["best_mse_domain"] + df["best_mse_boundary"]

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{PISR-NSGA-II Performance Metrics (Mean $\pm$ Std).}",
        r"  \label{tab:symbolic_stats}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{Domain MSE} & \textbf{BC MSE} & \textbf{Total MSE} & \textbf{Hypervolume} \\",
        r"    \midrule"
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            if pde_base in ONLY_2D and d == 1:
                continue
            if pde_base in ONLY_1D and d == 2:
                continue
            full_pde_name = f"{pde_base}_{d}D"
            sub = df[(df["pde"] == full_pde_name) & (df["method"] == "PI-NSGA-II")]
            if sub.empty: continue

            stats = sub.agg({
                "best_mse_domain":  ["mean", "std"],
                "best_mse_boundary":["mean", "std"],
                "mse_total":        ["mean", "std"],
                "hypervolume":      ["mean", "std"],
            })

            f_dom = fmt_sci_stat(stats.loc["mean","best_mse_domain"],  stats.loc["std","best_mse_domain"])
            f_bnd = fmt_sci_stat(stats.loc["mean","best_mse_boundary"],stats.loc["std","best_mse_boundary"])
            f_tot = fmt_sci_stat(stats.loc["mean","mse_total"],        stats.loc["std","mse_total"])
            f_hv  = f"{stats.loc['mean','hypervolume']:.4f}"

            lines.append(rf"    {pde_base} & {d}D & ${f_dom}$ & ${f_bnd}$ & ${f_tot}$ & {f_hv} \\")
            lines.append(r"    \midrule")

    if len(lines) > 9:
        lines[-1] = r"    \bottomrule"
    else:
        lines.append(r"    \bottomrule")

    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]
    out = os.path.join(TABLES_DIR, "symbolic_comparison.tex")
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


# ─── Tabla 2: Comparación global RK4/FDM vs DeepXDE vs PISR-NSGA-II ─────────────
def make_global_comparison_table():
    df = load_all()
    if df.empty:
        print("  [WARN] No data found for global comparison table.")
        return

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Performance Comparison: Numerical (RK4/FDM), Neural (DeepXDE), and Symbolic (PISR-NSGA-II) Methods (Total MSE).}",
        r"  \label{tab:global_comp_stats}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{RK4/FDM} & \textbf{DeepXDE} & \textbf{PISR-NSGA-II} \\",
        r"    \midrule"
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            if pde_base in ONLY_2D and d == 1:
                continue
            if pde_base in ONLY_1D and d == 2:
                continue

            full_name = f"{pde_base}_{d}D"
            sub = df[df["pde"] == full_name]

            # ── PISR-NSGA-II ──
            pi_sub = sub[sub["method"] == "PI-NSGA-II"]
            if not pi_sub.empty:
                s = pi_sub.agg({"mse_total": ["mean","std"]})
                f_pi = fmt_sci_stat(s.loc["mean","mse_total"], s.loc["std","mse_total"])
            else:
                f_pi = "—"

            # ── RK4/FDM ──
            num_sub = sub[sub["method"] == "RK4/FDM"]
            if not num_sub.empty:
                s = num_sub.agg({"mse_total": ["mean","std"]})
                f_num = fmt_sci_stat(s.loc["mean","mse_total"], s.loc["std","mse_total"])
            else:
                f_num = "—"

            # ── DeepXDE ──
            pinn_sub = sub[sub["method"] == "DeepXDE"]
            if not pinn_sub.empty:
                # DeepXDE puede tener mse_total_std ya calculado al promediar corridas
                m = pinn_sub["mse_total"].mean()
                s_col = "mse_total_std" if "mse_total_std" in pinn_sub.columns else "mse_total"
                s_val = pinn_sub[s_col].mean() if s_col == "mse_total_std" else pinn_sub["mse_total"].std()
                s_val = 0.0 if np.isnan(s_val) else s_val
                f_pinn = fmt_sci_stat(m, s_val)
            else:
                f_pinn = "—"

            # Formatear celdas con $…$ solo cuando no son «—»
            def wrap(v):
                return f"${v}$" if v != "—" else "---"

            lines.append(
                rf"    {pde_base} & {d}D & {wrap(f_num)} & {wrap(f_pinn)} & {wrap(f_pi)} \\"
            )
            lines.append(r"    \midrule")

    if len(lines) > 9:
        lines[-1] = r"    \bottomrule"
    else:
        lines.append(r"    \bottomrule")

    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]
    out = os.path.join(TABLES_DIR, "global_comparison.tex")
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


# ─── Copiar figuras ───────────────────────────────────────────────────────────
def copy_figures():
    for pde in PDE_BASE_NAMES:
        for d in [1, 2]:
            for ext in ["pdf", "png"]:
                fname = f"solution_{d}d_{pde}.{ext}"
                src = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(FIGS_DIR, fname))


def make_latex_report():
    lines = [
        r"\documentclass[10pt,twocolumn]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[margin=1.5cm]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{graphicx}",
        r"\usepackage{caption}",
        r"\usepackage{subcaption}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\graphicspath{{figures/}}",
        r"\begin{document}",
        r"\title{\textbf{PISR-NSGA-II: Multi-Objective Symbolic Regression for Non-Linear Physics}}",
        r"\author{Scientific Benchmark Report}",
        r"\maketitle",
        r"",
        r"% ─── SEEDS ANALYSIS ────────────────────────────────────────────────────────",
        r"\section{Symbolic Library and Physical Seed Templates}",
        r"Unlike standard symbolic regression that relies on simple arithmetic trees, PISR-NSGA-II is augmented with a rich library of physical motifs and orthogonal polynomials. These templates act as high-level building blocks, injecting inductive bias that allows the evolutionary algorithm to effectively capture the complex topologies of non-linear PDEs. Below is the complete dictionary of 39 seed structures utilized during initialization.",
        r"",
        r"\subsection{Orthogonal Polynomials \& Spectral Basis}",
        r"\begin{itemize}",
        r"    \item \textbf{Legendre} $P_n(x^2+y^2)$: Optimal for spherically symmetric systems.",
        r"    \item \textbf{Hermite} $H_n(x)e^{-x^2/2}$: The natural basis for the Quantum Harmonic Oscillator.",
        r"    \item \textbf{Laguerre} $L_n(x)e^{-x/2}$: Essential for radial decays (e.g., Hydrogen atom).",
        r"    \item \textbf{Chebyshev} $T_n(x)$: Minimax approximations for minimizing maximum errors.",
        r"    \item \textbf{Spectral Basis} $a\sin(\pi x) + b\cos(\pi x)$ and \textbf{Forced Spectral} $x(1-x)\sin(\pi x)$.",
        r"\end{itemize}",
        r"",
        r"\subsection{Asymptotic and Singular Operators}",
        r"\begin{itemize}",
        r"    \item \textbf{Padé-Exponential Attractor} $\frac{1+ax}{1+x^2}e^{-x^2}$: Captures both the rational behavior near the origin and exponential death at infinity.",
        r"    \item \textbf{Padé Approximants (2/2)} $\frac{1+ax+bx^2}{1+cx+dx^2}$: Universal meromorphic approximant.",
        r"    \item \textbf{Fractional Power Decay} $\exp(c \ln(x+\epsilon))$ and \textbf{Lorentzian Resonances} $\frac{1}{1+(ax)^2}$.",
        r"    \item \textbf{Rational Decay} $\frac{1}{1+x^2+y^2+t}$ and \textbf{Complex Rational} $\frac{1}{x+ic}$.",
        r"    \item \textbf{Thomas-Fermi Ratio} $\frac{1}{1+a\sqrt{x}+bx}$.",
        r"\end{itemize}",
        r"",
        r"\subsection{Wave Packets, Solitons and Fluid Flows}",
        r"\begin{itemize}",
        r"    \item \textbf{Solitons and Kinks:} $\tanh(x-ct)$, \textbf{Diagonal Interfaces} $\tanh(x+y-1)$, and \textbf{Double-Kink interactions} $\tanh(a(x-b))+\tanh(c(x-d))$.",
        r"    \item \textbf{Anisotropic Gaussians:} $e^{-(ax^2+bxy+cy^2)}$, allowing discovery of rotated/deformed wave fronts.",
        r"    \item \textbf{Kovasznay Flow Motif:} $y - \frac{e^{ax}\sin(by)}{c}$, providing the fundamental velocity-field structure for viscous steady flows.",
        r"    \item \textbf{Spatiotemporal Wave Packets:} $\sin(\pi(x-t)) e^{-(x+y)^2}$ and \textbf{Complex Plane Waves} $e^{i(kx-\omega t)}$.",
        r"    \item \textbf{Damped Oscillators:} $\frac{\sin(\omega x)}{1+x^2}$ and $e^{-ax}\sin(\omega x)$.",
        r"    \item \textbf{Exponential Shockwaves:} $e^{-(x-ct)^2}$ and \textbf{Modulated Solitons} $\tanh(ax)\sin(bx)$.",
        r"    \item \textbf{Log-Cosh (Dynamic Bratu):} $\ln\left(\frac{1}{\cosh^2(x+y+t)}\right)$ and \textbf{Sech Soliton} $\frac{1}{\cosh(ax)}$.",
        r"    \item \textbf{Non-linear Airy:} $e^{-x^{1.5}}$.",
        r"    \item \textbf{Unitary Rotation:} $\frac{x+iy}{\sqrt{x^2+y^2}}$.",
        r"    \item \textbf{Variable Amplitude:} $x(\pi+t)$ and \textbf{Separation of Variables} $\sin(\pi x)\sin(\pi y)e^{-ct}$.",
        r"    \item \textbf{Sinc function} $\frac{\sin(ax)}{ax}$ and \textbf{Self-similarity} $\frac{x}{\sqrt{t+1}}$.",
        r"\end{itemize}",
        r"",
        r"\subsection{Complex Plane Explorers}",
        r"Leveraging the algorithm's native complex-arithmetic engine, these seeds use imaginary dimensions as a continuous search space to resolve difficult topologies before projecting back to the real plane:",
        r"\begin{itemize}",
        r"    \item \textbf{Quantum Phase Chirp:} $e^{-(a+ib)x^2}$. Separates into decaying and oscillating components via Euler's formula.",
        r"    \item \textbf{Log-Periodic Oscillations:} $x^{a+ib} = x^a \cos(b \ln x) + i x^a \sin(b \ln x)$. Ideal for critical phenomena.",
        r"    \item \textbf{Conjugate Pole Resonances:} $\frac{1}{x-ic} + \frac{1}{x+ic}$. Teaches the algorithm conjugate symmetry to form stable real Lorentzians.",
        r"    \item \textbf{Translational Phase Interference:} $\sin(x+ic)$. Converts imaginary phase shifts into real amplitude control.",
        r"    \item \textbf{Topological Vortices:} $(x+iy)^n e^{-(x^2+y^2)}$. Directly injects orbital angular momentum (topological charge) for Gross-Pitaevskii and fluid dynamics.",
        r"\end{itemize}",
        r"",
        r"\subsection{Extreme Physical Regimes}",
        r"To tackle high-stiffness PDEs and phase transitions, we introduced templates that explicitly model non-linear boundary layers and sharp interfaces:",
        r"\begin{itemize}",
        r"    \item \textbf{Troesch Boundary Layer:} $c_0 + c_1 e^{a(x-1)}$. Designed for equations where the solution is flat across the domain but surges exponentially at the boundary.",
        r"    \item \textbf{Sommerfeld Fractional Rational:} $(1 + ax^b)^{-c}$. Vital for multi-power law scaling in atomic physics (Thomas-Fermi).",
        r"    \item \textbf{Rotated Phase Sigmoid:} $\tanh(a(x\sin\theta + y\cos\theta - d))$. Provides the sharp, directional interfaces required for Allen-Cahn phase separation and biological Fisher fronts.",
        r"\end{itemize}",
        r"",
        r"\subsection{Infinite Series Operators}",
        r"Standard genetic programming represents series expansions as bloated, deeply nested binary trees, which are penalized by parsimony pressure. To overcome this, PISR-NSGA-II implements a native \texttt{SERIES} operator that acts as a symbolic loop, accumulating exact AD derivatives for each term efficiently:",
        r"\begin{itemize}",
        r"    \item \textbf{Generalized SERIES Wrapper:} $\sum_{n=1}^N C_n \cdot \mathcal{T}(x,y,t,n)$. A meta-template that allows evolution to discover novel series structures by wrapping any random tree.",
        r"    \item \textbf{Taylor / Polynomial Series:} $\sum_{n=1}^N C_n x^n$. The foundation for local approximations.",
        r"    \item \textbf{Fourier Base Series:} $\sum_{n=1}^N C_n \sin(nx)$. Fundamental for periodic phenomena and spectral methods.",
        r"    \item \textbf{Frobenius Series:} $\sum_{n=1}^N C_n x^{n-0.5}$. Attacks singular ODEs by modeling non-integer power behavior.",
        r"    \item \textbf{Exponential Series:} $\sum_{n=1}^N C_n e^{-nx}$. Models multi-scale decaying processes.",
        r"\end{itemize}",
        r"",
        r"% ─── METRICS ───────────────────────────────────────────────────────────────",
        r"\section{Performance Metrics}",
        r"\input{tables/symbolic_comparison.tex}",
        r"",
        r"% ─── CONVERGENCE ───────────────────────────────────────────────────────────",
        r"\section{Global Parsimony \& Complexity}",
        r"\begin{figure*}[ht]",
        r"  \centering",
        r"  \includegraphics[width=\textwidth]{parsimony_pressure}",
        r"  \caption{Parsimony Pressure: Trade-off between PDE Accuracy and Expression Complexity (Total node count). Complexity represents the number of mathematical elements (operators, variables, and constants) in the symbolic tree.}",
        r"  \label{fig:parsimony}",
        r"\end{figure*}",
        r"",
        r"\begin{figure*}[ht]",
        r"  \centering",
        r"  \includegraphics[width=0.9\textwidth]{pareto_fronts_3d}",
        r"  \caption{3D Pareto Fronts: Physics vs. Boundary vs. Complexity. Dominated solutions are shown in transparent gray.}",
        r"  \label{fig:pareto3d}",
        r"\end{figure*}",
        r"",
        r"% ─── PDE ANALYSIS ──────────────────────────────────────────────────────────",
        r"\clearpage",
        r"\onecolumn",
        r"\section{Detailed PDE Analysis}",
        r"The following sections present the symbolic approximations discovered by PISR-NSGA-II and their multi-objective Pareto trade-offs for each benchmark problem.",
        r"",
        r"In problems where the analytical truth is unknown and the numerical ground truth (RK4/FDM) may fail due to singularities or extreme stiffness (e.g., Troesch, Lane-Emden, Thomas-Fermi), evaluating the absolute error solely against numerical solvers can be misleading. To address this, our visualizations include the absolute error between the Symbolic approximation (PISR-NSGA-II) and the Neural approximation (DeepXDE). When both physics-informed algorithms converge to identical structures ($\mathcal{E}_{PI, PINN} \to 0$), it provides strong empirical evidence that they have discovered the true physical manifold, outperforming classical numerical integration.",
        r""
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            if pde_base in ONLY_2D and d == 1: continue
            if pde_base in ONLY_1D and d == 2: continue
            
            pde_label = f"{pde_base}_{d}D"
            pde_label_tex = pde_label.replace("_", r"\_")
            sol_fig = f"solution_{d}d_{pde_base}.pdf"
            pairs_fig = f"pareto_pairs_{pde_label}.pdf"
            
            # Check if figures exist
            if os.path.exists(os.path.join(FIGS_DIR, sol_fig)) and os.path.exists(os.path.join(FIGS_DIR, pairs_fig)):
                lines += [
                    rf"\subsection{{{pde_label_tex} Visualization}}",
                    r"\begin{figure}[h!]",
                    r"  \centering",
                    rf"  \includegraphics[width=\textwidth]{{{sol_fig}}}",
                    rf"  \caption{{Symbolic vs Numerical solution for {pde_label_tex}.}}",
                    r"\end{figure}",
                    r"\begin{figure}[h!]",
                    r"  \centering",
                    rf"  \includegraphics[width=\textwidth]{{{pairs_fig}}}",
                    rf"  \caption{{Pairwise Pareto Trade-offs for {pde_label_tex}.}}",
                    r"\end{figure}",
                    r"\clearpage"
                ]

    lines += [
        r"\twocolumn",
        r"\section{Discovered Formulas}",
        r"\input{tables/formulas_table.tex}",
        r"\end{document}"
    ]

    out_tex = os.path.join(REPORT_DIR, "results.tex")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] Generated dynamic LaTeX report at {out_tex}")

if __name__ == "__main__":
    print("Generando tablas LaTeX…")
    make_symbolic_table()
    make_global_comparison_table()
    make_runtime_table()
    copy_figures()
    make_latex_report()
    print("\nReport generation complete. Tables in:", TABLES_DIR)

