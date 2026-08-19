#!/usr/bin/env python3
"""
report/generate_report.py — Genera tablas LaTeX centradas en PISR-NSGA-II.

Fuentes de datos:
  - PISR-NSGA-II / RK4/FDM : results/all_runs_summary.csv
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
    """Carga all_runs_summary.csv (PISR-NSGA-II + RK4/FDM)."""
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


def load_sota() -> pd.DataFrame:
    pattern = os.path.join(RESULTS_DIR, "**", "*_metrics.csv")
    files   = glob.glob(pattern, recursive=True)
    files  += glob.glob(os.path.join(RESULTS_DIR, "*_metrics.csv"))
    files   = list(set(files))

    if not files: return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            dfs.append(tmp)
        except Exception as e: print(f"  [WARN] Skipping {f}: {e}")

    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    
    # Map SOTA pde names back to PISR-NSGA-II standard names
    sota_to_pisr = {
        "airy": "Airy",
        "fisher": "Fisher",
        "duffing": "Duffing",
        "thomas_fermi": "Thomas-Fermi",
        "lane_emden": "Lane-Emden",
        "troesch": "Troesch",
        "painleve": "Painleve-I",
        "ginzburg_landau": "Ginzburg-Landau"
    }
    
    def map_pde_name(x):
        for sota, pisr in sota_to_pisr.items():
            if x.startswith(sota + "_"):
                return x.replace(sota + "_", pisr + "_")
        return x
        
    df["pde"] = df["pde"].apply(map_pde_name)
    
    # SOTA scripts output columns: method, pde, best_mse_domain, best_mse_boundary, mse_total, runtime_s
    return df


def load_all() -> pd.DataFrame:
    df_main = load_pi_rk4()
    df_pinn = load_pinn()
    df_sota = load_sota()
    
    if not df_main.empty: df_main["mse_total_std"] = 0.0
    
    frames = []
    if not df_main.empty: frames.append(df_main)
    if not df_pinn.empty: frames.append(df_pinn)
    if not df_sota.empty: frames.append(df_sota)
    
    if not frames: return pd.DataFrame()
    
    keep = ["method", "pde", "best_mse_domain", "best_mse_boundary", "mse_total", "mse_total_std", "runtime_s"]
    for df in frames:
        for c in keep:
            if c not in df.columns: df[c] = np.nan
            
    return pd.concat([df[keep] for df in frames], ignore_index=True)

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
        r"  \begin{tabular}{llccccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{RK4/FDM} & \textbf{DeepXDE} & \textbf{PySR} & \textbf{PISR-NSGA-II} \\",
        r"    \midrule"
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            if pde_base in ONLY_2D and d == 1: continue
            if pde_base in ONLY_1D and d == 2: continue
            full_name = f"{pde_base}_{d}D"
            sub = df[df["pde"] == full_name]

            t_pi  = sub[sub["method"] == "PISR-NSGA-II"]["runtime_s"].mean()
            t_num = sub[sub["method"] == "RK4/FDM"]["runtime_s"].mean()
            t_pinn = sub[sub["method"] == "DeepXDE"]["runtime_s"].mean()

            t_pysr = sub[sub["method"] == "PySR"]["runtime_s"].mean() if "PySR" in sub["method"].values else np.nan

            def f_t(v):
                if v is None or pd.isna(v) or not np.isfinite(v): return "---"
                if v < 0.001: return f"{v:.2e}s"
                return f"{v:.3f}s"

            lines.append(rf"    {pde_base} & {d}D & {f_t(t_num)} & {f_t(t_pinn)} & {f_t(t_pysr)} & {f_t(t_pi)} \\")
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
            sub = df[(df["pde"] == full_pde_name) & (df["method"] == "PISR-NSGA-II")]
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
        r"  \begin{tabular}{lccccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{RK4/FDM} & \textbf{DeepXDE} & \textbf{PySR} & \textbf{PISR-NSGA-II} \\",
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
            pi_sub = sub[sub["method"] == "PISR-NSGA-II"]
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

            # ── PySINDy logic removed ──

            # ── PySR ──
            pysr_sub = sub[sub["method"] == "PySR"]
            f_pysr = "—"
            if not pysr_sub.empty:
                m = pysr_sub["mse_total"].mean()
                s_col = "mse_total_std" if "mse_total_std" in pysr_sub.columns else "mse_total"
                s_val = pysr_sub[s_col].mean() if s_col == "mse_total_std" else pysr_sub["mse_total"].std()
                s_val = 0.0 if pd.isna(s_val) else s_val
                f_pysr = fmt_sci_stat(m, s_val)

            # Formatear celdas con $…$ solo cuando no son «—»
            def wrap(v):
                return f"${v}$" if v != "—" else "---"

            lines.append(
                rf"    {pde_base} & {d}D & {wrap(f_num)} & {wrap(f_pinn)} & {wrap(f_pysr)} & {wrap(f_pi)} \\"
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
        r"% ─── ALGORITHM OVERVIEW ────────────────────────────────────────────────────",
        r"\section{Algorithm Overview}",
        r"PISR-NSGA-II discovers closed-form symbolic solutions to ODEs/PDEs using classic NSGA-II (non-dominated sorting, crowding distance with a structural-diversity bonus, tournament selection, elitist truncation) over three objectives: domain residual, boundary residual, and tree size. No labeled solution data is used — only collocation points where the governing equation's residual is evaluated via exact automatic differentiation.",
        r"",
        r"\subsection{Exact Boundary Ansatz}",
        r"For Dirichlet boundary conditions, the candidate is not evaluated directly: the tree $N(x)$ (or $N(x,y)$) is composed as $U = L + B \cdot N$, where $L$ interpolates the boundary values exactly (linear in 1D, transfinite/Coons-patch interpolation in 2D) and $B$ vanishes identically on the boundary. This locks the boundary residual to zero by construction for every 1D problem and every 2D problem with simple Dirichlet conditions, leaving boundary error out of the search entirely. Problems with more complex boundary conditions (e.g. stream-function/velocity constraints in Navier-Stokes) fall back to a soft penalty term.",
        r"",
        r"\subsection{Structural Search Operators}",
        r"Mutation and crossover operate on the tree's additive-term decomposition (splitting a sum into independent summands) and, within each term, its multiplicative-factor decomposition (splitting a product/ratio into factors). Both levels support adding, deleting, or replacing a term/factor, and a smooth point-mutation that swaps a node for another in the same functional family (e.g. $\sin \leftrightarrow \cos \leftrightarrow \exp$ via Euler's formula) rather than an unrelated one — reducing destructive jumps between generations. A dedicated phase-rotation factor $e^{i\theta}$ (evolvable $\theta$) lets the search explore complex-plane rotations directly, exploiting the fact that the evaluation engine is complex-valued end to end.",
        r"",
        r"\subsection{Constant Optimization}",
        r"Every individual (not only the elite front) is locally polished each generation via adaptive gradient descent. Gradients of the residual with respect to each ERC constant are computed by complex-step differentiation — perturbing the constant along the imaginary axis and reading $\mathrm{Im}(f(c+ih))/h$ — which is exact to machine precision with no finite-difference truncation or cancellation error, at roughly half the tree evaluations of central differences. The one PDE in the benchmark suite whose residual is not holomorphic in its constants (Thomas-Fermi) falls back to finite differences.",
        r"",
        r"\subsection{Robustness}",
        r"Per-point squared residuals are capped before averaging into the domain loss, preventing a single outlier point (e.g. near a PDE singularity) from dominating the fitness signal and destabilizing selection. The training collocation batch is a large, stable subset of the domain, resampled only periodically rather than every generation, so that improvement between generations reflects genuine progress rather than sampling noise.",
        r"",
        r"% ─── METRICS ───────────────────────────────────────────────────────────────",
        r"\section{Performance Metrics}",
        r"\input{tables/global_comparison.tex}",
        r"\input{tables/runtime_comparison.tex}",
        r"\input{tables/symbolic_comparison.tex}",
        r"",
        r"% ─── DATA EFFICIENCY ───────────────────────────────────────────────────────",
        r"\section{Data Efficiency vs. State-of-the-Art}",
        r"Unlike PySR, which requires labeled ground-truth solution points to train, "
        r"PISR-NSGA-II discovers symbolic solutions using only the governing "
        r"differential equation and boundary conditions — zero labeled data. "
        r"Table~\ref{tab:data_efficiency} reports the number of labeled points each "
        r"method actually consumed alongside the final MSE achieved.",
        r"\input{tables/data_efficiency_comparison.tex}",
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
        r"In problems where the analytical truth is unknown and the numerical ground truth (RK4/FDM) may fail due to singularities or extreme stiffness (e.g., Troesch, Lane-Emden, Thomas-Fermi), evaluating the absolute error solely against numerical solvers can be misleading. Our visualizations now include the symbolic approximations (PISR-NSGA-II) against the data-driven state-of-the-art (PySR) and Neural approximations (DeepXDE). When physics-informed algorithms converge to identical structures without labeled data, it provides strong empirical evidence of physical discovery.",
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

