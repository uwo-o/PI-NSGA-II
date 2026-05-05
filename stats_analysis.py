#!/usr/bin/env python3
"""
stats_analysis.py — Análisis estadístico completo: PI-NSGA-II vs Tsoulos (GE)
  Lee all_runs_summary.csv generado por ./build/pi_nsga2 --runs N
  Genera:
    results/stats_table.csv              — media ± std, p-valor Wilcoxon, tamaño de efecto
    results/boxplot_hypervolume.png      — box plots de hipervolumen por PDE × método
    results/boxplot_mse.png              — box plots de MSE dominio y frontera
    results/convergence_summary.txt      — resumen textual de resultados

  Uso:
    python3 stats_analysis.py
    python3 stats_analysis.py --input results/all_runs_summary.csv
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.linewidth":   0.8,
    "grid.linewidth":   0.5,
})
warnings.filterwarnings("ignore")

# ─── Paleta de colores ────────────────────────────────────────────────────────
COLORS = {
    "Tsoulos":    "#E07535",
    "PI-NSGA-II": "#2E86C1",
}
PDE_ORDER   = ["Laplace", "Poisson", "Helmholtz", "Schrodinger"]
METHOD_LIST = ["Tsoulos", "PI-NSGA-II"]

# ─── Carga de datos ───────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] No encontrado: {path}")
        print("  Ejecuta primero:  ./build/pi_nsga2 --runs 20")
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"run", "method", "pde", "hypervolume",
                "best_mse_domain", "best_mse_boundary", "runtime_s"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Columnas faltantes: {missing}")
        sys.exit(1)
    return df

# ─── Test de Wilcoxon + tamaño de efecto r ───────────────────────────────────
def wilcoxon_test(a: np.ndarray, b: np.ndarray):
    """
    Mann-Whitney U test (equivalente a Wilcoxon rank-sum para muestras independientes).
    Retorna: (U, p_value, effect_size_r)
    effect_size r = Z / sqrt(n1 + n2)
    """
    if len(a) < 2 or len(b) < 2:
        return None, None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    # Convertir U a Z aproximado
    mu_U  = n1 * n2 / 2.0
    sig_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    Z = (U - mu_U) / (sig_U + 1e-12)
    r = abs(Z) / np.sqrt(n1 + n2)
    return U, p, r

def effect_label(r):
    """Interpretación del tamaño de efecto según Cohen (1988)."""
    if r < 0.1:  return "negligible"
    if r < 0.3:  return "small"
    if r < 0.5:  return "medium"
    return "large"

# ─── Tabla estadística completa ───────────────────────────────────────────────
def build_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = {
        "hypervolume":       "Hypervolume ↑",
        "best_mse_domain":   "Best MSE Domain ↓",
        "best_mse_boundary": "Best MSE BC ↓",
        "runtime_s":         "Runtime (s) ↓",
    }

    for pde in PDE_ORDER:
        sub = df[df["pde"] == pde]
        for metric_key, metric_name in metrics.items():
            row = {"PDE": pde, "Metric": metric_name}
            vals = {}
            for m in METHOD_LIST:
                v = sub[sub["method"] == m][metric_key].dropna().values
                vals[m] = v
                row[f"{m}_mean"]   = np.mean(v)   if len(v) > 0 else np.nan
                row[f"{m}_std"]    = np.std(v, ddof=1) if len(v) > 1 else np.nan
                row[f"{m}_median"] = np.median(v) if len(v) > 0 else np.nan

            # Wilcoxon test entre los dos métodos
            U, p, r = wilcoxon_test(vals["Tsoulos"], vals["PI-NSGA-II"])
            row["U_stat"]       = U
            row["p_value"]      = p
            row["effect_r"]     = r
            row["effect_label"] = effect_label(r) if r is not None else "n/a"

            # ¿Cuál método es mejor?
            k_mean = row.get("Tsoulos_mean")
            pi_mean = row.get("PI-NSGA-II_mean")
            if k_mean is not None and pi_mean is not None:
                if metric_key == "hypervolume":
                    row["winner"] = "Tsoulos" if k_mean > pi_mean else "PI-NSGA-II" if pi_mean > k_mean else "Empate"
                else:
                    row["winner"] = "Tsoulos" if k_mean < pi_mean else "PI-NSGA-II" if pi_mean < k_mean else "Empate"

            rows.append(row)

    return pd.DataFrame(rows)

# ─── Box plots — Hipervolumen ─────────────────────────────────────────────────
def plot_boxplot_hv(df: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(1, len(PDE_ORDER), figsize=(4 * len(PDE_ORDER), 5), sharey=False)
    fig.suptitle(
        "Hypervolume by Method and PDE Equation\n"
        "(N independent runs — higher is better)",
        fontsize=12, fontweight="bold"
    )

    for ax, pde in zip(axes, PDE_ORDER):
        sub = df[df["pde"] == pde]
        data_tsoulos = sub[sub["method"] == "Tsoulos"]["hypervolume"].values
        data_pi      = sub[sub["method"] == "PI-NSGA-II"]["hypervolume"].values

        bp = ax.boxplot(
            [data_tsoulos, data_pi],
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="x", markersize=5, alpha=0.5),
        )
        for patch, method in zip(bp["boxes"], METHOD_LIST):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.8)

        # Añadir jitter (puntos individuales)
        for j, (data, method) in enumerate(zip([data_tsoulos, data_pi], METHOD_LIST), 1):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(data))
            ax.scatter(np.full(len(data), j) + jitter, data,
                       color=COLORS[method], alpha=0.5, s=18, zorder=3)

        # p-value en el plot
        U, p, r = wilcoxon_test(data_tsoulos, data_pi)
        p_str = f"p={p:.3f}" if p is not None else ""
        ax.set_title(f"{pde}  ({p_str})", fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(METHOD_LIST, rotation=25)
        ax.set_ylabel("Hypervolume")
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # Leyenda global
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHOD_LIST]
    fig.legend(handles=patches, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Guardado: {outpath}")
    plt.close(fig)

# ─── Box plots — MSE dominio y frontera ───────────────────────────────────────
def plot_boxplot_mse(df: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, len(PDE_ORDER), figsize=(4 * len(PDE_ORDER), 9))
    fig.suptitle(
        "MSE Distribution by Method and PDE Equation\n"
        "Top: Domain MSE (PDE residual)  |  Bottom: Boundary MSE (BC condition)",
        fontsize=12, fontweight="bold"
    )

    for col, pde in enumerate(PDE_ORDER):
        sub = df[df["pde"] == pde]
        for row, metric in enumerate(["best_mse_domain", "best_mse_boundary"]):
            ax = axes[row][col]
            data_tsoulos = sub[sub["method"] == "Tsoulos"][metric].values
            data_pi      = sub[sub["method"] == "PI-NSGA-II"][metric].values

            bp = ax.boxplot(
                [data_tsoulos, data_pi],
                patch_artist=True,
                widths=0.45,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="x", markersize=5, alpha=0.5),
            )
            for patch, method in zip(bp["boxes"], METHOD_LIST):
                patch.set_facecolor(COLORS[method])
                patch.set_alpha(0.8)

            # Jitter
            for j, (data, method) in enumerate(zip([data_tsoulos, data_pi], METHOD_LIST), 1):
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(data))
                ax.scatter(np.full(len(data), j) + jitter, data,
                           color=COLORS[method], alpha=0.5, s=15, zorder=3)

            U, p, r = wilcoxon_test(data_tsoulos, data_pi)
            p_str = f"p={p:.3f}" if p is not None else ""
            label = "Domain MSE" if row == 0 else "Boundary MSE"
            ax.set_title(f"{pde} — {label}\n({p_str}, r={r:.2f} [{effect_label(r)}])"
                         if r is not None else f"{pde} — {label}",
                         fontsize=9, fontweight="bold")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(METHOD_LIST, rotation=25)
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylabel(label)
            ax.grid(True, axis="y", ls="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHOD_LIST]
    fig.legend(handles=patches, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Guardado: {outpath}")
    plt.close(fig)

# ─── Resumen textual ─────────────────────────────────────────────────────────
def print_and_save_summary(tbl: pd.DataFrame, outpath: str):
    n_runs = None
    lines = []
    lines.append("=" * 80)
    lines.append("  ANÁLISIS ESTADÍSTICO: PI-NSGA-II vs Tsoulos (GE)")
    lines.append("  Ecuaciones: Laplace / Poisson / Helmholtz / Schrodinger  —  Ω = [0,1]²")
    lines.append("=" * 80)
    lines.append("")

    hdr = f"{'PDE':<12} {'Métrica':<22} {'Tsoulos mean±std':>18} {'PI mean±std':>18} {'p-val':>8} {'r':>6} {'Ganador':>12}"
    lines.append(hdr)
    lines.append("-" * 80)

    winners = {"Tsoulos": 0, "PI-NSGA-II": 0, "Empate": 0}
    for _, row in tbl.iterrows():
        tm = row.get("Tsoulos_mean", np.nan)
        ts = row.get("Tsoulos_std",  np.nan)
        pm = row.get("PI-NSGA-II_mean", np.nan)
        ps = row.get("PI-NSGA-II_std",  np.nan)
        pv = row.get("p_value", np.nan)
        r  = row.get("effect_r", np.nan)
        w  = row.get("winner", "—")

        tm_s = f"{tm:.3e}±{ts:.2e}" if not np.isnan(tm) else "n/a"
        pm_s = f"{pm:.3e}±{ps:.2e}" if not np.isnan(pm) else "n/a"
        pv_s = f"{pv:.4f}" if pv is not None and not np.isnan(float(pv)) else "n/a"
        r_s  = f"{r:.3f}"  if r is not None and not np.isnan(float(r))  else "n/a"

        line = f"{row['PDE']:<12} {row['Metric']:<22} {tm_s:>18} {pm_s:>18} {pv_s:>8} {r_s:>6} {w:>12}"
        lines.append(line)
        if w in winners:
            winners[w] += 1

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  Victorias globales → Tsoulos: {winners['Tsoulos']}   PI-NSGA-II: {winners['PI-NSGA-II']}")
    lines.append("")
    lines.append("  Interpretación del tamaño de efecto r (Cohen):")
    lines.append("    r < 0.1 negligible | 0.1–0.3 small | 0.3–0.5 medium | >0.5 large")
    lines.append("  p < 0.05 → diferencia estadísticamente significativa.")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)
    with open(outpath, "w") as f:
        f.write(text + "\n")
    print(f"\n  Guardado: {outpath}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/all_runs_summary.csv")
    args = parser.parse_args()

    print(f"\n  Cargando: {args.input}")
    df = load_data(args.input)
    n_runs = df["run"].nunique()
    print(f"  Corridas encontradas: {n_runs}  |  "
          f"PDEs: {df['pde'].nunique()}  |  "
          f"Métodos: {df['method'].nunique()}\n")

    os.makedirs("results", exist_ok=True)

    # Tabla estadística
    tbl = build_stats_table(df)
    tbl.to_csv("results/stats_table.csv", index=False)
    print("  Guardado: results/stats_table.csv")

    # Resumen textual
    print_and_save_summary(tbl, "results/convergence_summary.txt")

    # Box plots
    print("\n  Generando gráficas...")
    plot_boxplot_hv(df,  "results/boxplot_hypervolume.png")
    plot_boxplot_mse(df, "results/boxplot_mse.png")

    print("\n  Análisis completo. Archivos en ./results/")

if __name__ == "__main__":
    main()
