#!/usr/bin/env python3
"""
report/generate_noise_table.py — Tabla y grafico de robustez a ruido para
PySR/PySINDy (ver sota/sota_benchmark_noise.sh).

Motivacion: PySR/PySINDy reciben hoy datos etiquetados PERFECTOS (el grid
exacto/numerico exportado por PISR-NSGA-II) — en un entorno real esos datos
vienen de mediciones con error. sota/run_pysr.py y sota/run_pysindy.py ahora
aceptan --noise_pct para simular eso; este script agrega los resultados
(limpios + con ruido) en una tabla LaTeX y una curva MSE-vs-ruido, para ver
cuanto se degrada cada metodo de comparacion cuando pierde el "dato perfecto"
que hoy le servimos en bandeja — algo que PISR-NSGA-II nunca recibe (cero
datos etiquetados, solo residuo de la EDP + condiciones de frontera).

Formato de archivos que busca (ver run_pysr.py/run_pysindy.py):
  results/[run_N/]<PDE>_pysr_metrics[_noiseNNpct].csv
  results/[run_N/]<PDE>_sindy_metrics[_noiseNNpct].csv
Cada uno con columna noise_pct (0.0 para los limpios, generados antes de
que existiera esta columna quedan en 0.0 por default).

Si todavia no se corrio el barrido de ruido (sota/sota_benchmark_noise.sh),
este script no falla: genera una tabla/figura vacias con una nota, para que
`generate_report.py` pueda incluirlas sin romper el PDF.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(REPORT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR  = os.path.join(REPORT_DIR, "tables")
FIGS_DIR    = os.path.join(REPORT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

SOTA_TO_PISR = {
    "airy": "Airy", "fisher": "Fisher", "duffing": "Duffing",
    "thomas_fermi": "Thomas-Fermi", "lane_emden": "Lane-Emden",
    "troesch": "Troesch", "painleve": "Painleve-I",
    "ginzburg_landau": "Ginzburg-Landau",
}


def _map_pde_name(x):
    for sota, pisr in SOTA_TO_PISR.items():
        if x.startswith(sota + "_"):
            return x.replace(sota + "_", pisr + "_")
    return x


def load_noise_data() -> pd.DataFrame:
    patterns = [
        os.path.join(RESULTS_DIR, "**", "*_pysr_metrics*.csv"),
        os.path.join(RESULTS_DIR, "**", "*_sindy_metrics*.csv"),
        os.path.join(RESULTS_DIR, "*_pysr_metrics*.csv"),
        os.path.join(RESULTS_DIR, "*_sindy_metrics*.csv"),
    ]
    files = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))

    if not files:
        return pd.DataFrame()

    dfs = []
    for f in sorted(files):
        try:
            tmp = pd.read_csv(f)
            if "noise_pct" not in tmp.columns:
                tmp["noise_pct"] = 0.0  # archivos generados antes de este cambio
            dfs.append(tmp)
        except Exception as e:
            print(f"  [WARN] Skipping {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["pde"] = df["pde"].apply(_map_pde_name)
    df["noise_pct"] = pd.to_numeric(df["noise_pct"], errors="coerce").fillna(0.0)
    df["mse_total"] = pd.to_numeric(df["mse_total"], errors="coerce")
    df = df.dropna(subset=["mse_total"])
    # Promedia si hay varias corridas (run_N) para la misma combinacion.
    df = df.groupby(["method", "pde", "noise_pct"], as_index=False).agg(
        mse_total=("mse_total", "mean"),
        mse_total_std=("mse_total", "std"),
        n_runs=("mse_total", "count"),
    )
    return df


def make_noise_table(df: pd.DataFrame):
    out = os.path.join(TABLES_DIR, "noise_robustness.tex")
    if df.empty or df["noise_pct"].nunique() < 2:
        # Sin datos de ruido todavia (solo el baseline 0%, o nada) — se deja
        # una tabla placeholder en vez de romper el \input del reporte.
        lines = [
            r"\begin{table}[ht]",
            r"  \centering",
            r"  \caption{Robustness to Measurement Noise (pending: run "
            r"\texttt{sota/sota\_benchmark\_noise.sh} to populate).}",
            r"  \label{tab:noise_robustness}",
            r"  \emph{No hay corridas con ruido $>0\%$ todavia.}",
            r"\end{table}",
        ]
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  [OK] {out} (placeholder, sin datos de ruido aun)")
        return

    noise_levels = sorted(df["noise_pct"].unique())
    header_cols = " & ".join([rf"\textbf{{{int(round(n*100))}\%}}" for n in noise_levels])

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Robustness to Measurement Noise: Total MSE vs.\ Gaussian noise level "
        r"(fraction of signal std) added to the labeled training data given to PySR/PySINDy. "
        r"PISR-NSGA-II uses zero labeled data and is unaffected by construction — included "
        r"only as a fixed reference line.}",
        r"  \label{tab:noise_robustness}",
        r"  \resizebox{\textwidth}{!}{",
        rf"  \begin{{tabular}}{{ll{'c' * len(noise_levels)}}}",
        r"    \toprule",
        rf"    \textbf{{PDE}} & \textbf{{Method}} & {header_cols} \\",
        r"    \midrule",
    ]

    for pde in sorted(df["pde"].unique()):
        for method in sorted(df[df["pde"] == pde]["method"].unique()):
            sub = df[(df["pde"] == pde) & (df["method"] == method)]
            cells = []
            for n in noise_levels:
                row = sub[sub["noise_pct"] == n]
                if row.empty:
                    cells.append("---")
                else:
                    m = row["mse_total"].iloc[0]
                    cells.append(f"${m:.2e}$" if np.isfinite(m) else "---")
            pde_tex = pde.replace("_", r"\_")
            lines.append(rf"    {pde_tex} & {method} & {' & '.join(cells)} \\")
        lines.append(r"    \midrule")

    if lines[-1] == r"    \midrule":
        lines[-1] = r"    \bottomrule"
    else:
        lines.append(r"    \bottomrule")
    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


def make_noise_plot(df: pd.DataFrame):
    out = os.path.join(FIGS_DIR, "noise_robustness.pdf")
    if df.empty or df["noise_pct"].nunique() < 2:
        # Placeholder: figura vacia con una nota, para que \includegraphics no
        # falle si el reporte se corre antes de tener datos de ruido.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Sin corridas con ruido $>0\\%$ todavia\n"
                           "(correr sota/sota_benchmark_noise.sh)",
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {out} (placeholder, sin datos de ruido aun)")
        return

    pdes = sorted(df["pde"].unique())
    n = len(pdes)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    colors = {"PySR": "#1f77b4", "PySINDy": "#d62728"}
    for i, pde in enumerate(pdes):
        ax = axes[i // ncols][i % ncols]
        sub_pde = df[df["pde"] == pde]
        for method in sorted(sub_pde["method"].unique()):
            sub = sub_pde[sub_pde["method"] == method].sort_values("noise_pct")
            if sub.empty:
                continue
            ax.plot(sub["noise_pct"] * 100, sub["mse_total"], marker="o",
                    color=colors.get(method, None), label=method)
        ax.set_yscale("log")
        ax.set_xlabel("Ruido gaussiano (% std señal)")
        ax.set_ylabel("MSE Total (vs. verdad limpia)")
        ax.set_title(pde)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()

    fig.suptitle("Degradación por ruido en los datos de entrenamiento (PySR / PySINDy)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def main():
    df = load_noise_data()
    if df.empty:
        print("  [INFO] No metrics files found at all (ni limpios ni con ruido).")
    make_noise_table(df)
    make_noise_plot(df)


if __name__ == "__main__":
    main()
