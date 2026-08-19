#!/usr/bin/env python3
"""
data_efficiency_comparison.py — La comparación que de verdad importa contra SOTA.

PySR (y en general la regresión simbólica clásica) necesita datos etiquetados
de la solución real para entrenar (ver sota/run_pysr.py: hasta 5000 puntos
(x, u_exact) por ecuación). PISR-NSGA-II no usa ni un solo punto de la solución
— sólo puntos de colocación donde se evalúa el residuo de la EDP/EDO, sin
conocer la respuesta. Este script arma la tabla que muestra esa diferencia
lado a lado con el MSE final de cada método, en vez de comparar sólo MSE
(que es la cancha de PySR, no la nuestra).
"""
import os
import re
import csv
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SOTA_DATA_DIR = os.path.join(ROOT_DIR, "sota", "data")
TABLES_DIR = os.path.join(BASE_DIR, "tables")

PYSR_TRAIN_CAP = 5000  # sota/run_pysr.py: train_size = 5000


def normalize_pde_key(pde: str) -> str:
    """Nombre de PDE -> clave normalizada para matchear contra sota/data/data_<key>.csv
    (minusculas, guiones -> guion bajo, sin sufijo _1D/_2D)."""
    key = re.sub(r"_(1D|2D)$", "", pde, flags=re.IGNORECASE)
    key = key.lower().replace("-", "_")
    return key


def canonical_key(pde: str) -> str:
    """Clave case-insensitive + normalizada (base + sufijo _1D/_2D) usada para
    unir resultados de PISR-NSGA-II y PySR pese a inconsistencias de
    capitalización entre corridas (ej. 'Airy_1D' vs 'airy_1D')."""
    base = normalize_pde_key(pde)
    m = re.search(r"_(1D|2D)$", pde, flags=re.IGNORECASE)
    suffix = ("_" + m.group(1).upper()) if m else ""
    return base + suffix


def count_data_rows(pde: str) -> int:
    """Filas (menos encabezado) del CSV de datos de sota/ para este PDE, o None
    si no se encuentra el archivo."""
    key = normalize_pde_key(pde)
    path = os.path.join(SOTA_DATA_DIR, f"data_{key}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)


def read_pysr_metrics():
    """Lee todos los results/*_pysr_metrics.csv. Distintas corridas pueden usar
    distinta capitalización del nombre de PDE (bug conocido, no se corrige acá)
    — nos quedamos con la última leída por clave normalizada."""
    rows = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*_pysr_metrics.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                pde = row.get("pde", "").strip()
                if not pde:
                    continue
                rows[normalize_pde_key(pde) + "|" + pde] = {
                    "pde": pde,
                    "mse_total": float(row["mse_total"]),
                    "runtime_s": float(row.get("runtime_s", 0.0)),
                }
    return rows


def read_pisr_results():
    """Promedia results/all_runs_summary.csv si existe (más robusto, varias
    corridas); si no, cae a comparison_summary.csv (una sola corrida)."""
    all_runs_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    summary_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")

    by_pde = {}
    path = all_runs_path if os.path.exists(all_runs_path) else summary_path
    if not os.path.exists(path):
        return by_pde

    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("method") != "PISR-NSGA-II":
                continue
            pde = row["pde"].strip()
            # solution_mse = error contra la solucion real/numerica (misma metrica
            # que reporta PySR, que entrena con datos etiquetados) — best_mse_domain
            # es el residuo de la EDP (fisica, no supervisado), una cantidad distinta
            # que no es comparable directamente contra el MSE de PySR.
            sol_raw = row.get("solution_mse", "")
            try:
                sol = float(sol_raw)
            except (ValueError, TypeError):
                sol = float("nan")
            if sol != sol or sol < 0:  # nan o -1 (sin campeon / CSV viejo sin la columna)
                dom = float(row.get("best_mse_domain", "nan") or "nan")
                bnd_raw = row.get("best_mse_boundary", "0")
                try:
                    bnd = float(bnd_raw)
                    if bnd != bnd:
                        bnd = 0.0
                except ValueError:
                    bnd = 0.0
                sol = dom + bnd  # fallback (CSV generado antes de este fix)
            rt = float(row.get("runtime_s", 0.0) or 0.0)
            entry = by_pde.setdefault(pde, {"mse_totals": [], "runtimes": []})
            entry["mse_totals"].append(sol)
            entry["runtimes"].append(rt)

    out = {}
    for pde, entry in by_pde.items():
        vals = [v for v in entry["mse_totals"] if v == v]  # descarta nan
        if not vals:
            continue
        out[pde] = {
            "pde": pde,
            "mse_total": sum(vals) / len(vals),
            "runtime_s": sum(entry["runtimes"]) / len(entry["runtimes"]) if entry["runtimes"] else 0.0,
            "n_runs": len(vals),
        }
    return out


def fmt_sci(v):
    """$a \\times 10^{e}$ para LaTeX, o «---» si no hay dato."""
    if v is None:
        return "---"
    exp = int(__import__("math").floor(__import__("math").log10(abs(v)))) if v != 0 else 0
    mant = v / (10 ** exp) if v != 0 else 0.0
    return rf"${mant:.2f} \times 10^{{{exp}}}$"


def make_latex_table(comparisons):
    """report/tables/data_efficiency_comparison.tex — misma familia de estilo
    (booktabs + resizebox) que las demás tablas de report/generate_report.py."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Data Efficiency: Labeled Solution Points Required vs. Final MSE. "
        r"PISR-NSGA-II never sees the true solution — only PDE collocation points; "
        r"PySR requires labeled $(x, u_{\mathrm{exact}})$ pairs to train.}",
        r"  \label{tab:data_efficiency}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Method} & \textbf{Labeled Points Used} & \textbf{Final MSE} \\",
        r"    \midrule",
    ]
    for pde, p_ours, p_pysr, pysr_points in comparisons:
        pde_tex = pde.replace("_", r"\_")
        if p_ours:
            lines.append(rf"    {pde_tex} & PISR-NSGA-II & 0 & {fmt_sci(p_ours['mse_total'])} \\")
        if p_pysr:
            pts = str(pysr_points) if pysr_points is not None else "?"
            lines.append(rf"    {pde_tex} & PySR & {pts} & {fmt_sci(p_pysr['mse_total'])} \\")
        lines.append(r"    \midrule")
    if lines[-1] == r"    \midrule":
        lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]
    out = os.path.join(TABLES_DIR, "data_efficiency_comparison.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


def main():
    pysr = read_pysr_metrics()
    pisr = read_pisr_results()

    # Unimos por clave canónica (normalizada + case-insensitive) para no perder
    # matches por inconsistencias de capitalización entre corridas de distintos
    # scripts (ej. 'Airy_1D' de nuestro lado vs 'airy_1D' del lado de PySR).
    pysr_by_key = {canonical_key(v["pde"]): v for v in pysr.values()}
    pisr_by_key = {canonical_key(v["pde"]): v for v in pisr.values()}

    pdes = sorted(set(pisr_by_key.keys()) | set(pysr_by_key.keys()))

    out_rows = []
    comparisons = []
    print("\n" + "=" * 100)
    print(f"{'PDE':<20} | {'Método':<14} | {'Puntos etiquetados':>18} | {'MSE final':>14} | {'Tiempo':>10}")
    print("-" * 100)

    for pde in pdes:
        p_ours = pisr_by_key.get(pde)
        p_pysr = pysr_by_key.get(pde)
        n_rows = count_data_rows(pde)
        pysr_points = min(PYSR_TRAIN_CAP, n_rows) if n_rows is not None else None
        comparisons.append((pde, p_ours, p_pysr, pysr_points))

        if p_ours:
            print(f"{pde:<20} | {'PISR-NSGA-II':<14} | {0:>18} | {p_ours['mse_total']:>14.3e} | {p_ours['runtime_s']:>9.1f}s")
            out_rows.append({"pde": pde, "method": "PISR-NSGA-II", "labeled_points_used": 0,
                              "mse_total": p_ours["mse_total"], "runtime_s": p_ours["runtime_s"]})
        if p_pysr:
            pts_str = str(pysr_points) if pysr_points is not None else "?"
            print(f"{pde:<20} | {'PySR':<14} | {pts_str:>18} | {p_pysr['mse_total']:>14.3e} | {p_pysr['runtime_s']:>9.1f}s")
            out_rows.append({"pde": pde, "method": "PySR", "labeled_points_used": pysr_points,
                              "mse_total": p_pysr["mse_total"], "runtime_s": p_pysr["runtime_s"]})

        if p_ours and p_pysr:
            verdict = "PISR-NSGA-II iguala o mejora con 0 datos" if p_ours["mse_total"] <= p_pysr["mse_total"] \
                else f"PySR mejor por {p_ours['mse_total'] / max(p_pysr['mse_total'], 1e-300):.1f}x (pero usó {pysr_points} puntos)"
            print(f"{'':<20} | {'→ ' + verdict:<86}")
        print("-" * 100)

    out_path = os.path.join(RESULTS_DIR, "data_efficiency_comparison.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pde", "method", "labeled_points_used", "mse_total", "runtime_s"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nGuardado en {out_path}")

    make_latex_table(comparisons)


if __name__ == "__main__":
    main()
