#!/usr/bin/env python3
import os

# Rutas dinámicas relativas a la ubicación del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

def get_formula(pde, dim, method):
    label = f"{pde}_{dim}D"
    import glob
    pattern = os.path.join(RESULTS_DIR, "**", f"expr_{label}_{method}.tex")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return "N/A"
    
    path = matches[0]
    with open(path, "r") as f:
        content = f.read().strip()
        # Eliminar signos de dólar y posibles prefijos de fórmula
        content = content.replace("$$", "").replace("$", "")
        for prefix in ["\\hat{u}(x,y) =", "\\hat{u}(x) ="]:
            content = content.replace(prefix, "")
        return content.strip()
    return "N/A"

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    pdes = ["Laplace", "Poisson", "Helmholtz", "Schrodinger"]
    dims = [1, 2]
    
    exacts = {
        ("Airy", 1): r"\text{Numerical (Ground Truth)}",
        ("Airy", 2): r"\text{Numerical (Ground Truth)}",
        ("Fisher", 1): r"\text{Numerical (Ground Truth)}",
        ("Fisher", 2): r"\text{Numerical (Ground Truth)}",
        ("Duffing", 1): r"\text{Numerical (Ground Truth)}",
        ("Duffing", 2): r"\text{Numerical (Ground Truth)}",
        ("Thomas-Fermi", 1): r"\text{Numerical (Ground Truth)}",
        ("Thomas-Fermi", 2): r"\text{Numerical (Ground Truth)}",
        ("Navier-Stokes", 2): r"y - \frac{e^{\lambda x}}{2 \pi \text{Re}} \sin(2 \pi y)",
        ("Navier-Stokes-Unsteady", 2): r"\sin(\pi x) \sin(\pi y) e^{-\lambda t}",
        ("Lane-Emden", 1): r"1 - x^2/6",
        ("Troesch", 1): r"\text{Numerical (Ground Truth)}",
        ("Ginzburg-Landau", 1): r"\text{Numerical (Ground Truth)}",
        ("Painleve-I", 1): r"\text{Numerical (Ground Truth)}"
    }
    
    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Comparison of Symbolic Expressions (1D and 2D).}",
        r"  \label{tab:formulas}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llp{14cm}}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Method} & \textbf{Symbolic Expression} $\hat{u}$ \\",
        r"    \midrule"
    ]
    
    pde_list = [
        "Airy", "Fisher", "Duffing", "Thomas-Fermi",
        "Navier-Stokes", "Navier-Stokes-Unsteady",
        "Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I"
    ]
    
    for pde in pde_list:
        for d in dims:
            if pde in ["Navier-Stokes", "Navier-Stokes-Unsteady"] and d == 1: continue
            if pde in ["Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I"] and d == 2: continue
            p_pi = get_formula(pde, d, "PI-NSGA-II")
            ex_eq = exacts.get((pde, d), "N/A")
            
            lines.append(rf"    \multirow{{2}}{{*}}{{{pde} ({d}D)}}")
            lines.append(rf"    & PI-NSGA-II & ${p_pi}$ \\")
            lines.append(rf"    & \textbf{{Exact}} & $\mathbf{{{ex_eq}}}$ \\")
            lines.append(r"    \midrule")
            
    lines += [r"    \bottomrule", r"  \end{tabular}", r"  }", r"\end{table*}"]
    
    outpath = os.path.join(TABLES_DIR, "formulas_table.tex")
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Guardado: {outpath}")

if __name__ == "__main__":
    main()
