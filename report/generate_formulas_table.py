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
        # La ecuacion de Thomas-Fermi real (u''=u^{1.5}/\sqrt{x}) no tiene solucion
        # cerrada conocida (por eso se resuelve por shooting/RK4 para la verdad
        # numerica) — esto es solo la funcion usada como ancla de frontera/bc,
        # no la solucion exacta de la EDO no lineal.
        ("Thomas-Fermi", 1): r"\frac{1}{x+0.5} \; \text{(ancla de frontera, no exacta — ver verdad numerica)}",
        ("Thomas-Fermi", 2): r"\frac{1}{x+y+0.5} \; \text{(ancla de frontera, no exacta — ver verdad numerica)}",
        ("Laplace", 1): r"x",
        ("Poisson", 1): r"\sin(\pi x)",
        ("HarmonicOscillator", 1): r"e^{-x^2/2}",
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
        "Laplace", "Poisson", "HarmonicOscillator",
        "Airy", "Fisher", "Duffing", "Thomas-Fermi",
        "Navier-Stokes", "Navier-Stokes-Unsteady",
        "Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I"
    ]
    sota_map = {
        "Thomas-Fermi": "thomas_fermi",
        "Lane-Emden": "lane_emden",
        "Painleve-I": "painleve",
        "Ginzburg-Landau": "ginzburg_landau"
    }
    def get_sota_formula(pde, d, method):
        prefix = "pysr" if method == "PySR" else "sindy"
        # sota_benchmark.sh guarda "{prefix}_{pde}_{d}D_run{N}.log" (con
        # sufijo de corrida) — antes esto buscaba sin el sufijo y nunca
        # encontraba nada, dejando la tabla vacia aunque el script sota si
        # habia corrido. Se prueban ambos patrones (con y sin sufijo _run1)
        # por compatibilidad con logs viejos generados a mano.
        base = os.path.join(ROOT_DIR, "sota", "results", f"{prefix}_{pde}_{d}D")
        log_path = base + ".log"
        if not os.path.exists(log_path):
            import glob as _glob
            candidates = sorted(_glob.glob(base + "_run*.log"))
            log_path = candidates[-1] if candidates else log_path
        if not os.path.exists(log_path): return "---"
        
        with open(log_path, "r") as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # PySR outputs "Best Discovered Equation by PySR for Airy:" then the equation
            if "Best Discovered Equation" in line and i+1 < len(lines):
                eq = lines[i+1].strip()
                if "---" in eq: continue
                try:
                    import sympy as sp
                    expr = sp.sympify(eq)
                    expr = expr.xreplace({n: round(n, 3) for n in expr.atoms(sp.Number)})
                    latex_str = sp.latex(expr)
                    return f"${latex_str}$"
                except Exception:
                    eq = eq.replace("_", r"\_").replace("%", r"\%").replace("^", r"\textasciicircum{}")
                    if len(eq) > 100: eq = eq[:97] + "..."
                    return f"\\texttt{{{eq}}}"
            # PySINDy parsing block removed
        return "---"
    
    for pde in pde_list:
        for d in dims:
            if pde in ["Navier-Stokes", "Navier-Stokes-Unsteady"] and d == 1: continue
            if pde in ["Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I",
                       "Laplace", "Poisson", "HarmonicOscillator"] and d == 2: continue
            p_pi = get_formula(pde, d, "PISR-EMOAD")
            ex_eq = exacts.get((pde, d), "N/A")
            
            p_pysr = get_sota_formula(pde, d, "PySR")
            # PySINDy solo corre en 1D (ver sota/run_pysindy.py: reformula el
            # BVP como sistema u'=v, v'=f(u,v) — no aplica limpio a 2D).
            p_sindy = get_sota_formula(pde, d, "PySINDy") if d == 1 else None
            n_rows = 4 if p_sindy else 3
            lines.append(rf"    \multirow{{{n_rows}}}{{*}}{{{pde} ({d}D)}}")
            lines.append(rf"    & PISR-NSGA-II & ${p_pi}$ \\")
            lines.append(rf"    & PySR & {p_pysr} \\")
            if p_sindy:
                lines.append(rf"    & PySINDy & {p_sindy} \\")
            lines.append(rf"    & \textbf{{Exact}} & $\mathbf{{{ex_eq}}}$ \\")
            lines.append(r"    \midrule")
            
    lines += [r"    \bottomrule", r"  \end{tabular}", r"  }", r"\end{table*}"]
    
    outpath = os.path.join(TABLES_DIR, "formulas_table.tex")
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Guardado: {outpath}")

if __name__ == "__main__":
    main()
