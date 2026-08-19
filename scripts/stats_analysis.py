#!/usr/bin/env python3
"""
stats_analysis.py — Análisis estadístico de rendimiento para PISR-NSGA-II.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def main():
    path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(path): 
        print(f"[ERROR] No se encontró {path}")
        return
    
    df = pd.read_csv(path)
    df["mse_total"] = pd.to_numeric(df["best_mse_domain"], errors='coerce') + pd.to_numeric(df["best_mse_boundary"], errors='coerce')
    # Para PISR-NSGA-II, mse_total (residuo de la EDP, fisica no supervisada) no es
    # comparable con PINN/RK4-FDM (error contra la solucion real) — si el CSV trae
    # la columna solution_mse (post-fix), la usamos para esas filas en su lugar.
    if "solution_mse" in df.columns:
        sol = pd.to_numeric(df["solution_mse"], errors="coerce")
        is_pi = df["method"] == "PISR-NSGA-II"
        valid_sol = is_pi & sol.notna() & (sol >= 0)
        df.loc[valid_sol, "mse_total"] = sol[valid_sol]
    df = df.dropna(subset=["mse_total"])
    
    # Filtrar métodos
    pdes = sorted(df["pde"].unique())

    print("\n" + "="*115)
    print(f"{'PDE':<22} | {'PISR-NSGA-II (Actual)':<26} | {'PINN':<26} | {'RK4/FDM':<18}")
    print("-" * 115)

    for pde in pdes:
        sub_pi   = df[(df["pde"] == pde) & (df["method"] == "PISR-NSGA-II")]
        sub_pinn = df[(df["pde"] == pde) & (df["method"] == "PINN")]
        sub_num  = df[(df["pde"] == pde) & (df["method"] == "RK4/FDM")]
        
        def get_stats(sub):
            errs = sub["mse_total"].values
            times = sub["runtime_s"].values
            m_err, s_err = (np.mean(errs), np.std(errs)) if len(errs)>0 else (0, 0)
            m_time = np.mean(times) if len(times)>0 else 0
            if len(errs) == 0:
                return "—", "—"
            return f"{m_err:.2e} ± {s_err:.2e}", f"{m_time:.2f}s"
            
        err_pi, time_pi = get_stats(sub_pi)
        err_pinn, time_pinn = get_stats(sub_pinn)
        err_num, time_num = get_stats(sub_num)
        
        print(f"{pde:<22} | {err_pi:<17} ({time_pi:<6}) | {err_pinn:<17} ({time_pinn:<6}) | {err_num:<11} ({time_num:<5})")

    print("-" * 115)

if __name__ == "__main__":
    main()
