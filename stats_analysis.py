#!/usr/bin/env python3
"""
stats_analysis.py — Análisis estadístico de rendimiento para PI-NSGA-II.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def main():
    path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(path): 
        print(f"[ERROR] No se encontró {path}")
        return
    
    df = pd.read_csv(path)
    df["mse_total"] = pd.to_numeric(df["best_mse_domain"], errors='coerce') + pd.to_numeric(df["best_mse_boundary"], errors='coerce')
    df = df.dropna(subset=["mse_total"])
    
    # Filtrar solo nuestro algoritmo
    df = df[df["method"] == "PI-NSGA-II"]
    pdes = sorted(df["pde"].unique())

    print("\n" + "="*80)
    print(f"{'PDE Case':<25} {'PI-NSGA-II (Mean ± Std)':<35} {'Runtime (s)':<15}")
    print("-" * 80)

    for pde in pdes:
        sub = df[df["pde"] == pde]
        
        errs = sub["mse_total"].values
        times = sub["runtime_s"].values
        
        m_err, s_err = (np.mean(errs), np.std(errs)) if len(errs)>0 else (0, 0)
        m_time = np.mean(times) if len(times)>0 else 0
        
        str_err = f"{m_err:.2e} ± {s_err:.2e}"
        str_time = f"{m_time:.2f}s"
        
        print(f"{pde:<25} {str_err:<35} {str_time:<15}")

    print("-" * 80)

if __name__ == "__main__":
    main()
