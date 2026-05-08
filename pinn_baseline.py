#!/usr/bin/env python3
"""
pinn_baseline.py  — PINN baseline usando DeepXDE (PyTorch backend).
Soporta problemas 1D y 2D.
"""

import os
import sys
import time
import argparse
import csv
import warnings
import glob
warnings.filterwarnings("ignore")

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import numpy as np
import torch
import deepxde as dde

dde.config.set_default_float("float64")
_pi = np.pi

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METHOD = "PINN"

# ── Ecuaciones exactas ────────────────────────────────────────────────────────
def get_exact_fn(pde_name, dim):
    if dim == 1:
        if pde_name == "Laplace":   return lambda x: x
        if pde_name == "Poisson":   return lambda x: np.sin(_pi * x)
        if pde_name == "Helmholtz": return lambda x: np.sin(_pi * x)
        if pde_name == "Schrodinger": return lambda x: np.exp(-(x - 0.5)**2)
    else:
        if pde_name == "Laplace":   return lambda x, y: np.sin(_pi * x) * np.sinh(_pi * y) / np.sinh(_pi)
        if pde_name == "Poisson":   return lambda x, y: np.sin(_pi * x) * np.sin(_pi * y)
        if pde_name == "Helmholtz": return lambda x, y: np.sin(_pi * x) * np.sin(_pi * y)
        if pde_name == "Schrodinger":
            return lambda x, y: np.exp(-((x - 0.5)**2 + (y - 0.5)**2))
    return None

def solve_pde(pde_name, dim, n_domain, n_boundary, epochs, lr):
    if dim == 1:
        geom = dde.geometry.Interval(0, 1)
    else:
        geom = dde.geometry.Rectangle([0, 0], [1, 1])

    exact_fn = get_exact_fn(pde_name, dim)

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        res = u_xx
        if dim == 2:
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            res += u_yy
        
        if pde_name == "Laplace":
            return res
        elif pde_name == "Poisson":
            if dim == 1:
                f = -_pi**2 * torch.sin(_pi * x[:, 0:1])
            else:
                f = -2 * _pi**2 * torch.sin(_pi * x[:, 0:1]) * torch.sin(_pi * x[:, 1:2])
            return res - f
        elif pde_name == "Helmholtz":
            k = 1.0
            if dim == 1:
                f = (k**2 - _pi**2) * torch.sin(_pi * x[:, 0:1])
            else:
                f = (k**2 - 2*_pi**2) * torch.sin(_pi * x[:, 0:1]) * torch.sin(_pi * x[:, 1:2])
            return res + k**2 * u - f
        elif pde_name == "Schrodinger":
            r2 = (x[:, 0:1] - 0.5)**2
            if dim == 2: r2 += (x[:, 1:2] - 0.5)**2
            V = 4.0 * r2
            E = 2.0 if dim == 1 else 4.0
            return res + (E - V) * u
        return res

    def bc_val(x):
        if dim == 1: return exact_fn(x[:, 0:1])
        return exact_fn(x[:, 0:1], x[:, 1:2])

    bc = dde.icbc.DirichletBC(geom, bc_val, lambda _, on_bnd: on_bnd)
    data = dde.data.PDE(geom, pde, bc, num_domain=n_domain, num_boundary=n_boundary, num_test=500)
    
    net = dde.nn.FNN([dim] + [64] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    
    model.compile("adam", lr=lr, loss_weights=[1, 100])
    model.train(iterations=epochs, display_every=1000)
    
    dde.optimizers.config.set_LBFGS_options(maxiter=2000)
    model.compile("L-BFGS")
    model.train()
    
    return model, exact_fn

def eval_model(model, exact_fn, pde_name, dim, run_dir):
    if dim == 1:
        pts = np.linspace(0, 1, 100).reshape(-1, 1)
        u_approx = model.predict(pts).ravel()
        u_exact  = exact_fn(pts[:, 0]).ravel()
    else:
        xs = np.linspace(0, 1, 50)
        xx, yy = np.meshgrid(xs, xs)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        u_approx = model.predict(pts).ravel()
        u_exact  = exact_fn(pts[:,0], pts[:,1]).ravel()

    mse_dom = float(np.mean((u_approx - u_exact)**2))
    
    # Boundary MSE
    if dim == 1:
        bnd_pts = np.array([[0.0], [1.0]])
        u_b_approx = model.predict(bnd_pts).ravel()
        u_b_exact  = exact_fn(bnd_pts[:, 0]).ravel()
    else:
        t = np.linspace(0, 1, 50)
        bnd_pts = np.concatenate([
            np.column_stack([t, np.zeros_like(t)]), np.column_stack([t, np.ones_like(t)]),
            np.column_stack([np.zeros_like(t), t]), np.column_stack([np.ones_like(t), t])
        ])
        u_b_approx = model.predict(bnd_pts).ravel()
        u_b_exact  = exact_fn(bnd_pts[:, 0], bnd_pts[:, 1]).ravel()
    
    mse_bnd = float(np.mean((u_b_approx - u_b_exact)**2))
    
    # Save grid/line CSV
    label = f"{pde_name}_{dim}D"
    grid_path = os.path.join(run_dir, f"grid_{label}_{METHOD}.csv")
    with open(grid_path, "w", newline="") as f:
        w = csv.writer(f)
        if dim == 1:
            w.writerow(["x", "u_exact", "u_approx"])
            for i in range(len(pts)):
                w.writerow([f"{pts[i,0]:.6f}", f"{u_exact[i]:.6f}", f"{u_approx[i]:.6f}"])
        else:
            w.writerow(["x", "y", "u_exact", "u_approx"])
            for i in range(len(pts)):
                w.writerow([f"{pts[i,0]:.6f}", f"{pts[i,1]:.6f}", f"{u_exact[i]:.6f}", f"{u_approx[i]:.6f}"])
    
    return mse_dom, mse_bnd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    pde_list = ["Laplace", "Poisson", "Helmholtz", "Schrodinger"]
    dims = [1, 2]

    all_rows = []
    for run_idx in range(args.runs):
        run_dir = RESULTS_DIR if args.runs == 1 else os.path.join(RESULTS_DIR, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        for d in dims:
            for name in pde_list:
                print(f"\n--- PINN: {name} {d}D (Run {run_idx}) ---")
                t0 = time.time()
                model, exact_fn = solve_pde(name, d, 1000 if d==1 else 2500, 100, args.epochs, 1e-3)
                rt = time.time() - t0
                
                mse_dom, mse_bnd = eval_model(model, exact_fn, name, d, run_dir)
                print(f"  MSE Dom: {mse_dom:.4e}, BC: {mse_bnd:.4e}")
                
                lbl = name + (f"_{d}D" if d==2 else "_1D") # Match C++ labels
                # But C++ uses name + "_1D" or "_2D"
                # Wait, let's check main.cpp labels: prob.name() + (prob.dim == 1 ? "_1D" : "_2D")
                label_cpp = f"{name}_{d}D"
                
                # Save individual Pareto CSV
                p_path = os.path.join(run_dir, f"{label_cpp}_{METHOD.lower()}_pareto.csv")
                with open(p_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["method", "pde", "dim", "mse_domain", "mse_boundary", "rank"])
                    w.writerow([METHOD, name, d, f"{mse_dom:.10f}", f"{mse_bnd:.10f}", 1])

                all_rows.append({
                    "run": run_idx, "method": METHOD, "pde": label_cpp, "dim": d,
                    "best_mse_domain": mse_dom, "best_mse_boundary": mse_bnd,
                    "runtime_s": rt
                })

    # Update global summaries
    all_runs_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    file_exists = os.path.exists(all_runs_path)
    with open(all_runs_path, "a", newline="") as f:
        fieldnames = ["run", "method", "pde", "pareto_size", "best_mse_domain", "best_mse_boundary",
                      "mean_mse_domain", "mean_mse_boundary", "hypervolume", "runtime_s"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: w.writeheader()
        for r in all_rows:
            w.writerow({
                "run": r["run"], "method": r["method"], "pde": r["pde"],
                "pareto_size": 1, "best_mse_domain": r["best_mse_domain"], "best_mse_boundary": r["best_mse_boundary"],
                "mean_mse_domain": r["best_mse_domain"], "mean_mse_boundary": r["best_mse_boundary"],
                "hypervolume": 0.0, "runtime_s": r["runtime_s"]
            })

if __name__ == "__main__":
    main()
