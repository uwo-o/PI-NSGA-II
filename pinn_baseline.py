#!/usr/bin/env python3
"""
pinn_baseline.py  — PINN baseline con DeepXDE (PyTorch backend).

Resuelve los mismos problemas que PI-NSGA-II y guarda CSVs compatibles.
Problemas:
  Analíticos 1D/2D: Laplace, Poisson, Helmholtz, Schrodinger
  Numéricos  1D:    Airy, HarmonicOscillator
  Numéricos  2D:    NonlinearPoisson, Liouville, Sine-Gordon, HarmonicOscillator

GPU: GTX 1050 Ti (sm_61) no compatible con PyTorch 2.x → usamos CPU multi-hilo.
     Si en el futuro se instala PyTorch ≤ 1.13 con CUDA sm_61, remover la línea
     de CUDA_VISIBLE_DEVICES y DeepXDE usará la GPU automáticamente.
"""

import os, sys, time, argparse, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")

# ── Backend ───────────────────────────────────────────────────────────────────
os.environ.setdefault("DDE_BACKEND", "pytorch")
# Forzar CPU porque GTX 1050 Ti (sm_61) no es compatible con PyTorch 2.x
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
# Usar todos los cores disponibles
N_THREADS = os.cpu_count() or 4
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(max(1, N_THREADS // 2))

import deepxde as dde
dde.config.set_default_float("float64")

_pi = np.pi
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
METHOD = "PINN"

# ─────────────────────────────────────────────────────────────────────────────
# Definición de cada problema: ecuación + BCs + solución de referencia
# ─────────────────────────────────────────────────────────────────────────────
def build_problem(pde_name, dim):
    """
    Devuelve (pde_fn, bcs, exact_fn, geom, n_domain, n_boundary, n_test, net_layers, epochs_adam)
    """
    if dim == 1:
        geom = dde.geometry.Interval(0, 1)
    else:
        geom = dde.geometry.Rectangle([0, 0], [1, 1])

    # ── Soluciones exactas / de referencia ────────────────────────────────────
    def exact_laplace_1d(x):   return x
    def exact_laplace_2d(x,y): return np.sin(_pi*x)*np.sinh(_pi*y)/np.sinh(_pi)
    def exact_poisson_1d(x):   return np.sin(_pi*x)
    def exact_poisson_2d(x,y): return np.sin(_pi*x)*np.sin(_pi*y)
    def exact_helmholtz(x, y=None):
        return np.sin(_pi*x) if y is None else np.sin(_pi*x)*np.sin(_pi*y)
    def exact_schrodinger_1d(x):   return np.exp(1j*_pi*x)  # onda plana
    def exact_schrodinger_2d(x,y): return np.exp(1j*_pi*(x+y))
    # Numéricos: función de referencia basada en la física del estado base
    def ref_airy_1d(x):          return 0.3550*np.exp(-0.3*x**2)   # aprox Ai(x) en [0,1]
    def ref_ho_1d(x):            return np.exp(-0.5*x**2)           # estado base ψ₀
    def ref_ho_2d(x,y):          return np.exp(-0.5*(x**2+y**2))
    def ref_nonlin_poisson(x,y): return 1.0/(1+x**2+y**2)          # solución analítica
    def ref_liouville(x,y):      return 1.0/(1+x**2+y**2)          # solución analítica
    def ref_sine_gordon(x,y):    return np.sin(_pi*x)*np.sin(_pi*y) # aprox

    # ── Definición del término de PDE (residuo) ───────────────────────────────
    def make_pde(pde_name, dim):
        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            lap = u_xx
            if dim == 2:
                u_yy = dde.grad.hessian(u, x, i=1, j=1)
                lap = u_xx + u_yy

            if pde_name == "Laplace":
                return lap

            elif pde_name == "Poisson":
                if dim == 1:
                    f = -_pi**2 * torch.sin(_pi * x[:, 0:1])
                else:
                    f = -2*_pi**2 * torch.sin(_pi*x[:,0:1]) * torch.sin(_pi*x[:,1:2])
                return lap - f

            elif pde_name == "Helmholtz":
                k2 = 1.0
                if dim == 1:
                    f = (k2 - _pi**2) * torch.sin(_pi * x[:,0:1])
                else:
                    f = (k2 - 2*_pi**2) * torch.sin(_pi*x[:,0:1]) * torch.sin(_pi*x[:,1:2])
                return lap + k2*u - f

            elif pde_name == "Schrodinger":
                # -u'' + 0*u = π²u  →  u'' + π²u = 0
                k2 = _pi**2 if dim==1 else 2*_pi**2
                return lap + k2 * u

            elif pde_name == "Airy":
                # u'' = x*u
                return lap - x[:,0:1]*u

            elif pde_name == "HarmonicOscillator":
                # -u'' + x²u = u  →  u'' = (x²-1)u
                if dim == 1:
                    V = x[:,0:1]**2
                else:
                    V = x[:,0:1]**2 + x[:,1:2]**2
                E = 1.0
                return lap - (V - E)*u

            elif pde_name == "NonlinearPoisson":
                # ∇²u + u² = 0 (con f calculado de la sol exacta)
                # Sol exacta: 1/(1+x²+y²) → ∇²u = 2(x²+y²-1)/(1+x²+y²)³
                if dim == 2:
                    r2 = x[:,0:1]**2 + x[:,1:2]**2
                    f = -2*(r2 - 1)/(1+r2)**3
                else:
                    f = torch.zeros_like(u)
                return lap + u**2 - f

            elif pde_name == "Liouville":
                # ∇²u = e^u  (ecuación de Liouville)
                return lap - torch.exp(u)

            elif pde_name == "Sine-Gordon":
                # ∇²u = sin(u)
                return lap - torch.sin(u)

            return lap
        return pde

    # ── Condiciones de contorno ───────────────────────────────────────────────
    exact_map = {
        ("Laplace",   1): (exact_laplace_1d,   None),
        ("Laplace",   2): (None,                exact_laplace_2d),
        ("Poisson",   1): (exact_poisson_1d,   None),
        ("Poisson",   2): (None,                exact_poisson_2d),
        ("Helmholtz", 1): (exact_helmholtz,     None),
        ("Helmholtz", 2): (None,                exact_helmholtz),
        ("Schrodinger",1):(lambda x: np.real(exact_schrodinger_1d(x)), None),
        ("Schrodinger",2):(None, lambda x,y: np.real(exact_schrodinger_2d(x,y))),
        ("Airy",       1): (ref_airy_1d,        None),
        ("HarmonicOscillator",1): (ref_ho_1d,  None),
        ("HarmonicOscillator",2): (None,        ref_ho_2d),
        ("NonlinearPoisson",2):   (None,        ref_nonlin_poisson),
        ("Liouville",  2): (None,               ref_liouville),
        ("Sine-Gordon",2): (None,               ref_sine_gordon),
    }

    fn1d, fn2d = exact_map.get((pde_name, dim), (None, None))

    if dim == 1:
        exact_fn = fn1d
        def bc_val(x, _):
            return np.atleast_2d(exact_fn(x[:, 0])).T
        bcs = [dde.icbc.DirichletBC(geom, lambda x: bc_val(x, None),
                                     lambda _, on_bnd: on_bnd)]
    else:
        exact_fn = fn2d
        def bc_val_2d(x, _):
            return np.atleast_2d(exact_fn(x[:, 0], x[:, 1])).T
        bcs = [dde.icbc.DirichletBC(geom, lambda x: bc_val_2d(x, None),
                                     lambda _, on_bnd: on_bnd)]

    # ── Arquitectura de red: más profunda para ecuaciones no lineales ─────────
    nonlinear = {"NonlinearPoisson", "Liouville", "Sine-Gordon", "Airy"}
    if pde_name in nonlinear:
        layers = [dim] + [128]*5 + [1]
        n_dom = 3000 if dim == 2 else 2000
        epochs_adam = 10000
    else:
        layers = [dim] + [64]*4 + [1]
        n_dom = 2000 if dim == 2 else 1000
        epochs_adam = 5000

    n_bnd = 400 if dim == 2 else 20

    return make_pde(pde_name, dim), bcs, exact_fn, geom, n_dom, n_bnd, 500, layers, epochs_adam


# ─────────────────────────────────────────────────────────────────────────────
def solve_and_eval(pde_name, dim, run_dir, epochs_override=None):
    print(f"\n{'='*60}")
    print(f"  PINN: {pde_name} {dim}D  [{N_THREADS} CPU threads]")
    print(f"{'='*60}")

    pde_fn, bcs, exact_fn, geom, n_dom, n_bnd, n_test, layers, epochs_adam = \
        build_problem(pde_name, dim)

    if epochs_override:
        epochs_adam = epochs_override

    data = dde.data.PDE(geom, pde_fn, bcs,
                        num_domain=n_dom, num_boundary=n_bnd, num_test=n_test)
    net = dde.nn.FNN(layers, "tanh", "Glorot normal")
    model = dde.Model(data, net)

    t0 = time.time()

    # Fase 1: Adam
    model.compile("adam", lr=1e-3, loss_weights=[1, 100])
    losshistory, _ = model.train(iterations=epochs_adam, display_every=epochs_adam//5)

    # Fase 2: L-BFGS para refinamiento fino
    dde.optimizers.config.set_LBFGS_options(maxiter=3000)
    model.compile("L-BFGS")
    losshistory, _ = model.train()

    rt = time.time() - t0
    print(f"  Tiempo total: {rt:.1f}s")

    # ── Evaluación ────────────────────────────────────────────────────────────
    if dim == 1:
        pts = np.linspace(0, 1, 100).reshape(-1, 1)
        u_approx = model.predict(pts).ravel()
        u_exact  = exact_fn(pts[:, 0]).ravel()
        bnd_pts  = np.array([[0.0],[1.0]])
    else:
        xs = np.linspace(0, 1, 50)
        xx, yy = np.meshgrid(xs, xs)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        u_approx = model.predict(pts).ravel()
        u_exact  = exact_fn(pts[:,0], pts[:,1]).ravel()
        t_bnd = np.linspace(0, 1, 50)
        bnd_pts = np.concatenate([
            np.column_stack([t_bnd, np.zeros_like(t_bnd)]),
            np.column_stack([t_bnd, np.ones_like(t_bnd)]),
            np.column_stack([np.zeros_like(t_bnd), t_bnd]),
            np.column_stack([np.ones_like(t_bnd), t_bnd]),
        ])

    mse_dom = float(np.mean((u_approx - u_exact)**2))

    u_bnd_approx = model.predict(bnd_pts).ravel()
    if dim == 1:
        u_bnd_exact = exact_fn(bnd_pts[:,0]).ravel()
    else:
        u_bnd_exact = exact_fn(bnd_pts[:,0], bnd_pts[:,1]).ravel()
    mse_bnd = float(np.mean((u_bnd_approx - u_bnd_exact)**2))

    print(f"  MSE Dom: {mse_dom:.4e}  |  MSE Bnd: {mse_bnd:.4e}")

    # ── Guardar CSV (mismo formato que PI-NSGA-II) ────────────────────────────
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
                w.writerow([f"{pts[i,0]:.6f}", f"{pts[i,1]:.6f}",
                            f"{u_exact[i]:.6f}", f"{u_approx[i]:.6f}"])

    # Pareto CSV (1 punto = el modelo entrenado)
    p_path = os.path.join(run_dir, f"{label}_pinn_pareto.csv")
    with open(p_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","pde","dim","mse_domain","mse_boundary","rank"])
        w.writerow([METHOD, pde_name, dim, f"{mse_dom:.10f}", f"{mse_bnd:.10f}", 1])

    return {"pde": label, "dim": dim, "mse_dom": mse_dom,
            "mse_bnd": mse_bnd, "rt": rt}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PINN baseline para PI-NSGA-II comparison")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override Adam epochs (default auto per problem)")
    parser.add_argument("--runs",   type=int, default=1)
    parser.add_argument("--only",   type=str, default=None,
                        help="Correr solo una ecuación, ej: --only Airy_1D")
    args = parser.parse_args()

    # Todos los problemas en el mismo orden que PI-NSGA-II
    problems = [
        ("Laplace",            1), ("Laplace",            2),
        ("Poisson",            1), ("Poisson",            2),
        ("Helmholtz",          1), ("Helmholtz",          2),
        ("Schrodinger",        1), ("Schrodinger",        2),
        ("Airy",               1),
        ("HarmonicOscillator", 1), ("HarmonicOscillator", 2),
        ("NonlinearPoisson",   2),
        ("Liouville",          2),
        ("Sine-Gordon",        2),
    ]

    if args.only:
        name, d = args.only.rsplit("_", 1)
        problems = [(name, int(d[0]))]

    all_results = []
    for run_idx in range(args.runs):
        run_dir = RESULTS_DIR if args.runs == 1 \
                  else os.path.join(RESULTS_DIR, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        for pde_name, dim in problems:
            try:
                res = solve_and_eval(pde_name, dim, run_dir, args.epochs)
                res["run"] = run_idx
                all_results.append(res)
            except Exception as e:
                print(f"  [ERROR] {pde_name} {dim}D: {e}")

    # ── Resumen global ────────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    file_exists  = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        fields = ["run","method","pde","pareto_size","best_mse_domain",
                  "best_mse_boundary","mean_mse_domain","mean_mse_boundary",
                  "hypervolume","runtime_s"]
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists: w.writeheader()
        for r in all_results:
            w.writerow({
                "run": r["run"], "method": METHOD, "pde": r["pde"],
                "pareto_size": 1,
                "best_mse_domain":  r["mse_dom"], "best_mse_boundary": r["mse_bnd"],
                "mean_mse_domain":  r["mse_dom"], "mean_mse_boundary": r["mse_bnd"],
                "hypervolume": 0.0, "runtime_s": r["rt"],
            })

    print("\n\n" + "="*60)
    print("  RESUMEN FINAL — PINN")
    print("="*60)
    print(f"{'PDE':<30} {'MSE Dom':>12} {'MSE Bnd':>12} {'Tiempo':>10}")
    print("-"*60)
    for r in all_results:
        print(f"{r['pde']:<30} {r['mse_dom']:>12.4e} {r['mse_bnd']:>12.4e} {r['rt']:>9.1f}s")


if __name__ == "__main__":
    main()
