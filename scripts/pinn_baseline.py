#!/usr/bin/env python3
"""
pinn_baseline.py  — PINN baseline con DeepXDE (PyTorch backend).

Resuelve los mismos problemas que PISR-NSGA-II y guarda CSVs compatibles.
Problemas:
  Analíticos 1D/2D: Laplace, Poisson, Helmholtz, Schrodinger
  Numéricos  1D:    Airy, HarmonicOscillator
  Numéricos  2D:    NonlinearPoisson, Liouville, Sine-Gordon, HarmonicOscillator

GPU: GTX 1050 Ti (sm_61) no compatible con PyTorch 2.x → usamos CPU multi-hilo.
     Si en el futuro se instala PyTorch ≤ 1.13 con CUDA sm_61, remover la línea
     de CUDA_VISIBLE_DEVICES y DeepXDE usará la GPU automáticamente.
"""

import os, sys, time, argparse, csv, warnings, subprocess
import numpy as np
warnings.filterwarnings("ignore")

# ── CUDA dynamic configuration ────────────────────────────────────────────────
use_cuda = False
try:
    # Probamos en un subproceso si CUDA realmente puede ejecutar tensores sin fallar (evita crashes por sm_61 en PyTorch 2.x)
    res = subprocess.run(
        [sys.executable, "-c", "import torch; torch.randn(1, device='cuda')"],
        capture_output=True, text=True, timeout=5
    )
    if res.returncode == 0:
        use_cuda = True
        print("[INFO] CUDA is functional. Running PINN on GPU.")
    else:
        print("[WARNING] CUDA is available but failed verification in subprocess (e.g. GPU compute capability mismatch).")
        print("[WARNING] Falling back to CPU.")
except Exception as e:
    print(f"[WARNING] Exception during CUDA subprocess check: {e}. Falling back to CPU.")

if not use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Backend ───────────────────────────────────────────────────────────────────
os.environ.setdefault("DDE_BACKEND", "pytorch")

import torch

# Usar todos los cores disponibles si se corre en CPU
N_THREADS = os.cpu_count() or 4
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(max(1, N_THREADS // 2))

import deepxde as dde
if use_cuda:
    dde.config.set_default_float("float32")
    print("[INFO] Using float32 precision for GPU training to optimize VRAM.")
else:
    dde.config.set_default_float("float64")
    print("[INFO] Using float64 precision for CPU training.")

_pi = np.pi
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
METHOD = "DeepXDE"

import torch.nn as nn

class SOTAPINNNet(nn.Module):
    def __init__(self, layers, activation="silu"):
        super().__init__()
        self.regularizer = None
        self.in_dim = layers[0]
        self.out_dim = layers[-1]
        self.hidden_dim = layers[1]
        self.num_layers = len(layers) - 2
        
        # Encoders
        self.encoder_u = nn.Linear(self.in_dim, self.hidden_dim)
        self.encoder_v = nn.Linear(self.in_dim, self.hidden_dim)
        
        # First hidden layer
        self.first_layer = nn.Linear(self.in_dim, self.hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers - 1)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)
        
        # Activation function
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sin":
            class Sin(nn.Module):
                def forward(self, x): return torch.sin(x)
            self.act = Sin()
        else:
            self.act = nn.SiLU()
            
        # Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # deepxde passes double/float64 tensors by default, ensure type matching
        x = x.to(self.encoder_u.weight.dtype)
        u = self.act(self.encoder_u(x))
        v = self.act(self.encoder_v(x))
        
        h = self.act(self.first_layer(x))
        for layer in self.hidden_layers:
            h_next = self.act(layer(h))
            h = (1.0 - h_next) * u + h_next * v
            
        return self.output_layer(h)

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
    def exact_schrodinger_1d(x):   return np.cos(_pi*x) 
    def exact_schrodinger_2d(x,y): return np.cos(_pi*(x+y))
    
    # Numéricos: función de referencia basada en la física del estado base
    def ref_airy_1d(x):          return np.exp(-0.5*x)
    def ref_airy_2d(x,y):        return np.exp(-0.5*(x+y))
    def ref_ho_1d(x):            return np.exp(-0.5*x**2)           # estado base ψ₀
    def ref_ho_2d(x,y):          return np.exp(-0.5*(x**2+y**2))
    def ref_nonlin_poisson(x,y): return 1.0/(1+x**2+y**2)          
    def ref_liouville(x,y):      return 1.0/(1+x**2+y**2)          
    def ref_sine_gordon(x,y):    return np.sin(_pi*x)*np.sin(_pi*y) 
    def ref_navier_stokes(x,y):
        Re = 20.0 # 1/0.05
        lam = Re / 2.0 - np.sqrt(Re**2 / 4.0 + 4.0 * _pi**2)
        return y - np.exp(lam * x) * np.sin(2.0 * _pi * y) / (2.0 * _pi * Re)
    def ref_fisher_1d(x):       return 1.0 / (1.0 + np.exp(-x))
    def ref_fisher_2d(x,y):     return 1.0 / (1.0 + np.exp(-(x + y)))
    def ref_duffing_1d(x):      return 1.0 / np.cosh(x)
    def ref_duffing_2d(x,y):    return 1.0 / np.cosh(x + y)
    def ref_tf_1d(x):           return 1.0 / (x + 0.5)
    def ref_tf_2d(x,y):         return 1.0 / (x + y + 0.5)
    def ref_bratu(x,y):         return np.log(2.0 / (np.cosh(x + y)**2))
    def ref_allen_cahn(x,y):
        eps = np.sqrt(0.01)
        return np.tanh((x + y - 1.0) / (eps * np.sqrt(2.0)))
    def ref_lane_emden(x):      return 1.0 - (x**2) / 6.0
    def ref_ns_unsteady(x,y):   return np.sin(_pi*x)*np.sin(_pi*y)
    def ref_troesch(x):         return np.sinh(3.0 * x) / np.sinh(3.0)
    def ref_ginzburg_landau(x): return np.tanh(x)
    def ref_painleve1(x):       return 0.5 * x**2

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
                # u'' = var*u
                if dim == 1:
                    return lap - x[:,0:1]*u
                else:
                    return lap - (x[:,0:1] + x[:,1:2])*u

            elif pde_name == "HarmonicOscillator":
                # -u'' + x²u = dim*u  →  u'' = (x² - dim)u
                if dim == 1:
                    V = x[:,0:1]**2
                    E = 1.0
                else:
                    V = x[:,0:1]**2 + x[:,1:2]**2
                    E = 2.0
                return lap - (V - E)*u

            elif pde_name == "NonlinearPoisson":
                # ∇²u + u² = f (con f calculado de la sol exacta)
                # Sol exacta: 1/(1+x²+y²) → ∇²u = 4(x²+y²-1)/(1+x²+y²)³
                # f = ∇²u + u² = [4(r²-1) + (1+r²)] / (1+r²)³ = (5r² - 3) / (1+r²)³
                r2 = x[:,0:1]**2 + (x[:,1:2]**2 if dim==2 else 0)
                f = (5*r2 - 3) / (1 + r2)**3
                return lap + u**2 - f

            elif pde_name == "Liouville":
                # ∇²u + e^u = f
                # f = ∇²u + exp(u_exact)
                r2 = x[:,0:1]**2 + (x[:,1:2]**2 if dim==2 else 0)
                u_exact = 1.0 / (1.0 + r2)
                f = 4*(r2 - 1)/(1+r2)**3 + torch.exp(u_exact)
                return lap + torch.exp(u) - f

            elif pde_name == "Sine-Gordon":
                # ∇²u - sin(u) = 0 (Match C++ pde_residual_ad)
                return lap - torch.sin(u)

            elif pde_name == "Fisher":
                # ∇²u + u*(1-u) = 0
                return lap + u * (1.0 - u)

            elif pde_name == "Duffing":
                # ∇²u + u + u³ = forcing
                if dim == 1:
                    forcing = torch.sin(_pi * x[:, 0:1])
                else:
                    forcing = torch.sin(_pi * x[:, 0:1]) * torch.sin(_pi * x[:, 1:2])
                return lap + u + u**3 - forcing

            elif pde_name == "Thomas-Fermi":
                # ∇²u = u^1.5 / sqrt(r)
                if dim == 1:
                    r = torch.abs(x[:, 0:1])
                else:
                    r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2)
                # Use abs(u) to prevent NaNs during training if u becomes negative
                return lap - torch.pow(torch.abs(u) + 1e-6, 1.5) / torch.sqrt(r + 1e-4)

            elif pde_name == "Navier-Stokes":
                psi_x = dde.grad.jacobian(u, x, i=0, j=0)
                psi_y = dde.grad.jacobian(u, x, i=0, j=1)
                psi_xx = dde.grad.hessian(u, x, i=0, j=0)
                psi_yy = dde.grad.hessian(u, x, i=1, j=1)
                L = psi_xx + psi_yy
                L_x = dde.grad.jacobian(L, x, i=0, j=0)
                L_y = dde.grad.jacobian(L, x, i=0, j=1)
                L_xx = dde.grad.hessian(L, x, i=0, j=0)
                L_yy = dde.grad.hessian(L, x, i=1, j=1)
                biharmonic = L_xx + L_yy
                nu = 0.05
                convective = psi_y * L_x - psi_x * L_y
                return convective - nu * biharmonic

            elif pde_name == "Navier-Stokes-Unsteady":
                psi_x = dde.grad.jacobian(u, x, i=0, j=0)
                psi_y = dde.grad.jacobian(u, x, i=0, j=1)
                psi_xx = dde.grad.hessian(u, x, i=0, j=0)
                psi_yy = dde.grad.hessian(u, x, i=1, j=1)
                L = psi_xx + psi_yy
                L_x = dde.grad.jacobian(L, x, i=0, j=0)
                L_y = dde.grad.jacobian(L, x, i=0, j=1)
                L_xx = dde.grad.hessian(L, x, i=0, j=0)
                L_yy = dde.grad.hessian(L, x, i=1, j=1)
                biharmonic = L_xx + L_yy
                nu = 0.01  # matches C++ k2=0.01
                convective = psi_y * L_x - psi_x * L_y
                return convective - nu * biharmonic

            elif pde_name == "Bratu":
                # ∇²u + 2 * exp(u) = 0
                return lap + 2.0 * torch.exp(u)

            elif pde_name == "Allen-Cahn":
                # 0.01 * ∇²u - (u³ - u) = 0
                return 0.01 * lap - (u**3 - u)

            elif pde_name == "Lane-Emden":
                # ∇²u + (2/x) * u' + u³ = 0
                u_x = dde.grad.jacobian(u, x, i=0, j=0)
                return lap + (2.0 / (x[:, 0:1] + 1e-6)) * u_x + u**3

            elif pde_name == "Troesch":
                # u'' = 2 * sinh(2u) (Match C++ pde_residual_ad)
                return lap - 2.0 * torch.sinh(2.0 * u)

            elif pde_name == "Ginzburg-Landau":
                # u'' + u - u³ = 0
                return lap + u - u**3

            elif pde_name == "Painleve-I":
                # u'' = u² + x + y
                if dim == 1:
                    var = x[:, 0:1]
                else:
                    var = x[:, 0:1] + x[:, 1:2]
                return lap - (u**2 + var)

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
        ("Airy",       2): (None,               ref_airy_2d),
        ("HarmonicOscillator",1): (ref_ho_1d,  None),
        ("HarmonicOscillator",2): (None,        ref_ho_2d),
        ("NonlinearPoisson",2):   (None,        ref_nonlin_poisson),
        ("Liouville",  2): (None,               ref_liouville),
        ("Sine-Gordon",2): (None,               ref_sine_gordon),
        ("Navier-Stokes",2): (None,             ref_navier_stokes),
        ("Navier-Stokes-Unsteady",2): (None,    ref_ns_unsteady),
        ("Bratu",      2): (None,               ref_bratu),
        ("Allen-Cahn", 2): (None,               ref_allen_cahn),
        ("Lane-Emden", 1): (ref_lane_emden,     None),
        ("Troesch",    1): (ref_troesch,        None),
        ("Ginzburg-Landau", 1): (ref_ginzburg_landau, None),
        ("Painleve-I", 1): (ref_painleve1,      None),
        ("Fisher",     1): (ref_fisher_1d,      None),
        ("Fisher",     2): (None,               ref_fisher_2d),
        ("Duffing",    1): (ref_duffing_1d,     None),
        ("Duffing",    2): (None,               ref_duffing_2d),
        ("Thomas-Fermi",1): (ref_tf_1d,          None),
        ("Thomas-Fermi",2): (None,               ref_tf_2d),
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
    nonlinear = {"NonlinearPoisson", "Liouville", "Sine-Gordon", "Airy", "Navier-Stokes", 
                 "Navier-Stokes-Unsteady", "Fisher", "Duffing", "Thomas-Fermi", 
                 "Bratu", "Allen-Cahn", "Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I"}
    
    # Memoria VRAM: GTX 1050 Ti tiene solo 4GB. 
    # Navier-Stokes con derivadas de 4to orden consume mucha memoria.
    if pde_name in nonlinear:
        layers = [dim] + [128]*5 + [1]
        if (pde_name == "Navier-Stokes" or pde_name == "Navier-Stokes-Unsteady") and use_cuda:
            n_dom = 1000  # Reduce memory usage on GPU for 4th-order derivatives
        else:
            n_dom = 3000 if dim == 2 else 2000
        epochs_adam = 10000
    else:
        layers = [dim] + [64]*4 + [1]
        n_dom = 2000 if dim == 2 else 1000
        epochs_adam = 5000

    n_bnd = 400 if dim == 2 else 20

    return make_pde(pde_name, dim), bcs, exact_fn, geom, n_dom, n_bnd, 500, layers, epochs_adam


# ─────────────────────────────────────────────────────────────────────────────
def load_exact_grid(pde_name, dim, run_dir):
    label = f"{pde_name}_{dim}D"
    for d in [run_dir, RESULTS_DIR]:
        path = os.path.join(d, f"grid_{label}_PISR-EMOAD.csv")
        if os.path.exists(path):
            try:
                import pandas as pd
                df = pd.read_csv(path)
                if dim == 1:
                    pts = df[["x"]].values
                else:
                    pts = df[["x", "y"]].values
                u_exact = df["u_exact"].values
                return pts, u_exact
            except Exception as e:
                print(f"[WARNING] Error reading {path}: {e}")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
def _solve_and_eval_inner(pde_name, dim, run_dir, epochs_override=None, is_test=False):
    print(f"\n{'='*60}")
    print(f"  PINN: {pde_name} {dim}D  [{N_THREADS} CPU threads]")
    print(f"{'='*60}")

    pde_fn, bcs, exact_fn, geom, n_dom, n_bnd, n_test, layers, epochs_adam = \
        build_problem(pde_name, dim)

    if epochs_override:
        epochs_adam = epochs_override

    if is_test:
        epochs_adam = 5

    data = dde.data.PDE(geom, pde_fn, bcs,
                        num_domain=n_dom, num_boundary=n_bnd, num_test=n_test)
    net = SOTAPINNNet(layers, activation="silu")
    model = dde.Model(data, net)

    t0 = time.time()

    # Fase 1: Adam con decaimiento de lr y resampler adaptativo (SOTA)
    resampler = dde.callbacks.PDEPointResampler(period=100)
    early_stop = dde.callbacks.EarlyStopping(min_delta=1e-9, patience=1000)
    
    model.compile("adam", lr=1e-3, decay=("step", 2000, 0.5), loss_weights=[1, 100])
    losshistory, _ = model.train(iterations=epochs_adam, 
                                 display_every=max(1, epochs_adam//10), 
                                 callbacks=[resampler, early_stop])

    # Fase 2: L-BFGS para refinamiento fino con pesos de pérdida sincronizados
    if not is_test:
        dde.optimizers.config.set_LBFGS_options(maxiter=3000)
        model.compile("L-BFGS", loss_weights=[1, 100])
        losshistory, _ = model.train()

    rt = time.time() - t0
    print(f"  Tiempo total: {rt:.1f}s")

    # ── Evaluación ────────────────────────────────────────────────────────────
    # Intentamos cargar la rejilla exacta del solver PISR-NSGA-II para comparación 1:1
    pts_exact, u_exact_grid = load_exact_grid(pde_name, dim, run_dir)

    if pts_exact is not None:
        pts = pts_exact
        u_approx = model.predict(pts).ravel()
        u_exact = u_exact_grid
    else:
        # Fallback si no existe la rejilla en disco
        if dim == 1:
            pts = np.linspace(0, 1, 101).reshape(-1, 1)
            u_approx = model.predict(pts).ravel()
            u_exact  = exact_fn(pts[:, 0]).ravel()
        else:
            xs = np.linspace(0, 1, 51)
            xx, yy = np.meshgrid(xs, xs)
            pts = np.column_stack([xx.ravel(), yy.ravel()])
            u_approx = model.predict(pts).ravel()
            u_exact  = exact_fn(pts[:,0], pts[:,1]).ravel()

    if dim == 1:
        bnd_pts  = np.array([[0.0],[1.0]])
    else:
        t_bnd = np.linspace(0, 1, 51)
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

    # ── Guardar CSV (mismo formato que PISR-NSGA-II) ────────────────────────────
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
        w.writerow(["method","pde","dim","mse_domain","mse_boundary","runtime_s","rank"])
        w.writerow([METHOD, pde_name, dim, f"{mse_dom:.10f}", f"{mse_bnd:.10f}", f"{rt:.4f}", 1])

    return {"pde": label, "dim": dim, "mse_dom": mse_dom,
            "mse_bnd": mse_bnd, "rt": rt}


def solve_and_eval(pde_name, dim, run_dir, epochs_override=None, is_test=False):
    try:
        return _solve_and_eval_inner(pde_name, dim, run_dir, epochs_override, is_test)
    except Exception as e:
        err_msg = str(e)
        if "out of memory" in err_msg.lower() and torch.cuda.is_available() and torch.get_default_device().type == "cuda":
            print(f"\n[WARNING] CUDA Out Of Memory detected during {pde_name} {dim}D. Falling back to CPU for this problem.")
            
            # Clean up GPU memory
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Temporarily switch default device to CPU
            orig_device = torch.get_default_device()
            torch.set_default_device("cpu")
            try:
                res = _solve_and_eval_inner(pde_name, dim, run_dir, epochs_override, is_test)
                # Restore original device after successful CPU run
                torch.set_default_device(orig_device)
                return res
            except Exception as cpu_e:
                # Restore original device even if CPU run fails
                torch.set_default_device(orig_device)
                raise cpu_e
        else:
            raise e


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DeepXDE baseline para PISR-NSGA-II comparison")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override Adam epochs (default auto per problem)")
    parser.add_argument("--runs",   type=int, default=1)
    parser.add_argument("--only",   type=str, default=None,
                        help="Correr solo una ecuación, ej: --only Airy_1D")
    parser.add_argument("--test",   action="store_true", help="Fast test mode")
    args = parser.parse_args()

    # The Hardcore Physics Benchmark (Matches C++ run_once selection)
    problems = [
        ("Airy",               1), ("Airy",               2),
        ("Fisher",             1), ("Fisher",             2),
        ("Duffing",            1), ("Duffing",            2),
        ("Thomas-Fermi",        1), ("Thomas-Fermi",        2),
        ("Navier-Stokes",      2),
        ("Navier-Stokes-Unsteady", 2),
        ("Lane-Emden",         1),
        ("Troesch",            1),
        ("Ginzburg-Landau",    1),
        ("Painleve-I",         1),
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
                res = solve_and_eval(pde_name, dim, run_dir, args.epochs, is_test=args.test)
                res["run"] = run_idx
                all_results.append(res)
                
                # Forzar limpieza de memoria después de cada problema
                import gc
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()
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
    print("  RESUMEN FINAL — DeepXDE")
    print("="*60)
    print(f"{'PDE':<30} {'MSE Dom':>12} {'MSE Bnd':>12} {'Tiempo':>10}")
    print("-"*60)
    for r in all_results:
        print(f"{r['pde']:<30} {r['mse_dom']:>12.4e} {r['mse_bnd']:>12.4e} {r['rt']:>9.1f}s")


if __name__ == "__main__":
    main()
