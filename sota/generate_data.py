import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import argparse
import os

# --- ODE Systems ---
def airy_system(t, y): return [y[1], t * y[0]]
def painleve_system(t, y): return [y[1], 6 * y[0]**2 + t]
def duffing_system(t, y, delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.0):
    return [y[1], gamma*np.cos(omega*t) - delta*y[1] - alpha*y[0] - beta*y[0]**3]
def thomas_fermi_system(t, y):
    t = max(t, 1e-5) # Avoid division by zero
    return [y[1], (1.0 / np.sqrt(t)) * (y[0]**1.5) if y[0] > 0 else 0]
def lane_emden_system(t, y, n=1.5):
    t = max(t, 1e-5) # Avoid division by zero
    return [y[1], -(2.0 / t) * y[1] - (y[0]**n if y[0] > 0 else 0)]
def troesch_system(t, y, mu=0.5):
    return [y[1], mu * np.sinh(mu * y[0])]

# --- PDE Solvers (Simple FDM for data generation) ---
def simulate_fisher_pde(x_pts=100, t_pts=5000):
    x = np.linspace(-10, 10, x_pts)
    t = np.linspace(0, 5, t_pts)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    u = np.zeros((t_pts, x_pts))
    u[0, :] = np.exp(-(x**2)) # Initial condition
    r = 1.0; D = 0.5
    for n in range(0, t_pts-1):
        for i in range(1, x_pts-1):
            uxx = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2)
            u[n+1, i] = u[n, i] + dt * (D * uxx + r * u[n, i] * (1 - u[n, i]))
        u[n+1, 0] = u[n+1, 1]
        u[n+1, -1] = u[n+1, -2]
    X, T = np.meshgrid(x, t)
    return pd.DataFrame({'t': T.flatten(), 'x': X.flatten(), 'u': u.flatten()})

def simulate_ginzburg_landau_pde(x_pts=100, t_pts=5000):
    x = np.linspace(-10, 10, x_pts)
    t = np.linspace(0, 5, t_pts)
    dx = x[1] - x[0]; dt = t[1] - t[0]
    u = np.zeros((t_pts, x_pts), dtype=complex)
    u[0, :] = np.exp(-(x**2)) * np.exp(1j * x)
    for n in range(0, t_pts-1):
        for i in range(1, x_pts-1):
            uxx = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2)
            u[n+1, i] = u[n, i] + dt * (uxx + u[n, i] - u[n, i] * np.abs(u[n, i])**2)
    X, T = np.meshgrid(x, t)
    # Return real part for simplicity in SR benchmarks
    return pd.DataFrame({'t': T.flatten(), 'x': X.flatten(), 'u': np.real(u).flatten()})

def generate_data(problem_name, output_path):
    print(f"Generating data for {problem_name}...")
    num_points = 1000
    
    # Check if PDE
    if problem_name == 'fisher': df = simulate_fisher_pde(); df.to_csv(output_path, index=False); return
    if problem_name == 'ginzburg_landau': df = simulate_ginzburg_landau_pde(); df.to_csv(output_path, index=False); return
    
    # ODE Setup
    t_span = [0.1, 5]
    y0 = [1.0, 0.0] # default
    if problem_name == 'airy': t_span = [0, 10]; y0 = [0.355028, -0.258819]; func = airy_system
    elif problem_name == 'painleve': t_span = [0, 1.2]; y0 = [0.0, 1.0]; func = painleve_system
    elif problem_name == 'duffing': t_span = [0, 50]; y0 = [1.0, 0.0]; func = duffing_system
    elif problem_name == 'thomas_fermi': t_span = [1e-3, 5]; y0 = [1.0, -1.0]; func = thomas_fermi_system
    elif problem_name == 'lane_emden': t_span = [1e-3, 5]; y0 = [1.0, 0.0]; func = lane_emden_system
    elif problem_name == 'troesch': t_span = [0, 1]; y0 = [0.0, 1.0]; func = troesch_system
    else: print(f"Skipping {problem_name} (Not implemented in Python numerical generator)."); return

    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(func, t_span, y0, method='Radau', t_eval=t_eval, rtol=1e-8, atol=1e-10)
    
    if not sol.success: print(f"Integration failed for {problem_name}"); return
    pd.DataFrame({'t': sol.t, 'u': sol.y[0]}).to_csv(output_path, index=False)
    print(f"Successfully generated {problem_name} data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--output', type=str, default='data.csv')
    args = parser.parse_args()
    generate_data(args.problem, os.path.join(os.path.dirname(__file__), args.output))
