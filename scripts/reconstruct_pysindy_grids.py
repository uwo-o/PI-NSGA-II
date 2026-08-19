import pandas as pd
import numpy as np
import sympy as sp
import os
from scipy.integrate import solve_ivp

# Formulas extracted from the LaTeX output
formulas = {
    "Airy_1D": "-1.0 * u",
    "Fisher_1D": "0.282 - 0.207 * u**3",
    "Duffing_1D": "28.937 * u**3 - 65.692 * u**2 + 49.689 * u - 13.008",
    "Thomas-Fermi_1D": "-1.0 * u**2",
    "Lane-Emden_1D": "104.841 * u**3 - 280.689 * u**2 + 251.484 * u - 75.664",
    "Troesch_1D": "-1.312 * u**3 + 2.352 * u**2 + 1.688 * u + 0.253",
    "Ginzburg-Landau_1D": "1.0 - 0.999 * u**2",
    "Painleve-I_1D": "11.647 * u**3 - 11.277 * u**2 + 4.629 * u + 0.084"
}

results_dir = "results"
u_sym = sp.Symbol('u')

for prob, eq_str in formulas.items():
    pisr_file = os.path.join(results_dir, f"grid_{prob}_PISR-EMOAD.csv")
    if not os.path.exists(pisr_file):
        continue
        
    df = pd.read_csv(pisr_file)
    if 'x' not in df.columns:
        # Some might use 't' as the spatial domain for SINDy in 1D? Usually 'x'.
        x_var = 't' if 't' in df.columns else None
    else:
        x_var = 'x'
        
    if not x_var: continue
        
    # Sort just in case
    df = df.sort_values(by=[x_var])
    x_vals = df[x_var].values
    
    # Parse equation
    try:
        expr = sp.sympify(eq_str)
        # Create a lambdified function for RHS
        f_rhs = sp.lambdify(u_sym, expr, "numpy")
        
        def ode_func(t, y):
            # y is a 1D array [u]
            return [f_rhs(y[0])]
            
        u0 = df['u_exact'].iloc[0]
        x_span = (x_vals[0], x_vals[-1])
        
        sol = solve_ivp(ode_func, x_span, [u0], t_eval=x_vals, method='RK45')
        
        if sol.success:
            u_approx = sol.y[0]
        else:
            print(f"Integration failed for {prob}")
            u_approx = np.zeros_like(x_vals) # fallback
            
        df_new = pd.DataFrame()
        df_new[x_var] = x_vals
        df_new['u_exact'] = df['u_exact'].values
        df_new['u_approx'] = u_approx
        
        out_file = os.path.join(results_dir, f"grid_{prob}_PySINDy.csv")
        df_new.to_csv(out_file, index=False)
        print(f"Reconstructed {out_file}")
        
    except Exception as e:
        print(f"Failed to evaluate {prob}: {e}")

print("Done generating corrected PySINDy grids.")
