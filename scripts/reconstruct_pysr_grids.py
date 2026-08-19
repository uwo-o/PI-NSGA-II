import pandas as pd
import numpy as np
import sympy as sp
import os
import re

formulas = {
    "Airy_1D": "exp(-x)",
    "Airy_2D": "exp(-x - y)",
    "Fisher_1D": "0.49999997*tanh(x*0.49999997) + 0.49999997",
    "Fisher_2D": "sin((x + y + 1.652747)*0.34147164)*0.9288274",
    "Duffing_1D": "exp(x*x*(-0.03650881*x*(-x - 1.0555391) - 0.50860727))",
    "Duffing_2D": "sin(exp(-(y - (-1)*0.18893717/exp(x + y*1.9665935)) + cos(x + 0.8701715)))",
    "Thomas-Fermi_1D": "1/(x + 0.50000006)",
    "Thomas-Fermi_2D": "1/(x + y + 0.4999997)",
    "Lane-Emden_1D": "0.99999976 - 0.16666669*x*x",
    "Troesch_1D": "x*sin(-0.43192714 + 0.63581115/cos(x))*1.4746897",
    "Ginzburg-Landau_1D": "tanh(x)",
    "Painleve-I_1D": "x*0.5*x"
}

results_dir = "results"
for prob, eq_str in formulas.items():
    pisr_file = os.path.join(results_dir, f"grid_{prob}_PISR-EMOAD.csv")
    if not os.path.exists(pisr_file):
        continue
        
    df = pd.read_csv(pisr_file)
    cols = list(df.columns)
    feature_cols = [c for c in ['x', 'y', 't'] if c in cols]
    
    # Create sympy expression
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    transformations = (standard_transformations + (implicit_multiplication_application,))
    
    try:
        expr = parse_expr(eq_str, transformations=transformations)
        
        # Evaluate for all rows
        u_approx = []
        for _, row in df.iterrows():
            subs = {sp.Symbol(c): row[c] for c in feature_cols}
            val = float(expr.evalf(subs=subs))
            u_approx.append(val)
            
        df_new = pd.DataFrame()
        for c in feature_cols:
            df_new[c] = df[c]
        df_new['u_exact'] = df['u_exact']
        df_new['u_approx'] = u_approx
        
        # Save PySR grid
        out_file = os.path.join(results_dir, f"grid_{prob}_PySR.csv")
        df_new.to_csv(out_file, index=False)
        print(f"Reconstructed {out_file} using equation: {eq_str}")
        
    except Exception as e:
        print(f"Failed to evaluate {prob}: {e}")

print("Done generating corrected PySR grids.")
