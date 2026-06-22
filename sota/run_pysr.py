import numpy as np
from pysr import PySRRegressor
import argparse

def run_pysr_benchmark(data_path, pde_name):
    """
    Runs PySR on the dataset to discover the analytical solution u(x,t).
    """
    print(f"--- Running PySR Benchmark for {pde_name} ---")
    
    import pandas as pd
    df = pd.read_csv(data_path)
    
    # Check if 1D (ODE) or 2D (PDE)
    cols = list(df.columns)
    feature_cols = [c for c in ['x', 'y', 't'] if c in cols]
    
    if not feature_cols:
        raise ValueError(f"Unknown format in dataset {data_path}")
        
    X = df[feature_cols].values
    var_names = feature_cols
    
    if 'u_exact' in df.columns:
        y = df['u_exact'].values
    elif 'u' in df.columns:
        y = df['u'].values
    else:
        raise ValueError(f"No target column found in {data_path}")

    # Subsample data for PySR training if too large (SINDy uses full grid, SR is too slow)
    train_size = 5000
    if X.shape[0] > train_size:
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], train_size, replace=False)
        X_train = X[idx]
        y_train = y[idx]
    else:
        X_train = X
        y_train = y

    # 1. Define the Regressor
    model = PySRRegressor(
        niterations=100,  # equivalent to generations
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[
            "cos",
            "sin",
            "exp",
            "inv(x) = 1/x",
            "tanh"
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        model_selection="best",
        elementwise_loss="loss(prediction, target) = (prediction - target)^2", 
        populations=15,
        population_size=33,
        maxsize=20, # Equivalent to our tree max depth/size
        random_state=42,
        progress=False
    )

    # 2. Fit the model
    import time
    start_t = time.time()
    print("Starting PySR Evolution...")
    model.fit(X_train, y_train, variable_names=var_names)
    runtime = time.time() - start_t

    # 3. Predict and save
    u_approx = model.predict(X)
    mse = np.mean((y - u_approx)**2)
    
    print(f"\nBest Discovered Equation by PySR for {pde_name}:")
    print(model.sympy())
    print("--------------------------------------------------\n")
    
    # Save Grid
    import os
    dim = len(var_names)
    suffix = f"_{dim}D"
    df['u_exact'] = y
    df['u_approx'] = u_approx
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(out_dir, "..", "results")
    os.makedirs(res_dir, exist_ok=True)
    
    grid_path = os.path.join(res_dir, f"grid_{pde_name}{suffix}_PySR.csv")
    df.to_csv(grid_path, index=False)
    
    # Save Metrics
    metrics_path = os.path.join(res_dir, f"{pde_name}{suffix}_pysr_metrics.csv")
    pd.DataFrame([{
        "method": "PySR",
        "pde": f"{pde_name}{suffix}",
        "best_mse_domain": mse,
        "best_mse_boundary": 0.0,
        "mse_total": mse,
        "runtime_s": runtime
    }]).to_csv(metrics_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PySR Baseline')
    parser.add_argument('--dataset', type=str, default='mock_data.csv', help='Path to numerical simulation data')
    parser.add_argument('--problem', type=str, default='Heat', help='Name of the PDE benchmark')
    args = parser.parse_args()
    
    run_pysr_benchmark(args.dataset, args.problem)
