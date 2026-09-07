import numpy as np
from pysr import PySRRegressor
import argparse

def run_pysr_benchmark(data_path, pde_name, run_id, noise_pct=0.0):
    """
    Runs PySR on the dataset to discover the analytical solution u(x,t).

    noise_pct: fraccion (0.05 = 5%) del std(y) de ruido gaussiano agregado a
    los datos de ENTRENAMIENTO — simula que en un ambiente real los datos
    etiquetados que PySR necesita vienen de mediciones/sensores con error, no
    perfectos como los generabamos antes (le "servimos en bandeja" datos
    limpios que PISR-NSGA-II nunca recibe, ya que no usa datos etiquetados en
    absoluto). El SCORING sigue siendo contra la verdad LIMPIA (y_true), no
    contra los datos ruidosos que vio el modelo — asi el MSE reportado mide
    genuinamente cuanto degrado la recuperacion de la fisica real, no el
    ajuste al ruido.
    """
    print(f"--- Running PySR Benchmark for {pde_name} (noise_pct={noise_pct}) ---")

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
        y_true = df['u_exact'].values
    elif 'u' in df.columns:
        y_true = df['u'].values
    else:
        raise ValueError(f"No target column found in {data_path}")

    # Ruido gaussiano relativo al std de la señal (para que "5%" signifique lo
    # mismo en una EDP con rango [0,1] que en una con rango [0,1000]). Semilla
    # fija (derivada de run_id) para que la misma corrida sea reproducible.
    if noise_pct > 0.0:
        rng = np.random.RandomState(1000 + run_id)
        sigma = noise_pct * np.std(y_true)
        y_noisy = y_true + rng.normal(0.0, sigma, size=y_true.shape)
    else:
        y_noisy = y_true

    # Subsample data for PySR training if too large (SINDy uses full grid, SR is too slow)
    train_size = 5000
    if X.shape[0] > train_size:
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], train_size, replace=False)
        X_train = X[idx]
        y_train = y_noisy[idx]
    else:
        X_train = X
        y_train = y_noisy

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

    # 3. Predict and save — scoring SIEMPRE contra y_true (limpia), aunque el
    # entrenamiento haya visto y_noisy. Esto mide cuanto degrada la
    # recuperacion de la fisica real, no el ajuste al ruido.
    u_approx = model.predict(X)
    mse = np.mean((y_true - u_approx)**2)

    print(f"\nBest Discovered Equation by PySR for {pde_name}:")
    print(model.sympy())
    print("--------------------------------------------------\n")

    # Save Grid
    import os
    dim = len(var_names)
    suffix = f"_{dim}D"
    df['u_exact'] = y_true
    df['u_approx'] = u_approx

    out_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(out_dir, "..", "results", f"run_{run_id}")
    os.makedirs(res_dir, exist_ok=True)

    # Sufijo de ruido en el nombre de archivo (solo si >0) para que corridas
    # con distintos noise_pct no se pisen entre si — sin sufijo cuando
    # noise_pct=0.0 para no romper compatibilidad con nombres/analisis viejos.
    noise_suffix = f"_noise{int(round(noise_pct * 100))}pct" if noise_pct > 0.0 else ""

    grid_path = os.path.join(res_dir, f"grid_{pde_name}{suffix}_PySR{noise_suffix}.csv")
    df.to_csv(grid_path, index=False)

    # Save Metrics
    metrics_path = os.path.join(res_dir, f"{pde_name}{suffix}_pysr_metrics{noise_suffix}.csv")
    pd.DataFrame([{
        "method": "PySR",
        "pde": f"{pde_name}{suffix}",
        "noise_pct": noise_pct,
        "best_mse_domain": mse,
        "best_mse_boundary": 0.0,
        "mse_total": mse,
        "runtime_s": runtime
    }]).to_csv(metrics_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PySR Baseline')
    parser.add_argument('--dataset', type=str, default='mock_data.csv', help='Path to numerical simulation data')
    parser.add_argument('--problem', type=str, default='Heat', help='Name of the PDE benchmark')
    parser.add_argument('--run_id', type=int, default=1, help='Run ID for multiple runs')
    parser.add_argument('--noise_pct', type=float, default=0.0,
                         help='Ruido gaussiano relativo (0.05 = 5%% del std de la señal) agregado a los datos de entrenamiento, simulando mediciones reales con error. 0 = datos limpios (default, comportamiento original).')
    args = parser.parse_args()

    run_pysr_benchmark(args.dataset, args.problem, args.run_id, args.noise_pct)
