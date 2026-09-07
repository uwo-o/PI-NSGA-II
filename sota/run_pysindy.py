import numpy as np
import pandas as pd
import argparse
import os
import time
import pysindy as ps


def run_pysindy_benchmark(data_path, pde_name, run_id, noise_pct=0.0):
    """
    Corre PySINDy sobre el grid 1D (x, u_exact) de una EDO de frontera.
    PySINDy descubre sistemas dX/dt = f(X) a partir de series/trayectorias, no
    EDPs de frontera directamente — se reformula el BVP como sistema de primer
    orden en x: estado [u, v=du/dx], y se le pide a SINDy que descubra
    dv/dx = f(u,v,x) (la ley que define la EDO original u''=f(u,u',x)).
    Solo 1D: en 2D no hay una nocion natural de "trayectoria" en la que
    aplicar SINDy sin reformular como sistema dinamico espacio-temporal, que
    esta fuera de alcance aca.
    """
    print(f"--- Running PySINDy Benchmark for {pde_name} ---")
    df = pd.read_csv(data_path)
    cols = list(df.columns)

    if "y" in cols:
        print(f"[SKIP] {pde_name}: PySINDy solo soporta 1D en esta implementacion "
              "(reformulacion del BVP como sistema u'=v, v'=f(u,v,x)); "
              "no aplica limpiamente a EDPs 2D.")
        return

    x_col = "x" if "x" in cols else "t"
    df = df.sort_values(by=x_col).reset_index(drop=True)
    x = df[x_col].values
    u_true = df["u_exact"].values if "u_exact" in cols else df["u"].values

    if len(x) < 5 or not np.all(np.isfinite(u_true)):
        print(f"[SKIP] {pde_name}: datos insuficientes o no finitos para PySINDy.")
        return

    # Ruido gaussiano relativo al std de la señal, simulando mediciones reales
    # con error (ver mismo criterio en run_pysr.py). Para PySINDy esto es
    # particularmente exigente: v=du/dx y u''=d²u/dx² se obtienen por
    # diferencias finitas SOBRE la señal ruidosa, y diferenciar amplifica el
    # ruido de alta frecuencia — es precisamente el modo de fallo real que
    # este experimento busca exponer (a diferencia de PISR-NSGA-II, que nunca
    # diferencia datos medidos: sus derivadas son AD exacto sobre el arbol
    # candidato, no sobre una señal ruidosa). El SCORING final sigue siendo
    # contra u_true limpia.
    if noise_pct > 0.0:
        rng = np.random.RandomState(2000 + run_id)
        sigma = noise_pct * np.std(u_true)
        u = u_true + rng.normal(0.0, sigma, size=u_true.shape)
    else:
        u = u_true

    dt = float(np.median(np.diff(x)))
    diff_method = ps.FiniteDifference(order=2)
    v = diff_method(u, t=dt)  # du/dx
    # d(v)/dx calculado DIRECTO de u (formula central de 2da derivada), no
    # rediferenciando v — v ya viene de una diferencia finita, y dejar que
    # SINDy la vuelva a diferenciar duplica el error numerico y hacia que el
    # ajuste divergiera con coeficientes absurdos (~100+). Pasando x_dot
    # explicito se evita esa doble diferenciacion.
    u_pp = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dt * dt)
    u_pp[0] = u_pp[1]; u_pp[-1] = u_pp[-2]  # bordes: la formula central no aplica
    X = np.column_stack([u, v])
    X_dot = np.column_stack([v, u_pp])

    # degree=1 (no cuadraticos/cubicos): para varios de estos problemas u y
    # v=du/dx quedan casi colineales (ej. Airy: u=e^-x, v=-u), lo que vuelve
    # el ajuste con terminos cuadraticos numericamente inestable (coeficientes
    # que se cancelan en pares ~100 y -100 sin converger a nada sparso) —
    # grado 1 evita esa ambiguedad al costo de no poder representar EDOs
    # genuinamente no lineales en u (esa es una limitacion real y honesta del
    # metodo aca, no algo a "arreglar" con mas hiperparametros).
    start_t = time.time()
    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=1),
        optimizer=ps.STLSQ(threshold=0.1),
    )
    model.fit(X, t=dt, x_dot=X_dot, feature_names=["u", "v"])
    runtime = time.time() - start_t

    eqs = model.equations(precision=4)
    # eqs[0] = du/dx = v (trivial, por construccion del sistema);
    # eqs[1] = dv/dx = f(u,v) es la ley de verdad descubierta.
    rhs = eqs[1] if len(eqs) > 1 else eqs[0]

    # Se imprime SOLO la expresion en la linea siguiente (sin "u''=" ni
    # ninguna otra etiqueta) — generate_formulas_table.py lee literalmente
    # la linea de abajo de "Best Discovered Equation..." y la intenta
    # parsear con sympy, igual que hace con el log de PySR.
    print(f"\nBest Discovered Equation by PySINDy for {pde_name}:")
    print(rhs)
    print("--------------------------------------------------\n")

    # Reconstruye la trayectoria integrando el modelo descubierto (shooting
    # desde el mismo punto inicial que la verdad) para poder comparar MSE y
    # graficar junto a PISR-NSGA-II/PySR.
    try:
        X_sim = model.simulate(X[0], t=x)
        u_approx = X_sim[:, 0]
    except Exception as e:
        print(f"[WARN] No se pudo simular la trayectoria descubierta: {e}")
        u_approx = np.full_like(u, np.nan)

    # Scoring SIEMPRE contra u_true (limpia), aunque el ajuste haya visto u
    # ruidosa — mismo criterio que run_pysr.py.
    valid = np.isfinite(u_approx)
    mse = float(np.mean((u_true[valid] - u_approx[valid]) ** 2)) if valid.any() else float("nan")

    dim = 1
    suffix = f"_{dim}D"
    out_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(out_dir, "..", "results", f"run_{run_id}")
    os.makedirs(res_dir, exist_ok=True)

    noise_suffix = f"_noise{int(round(noise_pct * 100))}pct" if noise_pct > 0.0 else ""

    df_out = pd.DataFrame({x_col: x, "u_exact": u_true, "u_approx": u_approx})
    df_out.to_csv(os.path.join(res_dir, f"grid_{pde_name}{suffix}_PySINDy{noise_suffix}.csv"), index=False)

    pd.DataFrame([{
        "method": "PySINDy",
        "pde": f"{pde_name}{suffix}",
        "noise_pct": noise_pct,
        "best_mse_domain": mse,
        "best_mse_boundary": 0.0,
        "mse_total": mse,
        "runtime_s": runtime,
    }]).to_csv(os.path.join(res_dir, f"{pde_name}{suffix}_sindy_metrics{noise_suffix}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySINDy Baseline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to numerical simulation data")
    parser.add_argument("--problem", type=str, required=True, help="Name of the PDE benchmark")
    parser.add_argument("--run_id", type=int, default=1, help="Run ID for multiple runs")
    parser.add_argument("--noise_pct", type=float, default=0.0,
                         help="Ruido gaussiano relativo (0.05 = 5%% del std de la señal) agregado a los datos de entrenamiento. 0 = datos limpios (default).")
    args = parser.parse_args()

    run_pysindy_benchmark(args.dataset, args.problem, args.run_id, args.noise_pct)
