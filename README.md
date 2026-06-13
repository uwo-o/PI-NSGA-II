# PISR-NSGA-II: Physics-Informed Multi-Objective Symbolic Regression

PISR-NSGA-II is a technical framework designed for the symbolic discovery of Partial Differential Equations (PDEs) and Ordinary Differential Equations (ODEs). It combines an exact Automatic Differentiation (AD) engine with a Pure Memetic NSGA-II optimization core to discover interpretable mathematical models that govern complex physical phenomena.

## Core Optimization Engine

### Pure Memetic NSGA-II
The framework implements a canonical NSGA-II (Deb et al., 2002) as its primary multi-objective search algorithm, optimized for three competitive objectives:
1.  **Domain Residual ($\mathcal{L}_{dom}$):** Exact PDE residual computed via second-order AD.
2.  **Boundary Error ($\mathcal{L}_{bc}$):** Mean Squared Error across all Dirichlet and Initial Conditions.
3.  **Structural Complexity:** Total node count of the symbolic expression tree (anti-bloat measure).

### Constraint Handling (Deb 2002)
Instead of arbitrary penalty functions, the engine enforces physical and mathematical consistency through strict constraint-domination rules:
*   **Non-Triviality:** Expressions lacking independent variables are flagged as infeasible.
*   **Dimensional Consistency:** Buckingham Pi theorem logic is enforced during tree construction and verified during evaluation.
*   **Numerical Stability:** Solutions generating NaNs or Inf during AD evaluation are categorically dominated by feasible candidates.

## Mathematical Architecture

### High-Fidelity AD Engine
The engine performs exact differentiation using recursive chain rules across expression trees. It supports second-order derivatives and complex-valued arithmetic, allowing the algorithm to navigate through the complex plane to resolve difficult physical topologies before projecting back to real solutions.

### Symbolic Basis and Library
The search space is augmented with physical motifs rather than simple arithmetic operators:
*   **Orthogonal Polynomials:** Legendre, Hermite, Chebyshev, and Laguerre bases.
*   **Infinite Series Operators:** Native `SERIES` node for Fourier (spectral) and Frobenius (singular) expansions.
*   **Complex Plane Explorers:** Quantum phase chirps and topological vortices.
*   **Extreme Regimes:** Specialized templates for boundary layers (Troesch) and fractional power laws (Thomas-Fermi).

### Physics-Guided Initialization (Priors)
Before evolution starts, the engine probes the target PDE to detect innate symmetries and properties:
*   **Lie Symmetries:** Detection of translational invariance to favor wave-packet structures.
*   **Scale Invariance:** Homogeneity analysis to identify self-similar solution manifolds.
*   **Differential ADN:** Detection of the maximum derivative order to filter out topologically insufficient trees.

## Optimization Pipeline

1.  **Structural Search:** The Memetic NSGA-II explores the symbolic topology while performing stochastic Hill-Climbing for coefficient optimization.
2.  **Committee RAR:** Residual-based Adaptive Refinement where a committee of elite and random individuals identifies high-uncertainty regions for resampling.
3.  **High-Precision Polishing:** Post-evolution stage using multi-stage coordinate descent to refine discovered physical constants to machine precision.

## CLI Parameters

The `PISR-NSGA-II` binary provides granular control over the physics-informed search:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--only <NAME>` | Target a specific PDE (e.g., `ThomasFermi_2D`, `Navier-Stokes_2D`). | All |
| `--pop <N>` | Population size for the structural search. | 300 |
| `--gen <N>` | Maximum number of generations. | 300 |
| `--domain <N>` | Number of collocation points for the domain residual. | 2000 |
| `--boundary <N>` | Number of points for boundary and initial conditions. | 500 |
| `--depth <N>` | Maximum symbolic tree depth. | 8 |
| `--sigma <F>` | Standard deviation for Gaussian ERC mutation. | 0.20 |
| `--stop <F>` | Global convergence threshold (MSE). | 1e-7 |
| `--cores <N>` | Number of CPU threads (0 = Auto-detect). | 1 |
| `--runs <N>` | Number of independent stochastic executions for statistical analysis. | 1 |
| `--test` | Debug mode with reduced population and generations. | Off |

## Execution

### Compilation
Requires a C++17 compliant compiler and CMake 3.10+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

### Full Pipeline
The pipeline executes the symbolic discovery, neural baselines (DeepXDE), statistical comparison, and LaTeX report generation:

```bash
./run_pipeline.sh
```

The comprehensive analysis is output to `report/results.pdf`.
