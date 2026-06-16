#pragma once
// =============================================================================
// pde_problems.hpp  —  Definición de Laplace, Poisson y Helmholtz
//   Dominio Ω = [0,1]², condiciones de Dirichlet.
// =============================================================================

#include "common.hpp"
#include "dimensions.hpp"
#include <vector>
#include <functional>

class Node;

// ─── Problema PDE ─────────────────────────────────────────────────────────────
struct PDEProblem {
    PDE    type;
    int    dim = 2;    // 1 o 2 dimensiones
    double k2 = 0.0;   // coeficiente k² en Helmholtz: ∇²u + k²u = f
    bool   is_numerical = false; // Indica si requiere validación vía NumericalSolver
    bool   is_unsteady = false;  // Indica si depende del tiempo
    std::vector<Complex> numerical_truth; // Malla de referencia pre-calculada

    // Dimensiones para coherencia física
    Dimension dim_u = Units::None;
    Dimension dim_x = Units::Length;
    Dimension dim_y = Units::Length;
    Dimension dim_t = Units::Time;

    // Puntos de colocación interior (rejilla uniforme)
    std::vector<Point> domain_points(int n) const;

    // Puntos de condición de frontera (Dirichlet sobre ∂Ω)
    std::vector<Point> boundary_points(int n) const;

    // Solución exacta conocida u*(x,y,t)
    virtual Complex exact(double x, double y, double t = 0.0) const;
    Complex numerical_exact(double x, double y, double t = 0.0) const;
    Complex pde_second_derivative(double x, Complex u) const;

    // Término fuente f(x,y,t) del lado derecho
    virtual Complex source(double x, double y, double t = 0.0) const;

    // Condición de frontera (valor de u en ∂Ω a tiempo t)
    virtual Complex bc(double x, double y, double t = 0.0) const;

    // Cálculo unificado del residuo (incluye AD o Híbrido AD-FDM)
    virtual Complex compute_residual(const Node* tree, const Point& p) const;

    // Cálculo unificado del error de frontera (Dirichlet, Neumann, etc.)
    virtual double compute_boundary_error(const Node* tree, const std::vector<Point>& bnd) const;

    // Residuo del PDE usando AD (para método propuesto)
    Complex pde_residual_ad(const AD& ad, double x, double y, double t = 0.0) const;

    // Nombre legible
    std::string name() const;
};

// ─── Análisis a priori de la física (Physics-Guided Search) ───────────────
struct PDEPriors {
    bool autonomous_x = false;   
    bool autonomous_y = false;   
    bool pole_at_origin = false; 
    bool even_parity_x = false;  
    
    // Priors de Nivel Superior
    bool scale_invariant = false; // Invarianza de escala (Auto-similaridad)
    int  max_deriv_order = 0;     // ADN Diferencial (Orden máximo de derivación)
    bool is_conservative = false; // Leyes de conservación (Divergencia nula)
};

PDEPriors probe_priors(const PDEProblem& prob);

// ─── Fabricación de problemas ─────────────────────────────────────────────────
PDEProblem make_laplace(int dim = 2);
PDEProblem make_poisson(int dim = 2);
PDEProblem make_helmholtz(int dim = 2, double k = 1.0);
PDEProblem make_schrodinger(int dim = 2);
PDEProblem make_nonlinear_poisson();
PDEProblem make_liouville();
PDEProblem make_sine_gordon();
PDEProblem make_airy(int dim);
PDEProblem make_harmonic_oscillator(int dim);
PDEProblem make_navier_stokes();
PDEProblem make_navier_stokes_unsteady();
PDEProblem make_fisher(int dim);
PDEProblem make_duffing(int dim);
PDEProblem make_thomas_fermi(int dim);
PDEProblem make_bratu();
PDEProblem make_allen_cahn();
PDEProblem make_lane_emden();
PDEProblem make_troesch();
PDEProblem make_ginzburg_landau();
PDEProblem make_painleve1();
