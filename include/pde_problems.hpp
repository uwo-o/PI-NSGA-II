#pragma once
// =============================================================================
// pde_problems.hpp  —  Definición de Laplace, Poisson y Helmholtz
//   Dominio Ω = [0,1]², condiciones de Dirichlet.
// =============================================================================

#include "common.hpp"
#include <vector>
#include <functional>


// ─── Problema PDE ─────────────────────────────────────────────────────────────
struct PDEProblem {
    PDE    type;
    int    dim = 2;    // 1 o 2 dimensiones
    double k2 = 0.0;   // coeficiente k² en Helmholtz: ∇²u + k²u = f
    bool   is_numerical = false; // Indica si requiere validación vía NumericalSolver
    std::vector<Complex> numerical_truth; // Malla de referencia pre-calculada

    // Puntos de colocación interior (rejilla uniforme)
    std::vector<Point> domain_points(int n) const;

    // Puntos de condición de frontera (Dirichlet sobre ∂Ω)
    std::vector<Point> boundary_points(int n) const;

    // Solución exacta conocida u*(x,y)
    virtual Complex exact(double x, double y) const;
    Complex numerical_exact(double x, double y) const;
    Complex pde_second_derivative(double x, Complex u) const;

    // Término fuente f(x,y) del lado derecho: ∇²u + k²u = f
    virtual Complex source(double x, double y) const;

    // Condición de frontera (valor de u en ∂Ω)
    virtual Complex bc(double x, double y) const;

    // Residuo del PDE usando AD (para método propuesto)
    // R = ∇²u + k²u - f
    virtual Complex pde_residual_ad(const AD& ad, double x, double y) const;

    // Nombre legible
    std::string name() const;
};

// ─── Fabricación de problemas ─────────────────────────────────────────────────
PDEProblem make_laplace(int dim = 2);
PDEProblem make_poisson(int dim = 2);
PDEProblem make_helmholtz(int dim = 2, double k = 1.0);
PDEProblem make_schrodinger(int dim = 2);
PDEProblem make_nonlinear_poisson();
PDEProblem make_liouville();
PDEProblem make_sine_gordon();
PDEProblem make_airy();
PDEProblem make_harmonic_oscillator(int dim);
