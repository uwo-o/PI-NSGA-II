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

    // Puntos de colocación interior (rejilla uniforme)
    std::vector<Point> domain_points(int n) const;

    // Puntos de condición de frontera (Dirichlet sobre ∂Ω)
    std::vector<Point> boundary_points(int n) const;

    // Solución exacta conocida u*(x,y)
    double exact(double x, double y) const;

    // Término fuente f(x,y) del lado derecho: ∇²u + k²u = f
    double source(double x, double y) const;

    // Condición de frontera (valor de u en ∂Ω)
    double bc(double x, double y) const;

    // Residuo del PDE usando AD (para método propuesto)
    // R = ∇²u + k²u - f
    double pde_residual_ad(const AD& ad, double x, double y) const;

    // Nombre legible
    std::string name() const;
};

// ─── Fabricación de problemas ─────────────────────────────────────────────────
PDEProblem make_laplace(int dim = 2);
PDEProblem make_poisson(int dim = 2);
PDEProblem make_helmholtz(int dim = 2, double k = 1.0);
PDEProblem make_schrodinger(int dim = 2);
