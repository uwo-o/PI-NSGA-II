// =============================================================================
// pde_problems.cpp  —  Implementación de Laplace, Poisson y Helmholtz
// =============================================================================

#include "pde_problems.hpp"
#include <cmath>
#include <stdexcept>

static constexpr double PI = M_PI;

// ─── domain_points: rejilla uniforme interior (x,y) ∈ (0,1)² ─────────────────
std::vector<Point> PDEProblem::domain_points(int n) const {
    int sq = (int)std::ceil(std::sqrt((double)n));
    std::vector<Point> pts;
    pts.reserve(sq * sq);
    for (int i = 1; i <= sq; ++i)
        for (int j = 1; j <= sq; ++j)
            pts.push_back({(double)i / (sq + 1), (double)j / (sq + 1)});
    return pts;
}

// ─── boundary_points: puntos en los 4 lados de ∂Ω ────────────────────────────
std::vector<Point> PDEProblem::boundary_points(int n) const {
    int per_side = std::max(2, n / 4);
    std::vector<Point> pts;
    pts.reserve(per_side * 4);
    for (int i = 0; i < per_side; ++i) {
        double t = (double)i / (per_side - 1);
        pts.push_back({t, 0.0}); // sur
        pts.push_back({t, 1.0}); // norte
        pts.push_back({0.0, t}); // oeste
        pts.push_back({1.0, t}); // este
    }
    return pts;
}

// ─── Solución exacta ──────────────────────────────────────────────────────────
double PDEProblem::exact(double x, double y) const {
    // Las 3 ecuaciones comparten: u*(x,y) = sin(πx)·sin(πy) [Poisson, Helmholtz]
    // Para Laplace usamos: sin(πx)·sinh(πy)/sinh(π)
    switch (type) {
        case PDE::LAPLACE:
            return std::sin(PI*x) * std::sinh(PI*y) / std::sinh(PI);
        case PDE::POISSON:
        case PDE::HELMHOLTZ:
            return std::sin(PI*x) * std::sin(PI*y);
        case PDE::SCHRODINGER: {
            double dx = x - 0.5;
            double dy = y - 0.5;
            return std::exp(-(dx*dx + dy*dy));
        }
    }
    return 0.0;
}

// ─── Término fuente f(x,y): ∇²u + k²u = f ────────────────────────────────────
double PDEProblem::source(double x, double y) const {
    switch (type) {
        case PDE::LAPLACE:
            return 0.0; // ∇²u = 0

        case PDE::POISSON:
            // u* = sin(πx)sin(πy) → ∇²u* = -2π²sin(πx)sin(πy)
            // Poisson: ∇²u = f → f = -2π²sin(πx)sin(πy)
            return -2.0 * PI * PI * std::sin(PI*x) * std::sin(PI*y);

        case PDE::HELMHOLTZ:
            // u* = sin(πx)sin(πy), ∇²u* = -2π²u*
            // ∇²u + k²u = f → f = (-2π² + k²)u*
            return (k2 - 2.0*PI*PI) * std::sin(PI*x) * std::sin(PI*y);
    }
    return 0.0;
}

// ─── Condición de frontera Dirichlet ──────────────────────────────────────────
double PDEProblem::bc(double x, double y) const {
    return exact(x, y);
}

double PDEProblem::pde_residual_ad(const AD& ad, double x, double y) const {
    double laplacian = ad.dxx + ad.dyy;
    if (type == PDE::SCHRODINGER) {
        // Ecuación de Schrödinger independiente del tiempo (dimensionless)
        // -∇²ψ + V(x,y)ψ = Eψ  =>  ∇²ψ + (E - V(x,y))ψ = 0
        // Para u = exp(-(x-0.5)² - (y-0.5)²), V = 4r² y E = 4
        double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
        double V = 4.0 * r2;
        double E = 4.0;
        return laplacian + (E - V) * ad.v;
    }
    return laplacian + k2 * ad.v - source(x, y);
}

// ─── Nombre ───────────────────────────────────────────────────────────────────
std::string PDEProblem::name() const { return pde_name(type); }

// ─── Fabricación ─────────────────────────────────────────────────────────────
PDEProblem make_laplace() {
    PDEProblem p;
    p.type = PDE::LAPLACE;
    p.k2   = 0.0;
    return p;
}
PDEProblem make_poisson() {
    PDEProblem p;
    p.type = PDE::POISSON;
    p.k2   = 0.0;
    return p;
}
PDEProblem make_helmholtz(double k) {
    PDEProblem p;
    p.type = PDE::HELMHOLTZ;
    p.k2   = k * k;
    return p;
}
PDEProblem make_schrodinger() {
    PDEProblem p;
    p.type = PDE::SCHRODINGER;
    p.k2   = 0.0;
    return p;
}
