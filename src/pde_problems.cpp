// =============================================================================
// pde_problems.cpp  —  Implementación de Laplace, Poisson y Helmholtz
// =============================================================================

#include "pde_problems.hpp"
#include <cmath>
#include <stdexcept>

static constexpr double PI = M_PI;

// ─── domain_points: rejilla uniforme interior (x,y) ∈ (0,1)² ─────────────────
std::vector<Point> PDEProblem::domain_points(int n) const {
    std::vector<Point> pts;
    if (dim == 1) {
        pts.reserve(n);
        for (int i = 1; i <= n; ++i)
            pts.push_back({(double)i / (n + 1), 0.0});
    } else {
        int sq = (int)std::ceil(std::sqrt((double)n));
        pts.reserve(sq * sq);
        for (int i = 1; i <= sq; ++i)
            for (int j = 1; j <= sq; ++j)
                pts.push_back({(double)i / (sq + 1), (double)j / (sq + 1)});
    }
    return pts;
}

// ─── boundary_points: puntos en los 4 lados de ∂Ω ────────────────────────────
std::vector<Point> PDEProblem::boundary_points(int n) const {
    std::vector<Point> pts;
    if (dim == 1) {
        pts.push_back({0.0, 0.0});
        pts.push_back({1.0, 0.0});
    } else {
        int per_side = std::max(2, n / 4);
        pts.reserve(per_side * 4);
        for (int i = 0; i < per_side; ++i) {
            double t = (double)i / (per_side - 1);
            pts.push_back({t, 0.0}); // sur
            pts.push_back({t, 1.0}); // norte
            pts.push_back({0.0, t}); // oeste
            pts.push_back({1.0, t}); // este
        }
    }
    return pts;
}

// ─── Solución exacta ──────────────────────────────────────────────────────────
double PDEProblem::exact(double x, double y) const {
    if (dim == 1) {
        switch (type) {
            case PDE::LAPLACE:   return x;
            case PDE::POISSON:   
            case PDE::HELMHOLTZ: return std::sin(PI * x);
            case PDE::SCHRODINGER: return std::exp(-(x - 0.5) * (x - 0.5));
        }
    } else {
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
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE:
                return 1.0 / (1.0 + x*x + y*y);
            case PDE::SINE_GORDON:
                return std::sin(PI*x) * std::sin(PI*y);
        }
    }
    return 0.0;
}

// ─── Término fuente f(x,y): ∇²u + k²u = f ────────────────────────────────────
double PDEProblem::source(double x, double y) const {
    if (dim == 1) {
        switch (type) {
            case PDE::LAPLACE: return 0.0;
            case PDE::POISSON: return -PI * PI * std::sin(PI * x);
            case PDE::HELMHOLTZ: return (k2 - PI * PI) * std::sin(PI * x);
        }
    } else {
        switch (type) {
            case PDE::LAPLACE: return 0.0;
            case PDE::POISSON:
                return -2.0 * PI * PI * std::sin(PI*x) * std::sin(PI*y);
            case PDE::HELMHOLTZ:
                return (k2 - 2.0*PI*PI) * std::sin(PI*x) * std::sin(PI*y);
            case PDE::NONLINEAR_POISSON: {
                double r2 = x*x + y*y;
                double den = 1.0 + r2;
                double laplacian = (8.0 * r2) / (den * den * den) - 4.0 / (den * den); // Corregido 2D
                double u_val = 1.0 / den;
                return laplacian + u_val * u_val;
            }
            case PDE::LIOUVILLE: {
                double r2 = x*x + y*y;
                double den = 1.0 + r2;
                double laplacian = (8.0 * r2) / (den * den * den) - 4.0 / (den * den);
                double u_val = 1.0 / den;
                return laplacian + std::exp(u_val);
            }
            case PDE::SINE_GORDON: {
                double u = std::sin(PI*x) * std::sin(PI*y);
                return -2.0 * PI * PI * u + std::sin(u);
            }
        }
    }
    return 0.0;
}

// ─── Condición de frontera Dirichlet ──────────────────────────────────────────
double PDEProblem::bc(double x, double y) const {
    return exact(x, y);
}

double PDEProblem::pde_residual_ad(const AD& ad, double x, double y) const {
    double laplacian = (dim == 1) ? ad.dxx : (ad.dxx + ad.dyy);
    if (type == PDE::SCHRODINGER) {
        double r2 = (x - 0.5) * (x - 0.5) + (dim == 2 ? (y - 0.5) * (y - 0.5) : 0.0);
        double V = 4.0 * r2;
        double E = (dim == 2) ? 4.0 : 2.0;
        return laplacian + (E - V) * ad.v;
    }
    if (type == PDE::NONLINEAR_POISSON) {
        return laplacian + ad.v * ad.v - source(x, y);
    }
    if (type == PDE::LIOUVILLE) {
        return laplacian + std::exp(ad.v) - source(x, y);
    }
    if (type == PDE::SINE_GORDON) {
        return laplacian + std::sin(ad.v) - source(x, y);
    }
    return laplacian + k2 * ad.v - source(x, y);
}

// ─── Nombre ───────────────────────────────────────────────────────────────────
std::string PDEProblem::name() const { return pde_name(type); }

// ─── Fabricación ─────────────────────────────────────────────────────────────
PDEProblem make_laplace(int dim) {
    PDEProblem p;
    p.type = PDE::LAPLACE;
    p.dim  = dim;
    p.k2   = 0.0;
    return p;
}
PDEProblem make_poisson(int dim) {
    PDEProblem p;
    p.type = PDE::POISSON;
    p.dim  = dim;
    p.k2   = 0.0;
    return p;
}
PDEProblem make_helmholtz(int dim, double k) {
    PDEProblem p;
    p.type = PDE::HELMHOLTZ;
    p.dim  = dim;
    p.k2   = k * k;
    return p;
}
PDEProblem make_schrodinger(int dim) {
    PDEProblem p;
    p.type = PDE::SCHRODINGER;
    p.dim  = dim;
    p.k2   = 0.0;
    return p;
}
PDEProblem make_nonlinear_poisson() {
    PDEProblem p;
    p.type = PDE::NONLINEAR_POISSON;
    p.dim  = 2;
    p.k2   = 0.0;
    return p;
}
PDEProblem make_liouville() {
    PDEProblem p;
    p.type = PDE::LIOUVILLE;
    p.dim  = 2;
    p.k2   = 0.0;
    return p;
}
PDEProblem make_sine_gordon() {
    PDEProblem p;
    p.type = PDE::SINE_GORDON;
    p.dim  = 2;
    p.k2   = 0.0;
    return p;
}
