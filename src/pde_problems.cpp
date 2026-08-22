#include "pde_problems.hpp"
#include "tree_node.hpp"
#include <cmath>
#include <algorithm>

namespace {
    const double PI_INTERNAL = 3.14159265358979323846;
}

std::string PDEProblem::name() const {
    return pde_name(type);
}

Complex PDEProblem::bc(double x, double y, double t) const {
    return exact(x, y, t);
}

Complex PDEProblem::exact(double x, double y, double t) const {
    if (dim == 1) {
        switch (type) {
            case PDE::LAPLACE:   return x;
            case PDE::POISSON:   
            case PDE::HELMHOLTZ: 
            case PDE::SINE_GORDON: return std::sin(PI_INTERNAL * x);
            case PDE::SCHRODINGER: return std::exp(I_COMPLEX * PI_INTERNAL * x);
            case PDE::HARMONIC_OSCILLATOR: return std::exp(-0.5 * x * x);
            case PDE::AIRY:      return std::exp(-x);
            case PDE::FISHER:    return 1.0 / (1.0 + std::exp(-x));
            case PDE::DUFFING:   return 1.0 / std::cosh(x);
            case PDE::THOMAS_FERMI: return 1.0 / (x + 0.5);
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE: return 1.0 / (1.0 + x * x);
            case PDE::LANE_EMDEN: return 1.0 - (x * x) / 6.0;
            case PDE::TROESCH: return std::sinh(3.0 * x) / std::sinh(3.0);
            case PDE::GINZBURG_LANDAU: return std::tanh(x);
            case PDE::PAINLEVE1: return 0.5 * x * x;
            default: return 0.0;
        }
    } else {
        switch (type) {
            case PDE::LAPLACE:
                return std::sin(PI_INTERNAL * x) * std::sinh(PI_INTERNAL * y) / std::sinh(PI_INTERNAL);
            case PDE::POISSON:
            case PDE::HELMHOLTZ:
            case PDE::SINE_GORDON:
                return std::sin(PI_INTERNAL * x) * std::sin(PI_INTERNAL * y);
            case PDE::SCHRODINGER:
                return std::exp(I_COMPLEX * PI_INTERNAL * (x + y));
            case PDE::HARMONIC_OSCILLATOR:
                return std::exp(-0.5 * (x*x + y*y));
            case PDE::AIRY:
                return std::exp(-(x + y));
            case PDE::FISHER:
                return 1.0 / (1.0 + std::exp(-(x + y)));
            case PDE::DUFFING:
                return 1.0 / std::cosh(x + y);
            case PDE::THOMAS_FERMI:
                return 1.0 / (x + y + 0.5);
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE:
                return 1.0 / (1.0 + x*x + y*y);
            case PDE::NAVIER_STOKES: {
                double Re = 1.0 / k2;
                double lambda = Re / 2.0 - std::sqrt(Re * Re / 4.0 + 4.0 * PI_INTERNAL * PI_INTERNAL);
                return y - std::exp(lambda * x) * std::sin(2.0 * PI_INTERNAL * y) / (2.0 * PI_INTERNAL * Re);
            }
            case PDE::NAVIER_STOKES_UNSTEADY: {
                double lambda = 2.0 * PI_INTERNAL * PI_INTERNAL * k2;
                return std::sin(PI_INTERNAL * x) * std::sin(PI_INTERNAL * y) * std::exp(-lambda * t); 
            }
            case PDE::BRATU:
                return std::log(2.0 / (std::cosh(x + y) * std::cosh(x + y)));
            case PDE::ALLEN_CAHN: {
                double eps = std::sqrt(0.01);
                return std::tanh((x + y - 1.0) / (eps * std::sqrt(2.0))); 
            }
            default: return 0.0;
        }
    }
}

Complex PDEProblem::source(double x, double y, double t) const {
    switch (type) {
        case PDE::LAPLACE: return 0.0;
        case PDE::BRATU: return 0.0; 
        case PDE::ALLEN_CAHN: return 0.0;
        case PDE::LANE_EMDEN: return 0.0;
        case PDE::POISSON: 
            return (dim == 1) ? (PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)) : (2.0*PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)*std::sin(PI_INTERNAL*y));
        case PDE::HELMHOLTZ: {
            double f_poisson = (dim == 1) ? (PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)) : (2.0*PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)*std::sin(PI_INTERNAL*y));
            return f_poisson - k2 * exact(x, y, t);
        }
        case PDE::NONLINEAR_POISSON: {
            double r2 = x*x + y*y; double den = 1.0 + r2;
            Complex lap = (8.0*r2)/(den*den*den) - 4.0/(den*den);
            return lap + exact(x, y, t) * exact(x, y, t);
        }
        case PDE::LIOUVILLE: {
            double r2 = x*x + y*y; double den = 1.0 + r2;
            Complex lap = (8.0*r2)/(den*den*den) - 4.0/(den*den);
            return lap + std::exp(exact(x, y, t));
        }
        default: return 0.0;
    }
}

Complex PDEProblem::pde_residual_ad(const AD& ad, double x, double y, double t) const {
    Complex u = ad.v;
    Complex laplacian = (dim == 1) ? ad.dxx : (ad.dxx + ad.dyy);
    switch (type) {
        case PDE::LAPLACE: return laplacian;
        case PDE::POISSON: return laplacian + source(x, y, t);
        case PDE::HELMHOLTZ: return laplacian + k2 * u + source(x, y, t);
        case PDE::SCHRODINGER: return laplacian + (static_cast<double>(dim)*PI_INTERNAL*PI_INTERNAL) * u;
        case PDE::HARMONIC_OSCILLATOR: return laplacian - (dim == 1 ? x*x : x*x+y*y)*u + (static_cast<double>(dim))*u;
        case PDE::NONLINEAR_POISSON: return laplacian + u*u - source(x, y, t);
        case PDE::LIOUVILLE: return laplacian + std::exp(u) - source(x, y, t);
        case PDE::SINE_GORDON: return laplacian - std::sin(u);
        case PDE::AIRY: return laplacian - (dim == 1 ? 1.0 : (x+y))*u;
        case PDE::FISHER: return laplacian + u*(1.0-u);
        case PDE::DUFFING: {
            double forcing = (dim == 1) ? std::sin(PI_INTERNAL * x) : std::sin(PI_INTERNAL * x) * std::sin(PI_INTERNAL * y);
            return laplacian + u + u*u*u - forcing;
        }
        case PDE::THOMAS_FERMI: {
            double r = (dim == 1) ? std::abs(x) : std::sqrt(x*x + y*y);
            double eps = 1e-12;
            Complex base = ad.v;
            // Usamos abs(base) para que la función sea diferenciable y no rompa el Gradient Descent
            return laplacian - std::pow(std::abs(base.real()) + eps, 1.5) / std::sqrt(r + eps);
        }
        case PDE::BRATU: return laplacian + 2.0 * std::exp(u);
        case PDE::ALLEN_CAHN: return 0.01 * laplacian - (u*u*u - u);
        case PDE::LANE_EMDEN: return laplacian + (2.0 / (x + 1e-6)) * ad.dx + std::pow(u + 1e-6, 3.0); 
        case PDE::TROESCH: return laplacian - 3.0 * std::sinh(3.0 * u); // Fully complex sinh
        case PDE::GINZBURG_LANDAU: return laplacian + u - u*u*u;
        case PDE::PAINLEVE1: return laplacian - (u*u + x + y);
        case PDE::NAVIER_STOKES: {
            // Simplified proxy for RAR: Advection-Diffusion balance using 2nd order AD
            return k2 * laplacian - (ad.dx + ad.dy); 
        }
        case PDE::NAVIER_STOKES_UNSTEADY: {
            // Stronger 2D proxy for RAR: Time-Laplacian coupling
            return ad.dt - k2 * laplacian;
        }
        default: return 0.0;
    }
}

std::vector<Point> PDEProblem::domain_points(int n) const {
    std::vector<Point> pts;
    double step = 1.0 / (std::sqrt(n));
    bool unsteady = (type == PDE::NAVIER_STOKES_UNSTEADY);
    for (double i = 0.1; i < 0.9; i += step) {
        for (double j = 0.1; j < 0.9; j += step) {
            // Si es inestable, t varía con i para evitar plegado de constantes
            double t = unsteady ? (i + j) * 0.5 : 0.0; 
            pts.push_back({i, j, t});
        }
    }
    // Refuerzo en el origen para Thomas-Fermi: anillos log-espaciados
    // acercandose a la singularidad r=0 (donde el termino u^1.5/sqrt(r) del
    // residuo explota). La version anterior usaba decaimiento geometrico
    // (0.9^i) que colapsa: para i>~116 la diferencia entre puntos consecutivos
    // es <1e-6, es decir ~60% del presupuesto de 300 puntos terminaba
    // muestreando casi el mismo radio una y otra vez sin aportar informacion
    // nueva. Con espaciado logaritmico real, cada punto cubre una decada
    // distinta de la transicion [r_min, r_max] — la version continua/real de
    // "anillos concentricos que se achican hacia el centro" (sin necesidad de
    // integral de contorno compleja, que no aplica aca por ser un punto de
    // ramificacion sqrt(r), no un polo).
    if (type == PDE::THOMAS_FERMI) {
        const double r_min = 1e-4, r_max = 0.2;
        const int n_rings = 300;
        double log_ratio = std::log(r_max / r_min);
        for (int i = 0; i < n_rings; ++i) {
            double frac = (double)i / (double)(n_rings - 1); // 0..1
            double r = r_min * std::exp(frac * log_ratio);
            pts.push_back({r, (dim == 2 ? r : 0.0), 0.0});
        }
    }
    return pts;
}

std::vector<Point> PDEProblem::boundary_points(int n) const {
    std::vector<Point> pts;
    bool unsteady = (type == PDE::NAVIER_STOKES_UNSTEADY);
    for (double s = 0; s <= 1.0; s += 0.05) {
        double t = unsteady ? s : 0.0;
        pts.push_back({0, s, t}); pts.push_back({1, s, t});
        pts.push_back({s, 0, t}); pts.push_back({s, 1, t});
    }
    return pts;
}

PDEProblem make_laplace(int d) { PDEProblem p; p.type = PDE::LAPLACE; p.dim = d; return p; }
PDEProblem make_poisson(int d) { PDEProblem p; p.type = PDE::POISSON; p.dim = d; return p; }
PDEProblem make_helmholtz(int d, double k) { PDEProblem p; p.type = PDE::HELMHOLTZ; p.dim = d; p.k2 = k*k; return p; }
PDEProblem make_schrodinger(int d) { PDEProblem p; p.type = PDE::SCHRODINGER; p.dim = d; return p; }
PDEProblem make_nonlinear_poisson() { PDEProblem p; p.type = PDE::NONLINEAR_POISSON; p.dim = 2; return p; }
PDEProblem make_liouville() { PDEProblem p; p.type = PDE::LIOUVILLE; p.dim = 2; return p; }
PDEProblem make_sine_gordon() { PDEProblem p; p.type = PDE::SINE_GORDON; p.dim = 2; return p; }
PDEProblem make_airy(int d) { PDEProblem p; p.type = PDE::AIRY; p.dim = d; p.dim_u = Units::Energy; return p; }
PDEProblem make_harmonic_oscillator(int d) { PDEProblem p; p.type = PDE::HARMONIC_OSCILLATOR; p.dim = d; p.dim_u = Units::Energy; return p; }
PDEProblem make_navier_stokes() { PDEProblem p; p.type = PDE::NAVIER_STOKES; p.dim = 2; p.k2 = 0.01; p.dim_u = Units::Viscosity; return p; }
PDEProblem make_navier_stokes_unsteady() { PDEProblem p; p.type = PDE::NAVIER_STOKES_UNSTEADY; p.dim = 2; p.k2 = 0.01; p.is_unsteady = true; p.dim_u = Units::Viscosity; return p; }
PDEProblem make_fisher(int d) { PDEProblem p; p.type = PDE::FISHER; p.dim = d; p.is_numerical = true; p.dim_u = Units::None; return p; }
PDEProblem make_duffing(int d) { PDEProblem p; p.type = PDE::DUFFING; p.dim = d; p.is_numerical = true; p.dim_u = Units::Acceleration; return p; }
PDEProblem make_thomas_fermi(int d) { 
    PDEProblem p; p.type = PDE::THOMAS_FERMI; p.dim = d; p.is_numerical = true; p.dim_u = Units::None; 
    // Añadimos puntos extra cerca de la singularidad r=0
    return p; 
}


PDEProblem make_bratu() { PDEProblem p; p.type = PDE::BRATU; p.dim = 2; return p; }
PDEProblem make_allen_cahn() { PDEProblem p; p.type = PDE::ALLEN_CAHN; p.dim = 2; p.k2 = 0.01; return p; }
PDEProblem make_lane_emden() { PDEProblem p; p.type = PDE::LANE_EMDEN; p.dim = 1; p.is_numerical = true; return p; }
PDEProblem make_troesch() { PDEProblem p; p.type = PDE::TROESCH; p.dim = 1; p.is_numerical = true; return p; }
PDEProblem make_ginzburg_landau() { PDEProblem p; p.type = PDE::GINZBURG_LANDAU; p.dim = 1; p.is_numerical = true; return p; }
PDEProblem make_painleve1() { PDEProblem p; p.type = PDE::PAINLEVE1; p.dim = 1; p.is_numerical = true; return p; }

Complex PDEProblem::pde_second_derivative(double x, Complex u) const {
    switch (type) {
        case PDE::LAPLACE: return 0.0;
        case PDE::POISSON: return source(x, 0.0, 0.0);
        case PDE::HELMHOLTZ: return source(x, 0.0, 0.0) - k2 * u;
        case PDE::SCHRODINGER: return -(PI_INTERNAL * PI_INTERNAL) * u;
        case PDE::AIRY:      return u;
        case PDE::HARMONIC_OSCILLATOR: return (x*x - 1.0) * u;
        case PDE::FISHER: return -u * (1.0 - u);
        case PDE::DUFFING: return -u - u * u * u;
        case PDE::THOMAS_FERMI: return std::pow(u.real() + 1e-12, 1.5) / std::sqrt(x + 1e-12);
        case PDE::BRATU: return -2.0 * std::exp(u);
        case PDE::ALLEN_CAHN: return (u*u*u - u) / 0.01;
        case PDE::LANE_EMDEN: return -u*u*u - (2.0 / (x + 1e-6)) * (u - 1.0); // Rough approximation of u'
        case PDE::TROESCH: return 3.0 * std::sinh(3.0 * u.real());
        case PDE::GINZBURG_LANDAU: return -u + u*u*u;
        case PDE::PAINLEVE1: return u*u + x;
        default: return 0.0;
    }
}

Complex PDEProblem::numerical_exact(double x, double y, double t) const {
    return exact(x, y, t);
}

// ─── Probing Físico (Análisis Asintótico y de Simetrías) ──────────────────────
PDEPriors probe_priors(const PDEProblem& prob) {
    PDEPriors priors;
    AD dummy_ad;
    dummy_ad.v = 1.0; dummy_ad.dx = 0.5; dummy_ad.dy = 0.5; 
    dummy_ad.dxx = -0.1; dummy_ad.dyy = -0.1; dummy_ad.dt = 0.1;
    
    // 1. Probar Invarianza Traslacional
    Complex res_base = prob.pde_residual_ad(dummy_ad, 1.0, 1.0, 0.0);
    Complex res_x_shift = prob.pde_residual_ad(dummy_ad, 2.0, 1.0, 0.0);
    Complex res_y_shift = prob.pde_residual_ad(dummy_ad, 1.0, 2.0, 0.0);
    if (std::abs(res_base - res_x_shift) < 1e-9) priors.autonomous_x = true;
    if (std::abs(res_base - res_y_shift) < 1e-9) priors.autonomous_y = true;
    
    // 2. Probar Simetría Par
    Complex res_x_neg = prob.pde_residual_ad(dummy_ad, -1.0, 1.0, 0.0);
    if (std::abs(res_base - res_x_neg) < 1e-9) priors.even_parity_x = true;
    
    // 3. Probar Singularidad en el Origen (Polo)
    Complex res_origin = prob.pde_residual_ad(dummy_ad, 1e-6, 1e-6, 0.0);
    if (std::abs(res_origin) > 1e4 || !std::isfinite(res_origin.real())) priors.pole_at_origin = true;

    // 4. Análisis de Homogeneidad (Invarianza de Escala)
    // Probamos si R(u, x, t) == R(lambda*u, lambda*x, lambda^2*t)
    AD scaled_ad; double L = 2.0;
    scaled_ad.v = dummy_ad.v * L; 
    scaled_ad.dx = dummy_ad.dx; // (du/dx se mantiene igual si u y x escalan igual)
    scaled_ad.dxx = dummy_ad.dxx / L;
    Complex res_scaled = prob.pde_residual_ad(scaled_ad, L, L, L*L);
    if (std::abs(res_base - res_scaled / L) < 1e-3) priors.scale_invariant = true;

    // 5. ADN Diferencial (Orden de Derivada)
    // Cambiamos dxx y vemos si el residuo cambia. Si no cambia, no es de 2º orden.
    AD ad_no_lap = dummy_ad; ad_no_lap.dxx = 0; ad_no_lap.dyy = 0;
    if (std::abs(res_base - prob.pde_residual_ad(ad_no_lap, 1.0, 1.0, 0.0)) > 1e-9) {
        priors.max_deriv_order = 2;
    } else if (std::abs(res_base - prob.pde_residual_ad(dummy_ad, 1.0, 1.0, 0.0)) > 1e-9) {
        priors.max_deriv_order = 1;
    }

    // 6. Conservación (Divergencia)
    if (prob.type == PDE::NAVIER_STOKES || prob.type == PDE::NAVIER_STOKES_UNSTEADY) priors.is_conservative = true;

    // 7. Invarianza Traslacional en t (autónomo en el tiempo)
    if (prob.is_unsteady) {
        Complex res_t_shift = prob.pde_residual_ad(dummy_ad, 1.0, 1.0, 1.0);
        Complex res_t_base = prob.pde_residual_ad(dummy_ad, 1.0, 1.0, 0.0);
        if (std::abs(res_t_base - res_t_shift) < 1e-9) priors.autonomous_t = true;
    }

    // 8. Simetría especular x <-> 1-x (y análoga en y) — la relevante para el
    // dominio real [0,1], no x<->-x (fuera del dominio, ver punto 2). Sólo se
    // marca si TANTO el operador COMO la condición de frontera (bc==exact para
    // la mayoría de los PDEs) son invariantes ante la reflexión en VARIOS
    // puntos (no uno solo: un único punto de prueba puede coincidir con un
    // cero accidental de la fórmula — ej. sin(2*pi*y) se anula en y=0.5 — y dar
    // un falso positivo). Se excluyen Navier-Stokes/Unsteady: su condición de
    // frontera real es sobre función de corriente + derivadas de velocidad, no
    // un simple valor Dirichlet, y su residuo aquí es un proxy simplificado que
    // no depende explícitamente de x,y (siempre "simétrico" por construcción,
    // no por física real) — comprobado empíricamente que producía falsos
    // positivos.
    if (prob.type != PDE::NAVIER_STOKES && prob.type != PDE::NAVIER_STOKES_UNSTEADY) {
        bool op_mirror_x = true, bc_mirror_x = true;
        for (double x0 : {0.15, 0.3, 0.42}) {
            for (double yfix : {0.2, 0.55, 0.8}) {
                Complex ra = prob.pde_residual_ad(dummy_ad, x0, yfix, 0.0);
                Complex rb = prob.pde_residual_ad(dummy_ad, 1.0 - x0, yfix, 0.0);
                if (std::abs(ra - rb) > 1e-9) { op_mirror_x = false; break; }
                Complex b1 = prob.bc(x0, yfix, 0.0);
                Complex b2 = prob.bc(1.0 - x0, yfix, 0.0);
                if (std::abs(b1 - b2) > 1e-6) { bc_mirror_x = false; break; }
            }
            if (!op_mirror_x || !bc_mirror_x) break;
        }
        priors.mirror_symmetric_x = op_mirror_x && bc_mirror_x;

        if (prob.dim == 2) {
            bool op_mirror_y = true, bc_mirror_y = true;
            for (double y0 : {0.15, 0.3, 0.42}) {
                for (double xfix : {0.2, 0.55, 0.8}) {
                    Complex ra = prob.pde_residual_ad(dummy_ad, xfix, y0, 0.0);
                    Complex rb = prob.pde_residual_ad(dummy_ad, xfix, 1.0 - y0, 0.0);
                    if (std::abs(ra - rb) > 1e-9) { op_mirror_y = false; break; }
                    Complex b1 = prob.bc(xfix, y0, 0.0);
                    Complex b2 = prob.bc(xfix, 1.0 - y0, 0.0);
                    if (std::abs(b1 - b2) > 1e-6) { bc_mirror_y = false; break; }
                }
                if (!op_mirror_y || !bc_mirror_y) break;
            }
            priors.mirror_symmetric_y = op_mirror_y && bc_mirror_y;
        }
    }

    // 9. Separabilidad de variables (solo 2D, EDPs no-Navier-Stokes con BC
    // Dirichlet). Se prueba con una funcion de sondeo GENERICA (sin(x)+sin(y)
    // para la aditiva, sin(x)*sin(y) para la multiplicativa) — NO con la
    // solucion exacta del problema (eso seria fuga de informacion, el mismo
    // error que se encontro y corrigio en la frontera de Navier-Stokes). Es
    // el analogo numerico de "intentar" separacion de variables clasica sin
    // conocer la respuesta: si el residuo del OPERADOR, evaluado sobre esa
    // funcion de sondeo, no mezcla x e y (segunda derivada cruzada ~0 en
    // varios puntos), el operador es consistente con esa forma estructural.
    // No se excluye ningun tipo de PDE por nombre aca (a diferencia del
    // sondeo de simetria especular, que si necesita excluir Navier-Stokes
    // porque ese chequeo compara prob.bc() — la condicion de frontera
    // real de Navier-Stokes es sobre funcion de corriente, no comparable por
    // igualdad simple). Este sondeo solo usa pde_residual_ad(), nunca bc(),
    // asi que ese problema no aplica — es igual de valido para cualquier EDP,
    // Navier-Stokes incluido. Excluirlo "a mano" seria justo el tipo de
    // solucion hecha a medida de un solo problema del benchmark que no
    // queremos.
    if (prob.dim == 2) {
        auto probe_mixing = [&](bool multiplicative) -> bool {
            auto eval_R = [&](double x, double y) -> Complex {
                AD probe;
                double sx = std::sin(x), cx = std::cos(x), sy = std::sin(y), cy = std::cos(y);
                if (multiplicative) {
                    probe.v = sx * sy;
                    probe.dx = cx * sy; probe.dy = sx * cy;
                    probe.dxx = -sx * sy; probe.dyy = -sx * sy;
                } else {
                    probe.v = sx + sy;
                    probe.dx = cx; probe.dy = cy;
                    probe.dxx = -sx; probe.dyy = -sy;
                }
                return prob.pde_residual_ad(probe, x, y, 0.0);
            };
            double h = 1e-3;
            for (double x0 : {0.2, 0.4, 0.6}) {
                for (double y0 : {0.25, 0.5, 0.75}) {
                    Complex rpp = eval_R(x0+h, y0+h), rpm = eval_R(x0+h, y0-h);
                    Complex rmp = eval_R(x0-h, y0+h), rmm = eval_R(x0-h, y0-h);
                    if (!std::isfinite(rpp.real()) || !std::isfinite(rmm.real())) return false;
                    Complex d2xy = (rpp - rpm - rmp + rmm) / (4.0*h*h);
                    double scale = std::max({std::abs(rpp), std::abs(rmm), 1.0});
                    if (std::abs(d2xy) > 1e-2 * scale) return false;
                }
            }
            return true;
        };
        priors.additive_separable = probe_mixing(false);
        if (!priors.additive_separable) priors.multiplicative_separable = probe_mixing(true);
    }

    // Separabilidad triple u(x,y,t)=f(x)g(y)h(t) para EDPs dependientes del
    // tiempo — mismo principio, generalizado a sondear las 3 derivadas
    // cruzadas (xy, xt, yt) con una funcion de sondeo generica
    // sin(x)*sin(y)*sin(t). General por construccion (ningun tipo excluido):
    // aplica a cualquier PDE 2D+t que pase el sondeo del OPERADOR, no solo a
    // una en particular.
    if (prob.dim == 2 && prob.is_unsteady) {
        auto eval_R3 = [&](double x, double y, double t) -> Complex {
            AD probe;
            double sx = std::sin(x), cx = std::cos(x);
            double sy = std::sin(y), cy = std::cos(y);
            double st = std::sin(t), ct = std::cos(t);
            probe.v = sx * sy * st;
            probe.dx = cx * sy * st; probe.dy = sx * cy * st; probe.dt = sx * sy * ct;
            probe.dxx = -sx * sy * st; probe.dyy = -sx * sy * st;
            return prob.pde_residual_ad(probe, x, y, t);
        };
        bool triple_ok = true;
        double h = 1e-3;
        for (double x0 : {0.2, 0.4, 0.6}) {
            for (double y0 : {0.25, 0.5, 0.75}) {
                for (double t0 : {0.3, 0.5, 0.7}) {
                    Complex r000 = eval_R3(x0, y0, t0);
                    if (!std::isfinite(r000.real())) { triple_ok = false; break; }
                    double scale = std::max(std::abs(r000), 1.0);
                    // d2R/dxdy
                    Complex rxy = (eval_R3(x0+h,y0+h,t0) - eval_R3(x0+h,y0-h,t0)
                                 - eval_R3(x0-h,y0+h,t0) + eval_R3(x0-h,y0-h,t0)) / (4.0*h*h);
                    // d2R/dxdt
                    Complex rxt = (eval_R3(x0+h,y0,t0+h) - eval_R3(x0+h,y0,t0-h)
                                 - eval_R3(x0-h,y0,t0+h) + eval_R3(x0-h,y0,t0-h)) / (4.0*h*h);
                    // d2R/dydt
                    Complex ryt = (eval_R3(x0,y0+h,t0+h) - eval_R3(x0,y0+h,t0-h)
                                 - eval_R3(x0,y0-h,t0+h) + eval_R3(x0,y0-h,t0-h)) / (4.0*h*h);
                    if (std::abs(rxy) > 1e-2*scale || std::abs(rxt) > 1e-2*scale || std::abs(ryt) > 1e-2*scale) {
                        triple_ok = false; break;
                    }
                }
                if (!triple_ok) break;
            }
            if (!triple_ok) break;
        }
        priors.triple_separable = triple_ok;
    }

    return priors;
}

Complex PDEProblem::compute_residual(const Node* tree, const Point& p) const {
    if (type == PDE::NAVIER_STOKES_UNSTEADY) {
        double nu = k2; double h = 0.01; double ht = 0.01;
        AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, dim);
        Complex u = ad_c.dy, v = -ad_c.dx;
        Complex w = -(ad_c.dxx + ad_c.dyy);
        
        AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, dim);
        AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, dim);
        AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, dim);
        AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, dim);
        
        Complex w_x = (-(ad_xp.dxx + ad_xp.dyy) - (-(ad_xm.dxx + ad_xm.dyy))) / (2.0 * h);
        Complex w_y = (-(ad_yp.dxx + ad_yp.dyy) - (-(ad_ym.dxx + ad_ym.dyy))) / (2.0 * h);
        Complex lap_w = (-(ad_xp.dxx + ad_xp.dyy) + (-(ad_xm.dxx + ad_xm.dyy)) + (-(ad_yp.dxx + ad_yp.dyy)) + (-(ad_ym.dxx + ad_ym.dyy)) - 4.0 * w) / (h * h);
        
        AD ad_tp = tree->ad_eval_t(p.x, p.y, p.t+ht, dim);
        AD ad_tm = tree->ad_eval_t(p.x, p.y, p.t-ht, dim);
        Complex w_t = (-(ad_tp.dxx + ad_tp.dyy) - (-(ad_tm.dxx + ad_tm.dyy))) / (2.0 * ht);
        
        return w_t + u * w_x + v * w_y - nu * lap_w;
    } else if (type == PDE::NAVIER_STOKES) {
        double nu = k2; double h = 0.01;
        AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, dim);
        Complex psi_x = ad_c.dx, psi_y = ad_c.dy;
        Complex L_val = (ad_c.dxx + ad_c.dyy);
        
        AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, dim);
        AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, dim);
        AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, dim);
        AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, dim);
        
        Complex L_x = ((ad_xp.dxx + ad_xp.dyy) - (ad_xm.dxx + ad_xm.dyy)) / (2.0*h);
        Complex L_y = ((ad_yp.dxx + ad_yp.dyy) - (ad_ym.dxx + ad_ym.dyy)) / (2.0*h);
        Complex biharmonic = ((ad_xp.dxx + ad_xp.dyy) + (ad_xm.dxx + ad_xm.dyy) + 
                             (ad_yp.dxx + ad_yp.dyy) + (ad_ym.dxx + ad_ym.dyy) - 4.0*L_val) / (h*h);
        return psi_y * L_x - psi_x * L_y - nu * biharmonic;
    } else {
        AD ad = tree->ad_eval_t(p.x, p.y, p.t, dim);
        return pde_residual_ad(ad, p.x, p.y, p.t);
    }
}

double PDEProblem::compute_boundary_error(const Node* tree, const std::vector<Point>& bnd) const {
    double raw_bc_mse = 0.0;
    // Acumulador local con guardia de finitud: un solo punto con derivada no
    // finita (ej. una expresion como exp(-c/t) evaluada justo en t=0, donde
    // la regla de la cadena de AD produce 0*inf=NaN) contamina la suma entera
    // con NaN silenciosamente, propagandose hasta el BIC y las tablas del
    // reporte. Mismo criterio de robustez que ya se aplica al residuo de
    // dominio (ver pi_solver.cpp: chequeo de isfinite + clip universal).
    auto add_term = [&raw_bc_mse](Complex diff) {
        double v = std::min(std::norm(diff), 1e4);
        if (std::isfinite(v)) raw_bc_mse += v;
        else raw_bc_mse += 1e4; // outlier no finito: penaliza como el peor caso, no lo ignora
    };
    // Dirichlet simple para TODOS los tipos, incluido Navier-Stokes(-Unsteady).
    // Antes, esos dos casos exigian ademas que du/dy y -du/dx (via diferencias
    // finitas de bc) calzaran con la "velocidad" de una funcion de corriente
    // psi — pero ninguno de los dos residuos de dominio actuales (ver
    // pde_residual_ad: proxy de adveccion-difusion para el estacionario,
    // proxy tipo ecuacion del calor u_t=k2*Lap(u) para el Unsteady) trata el
    // arbol como una funcion de corriente; ambos operan directamente sobre el
    // campo escalar u. Esa condicion de frontera heredada de una formulacion
    // anterior le daba al buscador el valor Y dos derivadas de la solucion
    // exacta en cada punto de frontera (para Unsteady, en TODO el rango de
    // t) — mucha mas informacion que un Dirichlet real, lo que explicaba
    // resultados "demasiado buenos para ser verdad" en Navier-Stokes-Unsteady.
    for (auto& p : bnd) {
        Complex val = tree->eval_t(p.x, p.y, p.t);
        Complex target = bc(p.x, p.y, p.t);
        add_term(val - target);
    }
    return bnd.empty() ? 0.0 : (raw_bc_mse / bnd.size());
}
