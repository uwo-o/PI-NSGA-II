#pragma once
// =============================================================================
// common.hpp  —  Tipos de datos y constantes globales
// =============================================================================

#include <vector>
#include <string>
#include <limits>
#include <complex>

using Complex = std::complex<double>;
const Complex I_COMPLEX(0.0, 1.0);
const double PI_VAL = 3.14159265358979323846;
const double E_VAL  = 2.71828182845904523536;

// ─── Tipos de PDE soportados ──────────────────────────────────────────────────
enum class PDE { 
    LAPLACE, POISSON, HELMHOLTZ, SCHRODINGER, 
    NONLINEAR_POISSON, LIOUVILLE, SINE_GORDON,
    AIRY, HARMONIC_OSCILLATOR, GROSS_PITAEVSKII,
    NAVIER_STOKES, NAVIER_STOKES_UNSTEADY,
    FISHER, DUFFING, THOMAS_FERMI,
    BRATU, ALLEN_CAHN, LANE_EMDEN,
    TROESCH, GINZBURG_LANDAU, PAINLEVE1
};

inline std::string pde_name(PDE t) {
    switch (t) {
        case PDE::LAPLACE:     return "Laplace";
        case PDE::POISSON:     return "Poisson";
        case PDE::HELMHOLTZ:   return "Helmholtz";
        case PDE::SCHRODINGER:       return "Schrodinger";
        case PDE::NONLINEAR_POISSON: return "NonlinearPoisson";
        case PDE::LIOUVILLE:         return "Liouville";
        case PDE::SINE_GORDON:       return "Sine-Gordon";
        case PDE::AIRY:              return "Airy";
        case PDE::HARMONIC_OSCILLATOR: return "HarmonicOscillator";
        case PDE::GROSS_PITAEVSKII:  return "Gross-Pitaevskii";
        case PDE::NAVIER_STOKES:     return "Navier-Stokes";
        case PDE::NAVIER_STOKES_UNSTEADY: return "Navier-Stokes-Unsteady";
        case PDE::FISHER:            return "Fisher";
        case PDE::DUFFING:           return "Duffing";
        case PDE::THOMAS_FERMI:      return "Thomas-Fermi";
        case PDE::BRATU:             return "Bratu";
        case PDE::ALLEN_CAHN:        return "Allen-Cahn";
        case PDE::LANE_EMDEN:        return "Lane-Emden";
        case PDE::TROESCH:           return "Troesch";
        case PDE::GINZBURG_LANDAU:   return "Ginzburg-Landau";
        case PDE::PAINLEVE1:         return "Painleve-I";
        default: return "Unknown";
    }
}
// ─── Tipos de Nodo del árbol ──────────────────────────────────────────────────
enum class NodeType {
    ADD, SUB, MUL, DIV, POW,
    SIN, COS, SINH, COSH, EXP, SQR, LOG, TANH,
    LEGENDRE, HERMITE, CHEBYSHEV, LAGUERRE,
    GAUSSIAN,
    VAR_X, VAR_Y, VAR_T, VAR_Z, VAR_N,
    ERC, CONST_I, CONST_PI, CONST_E,
    CONST_G, CONST_C, CONST_HBAR, CONST_KB, CONST_EPS0,
    SERIES,
    ROTATE,
    UNKNOWN
    };
// ─── Estructura Dual (Valor + Derivadas) para AD ──────────────────────────────
struct AD {
    Complex v;   // valor
    Complex dx, dy, dt;
    Complex dxx, dyy, dtt;
    
    AD(Complex val = 0.0) : v(val), dx(0), dy(0), dt(0), dxx(0), dyy(0), dtt(0) {}
};

// ─── Punto en el dominio ──────────────────────────────────────────────────────
struct Point {
    double x, y, t;
};

// ─── Clase base para individuo (para NSGA-II genérico) ────────────────────────
struct Individual {
    virtual ~Individual() = default;
    double mse_domain   = std::numeric_limits<double>::max();
    double mse_boundary = std::numeric_limits<double>::max();
    int    rank         = 0;
    double crowding     = 0.0;
    int    tree_size    = 0;
    NodeType root_type  = NodeType::UNKNOWN;

    // Restricciones NSGA-II (Deb, 2002)
    bool   is_feasible = true;
    double constraint_violation = 0.0; // 0 = feasible, >0 = infeasible
};

// ─── Estadísticas de convergencia por generación ─────────────────────────────
struct ConvergenceStats {
    int    gen;
    double best_mse_domain;   // Mejor MSE de dominio del batch de entrenamiento de esta generacion (ruidoso: mini-batch cambia cada gen)
    double best_mse_boundary; // Idem, frontera
    double best_total_mse;    // best_mse_domain + best_mse_boundary (mismo batch, mismo ruido)
    double best_val_mse;      // MSE del campeon del Hall of Fame contra la solucion exacta en grilla FIJA de validacion (senal confiable de progreso real)
};

// ─── Parámetros globales ──────────────────────────────────────────────────────
namespace Config {
    extern int    POP_SIZE;   
    extern int    MAX_GEN;   
    extern int    N_DOMAIN;  
    extern int    N_BOUNDARY;   
    extern double ERC_SIGMA;  
    extern int    MAX_TREE_DEPTH;
    
    extern double CROSSOVER_PROB;  
    extern double MUTATION_PROB;   
    extern int    TOURNAMENT_SIZE;    
    extern double STOP_THRESHOLD; 

    // Hybrid Committee RAR
    extern int    RAR_INTERVAL;      
    extern int    RAR_CANDIDATES;    
    extern double RAR_ADAPTIVE_RATIO;
    extern int    RAR_ELITE_COUNT;   
    extern double RAR_RANDOM_RATIO;  
    extern bool   GENERAL_MODE;

    // Si es false, se desactiva el ansatz de frontera exacta (U=L+B*N) — la
    // frontera vuelve a ser una penalizacion blanda real (compute_boundary_error)
    // en vez de forzarse a 0 por construccion, y el arbol resuelve la EDP
    // "a pelo". Distinto de GENERAL_MODE (que ademas apaga la frontera por
    // completo, ni siquiera como penalizacion blanda).
    extern bool   USE_ANSATZ;

    extern int    CORES;
    }