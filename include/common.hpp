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

// ─── Tipos de PDE soportados ──────────────────────────────────────────────────
enum class PDE { 
    LAPLACE, POISSON, HELMHOLTZ, SCHRODINGER, 
    NONLINEAR_POISSON, LIOUVILLE, SINE_GORDON,
    AIRY, HARMONIC_OSCILLATOR, GROSS_PITAEVSKII
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
        default: return "Unknown";
    }
}

// ─── Tipos de Nodo del árbol ──────────────────────────────────────────────────
enum class NodeType {
    ADD, SUB, MUL, DIV,
    SIN, COS, SINH, COSH, EXP, SQR,
    VAR_X, VAR_Y, ERC, CONST_I,
    UNKNOWN
};

// ─── Estructura Dual (Valor + Derivadas) para AD ──────────────────────────────
struct AD {
    Complex v;   // valor
    Complex dx, dy;
    Complex dxx, dyy;
    
    AD(Complex val = 0.0) : v(val), dx(0), dy(0), dxx(0), dyy(0) {}
};

// ─── Punto en el dominio ──────────────────────────────────────────────────────
struct Point {
    double x, y;
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
};

// ─── Estadísticas de convergencia por generación ─────────────────────────────
struct ConvergenceStats {
    int    gen;
    double best_mse_domain;
    double best_mse_boundary;
    double best_total_mse;
};

// ─── Parámetros globales ──────────────────────────────────────────────────────
namespace Config {
    constexpr int    POP_SIZE       = 500;   
    constexpr int    MAX_GEN        = 500;   
    constexpr int    N_DOMAIN       = 600;   
    constexpr int    N_BOUNDARY     = 200;   
    constexpr double ERC_SIGMA      = 0.20;  
    constexpr int    MAX_TREE_DEPTH = 8;    
    constexpr int    CODON_LENGTH   = 64;    
    constexpr double CROSSOVER_PROB = 0.80;  
    constexpr double MUTATION_PROB  = 0.3;  
    constexpr double PI_ALPHA       = 0.5;   
    constexpr int    TOURNAMENT_SIZE = 3;    
    constexpr double STOP_THRESHOLD  = 1e-7; // Alta precisión analítica
}