#pragma once
// =============================================================================
// common.hpp  —  Tipos de datos y constantes globales
// =============================================================================

#include <vector>
#include <string>
#include <limits>

// ─── Tipos de PDE soportados ──────────────────────────────────────────────────
enum class PDE { LAPLACE, POISSON, HELMHOLTZ, SCHRODINGER };

inline std::string pde_name(PDE t) {
    switch (t) {
        case PDE::LAPLACE:     return "Laplace";
        case PDE::POISSON:     return "Poisson";
        case PDE::HELMHOLTZ:   return "Helmholtz";
        case PDE::SCHRODINGER: return "Schrodinger";
    }
    return "Unknown";
}

// ─── Tipos de Nodo del árbol ──────────────────────────────────────────────────
enum class NodeType {
    ADD, SUB, MUL, DIV,
    SIN, COS, SINH, COSH, EXP, SQR,
    VAR_X, VAR_Y, ERC,
    UNKNOWN
};

// ─── Estructura Dual (Valor + Derivadas) para AD ──────────────────────────────
struct AD {
    double v;   // valor
    double dx, dy;
    double dxx, dyy;
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
    constexpr int    POP_SIZE       = 300;   
    constexpr int    MAX_GEN        = 1000;   
    constexpr int    N_DOMAIN       = 300;   // Incrementado para dar más peso al residuo PDE
    constexpr int    N_BOUNDARY     = 150;   // Reducido para evitar sobreajuste a la frontera
    constexpr double ERC_SIGMA      = 0.25;  
    constexpr int    MAX_TREE_DEPTH = 7;     // Reducido para evitar bloat e incrementar velocidad
    constexpr int    CODON_LENGTH   = 64;    
    constexpr double CROSSOVER_PROB = 0.85;  
    constexpr double MUTATION_PROB  = 0.4;  // .3 
    constexpr double PI_ALPHA       = 0.4;   // Balance Frontera (alpha) vs PDE (beta = 1-alpha)
    constexpr int    TOURNAMENT_SIZE = 4;    // Presión de selección aumentada
    constexpr double STOP_THRESHOLD  = 1e-7; // Alta precisión analítica
}