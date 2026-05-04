#pragma once
// =============================================================================
// common.hpp  —  Tipos base, parámetros globales y struct AD
// =============================================================================

#include <vector>
#include <string>
#include <cmath>
#include <limits>

// ─── PDE types ────────────────────────────────────────────────────────────────
enum class PDE { LAPLACE, POISSON, HELMHOLTZ, SCHRODINGER };

inline std::string pde_name(PDE p) {
    switch (p) {
        case PDE::LAPLACE:   return "Laplace";
        case PDE::POISSON:   return "Poisson";
        case PDE::HELMHOLTZ:   return "Helmholtz";
        case PDE::SCHRODINGER: return "Schrodinger";
    }
    return "Unknown";
}

// ─── 2-D point ────────────────────────────────────────────────────────────────
struct Point { double x, y; };

// ─── Tipos de nodo (NodeType) ─────────────────────────────────────────────────
enum class NodeType {
    VAR_X, VAR_Y, ERC,            // terminales
    ADD, SUB, MUL,                 // operadores binarios
    SIN,  COS,                     // trigonométricas
    SINH, COSH, TANH,              // hiperbólicas  (Laplace, difusión, onda)
    EXP,                           // exponencial   (calor, amortiguamiento)
    SQRT,                          // raíz cuadrada (Bessel, potencial)
    LOG,                           // log(|·|+ε)    (Laplace polar, Green)
    ATAN,                          // arctan        (Laplace polar)
    UNKNOWN                        // fallback
};

// ─── Automatic Differentiation value ─────────────────────────────────────────
// Holds u(x,y) and its first/second partials computed in a single tree pass.
struct AD {
    double v   = 0.0; // u
    double dx  = 0.0; // ∂u/∂x
    double dy  = 0.0; // ∂u/∂y
    double dxx = 0.0; // ∂²u/∂x²
    double dyy = 0.0; // ∂²u/∂y²
};

// ─── NSGA-II individual (method-agnostic) ─────────────────────────────────────
struct Individual {
    double mse_domain   = std::numeric_limits<double>::max();
    double mse_boundary = std::numeric_limits<double>::max();
    int    rank         = 0;
    double crowding     = 0.0;
    int    tree_size    = 0;
    NodeType root_type  = NodeType::UNKNOWN;
};

// ─── Parámetros globales ──────────────────────────────────────────────────────
namespace Config {
    constexpr int    POP_SIZE       = 120;   // +50% diversidad en frente de Pareto
    constexpr int    MAX_GEN        = 200;   // 2x generaciones → mejor convergencia
    constexpr int    N_DOMAIN       = 400;   // más puntos de colocación interiores
    constexpr int    N_BOUNDARY     = 80;    // mejor representación de ∂Ω
    constexpr double ERC_SIGMA      = 0.20;  // mayor exploración de constantes
    constexpr int    MAX_TREE_DEPTH = 7;    // expresiones más complejas
    constexpr int    CODON_LENGTH   = 64;    // genotipos BNF más largos
    constexpr double CROSSOVER_PROB  = 0.85;
    constexpr double MUTATION_PROB   = 0.15;
    constexpr double ALPHA_WEIGHT    = 0.15; // Peso para Dominio (Beta = 1 - Alpha para BC)
}
