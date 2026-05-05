#pragma once
// =============================================================================
// tsoulos_ge.hpp  —  Método Tsoulos & Lagaris (2006) con Gramática Evolutiva (GE)
//   Genotipo: vector de enteros (codones)
//   Fenotipo: árbol de expresión (NodePtr) construido por la gramática BNF
//   Derivadas: diferencias finitas (NO simbólicas)
// =============================================================================

#include "common.hpp"
#include "tree_node.hpp"
#include "pde_problems.hpp"
#include <vector>
#include <random>

// ─── Individuo Tsoulos (GE) ──────────────────────────────────────────────────
struct TsoulosIndividual : public Individual {
    std::vector<int> codons;    // genotipo: secuencia de enteros
    NodePtr          tree;      // fenotipo: árbol generado por BNF

    TsoulosIndividual() = default;
    TsoulosIndividual(const TsoulosIndividual& other) : Individual(other), codons(other.codons) {
        if (other.tree) tree = other.tree->clone();
    }
    TsoulosIndividual(TsoulosIndividual&&) noexcept = default;
    TsoulosIndividual& operator=(const TsoulosIndividual& other) {
        if (this != &other) {
            Individual::operator=(other);
            codons = other.codons;
            tree = other.tree ? other.tree->clone() : nullptr;
        }
        return *this;
    }
    TsoulosIndividual& operator=(TsoulosIndividual&&) noexcept = default;

    // Mapea codones → árbol usando la gramática BNF
    void decode();

    // Evalúa MSE dominio + MSE frontera con diferencias finitas
    void evaluate(const PDEProblem& prob,
                  const std::vector<Point>& dom,
                  const std::vector<Point>& bnd);
};

// ─── Clase Algoritmo Tsoulos+NSGA-II ─────────────────────────────────────────
class TsoulosSolver {
public:
    explicit TsoulosSolver(const PDEProblem& prob, unsigned seed = 42);

    // Corre MAX_GEN generaciones de NSGA-II con representación BNF
    // Devuelve la población final ordenada por rango
    std::vector<TsoulosIndividual> run(int pop_size  = Config::POP_SIZE,
                                       int max_gen   = Config::MAX_GEN);

    // Retorna el frente de Pareto (rank == 1) de la última ejecución
    std::vector<TsoulosIndividual> pareto_front() const;

private:
    PDEProblem             prob_;
    std::mt19937           gen_;
    std::vector<Point>     dom_pts_;
    std::vector<Point>     bnd_pts_;
    std::vector<TsoulosIndividual> population_;

    TsoulosIndividual random_individual();
    TsoulosIndividual crossover(const TsoulosIndividual& a, const TsoulosIndividual& b);
    void              mutate(TsoulosIndividual& ind);
};

// ─── Función libre: gramática BNF → árbol ─────────────────────────────────────
// Construye un árbol de expresión siguiendo la gramática BNF de Tsoulos.
// Los codones se consumen circularmente (wrapping).
NodePtr bnf_to_tree(const std::vector<int>& codons,
                    int& idx, int depth, int max_depth);
