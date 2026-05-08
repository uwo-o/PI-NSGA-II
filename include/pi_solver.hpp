#pragma once
#include "common.hpp"
#include "pde_problems.hpp"
#include "tree_node.hpp"
#include <random>
#include <memory>

// ─── Individuo de PI-NSGA-II ──────────────────────────────────────────────────
struct PIIndividual : public Individual {
    NodePtr tree;

    void evaluate(const PDEProblem& prob, 
                  const std::vector<Point>& dom, 
                  const std::vector<Point>& bnd);
};

// ─── Solver de PI-NSGA-II ─────────────────────────────────────────────────────
class PISolver {
public:
    PISolver(const PDEProblem& prob, unsigned seed = 42);

    std::vector<PIIndividual> run(int pop_size, int max_gen);
    std::vector<PIIndividual> pareto_front() const;
    
    // Estadísticas de convergencia
    const std::vector<ConvergenceStats>& history() const { return history_; }

private:
    const PDEProblem& prob_;
    std::mt19937 gen_;
    std::vector<PIIndividual> population_;
    std::vector<ConvergenceStats> history_;

    // Puntos de evaluación (Fijos)
    std::vector<Point> dom_pts_;
    std::vector<Point> bnd_pts_;

    // Operadores evolutivos
    PIIndividual random_individual();
    PIIndividual make_offspring(const PIIndividual& a, const PIIndividual& b);

    // Memetic Hill Climbing (Refinamiento de constantes)
    void hill_climb_constants(PIIndividual& ind, int iterations);
};
