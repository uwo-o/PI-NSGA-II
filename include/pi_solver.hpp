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
                  
    // Evaluación robusta en rejilla fija para el Hall of Fame
    double get_validation_mse(const PDEProblem& prob, 
                              const std::vector<Point>& val_dom, 
                              const std::vector<Point>& val_bnd);
};

// ─── Solver de PI-NSGA-II ─────────────────────────────────────────────────────
class PISolver {
public:
    PISolver(const PDEProblem& prob, unsigned seed = 42);

    std::vector<PIIndividual> run(int pop_size, int max_gen);
    std::vector<PIIndividual> pareto_front() const;
    
    const std::vector<ConvergenceStats>& history() const { return history_; }

private:
    const PDEProblem& prob_;
    std::mt19937 gen_;
    std::vector<PIIndividual> population_;
    std::vector<ConvergenceStats> history_;

    // Puntos Dinámicos (Training)
    std::vector<Point> dom_pts_, bnd_pts_;
    std::vector<Point> val_dom_pts_, val_bnd_pts_;
    std::vector<Point> fixed_dom_pts_, fixed_bnd_pts_;
    
    // Hall of Fame (Elite Robusto)
    PIIndividual best_ever_;
    bool has_best_ever_ = false;
    int stagnation_counter_ = 0;
    int cataclysm_count_ = 0; // Para limitar a una sola inyección
    double last_best_mse_ = 1e18;

    PIIndividual random_individual();
    PIIndividual random_individual_special();
    PIIndividual make_offspring(const PIIndividual& a, const PIIndividual& b);
    void hill_climb_constants(PIIndividual& ind, int iterations);
    
    void update_hall_of_fame();
};
