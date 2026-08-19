#pragma once
#include "common.hpp"
#include "pde_problems.hpp"
#include "tree_node.hpp"
#include <random>
#include <memory>

// ─── Individuo de PISR-NSGA-II ──────────────────────────────────────────────────
struct PIIndividual : public Individual {
    NodePtr tree;
    double sample_dom_variance = 0.0;

    void evaluate(const PDEProblem& prob, 
                  const std::vector<Point>& dom, 
                  const std::vector<Point>& bnd,
                  int current_gen = 100);
                  
    // Evaluación robusta en rejilla fija para el Hall of Fame
    double get_validation_mse(const PDEProblem& prob, 
                             const std::vector<Point>& val_dom, 
                             const std::vector<Point>& val_bnd);

    bool is_physically_complete(const PDEProblem& prob) const;
    };
// ─── Solver de PISR-NSGA-II ─────────────────────────────────────────────────────
class PISolver {
public:
    PISolver(const PDEProblem& prob, unsigned seed = 42);

    std::vector<PIIndividual> run(int pop_size, int max_gen);
    std::vector<PIIndividual> pareto_front() const;
    
    const std::vector<ConvergenceStats>& history() const { return history_; }

    // Campeon del Hall of Fame: unica fuente de verdad para "el mejor
    // individuo de la corrida", ya validado contra la grilla fija (no un
    // scan del mse de entrenamiento de la poblacion final, que puede estar
    // contaminado por overfitting al mini-batch de la ultima generacion).
    bool has_champion() const { return has_best_ever_; }
    const PIIndividual& champion() const { return best_ever_; }

private:
    PDEProblem prob_;
    PDEPriors priors_;
    std::mt19937 gen_;
    std::vector<PIIndividual> population_;
    std::vector<ConvergenceStats> history_;

    // Puntos Dinámicos (Training)
    std::vector<Point> dom_pts_, bnd_pts_;
    std::vector<Point> val_dom_pts_, val_bnd_pts_;

    // Hall of Fame (Elite Robusto)
    PIIndividual best_ever_;
    bool has_best_ever_ = false;
    int current_gen_ = 0;
    int max_gen_ = 0;

    PIIndividual random_individual();
    PIIndividual random_individual_special();
    PIIndividual make_offspring(const PIIndividual& a, const PIIndividual& b);
    void hill_climb_constants(PIIndividual& ind, int iterations, std::mt19937& thread_gen);
    void gradient_descent_constants(PIIndividual& ind, int iterations,
                                     const std::vector<Point>* dom = nullptr,
                                     const std::vector<Point>* bnd = nullptr);
    void nelder_mead_polish(PIIndividual& ind, int max_iter = 500);
    void differential_evolution_polish(PIIndividual& ind, int max_iter = 100);
public:
    void polish_constants(PIIndividual& ind); // New high-precision polisher

    // Error de SOLUCION (vs. verdad conocida/numerica) en la grilla fija de
    // validacion, para cualquier individuo (tipicamente el campeon ya pulido).
    // Distinto de ind.mse_domain, que es el residuo de la EDP (fisica, no
    // supervisado) — ver comentario en compute_stats() en main.cpp.
    double solution_mse(PIIndividual& ind) { return ind.get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_); }
    
private:
    // Hybrid Committee RAR
    void apply_committee_rar();
    Point generate_random_point();

    void update_hall_of_fame();
};
