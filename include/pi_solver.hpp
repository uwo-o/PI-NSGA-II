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

    // dom_weights (opcional, mismo tamaño y orden que dom): peso multiplicativo
    // por punto en la suma del residuo — inspirado en el analisis NTK de
    // PINNs (Wang et al. 2022): puntos donde el residuo tarda mas en bajar a
    // lo largo de generaciones reciben mas peso, en vez de promediar todo
    // parejo (lo que "aplana" la señal justo donde mas importa, ej. cerca de
    // una singularidad). Ver PISolver::point_weights_ / update_point_weights().
    void evaluate(const PDEProblem& prob,
                  const std::vector<Point>& dom,
                  const std::vector<Point>& bnd,
                  int current_gen = 100,
                  const std::vector<double>* dom_weights = nullptr);
                  
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

    // Snapshot de la poblacion final tomado ANTES del pulido de constantes
    // por gradiente de esa misma generacion (ver comentario en run(), justo
    // antes del loop "Pulido de constantes para TODA la poblacion"). El
    // pulido converge a todos los individuos de un mismo tree_size al mismo
    // optimo local de mse_domain, lo que colapsa el frente de Pareto visual
    // a un puñado de clusters — este snapshot preserva la diversidad genuina
    // de mse_domain dentro de cada tamaño, util para graficar el frente
    // "crudo" tal como emergio de la evolucion.
    const std::vector<PIIndividual>& unpolished_final_population() const { return unpolished_snapshot_; }

    // Archivo historico: estadisticas (mse_domain, mse_boundary, tree_size,
    // rank) de CADA individuo de CADA generacion, no solo la poblacion final.
    // Motivacion: incluso guardando toda la poblacion final (150 individuos,
    // todos los ranks — ver save_pareto_csv en main.cpp, ya no filtra a
    // rank==1), sigue siendo una sola generacion; el pulido de constantes por
    // gradiente converge a los individuos de un mismo tree_size al mismo
    // optimo local (ver unpolished_final_population arriba), lo que colapsa
    // la variedad visual incluso ahi. Generaciones anteriores, antes de
    // converger del todo, tienen combinaciones (tree_size, mse_domain)
    // genuinamente distintas que ya no sobreviven a la generacion final —
    // acumularlas todas da muchos mas puntos reales para graficar el
    // trade-off precision/complejidad. Solo se guardan las 4 estadisticas
    // numericas (no el arbol) para que el costo extra sea despreciable.
    struct ArchiveEntry { int gen; double mse_domain; double mse_boundary; int tree_size; int rank; };
    const std::vector<ArchiveEntry>& population_archive() const { return population_archive_; }

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
    std::vector<PIIndividual> unpolished_snapshot_;
    std::vector<ArchiveEntry> population_archive_;
    std::vector<ConvergenceStats> history_;

    // Puntos Dinámicos (Training)
    std::vector<Point> dom_pts_, bnd_pts_;
    std::vector<Point> val_dom_pts_, val_bnd_pts_;

    // Hall of Fame (Elite Robusto)
    PIIndividual best_ever_;
    bool has_best_ever_ = false;
    int current_gen_ = 0;

    // Si el ansatz de frontera exacta aplica, mse_boundary es constante (0)
    // para todo individuo — se excluye como objetivo de NSGA-II (dominancia +
    // crowding distance) en vez de dejarlo empatado, ver comentario junto a
    // fast_non_dominated_sort en pi_solver.cpp. Calculado una vez en run().
    bool use_boundary_objective_ = true;
    int max_gen_ = 0;

    // Peso adaptativo por punto de dominio (mismo tamaño/orden que dom_pts_),
    // inspirado en el analisis NTK de PINNs (Wang et al. 2022) — ver
    // update_point_weights(). Complementa RAR (que decide QUE puntos entran
    // al batch): esto decide CUANTO pesa cada uno dentro del batch actual.
    std::vector<double> point_weights_;
    void update_point_weights(const std::vector<Point>& pts);

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
