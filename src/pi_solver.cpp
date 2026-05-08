// =============================================================================
// pi_solver.cpp  —  Método Propuesto: PI-NSGA-II con AD simbólico exacto
// =============================================================================

#include "pi_solver.hpp"
#include "nsga2.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// ─── PIIndividual::evaluate ───────────────────────────────────────────────────
void PIIndividual::evaluate(const PDEProblem& prob,
                            const std::vector<Point>& dom,
                            const std::vector<Point>& bnd)
{
    if (!tree) { mse_domain = 1e10; mse_boundary = 1e10; tree_size = 0; root_type = NodeType::UNKNOWN; return; }
    tree_size = tree->count_nodes();
    root_type = tree->get_type();

    // MSE dominio — residuo del PDE con AD simbólico exacto y pesado espacial
    double sum_dom = 0.0;
    double total_w = 0.0;
    for (auto& p : dom) {
        AD ad = tree->ad_eval(p.x, p.y);
        double res = prob.pde_residual_ad(ad, p.x, p.y);
        if (!std::isfinite(res)) { mse_domain = 1e10; mse_boundary = 1e10; return; }
        
        // Pesado espacial: más peso cerca de los bordes para evitar "coincidir solo al centro"
        double dist_x = std::min(p.x, 1.0 - p.x);
        double dist_y = std::min(p.y, 1.0 - p.y);
        double min_dist = std::min(dist_x, dist_y);
        double weight = 1.0 / (min_dist + 0.1); 
        
        sum_dom += weight * res * res;
        total_w += weight;
    }
    mse_domain = (sum_dom / total_w); 

    // --- PENALIZACIÓN DIMENSIONAL Y DE COMPLEJIDAD (2D) ---
    if (prob.dim == 2) {
        bool has_x = tree->uses_variable(NodeType::VAR_X);
        bool has_y = tree->uses_variable(NodeType::VAR_Y);
        if (!has_x || !has_y) { mse_domain *= 100.0; }
        if (tree_size < 5) { mse_domain *= 10.0; }
    }

    // MSE frontera — diferencia con condición de Dirichlet
    double sum_bnd = 0.0;
    for (auto& p : bnd) {
        double u    = tree->eval(p.x, p.y);
        double diff = u - prob.bc(p.x, p.y);
        if (!std::isfinite(diff)) { mse_boundary = 1e10; return; }
        sum_bnd += diff * diff;
    }
    
    double raw_bc_mse = sum_bnd / (double)bnd.size();
    mse_boundary = raw_bc_mse * Config::BC_WEIGHT;
    
    // Penalización de anclaje continua
    mse_domain += 100.0 * raw_bc_mse; 
}

// ─── PISolver ─────────────────────────────────────────────────────────────────
PISolver::PISolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed)
{
    // Generar puntos iniciales
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
}

PIIndividual PISolver::random_individual() {
    PIIndividual ind;
    ind.tree = random_tree(Config::MAX_TREE_DEPTH, gen_);
    ind.evaluate(prob_, dom_pts_, bnd_pts_);
    return ind;
}

PIIndividual PISolver::make_offspring(const PIIndividual& a, const PIIndividual& b) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    PIIndividual child;
    if (prob(gen_) < Config::CROSSOVER_PROB) {
        auto [c1, c2] = tree_crossover(a.tree, b.tree, gen_);
        child.tree = std::move(c1);
    } else {
        child.tree = a.tree->clone();
    }
    if (prob(gen_) < Config::MUTATION_PROB)
        child.tree = tree_mutate(child.tree, gen_);
    if (prob(gen_) < 0.4) 
        child.tree->mutate_erc(gen_, Config::ERC_SIGMA);
    return child;
}

std::vector<PIIndividual> PISolver::run(int pop_size, int max_gen) {
    population_.clear();
    history_.clear();
    population_.reserve(pop_size);
    for (int i = 0; i < pop_size; ++i)
        population_.push_back(random_individual());

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int g = 0; g < max_gen; ++g) {
        // --- REMUESTREO ALEATORIO UNIFORME ---
        // Generamos nuevos puntos cada generación para cubrir todo el dominio sin SA
        dom_pts_.clear();
        for(int i=0; i<Config::N_DOMAIN; ++i) {
            dom_pts_.push_back({dist(gen_), dist(gen_)});
        }
        // Re-evaluar población con los nuevos puntos para evitar "overfitting" a la muestra anterior
        for(auto& ind : population_) {
            ind.evaluate(prob_, dom_pts_, bnd_pts_);
        }

        std::vector<PIIndividual> offspring;
        offspring.reserve(pop_size);

        while ((int)offspring.size() < pop_size) {
            int p1 = tournament_select(population_, gen_);
            int p2 = tournament_select(population_, gen_);
            PIIndividual child = make_offspring(population_[p1], population_[p2]);
            hill_climb_constants(child, 20); 
            child.evaluate(prob_, dom_pts_, bnd_pts_);
            offspring.push_back(std::move(child));
        }

        std::vector<PIIndividual> combined;
        combined.reserve(pop_size * 2);
        for (auto& x : population_) combined.push_back(std::move(x)); 
        for (auto& x : offspring)   combined.push_back(std::move(x));
        population_ = nsga2_select_next(std::move(combined), pop_size);

        double b_dom = 1e18, b_bnd = 1e18, b_tot = 1e18;
        for (auto& ind : population_) {
            if (ind.rank == 1) {
                b_dom = std::min(b_dom, ind.mse_domain);
                b_bnd = std::min(b_bnd, ind.mse_boundary);
                b_tot = std::min(b_tot, ind.mse_domain + ind.mse_boundary);
            }
        }
        history_.push_back({g, b_dom, b_bnd, b_tot});

        if (g % 25 == 0) {
            std::cout << "  [PI/" << prob_.name() << "] gen=" << g
                      << "  best_dom=" << std::scientific << std::setprecision(3) << b_dom
                      << "  best_bnd=" << b_bnd << std::defaultfloat << "\n";
        }

        if (b_tot < Config::STOP_THRESHOLD) {
            std::cout << "  [PI/" << prob_.name() << "] Convergencia alcanzada en gen " << g 
                      << " (error " << std::scientific << std::setprecision(2) << b_tot 
                      << " < " << Config::STOP_THRESHOLD << ")\n" << std::defaultfloat;
            break;
        }
    }
    return std::move(population_);
}

std::vector<PIIndividual> PISolver::pareto_front() const {
    std::vector<PIIndividual> front;
    for (auto& ind : population_) {
        if (ind.rank == 1) {
            PIIndividual copy;
            copy.mse_domain = ind.mse_domain;
            copy.mse_boundary = ind.mse_boundary;
            copy.rank = ind.rank;
            copy.crowding = ind.crowding;
            copy.tree_size = ind.tree_size;
            copy.root_type = ind.root_type;
            if (ind.tree) copy.tree = ind.tree->clone();
            front.push_back(std::move(copy));
        }
    }
    return front;
}

void PISolver::hill_climb_constants(PIIndividual& ind, int iterations) {
    if (!ind.tree) return;
    std::vector<double*> ercs;
    ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;
    std::normal_distribution<double> noise(0.0, Config::ERC_SIGMA);
    std::uniform_int_distribution<int> select_erc(0, ercs.size() - 1);
    for (int i = 0; i < iterations; ++i) {
        double old_err = ind.mse_domain + ind.mse_boundary;
        int idx = select_erc(gen_);
        double old_val = *ercs[idx];
        *ercs[idx] += noise(gen_);
        ind.evaluate(prob_, dom_pts_, bnd_pts_);
        double new_err = ind.mse_domain + ind.mse_boundary;
        if (new_err > old_err) {
            *ercs[idx] = old_val;
            ind.evaluate(prob_, dom_pts_, bnd_pts_); 
        }
    }
}
