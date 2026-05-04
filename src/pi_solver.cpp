// =============================================================================
// pi_solver.cpp  —  Método Propuesto: PI-NSGA-II con AD simbólico exacto
// =============================================================================

#include "pi_solver.hpp"
#include "nsga2.hpp"
#include <iostream>
#include <algorithm>

// ─── PIIndividual::evaluate ───────────────────────────────────────────────────
void PIIndividual::evaluate(const PDEProblem& prob,
                            const std::vector<Point>& dom,
                            const std::vector<Point>& bnd)
{
    if (!tree) { mse_domain = 1e10; mse_boundary = 1e10; tree_size = 0; root_type = NodeType::UNKNOWN; return; }
    tree_size = tree->count_nodes();
    root_type = tree->get_type();

    // MSE dominio — residuo del PDE con AD simbólico exacto y pesado espacial (Alpha)
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
        double weight = 1.0 / (min_dist + 0.1); // El peso aumenta cerca de los bordes (dist -> 0)
        
        sum_dom += weight * res * res;
        total_w += weight;
    }
    mse_domain = Config::ALPHA_WEIGHT * (sum_dom / total_w);

    // MSE frontera — diferencia con condición de Dirichlet (Beta = 1 - Alpha)
    double sum_bnd = 0.0;
    for (auto& p : bnd) {
        double u    = tree->eval(p.x, p.y);
        double diff = u - prob.bc(p.x, p.y);
        if (!std::isfinite(diff)) { mse_boundary = 1e10; return; }
        sum_bnd += diff * diff;
    }
    mse_boundary = (1.0 - Config::ALPHA_WEIGHT) * (sum_bnd / (double)bnd.size());
}

// ─── PISolver ─────────────────────────────────────────────────────────────────
PISolver::PISolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed)
{
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
        child.tree = std::move(c1); // tomamos el primer hijo
    } else {
        child.tree = a.tree->clone();
    }
    // Mutación
    if (prob(gen_) < Config::MUTATION_PROB)
        child.tree = tree_mutate(child.tree, gen_);
    // Mutación paramétrica ERC con probabilidad adicional
    if (prob(gen_) < 0.3)
        child.tree->mutate_erc(gen_, Config::ERC_SIGMA);
    return child;
}

std::vector<PIIndividual> PISolver::run(int pop_size, int max_gen) {
    // Inicialización
    population_.clear();
    population_.reserve(pop_size);
    for (int i = 0; i < pop_size; ++i)
        population_.push_back(random_individual());

    // Bucle generacional NSGA-II
    for (int g = 0; g < max_gen; ++g) {
        std::vector<PIIndividual> offspring;
        offspring.reserve(pop_size);

        while ((int)offspring.size() < pop_size) {
            int p1 = tournament_select(population_, gen_);
            int p2 = tournament_select(population_, gen_);
            PIIndividual child = make_offspring(population_[p1], population_[p2]);
            child.evaluate(prob_, dom_pts_, bnd_pts_);
            offspring.push_back(std::move(child));
        }

        std::vector<PIIndividual> combined;
        combined.reserve(pop_size * 2);
        for (auto& x : population_) combined.push_back(std::move(x)); // unique_ptr must move!
        for (auto& x : offspring)   combined.push_back(std::move(x));
        population_ = nsga2_select_next(std::move(combined), pop_size);

        if (g % 25 == 0) {
            auto& best = population_.front();
            std::cout << "  [PI/"   << prob_.name() << "] gen=" << g
                      << "  dom="   << best.mse_domain
                      << "  bnd="   << best.mse_boundary << "\n";
        }
    }
    return population_;
}

std::vector<PIIndividual> PISolver::pareto_front() const {
    std::vector<PIIndividual> front;
    for (auto& ind : population_)
        if (ind.rank == 1) front.push_back(ind);
    return front;
}
