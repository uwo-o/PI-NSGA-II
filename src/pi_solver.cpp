// =============================================================================
// pi_solver.cpp  —  Método Propuesto: PI-NSGA-II con Elitismo Robusto
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

    double sum_dom = 0.0, total_w = 0.0;
    for (auto& p : dom) {
        AD ad = tree->ad_eval(p.x, p.y);
        double res = prob.pde_residual_ad(ad, p.x, p.y);
        if (!std::isfinite(res)) { mse_domain = 1e10; mse_boundary = 1e10; return; }
        double weight = 1.0 / (std::min(std::min(p.x, 1.0-p.x), std::min(p.y, 1.0-p.y)) + 0.1); 
        sum_dom += weight * res * res;
        total_w += weight;
    }
    mse_domain = (sum_dom / total_w); 

    if (prob.dim == 2) {
        if (!tree->uses_variable(NodeType::VAR_X) || !tree->uses_variable(NodeType::VAR_Y)) mse_domain *= 100.0;
        if (tree_size < 5) mse_domain *= 10.0;
    }

    double sum_bnd = 0.0;
    for (auto& p : bnd) {
        double u = tree->eval(p.x, p.y);
        double diff = u - prob.bc(p.x, p.y);
        if (!std::isfinite(diff)) { mse_boundary = 1e10; return; }
        sum_bnd += diff * diff;
    }
    double raw_bc_mse = sum_bnd / (double)bnd.size();
    mse_boundary = raw_bc_mse * Config::BC_WEIGHT;
    mse_domain += 100.0 * raw_bc_mse; 
}

double PIIndividual::get_validation_mse(const PDEProblem& prob, 
                                        const std::vector<Point>& val_dom, 
                                        const std::vector<Point>& val_bnd) 
{
    if (!tree) return 1e18;
    double sum_dom = 0.0;
    for (auto& p : val_dom) {
        AD ad = tree->ad_eval(p.x, p.y);
        double res = prob.pde_residual_ad(ad, p.x, p.y);
        if (!std::isfinite(res)) return 1e18;
        sum_dom += res * res;
    }
    double sum_bnd = 0.0;
    for (auto& p : val_bnd) {
        double diff = tree->eval(p.x, p.y) - prob.bc(p.x, p.y);
        sum_bnd += diff * diff;
    }
    return (sum_dom / val_dom.size()) + (sum_bnd / val_bnd.size());
}

// ─── PISolver ─────────────────────────────────────────────────────────────────
PISolver::PISolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed)
{
    // Puntos de entrenamiento iniciales
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);

    // REJILLA DE VALIDACIÓN FIJA (Para el Elitismo Robusto)
    // Usamos 400 puntos (20x20) fijos para medir la calidad real sin ruido.
    int n_val = (prob.dim == 1) ? 200 : 400;
    val_dom_pts_ = prob_.domain_points(n_val); 
    val_bnd_pts_ = prob_.boundary_points(100);
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

void PISolver::update_hall_of_fame() {
    for (auto& ind : population_) {
        if (ind.rank == 1) {
            double v_mse = ind.get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_);
            if (!has_best_ever_ || v_mse < best_ever_.mse_domain) { // Usamos mse_domain para guardar el v_mse
                best_ever_.tree = ind.tree->clone();
                best_ever_.mse_domain = v_mse; // Guardamos el valor de validación aquí
                has_best_ever_ = true;
            }
        }
    }
}

std::vector<PIIndividual> PISolver::run(int pop_size, int max_gen) {
    population_.clear();
    history_.clear();
    has_best_ever_ = false;

    for (int i = 0; i < pop_size; ++i)
        population_.push_back(random_individual());

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int g = 0; g < max_gen; ++g) {
        // 1. Remuestreo aleatorio
        dom_pts_.clear();
        for(int i=0; i<Config::N_DOMAIN; ++i) 
            dom_pts_.push_back({dist(gen_), dist(gen_)});
        
        for(auto& ind : population_) ind.evaluate(prob_, dom_pts_, bnd_pts_);

        // 2. Selección NSGA-II
        // (Ya se hizo al final de la gen anterior, excepto en la Gen 0)

        // 3. Actualizar Hall of Fame (cada generación para máxima seguridad)
        update_hall_of_fame();

        // 4. Crear descendencia
        std::vector<PIIndividual> offspring;
        offspring.reserve(pop_size);
        
        // Inyectar el MEJOR GLOBAL (Elitismo Robusto)
        if (has_best_ever_) {
            PIIndividual elite;
            elite.tree = best_ever_.tree->clone();
            elite.evaluate(prob_, dom_pts_, bnd_pts_);
            offspring.push_back(std::move(elite));
        }

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

        // Estadísticas (basadas en el mejor del Hall of Fame para estabilidad)
        double b_tot = has_best_ever_ ? best_ever_.mse_domain : 1e18;
        history_.push_back({g, 0.0, 0.0, b_tot});

        if (g % 25 == 0) {
            std::cout << "  [PI/" << prob_.name() << "] gen=" << g
                      << "  val_mse=" << std::scientific << std::setprecision(3) << b_tot 
                      << " (HoF)" << std::defaultfloat << "\n";
        }

        if (b_tot < Config::STOP_THRESHOLD) {
            std::cout << "  [PI/" << prob_.name() << "] Convergencia alcanzada (HoF error " 
                      << b_tot << " < " << Config::STOP_THRESHOLD << ")\n";
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
