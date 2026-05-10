// =============================================================================
// pi_solver.cpp  —  PI-NSGA-II con Elitismo Robusto y Log Estándar
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
    if (!tree) return;
    tree_size = tree->count_nodes();

    // 1. Residuo PDE (Domain) - Usa AD con dimensión optimizada
    double sum_dom = 0.0;
    double total_w = 0.0;
    int dim = prob.dim;
    for (auto& p : dom) {
        AD ad = tree->ad_eval(p.x, p.y, dim);
        double res = prob.pde_residual_ad(ad, p.x, p.y);
        if (!std::isfinite(res)) { mse_domain = 1e10; mse_boundary = 1e10; return; }
        
        double dist = std::min({p.x, 1.0-p.x, p.y, 1.0-p.y});
        double weight = 1.0 / (dist + 0.1); 
        sum_dom += weight * res * res;
        total_w += weight;
    }
    double pde_mse = (sum_dom / total_w); 

    // 2. MSE Frontera Estándar - Usa eval() rápido (sin derivadas)
    double sum_bnd = 0.0;
    for (auto& p : bnd) {
        double u_val = tree->eval(p.x, p.y);
        double bc_val = prob.bc(p.x, p.y);
        double diff = u_val - bc_val;
        if (!std::isfinite(diff)) { mse_boundary = 1e10; return; }
        sum_bnd += diff * diff;
    }
    double raw_bc_mse = sum_bnd / (double)bnd.size();

    double alpha = Config::PI_ALPHA;
    double beta  = 1.0 - alpha;

    mse_domain = beta * pde_mse + alpha * raw_bc_mse; 
    mse_boundary = raw_bc_mse; 
}

double PIIndividual::get_validation_mse(const PDEProblem& prob, 
                                        const std::vector<Point>& val_dom, 
                                        const std::vector<Point>& val_bnd) 
{
    if (!tree) return 1e18;
    double sum_dom = 0.0;
    for (auto& p : val_dom) {
        AD ad = tree->ad_eval(p.x, p.y, prob.dim);
        double res = prob.pde_residual_ad(ad, p.x, p.y);
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
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
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

PIIndividual PISolver::random_individual_special() {
    PIIndividual ind;
    ind.tree = random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_.dim);
    ind.evaluate(prob_, dom_pts_, bnd_pts_);
    // Refuerzo agresivo: 100 iteraciones de ajuste de constantes al nacer
    hill_climb_constants(ind, 100); 
    return ind;
}

PIIndividual PISolver::make_offspring(const PIIndividual& a, const PIIndividual& b) {
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    PIIndividual child;
    bool is_elite = (a.rank == 1 || b.rank == 1);

    if (p_dist(gen_) < Config::CROSSOVER_PROB) {
        auto [c1, c2] = tree_crossover(a.tree, b.tree, gen_);
        child.tree = std::move(c1);
    } else {
        child.tree = a.tree->clone();
    }

    if (is_elite) {
        // Individuos élite: Mutación de parámetros para ajuste fino (70%) vs Estructural (30%)
        if (p_dist(gen_) < 0.7) child.tree->mutate_erc(gen_, Config::ERC_SIGMA * 0.5); 
        else                    child.tree = tree_mutate(child.tree, gen_);
    } else {
        // Individuos normales: Mutación estructural estándar
        if (p_dist(gen_) < Config::MUTATION_PROB)
            child.tree = tree_mutate(child.tree, gen_);
        if (p_dist(gen_) < 0.4) 
            child.tree->mutate_erc(gen_, Config::ERC_SIGMA);
    }
    
    return child;
}

void PISolver::update_hall_of_fame() {
    for (auto& ind : population_) {
        if (ind.rank == 1) {
            double v_mse = ind.get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_);
            if (!has_best_ever_ || v_mse < best_ever_.mse_domain) { 
                best_ever_.tree = ind.tree->clone();
                best_ever_.mse_domain = v_mse; 
                has_best_ever_ = true;
            }
        }
    }
}

std::vector<PIIndividual> PISolver::run(int pop_size, int max_gen) {
    population_.clear();
    history_.clear();
    has_best_ever_ = false;
    cataclysm_count_ = 0;
    stagnation_counter_ = 0;

    int n_special = static_cast<int>(0.2 * pop_size);
    for (int i = 0; i < pop_size; ++i) {
        if (i < n_special) population_.push_back(random_individual_special());
        else               population_.push_back(random_individual());
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Generación Única de Puntos (Estáticos para máxima estabilidad)
    dom_pts_.clear();
    bnd_pts_.clear();
    for(int i=0; i<Config::N_DOMAIN; ++i) dom_pts_.push_back({dist(gen_), dist(gen_)});
    for(int i=0; i<Config::N_BOUNDARY; ++i) bnd_pts_.push_back({dist(gen_), dist(gen_)});

    for (int g = 0; g < max_gen; ++g) {
        // 1. Verificar Estancamiento
        if (has_best_ever_) {
            if (best_ever_.mse_domain < last_best_mse_ * 0.999) { // Mejora significativa (>0.1%)
                last_best_mse_ = best_ever_.mse_domain;
                stagnation_counter_ = 0;
            } else {
                stagnation_counter_++;
            }
        }

        // 2. Ejecutar Cataclismo (Reinicio Suave) si hay estancamiento (50 gens)
        // 2. Ejecutar Cataclismo (Reinicio Suave) si hay estancamiento (100 gens)
        // 2. Ejecutar Cataclismo (Reinicio Suave) o Early Stop
        if (stagnation_counter_ >= 100) {
            if (cataclysm_count_ < 1) {
                std::cout << "  [!] Cataclismo: Estancamiento detectado (" << stagnation_counter_ 
                          << " gens). Reinyectando diversidad especializada (Intento 1/1)...\n";
                
                std::vector<PIIndividual> next_pop;
                for (auto& ind : population_) if (ind.rank == 1) next_pop.push_back(std::move(ind));
                
                while ((int)next_pop.size() < pop_size) {
                    PIIndividual new_ind = random_individual_special();
                    hill_climb_constants(new_ind, 200); 
                    next_pop.push_back(std::move(new_ind));
                }
                population_ = std::move(next_pop);
                cataclysm_count_++;
                stagnation_counter_ = 0; 
            } else {
                std::cout << "  [!] Early Stop: Sin mejoras tras inyección de genes. Terminando en gen " << g << ".\n";
                break; // Detener ejecución completamente
            }
        }

        // Eliminamos la re-evaluación global. Los individuos ya vienen evaluados.
        update_hall_of_fame();

        std::vector<PIIndividual> offspring;
        offspring.reserve(pop_size);
        
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
            child.evaluate(prob_, dom_pts_, bnd_pts_); // Evaluamos solo al nacer
            offspring.push_back(std::move(child));
        }

        std::vector<PIIndividual> combined;
        combined.reserve(pop_size * 2);
        for (auto& x : population_) combined.push_back(std::move(x)); 
        for (auto& x : offspring)   combined.push_back(std::move(x));
        population_ = nsga2_select_next(std::move(combined), pop_size);
        
        // Pulido de Élite: Solo en los top sujetos (Rank 1) y con más intensidad
        for (auto& ind : population_) {
            if (ind.rank == 1) {
                hill_climb_constants(ind, 50); // Más iteraciones pero solo a los mejores
            }
        }

        // b_tot para parada: el mínimo entre el mejor validado y el mejor de entrenamiento actual
        double current_best_train = 1e18;
        for (auto& ind : population_) {
            if (ind.rank == 1) current_best_train = std::min(current_best_train, ind.mse_domain);
        }

        double b_tot = has_best_ever_ ? best_ever_.mse_domain : current_best_train;
        history_.push_back({g, 0.0, 0.0, b_tot});

        if (g % 25 == 0) {
            double b_dom = 1e18, b_bnd = 1e18;
            for (auto& ind : population_) {
                if (ind.rank == 1) {
                    b_dom = std::min(b_dom, ind.mse_domain);
                    b_bnd = std::min(b_bnd, ind.mse_boundary);
                }
            }
            double v_mse = has_best_ever_ ? best_ever_.mse_domain : 1e18;
            std::cout << "  [PI/" << prob_.name() << "] gen=" << g
                      << "  best_dom=" << std::scientific << std::setprecision(3) << b_dom
                      << "  best_bnd=" << b_bnd 
                      << "  (val_best=" << v_mse << ")" << std::defaultfloat << "\n";
        }

        if (b_tot < Config::STOP_THRESHOLD) break;
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
    
    // Optimización: Hill Climbing con Mini-Batch (Submuestreo)
    // Usamos solo un 10% de los puntos para acelerar el ajuste de constantes
    std::vector<Point> mini_dom, mini_bnd;
    std::sample(dom_pts_.begin(), dom_pts_.end(), std::back_inserter(mini_dom), 40, gen_);
    std::sample(bnd_pts_.begin(), bnd_pts_.end(), std::back_inserter(mini_bnd), 20, gen_);

    std::normal_distribution<double> noise(0.0, Config::ERC_SIGMA);
    std::uniform_int_distribution<int> select_erc(0, (int)ercs.size() - 1);

    for (int i = 0; i < iterations; ++i) {
        int idx_target = select_erc(gen_);
        double old_val = *ercs[idx_target];

        // Evaluación previa en mini-batch
        ind.evaluate(prob_, mini_dom, mini_bnd);
        double old_dom = ind.mse_domain;
        double old_bnd = ind.mse_boundary;

        *ercs[idx_target] += noise(gen_);
        ind.evaluate(prob_, mini_dom, mini_bnd);

        // Si el nuevo estado es dominado por el viejo (es peor), revertimos
        bool old_dominates_new = (old_dom <= ind.mse_domain && old_bnd <= ind.mse_boundary) &&
                                 (old_dom < ind.mse_domain  || old_bnd < ind.mse_boundary);
        
        if (old_dominates_new) {
             *ercs[idx_target] = old_val;
        }
    }
    // Re-evaluación final con todos los puntos para mantener consistencia
    ind.evaluate(prob_, dom_pts_, bnd_pts_);
}
