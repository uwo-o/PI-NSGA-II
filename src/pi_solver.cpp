// =============================================================================
// pi_solver.cpp  —  PI-NSGA-II con Elitismo Robusto y Log Estándar
// =============================================================================

#include "pi_solver.hpp"
#include "nsga2.hpp"
#include "numerical_solver.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// ─── PIIndividual::evaluate ───────────────────────────────────────────────────
void PIIndividual::evaluate(const PDEProblem& prob, 
                          const std::vector<Point>& dom, 
                          const std::vector<Point>& bnd) 
{
    if (!tree) { mse_domain = 1e18; mse_boundary = 1e18; return; }
    
    double pde_mse = 0.0;
    for (auto& p : dom) {
        AD ad = tree->ad_eval(p.x, p.y, prob.dim);
        Complex res = prob.pde_residual_ad(ad, p.x, p.y);
        if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) {
            mse_domain = 1e18; mse_boundary = 1e18; return;
        }
        pde_mse += std::norm(res); 
    }
    if (!dom.empty()) pde_mse /= dom.size();

    double raw_bc_mse = 0.0;
    for (auto& p : bnd) {
        Complex val = tree->eval(p.x, p.y);
        Complex bc_val = prob.bc(p.x, p.y);
        Complex diff = val - bc_val;
        if (!std::isfinite(diff.real()) || !std::isfinite(diff.imag())) {
            mse_domain = 1e18; mse_boundary = 1e18; return;
        }
        raw_bc_mse += std::norm(diff);
    }
    if (!bnd.empty()) raw_bc_mse /= bnd.size();

    tree_size = tree->count_nodes();
    root_type = tree->get_type();

    double alpha = 20.0; 
    double beta = 1.0;
    mse_domain = beta * pde_mse + alpha * raw_bc_mse; 
    mse_boundary = raw_bc_mse; 
}

double PIIndividual::get_validation_mse(const PDEProblem& prob, 
                                        const std::vector<Point>& val_dom, 
                                        const std::vector<Point>& val_bnd) 
{
    if (!tree) return 1e18;
    
    // Si tenemos verdad numérica, comparamos contra ella (MSE directo)
    if (prob.is_numerical && !prob.numerical_truth.empty()) {
        double sum_val = 0.0;
        int n_pts = 0;
        if (prob.dim == 1) {
            // Usar interpolación: nunca indexamos numerical_truth directamente
            for (auto& pt : val_dom) {
                Complex u_approx = tree->eval(pt.x, pt.y);
                Complex u_exact  = prob.numerical_exact(pt.x, pt.y);
                if (!std::isfinite(u_approx.real())) continue;
                sum_val += std::norm(u_approx - u_exact);
                ++n_pts;
            }
            return (n_pts > 0) ? sum_val / n_pts : 1e18;
        } else {
            int N = (int)std::sqrt(prob.numerical_truth.size());
            double h = 1.0 / (N - 1);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    Complex val = tree->eval(i * h, j * h);
                    if (!std::isfinite(val.real())) continue;
                    sum_val += std::norm(val - prob.numerical_truth[i * N + j]);
                    ++n_pts;
                }
            }
            return (n_pts > 0) ? sum_val / n_pts : 1e18;
        }
    }

    double sum_dom = 0.0;
    for (auto& p : val_dom) {
        AD ad = tree->ad_eval(p.x, p.y, prob.dim);
        Complex res = prob.pde_residual_ad(ad, p.x, p.y);
        if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) return 1e18;
        sum_dom += std::norm(res);
    }
    double sum_bnd = 0.0;
    for (auto& p : val_bnd) {
        Complex val = tree->eval(p.x, p.y);
        Complex bc_val = prob.bc(p.x, p.y);
        Complex diff = val - bc_val;
        if (!std::isfinite(diff.real()) || !std::isfinite(diff.imag())) return 1e18;
        sum_bnd += std::norm(diff);
    }
    double r_dom = val_dom.empty() ? 0.0 : sum_dom / val_dom.size();
    double r_bnd = val_bnd.empty() ? 0.0 : sum_bnd / val_bnd.size();
    return r_dom + r_bnd;
}

// ─── PISolver ─────────────────────────────────────────────────────────────────
PISolver::PISolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed)
{
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
    int n_val = (prob.dim == 1) ? 200 : 400;
    val_dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    val_bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
}

PIIndividual PISolver::random_individual() {
    PIIndividual ind;
    ind.tree = random_tree(Config::MAX_TREE_DEPTH, gen_);
    ind.evaluate(prob_, dom_pts_, bnd_pts_);
    return ind;
}

PIIndividual PISolver::random_individual_special() {
    PIIndividual ind;
    ind.tree = random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_);
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
        // 1. Regenerar Puntos Dinámicos cada 20 generaciones (EVITA TRAMPAS Y MEMORIZACIÓN)
        if (g % 20 == 0) {
            dom_pts_.clear();
            bnd_pts_.clear();
            
            // Puntos de Dominio (Interior)
            for(int i=0; i<Config::N_DOMAIN; ++i) dom_pts_.push_back({dist(gen_), dist(gen_)});
            
            // Puntos de Frontera (Bordes Reales: x=0, x=1, y=0, y=1)
            int pts_per_side = Config::N_BOUNDARY / 4;
            for(int i=0; i<pts_per_side; ++i) {
                double t = dist(gen_);
                bnd_pts_.push_back({0.0, t}); // Izquierda
                bnd_pts_.push_back({1.0, t}); // Derecha
                bnd_pts_.push_back({t, 0.0}); // Abajo
                bnd_pts_.push_back({t, 1.0}); // Arriba
            }
            
            // Re-evaluar población con la nueva "física"
            for (auto& ind : population_) ind.evaluate(prob_, dom_pts_, bnd_pts_);
        }

        // 2. Hall of Fame y Estancamiento
        update_hall_of_fame();
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

        // ── Condición de parada ───────────────────────────────────────────────
        // Usamos el mejor MSE de ENTRENAMIENTO de la generación actual.
        // best_ever_.mse_domain almacena la validación (útil para exportar),
        // pero la parada debe basarse en el fitness de entrenamiento.
        double stop_threshold = Config::STOP_THRESHOLD;
        if (prob_.is_numerical) stop_threshold = 1e-4; // RK4 tiene error de discretización

        // Parada A: fitness de entrenamiento bajo el umbral
        if (current_best_train < stop_threshold) {
            std::cout << "  [!] Convergencia alcanzada (MSE=" << current_best_train
                      << "). Terminando en gen " << g << ".\n";
            break;
        }

        // Parada B: ambos componentes esencialmente cero (solución perfecta)
        double b_dom_cur = 1e18, b_bnd_cur = 1e18;
        for (auto& ind : population_) {
            if (ind.rank == 1) {
                b_dom_cur = std::min(b_dom_cur, ind.mse_domain);
                b_bnd_cur = std::min(b_bnd_cur, ind.mse_boundary);
            }
        }
        if (b_dom_cur < 1e-12 && b_bnd_cur < 1e-12) {
            std::cout << "  [!] Solución perfecta (dom=" << b_dom_cur
                      << ", bnd=" << b_bnd_cur << "). Terminando en gen " << g << ".\n";
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

// ─── Solucionador de Mínimos Cuadrados vía Householder QR ──────────────────────
static void solve_qr_internal(std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x) {
    int m = (int)A.size();    
    int n = (int)A[0].size(); 
    
    for (int k = 0; k < n; ++k) {
        double norm = 0;
        for (int i = k; i < m; ++i) norm += A[i][k] * A[i][k];
        norm = std::sqrt(norm);
        
        double s = (A[k][k] > 0) ? -norm : norm;
        double u1 = A[k][k] - s;
        
        std::vector<double> v(m, 0.0);
        v[k] = 1.0;
        for (int i = k + 1; i < m; ++i) {
            A[i][k] /= u1;
            v[i] = A[i][k];
        }
        A[k][k] = s;
        double tau = -u1 / s;
        
        for (int j = k + 1; j < n; ++j) {
            double sum = A[k][j];
            for (int i = k + 1; i < m; ++i) sum += v[i] * A[i][j];
            sum *= tau;
            A[k][j] -= sum;
            for (int i = k + 1; i < m; ++i) A[i][j] -= sum * v[i];
        }
        
        double sum_b = b[k];
        for (int i = k + 1; i < m; ++i) sum_b += v[i] * b[i];
        sum_b *= tau;
        b[k] -= sum_b;
        for (int i = k + 1; i < m; ++i) b[i] -= sum_b * v[i];
    }
    
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) sum -= A[i][j] * x[j];
        if (std::abs(A[i][i]) > 1e-12) x[i] = sum / A[i][i];
        else x[i] = 0.0;
    }
}

void PISolver::hill_climb_constants(PIIndividual& ind, int iterations) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs;
    ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    int n_pts_mini = 40; 
    std::vector<Point> mini_dom;
    std::sample(dom_pts_.begin(), dom_pts_.end(), std::back_inserter(mini_dom), n_pts_mini, gen_);
    
    const double epsilon = 1e-6;
    int n_ercs = ercs.size();
    int n_vars = n_ercs * 2; // Real e Imaginaria por cada ERC

    for (int iter = 0; iter < 2; ++iter) {
        std::vector<std::vector<double>> J(n_pts_mini * 2, std::vector<double>(n_vars));
        std::vector<double> r(n_pts_mini * 2);
        
        for (int i = 0; i < n_pts_mini; ++i) {
            Complex val_orig = ind.tree->eval(mini_dom[i].x, mini_dom[i].y);
            Complex target = prob_.exact(mini_dom[i].x, mini_dom[i].y);
            Complex residual = target - val_orig;
            
            if (!std::isfinite(residual.real()) || !std::isfinite(residual.imag())) return;

            r[i*2] = residual.real();
            r[i*2+1] = residual.imag();

            for (int j = 0; j < n_ercs; ++j) {
                Complex old_v = *ercs[j];
                // Derivada numérica respecto a la parte REAL
                *ercs[j] = old_v + epsilon;
                Complex val_plus = ind.tree->eval(mini_dom[i].x, mini_dom[i].y);
                Complex grad = (val_plus - val_orig) / epsilon;
                *ercs[j] = old_v;

                if (!std::isfinite(grad.real()) || !std::isfinite(grad.imag())) return;

                // Columna j*2: Sensibilidad a la parte REAL
                J[i*2][j*2]     = grad.real();
                J[i*2+1][j*2]   = grad.imag();

                // Columna j*2+1: Sensibilidad a la parte IMAGINARIA (Cauchy-Riemann)
                // d/d(Im) = i * d/d(Re)  =>  Re(i*grad) = -Im(grad), Im(i*grad) = Re(grad)
                J[i*2][j*2+1]   = -grad.imag();
                J[i*2+1][j*2+1] = grad.real();
            }
        }

        std::vector<double> delta_c;
        solve_qr_internal(J, r, delta_c);

        for (int j = 0; j < n_ercs; ++j) {
            double dr = delta_c[j*2];
            double di = delta_c[j*2+1];
            if (std::isfinite(dr) && std::isfinite(di)) {
                *ercs[j] += Complex(dr * 0.4, di * 0.4); // Salto amortiguado
            }
        }
    }
    ind.evaluate(prob_, dom_pts_, bnd_pts_);
}
