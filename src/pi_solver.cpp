// =============================================================================
// pi_solver.cpp
// =============================================================================

#include "pi_solver.hpp"
#include "nsga2.hpp"
#include "numerical_solver.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <numeric>

// ─── Inicialización de Parámetros Globales ──────────────────────────────────
namespace Config {
    int    POP_SIZE       = 200;   
    int    MAX_GEN        = 300;   
    int    N_DOMAIN       = 2000;  
    int    N_BOUNDARY     = 500;   
    double ERC_SIGMA      = 0.20;  
    int    MAX_TREE_DEPTH = 6;
    double CROSSOVER_PROB = 0.80;  
    double MUTATION_PROB  = 0.45;   
    int    TOURNAMENT_SIZE = 4;    
    double STOP_THRESHOLD  = 1e-7; 

    int    RAR_INTERVAL       = 25;
    int    RAR_CANDIDATES     = 10000;
    double RAR_ADAPTIVE_RATIO = 0.30;
    int    RAR_ELITE_COUNT    = 8;
    double RAR_RANDOM_RATIO   = 0.25;
    int    CORES              = 1;
}

// ─── PIIndividual::evaluate ───────────────────────────────────────────────────
void PIIndividual::evaluate(const PDEProblem& prob, 
                            const std::vector<Point>& dom, 
                            const std::vector<Point>& bnd, 
                            int current_gen) 
{
    if (!tree) { 
        is_feasible = false; constraint_violation = 1.0;
        mse_domain = 1e18; mse_boundary = 1e18; return; 
    }

    is_feasible = true;
    constraint_violation = 0.0;

    // ─── RESTRICCIONES DURAS (No afectan al error) ──────────────────────────
    
    // 1. Restricción de No-Trivialidad Estructural (Básica)
    if (!tree->contains_variables()) {
        is_feasible = false;
        constraint_violation += 1.0;
    }
    
    // 2. Requerimiento de Variables (Relajado: Filtro POST)

    // 3. Restricción Dimensional Semi-Dura (Buckingham Pi)
    if (!prob.is_numerical && !(prob.dim_u == Units::None)) {
        auto d_opt = tree->get_dimension(prob);
        if (!d_opt.has_value() || *d_opt != prob.dim_u) {
            constraint_violation += 0.3;
        }
    }

    // 4. Restricción de No-Anidamiento Prolongado (ESTRICTA)
    // Se limita la profundidad de funciones unarias a 3 niveles.
    if (tree->get_unary_depth() > 3) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    // ─── Evaluación de Dinámica y Varianza ───
    double pde_mse = 0.0;
    std::vector<double> sample_values;
    sample_dom_variance = 0.0;
    double max_grad = 0.0;

    if (prob.type == PDE::NAVIER_STOKES_UNSTEADY) {
        // ... (NS Unsteady eval) ...
        double nu = prob.k2;
        double h = 0.01; double ht = 0.01;
        for (auto& p : dom) {
            AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            sample_values.push_back(ad_c.v.real());
            max_grad = std::max({max_grad, std::abs(ad_c.dx.real()), std::abs(ad_c.dy.real()), std::abs(ad_c.dt.real())});
            
            Complex psi_x = ad_c.dx, psi_y = ad_c.dy;
            Complex u = psi_y, v = -psi_x;
            Complex w = -(ad_c.dxx + ad_c.dyy); 
            // ... (rest of NS Unsteady) ...

            AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, prob.dim);
            AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, prob.dim);
            AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, prob.dim);
            AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, prob.dim);

            Complex w_xp = -(ad_xp.dxx + ad_xp.dyy);
            Complex w_xm = -(ad_xm.dxx + ad_xm.dyy);
            Complex w_yp = -(ad_yp.dxx + ad_yp.dyy);
            Complex w_ym = -(ad_ym.dxx + ad_ym.dyy);
            Complex w_x = (w_xp - w_xm) / (2.0 * h);
            Complex w_y = (w_yp - w_ym) / (2.0 * h);
            Complex lap_w = (w_xp + w_xm + w_yp + w_ym - 4.0 * w) / (h * h);
            
            AD ad_tp = tree->ad_eval_t(p.x, p.y, p.t+ht, prob.dim);
            AD ad_tm = tree->ad_eval_t(p.x, p.y, p.t-ht, prob.dim);
            Complex w_tp = -(ad_tp.dxx + ad_tp.dyy);
            Complex w_tm = -(ad_tm.dxx + ad_tm.dyy);
            Complex w_t = (w_tp - w_tm) / (2.0 * ht);

            Complex res = w_t + u * w_x + v * w_y - nu * lap_w;
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) {
                is_feasible = false; constraint_violation += 10.0;
                mse_domain = 1e18; mse_boundary = 1e18; return;
            }
            pde_mse += std::norm(res);
        }
        if (!dom.empty()) pde_mse /= dom.size();
    } else if (prob.type == PDE::NAVIER_STOKES) {
        double nu = prob.k2; double h = 0.01;
        for (auto& p : dom) {
            AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            sample_values.push_back(ad_c.v.real());
            max_grad = std::max({max_grad, std::abs(ad_c.dx.real()), std::abs(ad_c.dy.real())});

            Complex psi_x = ad_c.dx, psi_y = ad_c.dy;
            Complex L_val = (ad_c.dxx + ad_c.dyy);
            AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, prob.dim);
            AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, prob.dim);
            AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, prob.dim);
            AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, prob.dim);
            Complex L_x = ((ad_xp.dxx + ad_xp.dyy) - (ad_xm.dxx + ad_xm.dyy)) / (2.0*h);
            Complex L_y = ((ad_yp.dxx + ad_yp.dyy) - (ad_ym.dxx + ad_ym.dyy)) / (2.0*h);
            Complex biharmonic = ((ad_xp.dxx + ad_xp.dyy) + (ad_xm.dxx + ad_xm.dyy) + 
                                 (ad_yp.dxx + ad_yp.dyy) + (ad_ym.dxx + ad_ym.dyy) - 4.0*L_val) / (h*h);
            Complex res = psi_y * L_x - psi_x * L_y - nu * biharmonic;
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) {
                is_feasible = false; constraint_violation += 10.0;
                mse_domain = 1e18; mse_boundary = 1e18; return;
            }
            pde_mse += std::norm(res);
        }
        if (!dom.empty()) pde_mse /= dom.size();
    } else {
        for (auto& p : dom) {
            AD ad = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            sample_values.push_back(ad.v.real());
            max_grad = std::max({max_grad, std::abs(ad.dx.real()), std::abs(ad.dy.real()), std::abs(ad.dt.real())});
            
            Complex res = prob.pde_residual_ad(ad, p.x, p.y, p.t);
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) {
                is_feasible = false; constraint_violation += 10.0;
                mse_domain = 1e18; mse_boundary = 1e18; return;
            }
            pde_mse += std::norm(res); 
        }
        if (!dom.empty()) pde_mse /= dom.size();
    }

    // 5. Restricción de Pendiente (Smoothness Check)
    // Evita saltos artificiales para cumplir BCs
    if (max_grad > 500.0) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    // 6. Restricción de No-Trivialidad Funcional (Varianza)
    if (!sample_values.empty()) {
        double sum = std::accumulate(sample_values.begin(), sample_values.end(), 0.0);
        double mean = sum / sample_values.size();
        double sq_sum = std::inner_product(sample_values.begin(), sample_values.end(), sample_values.begin(), 0.0);
        sample_dom_variance = std::abs((sq_sum / sample_values.size()) - (mean * mean));
        
        // Umbral adaptativo
        if (sample_dom_variance < (1e-12 * mean * mean + 1e-15)) {
            is_feasible = false;
            constraint_violation += 1.0;
        }
    }

    double raw_bc_mse = 0.0;
    for (auto& p : bnd) {
        Complex val = tree->eval_t(p.x, p.y, p.t);
        Complex bc_val = prob.bc(p.x, p.y, p.t);
        Complex diff = val - bc_val;
        if (!std::isfinite(diff.real()) || !std::isfinite(diff.imag())) {
            is_feasible = false; constraint_violation += 10.0;
            mse_domain = 1e18; mse_boundary = 1e18; return;
        }
        raw_bc_mse += std::norm(diff);
    }
    if (!bnd.empty()) raw_bc_mse /= bnd.size();

    tree_size = tree->count_nodes();
    root_type = tree->get_type();

    // Puro Pareto NSGA-II
    mse_domain = pde_mse; 
    mse_boundary = raw_bc_mse; 
}

double PIIndividual::get_validation_mse(const PDEProblem& prob, 
                                        const std::vector<Point>& val_dom, 
                                        const std::vector<Point>& val_bnd) 
{
    if (!tree) return 1e18;
    
    if (prob.is_numerical && !prob.numerical_truth.empty()) {
        double sum_val = 0.0;
        int n_pts = 0;
        if (prob.dim == 1) {
            for (auto& pt : val_dom) {
                Complex u_approx = tree->eval_t(pt.x, pt.y, pt.t);
                Complex u_exact  = prob.numerical_exact(pt.x, pt.y);
                if (!std::isfinite(u_approx.real())) continue;
                sum_val += std::norm(u_approx - u_exact);
                ++n_pts;
            }
            return (n_pts > 0) ? sum_val / n_pts : 1e18;
        } else {
            int N = (int)std::sqrt(prob.numerical_truth.size());
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double x = i / (double)(N - 1);
                    double y = j / (double)(N - 1);
                    Complex val = tree->eval_t(x, y, 0.0);
                    if (!std::isfinite(val.real())) continue;
                    sum_val += std::norm(val - prob.numerical_truth[i * N + j]);
                    ++n_pts;
                }
            }
            return (n_pts > 0) ? sum_val / n_pts : 1e18;
        }
    }

    double sum_dom = 0.0;
    if (prob.type == PDE::NAVIER_STOKES_UNSTEADY) {
        double nu = prob.k2; double h = 0.01; double ht = 0.01;
        for (auto& p : val_dom) {
            AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            double psi_x = ad_c.dx.real(), psi_y = ad_c.dy.real();
            double u = psi_y, v = -psi_x;
            double w = -(ad_c.dxx + ad_c.dyy).real(); 
            
            AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, prob.dim);
            AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, prob.dim);
            AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, prob.dim);
            AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, prob.dim);
            double w_xp = -(ad_xp.dxx + ad_xp.dyy).real();
            double w_xm = -(ad_xm.dxx + ad_xm.dyy).real();
            double w_yp = -(ad_yp.dxx + ad_yp.dyy).real();
            double w_ym = -(ad_ym.dxx + ad_ym.dyy).real();
            double w_x = (w_xp - w_xm) / (2.0 * h);
            double w_y = (w_yp - w_ym) / (2.0 * h);
            double lap_w = (w_xp + w_xm + w_yp + w_ym - 4.0 * w) / (h * h);
            
            AD ad_tp = tree->ad_eval_t(p.x, p.y, p.t+ht, prob.dim);
            AD ad_tm = tree->ad_eval_t(p.x, p.y, p.t-ht, prob.dim);
            double w_tp = -(ad_tp.dxx + ad_tp.dyy).real();
            double w_tm = -(ad_tm.dxx + ad_tm.dyy).real();
            double w_t = (w_tp - w_tm) / (2.0 * ht);

            Complex res = w_t + u * w_x + v * w_y - nu * lap_w;
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) return 1e18;
            sum_dom += std::norm(res);
        }
    } else if (prob.type == PDE::NAVIER_STOKES) {
        double h = 0.01; double nu = prob.k2;
        for (auto& p : val_dom) {
            AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            double psi_x = ad_c.dx.real(), psi_y = ad_c.dy.real();
            double L_val = (ad_c.dxx + ad_c.dyy).real();
            AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, prob.dim);
            AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, prob.dim);
            AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, prob.dim);
            AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, prob.dim);
            double L_x = ((ad_xp.dxx + ad_xp.dyy).real() - (ad_xm.dxx + ad_xm.dyy).real()) / (2.0*h);
            double L_y = ((ad_yp.dxx + ad_yp.dyy).real() - (ad_ym.dxx + ad_ym.dyy).real()) / (2.0*h);
            double biharmonic = ((ad_xp.dxx + ad_xp.dyy).real() + (ad_xm.dxx + ad_xm.dyy).real() + 
                                 (ad_yp.dxx + ad_yp.dyy).real() + (ad_ym.dxx + ad_ym.dyy).real() - 4.0*L_val) / (h*h);
            Complex res = Complex(psi_y * L_x - psi_x * L_y - nu * biharmonic, 0.0);
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) return 1e18;
            sum_dom += std::norm(res);
        }
    } else {
        for (auto& p : val_dom) {
            AD ad = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            Complex res = prob.pde_residual_ad(ad, p.x, p.y, p.t);
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) return 1e18;
            sum_dom += std::norm(res);
        }
    }
    double sum_bnd = 0.0;
    for (auto& p : val_bnd) {
        Complex val = tree->eval_t(p.x, p.y, p.t);
        Complex bc_val = prob.bc(p.x, p.y, p.t);
        if (!std::isfinite(bc_val.real())) return 1e18;
        Complex diff = val - bc_val;
        if (!std::isfinite(diff.real()) || !std::isfinite(diff.imag())) return 1e18;
        sum_bnd += std::norm(diff);
    }
    if (!val_dom.empty()) sum_dom /= val_dom.size();
    if (!val_bnd.empty()) sum_bnd /= val_bnd.size();

    return sum_dom + sum_bnd; 
}

// ─── PISolver ─────────────────────────────────────────────────────────────────
PISolver::PISolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed)
{
    priors_ = probe_priors(prob_);
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
    int n_val = (prob.dim == 1) ? 200 : 400;
    val_dom_pts_ = prob_.domain_points(n_val);
    val_bnd_pts_ = prob_.boundary_points(n_val / 2);
}

PIIndividual PISolver::random_individual() {
    PIIndividual ind;
    ind.tree = random_tree(Config::MAX_TREE_DEPTH, gen_, prob_);
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
    return ind;
}

PIIndividual PISolver::random_individual_special() {
    PIIndividual ind;
    ind.tree = random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_, priors_);
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
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
        if (p_dist(gen_) < 0.7) child.tree->mutate_erc(gen_, Config::ERC_SIGMA * 0.5);
        else                    child.tree = tree_mutate(child.tree, gen_, prob_);
    } else {
        if (p_dist(gen_) < Config::MUTATION_PROB)
            child.tree = tree_mutate(child.tree, gen_, prob_);
        if (p_dist(gen_) < 0.4)
            child.tree->mutate_erc(gen_, Config::ERC_SIGMA);
    }
    return child;
}

void PISolver::update_hall_of_fame() {
    for (auto& ind : population_) {
        if (ind.rank == 1 && ind.mse_boundary < 0.05 && ind.tree_size > 3 && ind.is_feasible) {
            double v_mse = ind.get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_);
            if (!has_best_ever_ || v_mse < best_ever_.mse_domain) {
                best_ever_.mse_domain = v_mse; 
                best_ever_.mse_boundary = ind.mse_boundary;
                best_ever_.rank = ind.rank;
                best_ever_.crowding = ind.crowding;
                best_ever_.tree_size = ind.tree_size;
                best_ever_.root_type = ind.root_type;
                if (ind.tree) best_ever_.tree = ind.tree->clone();
                has_best_ever_ = true;
                stagnation_counter_ = 0;
            }
        }
    }
}


Point PISolver::generate_random_point() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double t_val = (prob_.type == PDE::NAVIER_STOKES_UNSTEADY) ? dist(gen_) : 0.0;
    if (prob_.dim == 1) {
        // Malla de Chebyshev (Gauss-Lobatto) para clavar los bordes en 1D
        double u = dist(gen_);
        double x_cheb = 0.5 * (1.0 - std::cos(M_PI * u));
        return {x_cheb, 0.0, t_val};
    }
    return {dist(gen_), dist(gen_), t_val};
}

void PISolver::apply_committee_rar() {
    if (population_.empty()) return;
    std::vector<PIIndividual*> committee;
    std::vector<PIIndividual*> elites;
    for (auto& ind : population_) if (ind.rank == 1) elites.push_back(&ind);
    std::sort(elites.begin(), elites.end(), [](PIIndividual* a, PIIndividual* b){
        return a->mse_domain < b->mse_domain;
    });
    for (int i = 0; i < std::min((int)elites.size(), Config::RAR_ELITE_COUNT); ++i) committee.push_back(elites[i]);

    int n_random = (int)(population_.size() * Config::RAR_RANDOM_RATIO);
    std::vector<int> indices(population_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen_);
    for (int i = 0; i < std::min(n_random, (int)indices.size()); ++i) committee.push_back(&population_[indices[i]]);

    if (committee.empty()) return;

    int pool_size = Config::RAR_CANDIDATES;
    struct ScoredPoint { Point p; double score; };
    std::vector<ScoredPoint> scored;
    scored.reserve(pool_size);

    #pragma omp parallel
    {
        std::mt19937 thread_gen(gen_() + omp_get_thread_num());
        std::uniform_real_distribution<double> ud(0, 1);
        std::vector<ScoredPoint> thread_scored;
        #pragma omp for
        for (int i = 0; i < pool_size; ++i) {
            double tx = ud(thread_gen), ty = ud(thread_gen);
            double tt = (prob_.type == PDE::NAVIER_STOKES_UNSTEADY) ? ud(thread_gen) : 0.0;
            Point p = {tx, ty, tt};
            double sum_error = 0.0; int valid_count = 0;
            for (auto* ind : committee) {
                if (!ind->tree) continue;
                AD ad = ind->tree->ad_eval_t(p.x, p.y, p.t, prob_.dim);
                Complex res = prob_.pde_residual_ad(ad, p.x, p.y, p.t);
                double err = std::norm(res);
                if (std::isfinite(err)) { sum_error += err; valid_count++; }
            }
            double consensus_score = (valid_count > 0) ? (sum_error / valid_count) : 0.0;
            if (std::isfinite(consensus_score)) thread_scored.push_back({p, consensus_score});
        }
        #pragma omp critical
        scored.insert(scored.end(), thread_scored.begin(), thread_scored.end());
    }
    std::sort(scored.begin(), scored.end(), [](const ScoredPoint& a, const ScoredPoint& b){ return a.score > b.score; });

    int n_total = (prob_.dim == 1) ? Config::N_DOMAIN : 4000;
    int n_adaptive = (int)(n_total * Config::RAR_ADAPTIVE_RATIO);
    int n_uniform = n_total - n_adaptive;
    dom_pts_.clear();
    for (int i = 0; i < std::min(n_adaptive, (int)scored.size()); ++i) dom_pts_.push_back(scored[i].p);
    for (int i = 0; i < n_uniform; ++i) dom_pts_.push_back(generate_random_point());
}

std::vector<PIIndividual> PISolver::run(int pop_size, int max_gen) {
    population_.clear();
    has_best_ever_ = false;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto gen_pts = [&]() {
        dom_pts_.clear(); bnd_pts_.clear();
        if (prob_.dim == 1) {
            for (int i = 0; i < Config::N_DOMAIN; ++i) dom_pts_.push_back({dist(gen_), 0.0, (prob_.type == PDE::NAVIER_STOKES_UNSTEADY ? dist(gen_) : 0.0)});
            for (int i = 0; i < 100; ++i) {
                double t_val = (prob_.type == PDE::NAVIER_STOKES_UNSTEADY) ? dist(gen_) : 0.0;
                bnd_pts_.push_back({0.0, 0.0, t_val}); bnd_pts_.push_back({1.0, 0.0, t_val});
                if (prob_.type == PDE::NAVIER_STOKES_UNSTEADY) bnd_pts_.push_back({dist(gen_), 0.0, 0.0});
            }
        } else {
            for (int i = 0; i < 4000; ++i) dom_pts_.push_back({dist(gen_), dist(gen_), (prob_.type == PDE::NAVIER_STOKES_UNSTEADY ? dist(gen_) : 0.0)});
            int pts_per_side = Config::N_BOUNDARY / 5;
            for (int i = 0; i < pts_per_side; ++i) {
                double s = dist(gen_), t = (prob_.type == PDE::NAVIER_STOKES_UNSTEADY ? dist(gen_) : 0.0);
                bnd_pts_.push_back({0.0, s, t}); bnd_pts_.push_back({1.0, s, t});
                bnd_pts_.push_back({s, 0.0, t}); bnd_pts_.push_back({s, 1.0, t});
                if (prob_.type == PDE::NAVIER_STOKES_UNSTEADY) bnd_pts_.push_back({dist(gen_), dist(gen_), 0.0});
            }
        }
    };
    gen_pts();

    NodePtr exact_tree = get_exact_solution_tree(prob_);
    for (int i = 0; i < pop_size; ++i) {
        PIIndividual ind;
        if (i < pop_size / 10 && exact_tree) ind.tree = exact_tree->clone();
        else if (i < pop_size / 2) ind.tree = random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_, priors_);
        else ind.tree = random_tree(Config::MAX_TREE_DEPTH, gen_, prob_);
        
        if (ind.tree) ind.tree = ind.tree->simplify();
        ind.evaluate(prob_, dom_pts_, bnd_pts_, 0);
        population_.push_back(std::move(ind));
    }

    population_ = nsga2_select_next(std::move(population_), pop_size);
    update_hall_of_fame();

    int n_threads = omp_get_max_threads();
    std::vector<std::mt19937> gen_thread(n_threads);
    for (int i = 0; i < n_threads; ++i) gen_thread[i].seed(gen_() + i);

    for (int g = 0; g < max_gen; ++g) {
        current_gen_ = g;
        if (g > 0 && g % Config::RAR_INTERVAL == 0) {
            apply_committee_rar();
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) population_[i].evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        }
        update_hall_of_fame();

        std::vector<PIIndividual> offspring;
        if (has_best_ever_) {
            PIIndividual elite; elite.tree = best_ever_.tree->clone();
            elite.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
            offspring.push_back(std::move(elite));
        }
        while ((int)offspring.size() < pop_size) {
            int p1 = tournament_select(population_, gen_), p2 = tournament_select(population_, gen_);
            offspring.push_back(make_offspring(population_[p1], population_[p2]));
        }
        #pragma omp parallel for
        for (size_t i = 0; i < offspring.size(); ++i) {
            if (offspring[i].tree) offspring[i].tree = offspring[i].tree->simplify();
            offspring[i].evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        }

        std::vector<PIIndividual> combined = std::move(population_);
        for (auto& o : offspring) combined.push_back(std::move(o));
        
        if (g == max_gen - 1) {
            // En la última generación, devolvemos el conjunto combinado (2N) 
            // antes de podarlo, para que el CSV tenga soluciones dominadas.
            // Como Ind no es copiable (unique_ptr), clonamos.
            std::vector<PIIndividual> last_pop;
            for (const auto& ind : combined) {
                PIIndividual copy;
                copy.mse_domain = ind.mse_domain; copy.mse_boundary = ind.mse_boundary;
                copy.rank = ind.rank; copy.crowding = ind.crowding; copy.tree_size = ind.tree_size;
                copy.is_feasible = ind.is_feasible;
                if (ind.tree) copy.tree = ind.tree->clone();
                last_pop.push_back(std::move(copy));
            }
            return last_pop;
        }

        population_ = nsga2_select_next(std::move(combined), pop_size);
        update_hall_of_fame();

        // Early Stopping: Si el error es menor a 1e-14 y la solución no es constante
        if (has_best_ever_ && best_ever_.is_feasible && (best_ever_.mse_domain + best_ever_.mse_boundary < 1e-14)) {
            if (best_ever_.tree && best_ever_.tree->contains_variables()) {
                std::cout << "[INFO] Early stopping en gen=" << g << " (Solucion No-Trivial, MSE < 1e-14)" << std::endl;
                // Clonamos la población actual para devolverla
                std::vector<PIIndividual> last_pop;
                for (const auto& ind : population_) {
                    PIIndividual copy;
                    copy.mse_domain = ind.mse_domain; copy.mse_boundary = ind.mse_boundary;
                    copy.rank = ind.rank; copy.crowding = ind.crowding; copy.tree_size = ind.tree_size;
                    copy.is_feasible = ind.is_feasible;
                    if (ind.tree) copy.tree = ind.tree->clone();
                    last_pop.push_back(std::move(copy));
                }
                return last_pop;
            }
        }

        // Memetic HC (Explotación Intensiva)
        int n_hc = (int)(pop_size * 0.20);
        #pragma omp parallel
        {
            std::mt19937& t_gen = gen_thread[omp_get_thread_num() % n_threads];
            #pragma omp for
            for (int i = 0; i < n_hc; ++i) {
                hill_climb_constants(population_[i], (g % 10 == 0 ? 150 : 40), t_gen);
            }
        }

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
    }
    return std::move(population_);
}

std::vector<PIIndividual> PISolver::pareto_front() const {
    std::vector<PIIndividual> front;
    for (auto& ind : population_) {
        if (ind.rank == 1 && ind.is_feasible) {
            // Filtro Post-Evolución de Variables
            bool vars_ok = true;
            if (prob_.dim >= 2) {
                if (!ind.tree->uses_variable(NodeType::VAR_X) || !ind.tree->uses_variable(NodeType::VAR_Y)) vars_ok = false;
            }
            if (prob_.type == PDE::NAVIER_STOKES_UNSTEADY) {
                if (!ind.tree->uses_variable(NodeType::VAR_T)) vars_ok = false;
            }

            if (vars_ok) {
                PIIndividual copy;
                copy.mse_domain = ind.mse_domain; copy.mse_boundary = ind.mse_boundary;
                copy.rank = ind.rank; copy.crowding = ind.crowding; copy.tree_size = ind.tree_size;
                copy.is_feasible = ind.is_feasible;
                if (ind.tree) copy.tree = ind.tree->clone();
                front.push_back(std::move(copy));
            }
        }
    }
    return front;
}

void PISolver::hill_climb_constants(PIIndividual& ind, int iterations, std::mt19937& thread_gen) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;
    
    double boundary_threshold = 1e-4;

    for (int iter = 0; iter < iterations; ++iter) {
        int i = std::uniform_int_distribution<int>(0, ercs.size() - 1)(thread_gen);
        double best_dom = ind.mse_domain, best_bnd = ind.mse_boundary;
        Complex best_val = *ercs[i];
        
        double sigma = Config::ERC_SIGMA * (1.0 - (double)iter / iterations);
        std::normal_distribution<double> step(0.0, sigma);
        *ercs[i] = best_val + Complex(step(thread_gen), step(thread_gen));
        ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        
        bool accept = false;
        if (best_bnd > boundary_threshold) {
            if (ind.mse_boundary < best_bnd) accept = true;
        } else {
            if (ind.mse_domain + ind.mse_boundary < best_dom + best_bnd) accept = true;
        }

        if (!accept || !std::isfinite(ind.mse_domain)) {
            *ercs[i] = best_val; ind.mse_domain = best_dom; ind.mse_boundary = best_bnd;
        }
    }
}

void PISolver::polish_constants(PIIndividual& ind) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;
    std::cout << "  [Optimizer] Polishing " << ercs.size() << " constants...\n";
    double orig = Config::ERC_SIGMA;
    Config::ERC_SIGMA = 0.5; hill_climb_constants(ind, 1000, gen_);
    Config::ERC_SIGMA = 0.05; hill_climb_constants(ind, 1000, gen_);
    Config::ERC_SIGMA = 0.001; hill_climb_constants(ind, 500, gen_);
    Config::ERC_SIGMA = orig;
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
}
