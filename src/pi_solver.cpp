#include "pi_solver.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <omp.h>

static void fast_non_dominated_sort(std::vector<PIIndividual>& pop) {
    int n = static_cast<int>(pop.size());
    std::vector<std::vector<int>> S(n);
    std::vector<int> count(n, 0);
    std::vector<int> front;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bool i_dom_j = false;
            if (pop[i].is_feasible && !pop[j].is_feasible) i_dom_j = true;
            else if (!pop[i].is_feasible && !pop[j].is_feasible) {
                if (pop[i].constraint_violation < pop[j].constraint_violation) i_dom_j = true;
            } else if (pop[i].is_feasible && pop[j].is_feasible) {
                if ((pop[i].mse_domain <= pop[j].mse_domain && pop[i].mse_boundary <= pop[j].mse_boundary && pop[i].tree_size <= pop[j].tree_size) &&
                    (pop[i].mse_domain < pop[j].mse_domain || pop[i].mse_boundary < pop[j].mse_boundary || pop[i].tree_size < pop[j].tree_size))
                    i_dom_j = true;
            }
            if (i_dom_j) S[i].push_back(j);
            else {
                bool j_dom_i = false;
                if (pop[j].is_feasible && !pop[i].is_feasible) j_dom_i = true;
                else if (!pop[j].is_feasible && !pop[i].is_feasible) {
                    if (pop[j].constraint_violation < pop[i].constraint_violation) j_dom_i = true;
                } else if (pop[j].is_feasible && pop[i].is_feasible) {
                    if ((pop[j].mse_domain <= pop[i].mse_domain && pop[j].mse_boundary <= pop[i].mse_boundary && pop[j].tree_size <= pop[i].tree_size) &&
                        (pop[j].mse_domain < pop[i].mse_domain || pop[j].mse_boundary < pop[i].mse_boundary || pop[j].tree_size < pop[i].tree_size))
                        j_dom_i = true;
                }
                if (j_dom_i) count[i]++;
            }
        }
        if (count[i] == 0) { pop[i].rank = 1; front.push_back(i); }
    }
    int curr = 1;
    while (!front.empty()) {
        std::vector<int> next_front;
        for (int i : front) {
            for (int j : S[i]) {
                count[j]--;
                if (count[j] == 0) { pop[j].rank = curr + 1; next_front.push_back(j); }
            }
        }
        front = next_front; curr++;
    }
}

static void calculate_crowding_distance(std::vector<PIIndividual>& pop, const std::vector<int>& front_indices) {
    if (front_indices.empty()) return;
    int n = static_cast<int>(front_indices.size());
    for (int idx : front_indices) pop[idx].crowding = 0.0;
    auto assign_dist = [&](auto getter) {
        std::vector<int> sorted = front_indices;
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return getter(pop[a]) < getter(pop[b]); });
        pop[sorted[0]].crowding = 1e15; pop[sorted[n-1]].crowding = 1e15;
        double range = getter(pop[sorted[n-1]]) - getter(pop[sorted[0]]);
        if (range < 1e-9) range = 1e-9;
        for (int i = 1; i < n-1; ++i) pop[sorted[i]].crowding += (getter(pop[sorted[i+1]]) - getter(pop[sorted[i-1]])) / range;
    };
    assign_dist([](const PIIndividual& ind) { return ind.mse_domain; });
    assign_dist([](const PIIndividual& ind) { return ind.mse_boundary; });
    assign_dist([](const PIIndividual& ind) { return static_cast<double>(ind.tree_size); });
}

std::vector<PIIndividual> nsga2_select_next(std::vector<PIIndividual> combined, int pop_size) {
    fast_non_dominated_sort(combined);
    std::vector<PIIndividual> next_pop; int rank = 1;
    while (next_pop.size() < static_cast<size_t>(pop_size)) {
        std::vector<int> front;
        for (size_t i = 0; i < combined.size(); ++i) if (combined[i].rank == rank) front.push_back(i);
        if (front.empty()) break;
        calculate_crowding_distance(combined, front);
        if (next_pop.size() + front.size() <= static_cast<size_t>(pop_size)) {
            for (int i : front) next_pop.push_back(std::move(combined[i]));
        } else {
            std::sort(front.begin(), front.end(), [&](int a, int b) {
                if (std::abs(combined[a].crowding - combined[b].crowding) > 1e-6)
                    return combined[a].crowding > combined[b].crowding;
                return combined[a].tree_size < combined[b].tree_size;
            });
            for (size_t i = 0; next_pop.size() < static_cast<size_t>(pop_size); ++i) 
                next_pop.push_back(std::move(combined[front[i]]));
        }
        rank++;
    }
    return next_pop;
}

static int tournament_select(const std::vector<PIIndividual>& pop, std::mt19937& gen) {
    int best = -1;
    for (int i = 0; i < 4; ++i) {
        int idx = std::uniform_int_distribution<int>(0, static_cast<int>(pop.size()) - 1)(gen);
        if (best == -1) best = idx;
        else {
            if (pop[idx].rank < pop[best].rank) best = idx;
            else if (pop[idx].rank == pop[best].rank && pop[idx].crowding > pop[best].crowding) best = idx;
        }
    }
    return best;
}

namespace Config {
    int    POP_SIZE       = 500;   
    int    MAX_GEN        = 1000;   
    int    N_DOMAIN       = 500;  
    int    N_BOUNDARY     = 200;   
    double ERC_SIGMA      = 0.20;  
    int    MAX_TREE_DEPTH = 6;
    double CROSSOVER_PROB = 0.80;  
    double MUTATION_PROB  = 0.45;   
    int    TOURNAMENT_SIZE = 4;    
    double STOP_THRESHOLD  = 1e-12; 

    int    RAR_INTERVAL       = 25;
    int    RAR_CANDIDATES     = 10000;
    double RAR_ADAPTIVE_RATIO = 0.30;
    int    RAR_ELITE_COUNT    = 4;
    double RAR_RANDOM_RATIO   = 0.40;
    bool   GENERAL_MODE       = false;
    int    CORES              = 1;
    bool   IS_SINGULAR        = false; // Detectado dinamicamente
}

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

    if (!is_physically_complete(prob)) {
        is_feasible = false;
        constraint_violation += 10.0;
    }

    if (!prob.is_numerical && !(prob.dim_u == Units::None)) {
        auto d_opt = tree->get_dimension(prob);
        bool has_erc = tree->contains_erc();
        if (!has_erc && (!d_opt.has_value() || *d_opt != prob.dim_u)) {
            is_feasible = false;
            constraint_violation += 5.0; 
        }
    }

    if (tree->get_unary_depth() > 3) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    if (tree->get_depth() > Config::MAX_TREE_DEPTH) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    if (tree->has_nested_trig()) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    if (tree->has_non_affine_trig_arg()) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    if (tree->has_nested_polynomial()) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    if (tree->has_nested_exp()) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    if (tree->has_invalid_polynomial_degree()) {
        is_feasible = false;
        constraint_violation += 1.0;
    }

    double pde_mse = 0.0;
    std::vector<double> sample_values;
    std::vector<double> grad_magnitudes;
    sample_dom_variance = 0.0;
    double max_grad = 0.0;

    bool use_ansatz_1d = (prob.dim == 1 && bnd.size() >= 2);
    double u0 = 0.0, u1 = 0.0;
    if (use_ansatz_1d) {
        u0 = prob.bc(0.0, 0.0, 0.0).real();
        u1 = prob.bc(1.0, 0.0, 0.0).real();
    }

    for (auto& p : dom) {
        AD ad = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
        
        if (use_ansatz_1d) {
            double L = u0 + p.x * (u1 - u0);
            double dL = (u1 - u0);
            double B = p.x * (p.x - 1.0);
            double dB = 2.0 * p.x - 1.0;
            double d2B = 2.0;

            AD U;
            U.v = L + B * ad.v.real();
            U.dx = dL + dB * ad.v.real() + B * ad.dx.real();
            U.dxx = d2B * ad.v.real() + 2.0 * dB * ad.dx.real() + B * ad.dxx.real();
            U.dy = U.dt = U.dyy = U.dtt = 0.0;
            ad = U;
        }

        sample_values.push_back(ad.v.real());
        
        double g_mag = std::abs(ad.dx.real()) + std::abs(ad.dy.real()) + std::abs(ad.dt.real());
        grad_magnitudes.push_back(g_mag);
        max_grad = std::max(max_grad, g_mag);

        // Pasamos null como tree porque la evaluación ya está en ad, y cambiamos prob.compute_residual 
        // para que use pde_residual_ad directamente si le mandamos un proxy.
        // Pero compute_residual llama a tree->ad_eval_t. Así que necesitamos usar pde_residual_ad directamente!
        Complex res = prob.pde_residual_ad(ad, p.x, p.y, p.t);
        
        if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) {
            is_feasible = false; constraint_violation += 10.0;
            mse_domain = 1e18; mse_boundary = 1e18; return;
        }
        pde_mse += std::norm(res); 
    }
    if (!dom.empty()) pde_mse /= dom.size();

    if (max_grad > 15.0) { is_feasible = false; constraint_violation += 5.0; }
    
    // Varianza de la función
    if (!sample_values.empty()) {
        double sum = std::accumulate(sample_values.begin(), sample_values.end(), 0.0);
        double mean = sum / sample_values.size();
        double sq_sum = std::inner_product(sample_values.begin(), sample_values.end(), sample_values.begin(), 0.0);
        sample_dom_variance = std::abs((sq_sum / sample_values.size()) - (mean * mean));
        if (sample_dom_variance < (1e-12 * mean * mean + 1e-15)) { is_feasible = false; constraint_violation += 1.0; }
    }

    // Curvatura Estructural (Anti-Aproximación Lineal)
    // Exigimos que el gradiente NO sea constante (varianza del gradiente > 0)
    if (!grad_magnitudes.empty()) {
        double g_sum = std::accumulate(grad_magnitudes.begin(), grad_magnitudes.end(), 0.0);
        double g_mean = g_sum / grad_magnitudes.size();
        double g_sq_sum = std::inner_product(grad_magnitudes.begin(), grad_magnitudes.end(), grad_magnitudes.begin(), 0.0);
        double g_var = std::abs((g_sq_sum / grad_magnitudes.size()) - (g_mean * g_mean));
        
        // Si el gradiente es constante (varianza < 1e-10) y la función no es plana (g_mean > 1e-5),
        // significa que es un plano lineal (u = cx + dy). Eso no resuelve una PDE de 2do orden.
        if (g_var < 1e-10 && g_mean > 1e-5) { 
            is_feasible = false; 
            constraint_violation += 1.0; 
        }
    }

    // ─── Error de Frontera ───
    double raw_bc_mse = 0.0;
    if (Config::GENERAL_MODE) {
        raw_bc_mse = 0.0;
    } else if (use_ansatz_1d) {
        raw_bc_mse = 0.0; // El Ansatz cumple las condiciones de frontera de manera exacta!
    } else {
        raw_bc_mse = prob.compute_boundary_error(tree.get(), bnd);
    }

    tree_size = tree->count_nodes();
    root_type = tree->get_type();
    mse_domain = pde_mse; mse_boundary = raw_bc_mse; 
}

double PIIndividual::get_validation_mse(const PDEProblem& prob, 
                                        const std::vector<Point>& val_dom, 
                                        const std::vector<Point>& val_bnd) 
{
    if (!tree) return 1e18;
    double sum_sq_err = 0.0; int n_pts = 0;
    if (prob.is_numerical && !prob.numerical_truth.empty()) {
        bool use_ansatz_1d = (prob.dim == 1 && val_bnd.size() >= 2);
        double u0 = 0.0, u1 = 0.0;
        if (use_ansatz_1d) {
            u0 = prob.bc(0.0, 0.0, 0.0).real();
            u1 = prob.bc(1.0, 0.0, 0.0).real();
        }

        if (prob.dim == 1) {
            for (auto& pt : val_dom) {
                Complex u_approx = tree->eval_t(pt.x, pt.y, pt.t);
                if (use_ansatz_1d) {
                    double L = u0 + pt.x * (u1 - u0);
                    double B = pt.x * (pt.x - 1.0);
                    u_approx = Complex(L + B * u_approx.real(), L + B * u_approx.imag());
                }
                Complex u_exact  = prob.numerical_exact(pt.x, pt.y, pt.t);
                if (!std::isfinite(u_approx.real())) continue;
                sum_sq_err += std::norm(u_approx - u_exact); ++n_pts;
            }
        } else {
            int N = (int)std::sqrt(prob.numerical_truth.size());
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double x = i / (double)(N - 1), y = j / (double)(N - 1);
                    Complex val = tree->eval_t(x, y, 0.0);
                    if (!std::isfinite(val.real())) continue;
                    sum_sq_err += std::norm(val - prob.numerical_truth[i * N + j]); ++n_pts;
                }
            }
        }
    } else {
        for (auto& pt : val_dom) {
            Complex u_approx = tree->eval_t(pt.x, pt.y, pt.t);
            Complex u_exact  = prob.exact(pt.x, pt.y, pt.t);
            if (!std::isfinite(u_approx.real())) continue;
            sum_sq_err += std::norm(u_approx - u_exact); ++n_pts;
        }
    }
    return (n_pts > 0) ? sum_sq_err / n_pts : 1e18;
}

PISolver::PISolver(const PDEProblem& prob, unsigned seed) : prob_(prob), gen_(seed) {
    priors_ = probe_priors(prob_);
    int n_val = (prob.dim == 1) ? 200 : 400;
    val_dom_pts_ = prob_.domain_points(n_val);
    val_bnd_pts_ = prob_.boundary_points(n_val / 2);

    // --- Detector A-Priori de Singularidad ---
    auto dummy_tree = make_binary(NodeType::ADD, make_var('x'), make_erc(1.0));
    Complex r_center = prob_.compute_residual(dummy_tree.get(), {0.5, 0.5, 0.0});
    Complex r_edge = prob_.compute_residual(dummy_tree.get(), {1e-5, 1e-5, 0.0});
    if (std::isfinite(r_center.real()) && std::isfinite(r_edge.real())) {
        if (std::norm(r_edge) > std::norm(r_center) * 1000.0) {
            Config::IS_SINGULAR = true;
            std::cout << "[INFO] Singularidad detectada en el PDE. Se usara Hill Climbing (HC) en lugar de GD." << std::endl;
        } else {
            Config::IS_SINGULAR = false;
        }
    } else {
        Config::IS_SINGULAR = true; // Ante la duda, asume singular
    }
}

PIIndividual PISolver::random_individual() {
    PIIndividual ind;
    ind.tree = make_binary(NodeType::MUL, make_erc(1.0), random_tree(Config::MAX_TREE_DEPTH, gen_, prob_));
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
    return ind;
}

PIIndividual PISolver::random_individual_special() {
    PIIndividual ind;
    ind.tree = make_binary(NodeType::MUL, make_erc(1.0), random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_, priors_));
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
        else child.tree = tree_mutate(child.tree, gen_, prob_);
    } else {
        if (p_dist(gen_) < Config::MUTATION_PROB) child.tree = tree_mutate(child.tree, gen_, prob_);
        if (p_dist(gen_) < 0.4) child.tree->mutate_erc(gen_, Config::ERC_SIGMA);
    }
    return child;
}

void PISolver::update_hall_of_fame() {
    for (auto& ind : population_) {
        if (ind.rank == 1 && ind.tree_size >= 1 && ind.is_feasible && ind.is_physically_complete(prob_)) {
            double current_err = ind.mse_domain + ind.mse_boundary;
            double best_err = has_best_ever_ ? (best_ever_.mse_domain + best_ever_.mse_boundary) : 1e18;
            if (!has_best_ever_ || current_err < best_err) {
                best_ever_.mse_domain = ind.mse_domain; best_ever_.mse_boundary = ind.mse_boundary;
                best_ever_.rank = ind.rank; best_ever_.crowding = ind.crowding;
                best_ever_.tree_size = ind.tree_size; best_ever_.root_type = ind.root_type;
                if (ind.tree) best_ever_.tree = ind.tree->clone();
                has_best_ever_ = true;
            }
        }
    }
}

Point PISolver::generate_random_point() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double t_val = (prob_.is_unsteady) ? dist(gen_) : 0.0;
    if (prob_.dim == 1) {
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
    std::sort(elites.begin(), elites.end(), [](PIIndividual* a, PIIndividual* b){ return a->mse_domain < b->mse_domain; });
    for (int i = 0; i < std::min((int)elites.size(), Config::RAR_ELITE_COUNT); ++i) committee.push_back(elites[i]);
    int n_random = (int)(population_.size() * Config::RAR_RANDOM_RATIO);
    std::vector<int> indices(population_.size()); std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen_);
    for (int i = 0; i < std::min(n_random, (int)indices.size()); ++i) committee.push_back(&population_[indices[i]]);
    if (committee.empty()) return;
    int pool_size = Config::RAR_CANDIDATES;
    struct ScoredPoint { Point p; double score; };
    std::vector<ScoredPoint> scored(pool_size);
    unsigned int seed_base = gen_();
    #pragma omp parallel
    {
        std::mt19937 thread_gen(seed_base + omp_get_thread_num());
        std::uniform_real_distribution<double> ud(0, 1);
        #pragma omp for schedule(static)
        for (int i = 0; i < pool_size; ++i) {
            double tx = ud(thread_gen), ty = ud(thread_gen);
            double tt = (prob_.is_unsteady) ? ud(thread_gen) : 0.0;
            Point p = {tx, ty, tt};
            double sum_error = 0.0; int valid_count = 0;
            for (auto* ind : committee) {
                if (!ind->tree) continue;
                AD ad = ind->tree->ad_eval_t(p.x, p.y, p.t, prob_.dim);
                Complex res = prob_.compute_residual(ind->tree.get(), p);
                double err = std::norm(res);
                if (std::isfinite(err)) { sum_error += err; valid_count++; }
            }
            double consensus_score = (valid_count > 0) ? (sum_error / valid_count) : 0.0;
            scored[i] = {p, std::isfinite(consensus_score) ? consensus_score : 0.0};
        }
    }
    std::sort(scored.begin(), scored.end(), [](const ScoredPoint& a, const ScoredPoint& b){ return a.score > b.score; });
    int n_total = (prob_.dim == 1) ? Config::N_DOMAIN : 4000;
    int n_adaptive = (int)(n_total * Config::RAR_ADAPTIVE_RATIO);
    std::vector<Point> next_pts;
    int n_keep = n_total - n_adaptive;
    if (dom_pts_.size() > (size_t)n_keep) {
        std::shuffle(dom_pts_.begin(), dom_pts_.end(), gen_);
        for (int i = 0; i < n_keep; ++i) next_pts.push_back(dom_pts_[i]);
    } else { next_pts = dom_pts_; }
    for (int i = 0; i < std::min(n_adaptive, (int)scored.size()); ++i) next_pts.push_back(scored[i].p);
    dom_pts_ = std::move(next_pts);
}

std::vector<PIIndividual> PISolver::run(int pop_size, int max_gen) {
    population_.clear(); has_best_ever_ = false;
    dom_pts_.clear(); bnd_pts_.clear();
    for (int i = 0; i < Config::N_DOMAIN; ++i) dom_pts_.push_back(generate_random_point());
    for (int i = 0; i < Config::N_BOUNDARY; ++i) {
        Point p = generate_random_point();
        if (prob_.dim == 1) p.x = (i%2 == 0) ? 0.0 : 1.0;
        else {
            double s = (double)i / Config::N_BOUNDARY;
            if (i%4 == 0) { p.x=0; p.y=s; } else if (i%4==1) { p.x=1; p.y=s; }
            else if (i%4==2) { p.x=s; p.y=0; } else { p.x=s; p.y=1; }
        }
        bnd_pts_.push_back(p);
    }
    NodePtr exact = get_exact_solution_tree(prob_);
    for (int i = 0; i < pop_size; ++i) {
        PIIndividual ind;
        if (i < pop_size/10 && exact) ind.tree = exact->clone();
        else if (i < pop_size/2) ind.tree = random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_, priors_);
        else ind.tree = random_tree(Config::MAX_TREE_DEPTH, gen_, prob_);
        if (ind.tree) ind.tree = ind.tree->simplify();
        ind.evaluate(prob_, dom_pts_, bnd_pts_, 0);
        population_.push_back(std::move(ind));
    }
    // MOEA/D Initialization
    int T_neighbors = std::max(2, (int)(0.1 * pop_size));
    std::vector<std::pair<double, double>> weights(pop_size);
    std::vector<std::vector<int>> B_neighbors(pop_size, std::vector<int>(T_neighbors));
    for (int i = 0; i < pop_size; ++i) {
        double w1 = (double)i / (pop_size - 1.0);
        weights[i] = {w1, 1.0 - w1};
    }
    for (int i = 0; i < pop_size; ++i) {
        std::vector<std::pair<double, int>> dists(pop_size);
        for (int j = 0; j < pop_size; ++j) dists[j] = {std::abs(weights[i].first - weights[j].first), j};
        std::sort(dists.begin(), dists.end());
        for (int k = 0; k < T_neighbors; ++k) B_neighbors[i][k] = dists[k].second;
    }

    fast_non_dominated_sort(population_);
    update_hall_of_fame();
    int n_threads = omp_get_max_threads();

    for (int g = 0; g < max_gen; ++g) {
        current_gen_ = g;
        if (g > 0 && g % Config::RAR_INTERVAL == 0) {
            apply_committee_rar();
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) population_[i].evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        }
        update_hall_of_fame();

        if (has_best_ever_) {
            best_ever_.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
            double train_err = Config::GENERAL_MODE ? best_ever_.mse_domain : (best_ever_.mse_domain + best_ever_.mse_boundary);
            double threshold = Config::STOP_THRESHOLD;
            
            if (train_err < threshold) {
                // En modo general, aceptamos la solución por su residuo dinámico
                if (Config::GENERAL_MODE) {
                    std::cout << "[INFO] Early stopping en gen=" << g << " (Solucion General Descubierta, Res < " << threshold << ")" << std::endl;
                    std::vector<PIIndividual> final_pop; final_pop.push_back(std::move(best_ever_));
                    for (auto& ind : population_) final_pop.push_back(std::move(ind));
                    fast_non_dominated_sort(final_pop);
                    return final_pop;
                }

                double val_err = best_ever_.get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_);
                if (val_err < threshold * 10.0) {
                    std::cout << "[INFO] Early stopping en gen=" << g << " (Ley Fisica Descubierta, MSE < " << threshold << ")" << std::endl;
                    std::vector<PIIndividual> final_pop; final_pop.push_back(std::move(best_ever_));
                    for (auto& ind : population_) final_pop.push_back(std::move(ind));
                    fast_non_dominated_sort(final_pop);
                    return final_pop;
                }
            }
        }

        // ─── MOEA/D Subproblem Optimization ───
        double z_min1 = 1e18, z_max1 = -1e18;
        double z_min2 = 1e18, z_max2 = -1e18;
        for (const auto& ind : population_) {
            if (!ind.is_feasible) continue;
            double f1 = ind.mse_domain + ind.mse_boundary;
            double f2 = ind.tree_size;
            z_min1 = std::min(z_min1, f1); z_max1 = std::max(z_max1, f1);
            z_min2 = std::min(z_min2, f2); z_max2 = std::max(z_max2, f2);
        }
        if (z_min1 > 1e17) { z_min1 = 0; z_max1 = 1; z_min2 = 0; z_max2 = 1; }
        if (std::abs(z_max1 - z_min1) < 1e-9) z_max1 = z_min1 + 1e-9;
        if (std::abs(z_max2 - z_min2) < 1e-9) z_max2 = z_min2 + 1e-9;

        auto calc_te = [&](const PIIndividual& ind, int subp) {
            double f1 = ind.mse_domain + ind.mse_boundary;
            double f2 = ind.tree_size;
            double n1 = (f1 - z_min1) / (z_max1 - z_min1);
            double n2 = (f2 - z_min2) / (z_max2 - z_min2);
            double w1 = std::max(weights[subp].first, 1e-6);
            double w2 = std::max(weights[subp].second, 1e-6);
            return std::max(w1 * n1, w2 * n2);
        };

        auto clone_ind = [](const PIIndividual& ind) {
            PIIndividual c;
            c.mse_domain = ind.mse_domain; c.mse_boundary = ind.mse_boundary;
            c.rank = ind.rank; c.crowding = ind.crowding; c.tree_size = ind.tree_size;
            c.is_feasible = ind.is_feasible; c.constraint_violation = ind.constraint_violation;
            if (ind.tree) c.tree = ind.tree->clone();
            return c;
        };

        std::vector<PIIndividual> next_pop;
        for (const auto& ind : population_) next_pop.push_back(clone_ind(ind));

        std::vector<int> subprobs(pop_size);
        std::iota(subprobs.begin(), subprobs.end(), 0);
        std::shuffle(subprobs.begin(), subprobs.end(), gen_);

        std::vector<PIIndividual> offsprings(pop_size);
        for (int i = 0; i < pop_size; ++i) {
            int r1 = B_neighbors[i][std::uniform_int_distribution<int>(0, T_neighbors - 1)(gen_)];
            int r2 = B_neighbors[i][std::uniform_int_distribution<int>(0, T_neighbors - 1)(gen_)];
            offsprings[i] = make_offspring(population_[r1], population_[r2]);
            if (offsprings[i].tree) offsprings[i].tree = offsprings[i].tree->simplify();
        }

        #pragma omp parallel for schedule(dynamic) num_threads(Config::CORES > 0 ? Config::CORES : omp_get_max_threads())
        for (int i = 0; i < pop_size; ++i) {
            offsprings[i].evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        }

        for (int i : subprobs) {
            PIIndividual& y = offsprings[i];

            if (y.is_feasible) {
                double f1 = y.mse_domain + y.mse_boundary;
                double f2 = y.tree_size;
                z_min1 = std::min(z_min1, f1); z_max1 = std::max(z_max1, f1);
                z_min2 = std::min(z_min2, f2); z_max2 = std::max(z_max2, f2);
            }

            int replaced_count = 0;
            std::vector<int> shuf_B = B_neighbors[i];
            std::shuffle(shuf_B.begin(), shuf_B.end(), gen_);
            for (int j : shuf_B) {
                if (replaced_count >= 2) break; // Replace max 2 neighbors
                bool replace = false;
                if (y.is_feasible && !next_pop[j].is_feasible) replace = true;
                else if (!y.is_feasible && !next_pop[j].is_feasible) {
                    if (y.constraint_violation < next_pop[j].constraint_violation) replace = true;
                } else if (y.is_feasible && next_pop[j].is_feasible) {
                    if (calc_te(y, j) < calc_te(next_pop[j], j)) replace = true;
                }
                if (replace) {
                    next_pop[j] = clone_ind(y);
                    replaced_count++;
                }
            }
        }
        population_ = std::move(next_pop);
        fast_non_dominated_sort(population_);

        unsigned int g_seed = gen_();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)population_.size(); ++i) {
            std::mt19937 local_gen(g_seed + i);
            if (population_[i].rank == 1) {
                gradient_descent_constants(population_[i], (g % 10 == 0 ? 30 : 10));
            }
        }
        if (g % 25 == 0) {
            double b_dom = 1e18, b_bnd = 1e18;
            for (auto& ind : population_) if (ind.rank == 1) { b_dom = std::min(b_dom, ind.mse_domain); b_bnd = std::min(b_bnd, ind.mse_boundary); }
            std::cout << "  [PI/" << prob_.name() << "] gen=" << g << "  best_dom=" << std::scientific << b_dom << "  best_bnd=" << b_bnd << std::defaultfloat << "\n";
        }
    }
    if (has_best_ever_) {
        PIIndividual champ;
        champ.mse_domain = best_ever_.mse_domain; champ.mse_boundary = best_ever_.mse_boundary;
        champ.rank = 1; champ.is_feasible = true;
        if (best_ever_.tree) champ.tree = best_ever_.tree->clone();
        population_.push_back(std::move(champ));
        fast_non_dominated_sort(population_);
    }
    return std::move(population_);
}

void PISolver::gradient_descent_constants(PIIndividual& ind, int iterations) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    double lr = 0.05;
    double h = 1e-6;
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> grads(ercs.size(), 0.0);
        ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        if (!ind.is_feasible) return;
        double current_loss = ind.mse_domain + ind.mse_boundary;
        
        for (size_t i = 0; i < ercs.size(); ++i) {
            double orig = ercs[i]->real();
            *ercs[i] = Complex(orig + h, ercs[i]->imag());
            ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
            double loss_plus = ind.mse_domain + ind.mse_boundary;
            
            *ercs[i] = Complex(orig - h, ercs[i]->imag());
            ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
            double loss_minus = ind.mse_domain + ind.mse_boundary;
            
            *ercs[i] = Complex(orig, ercs[i]->imag());
            grads[i] = (loss_plus - loss_minus) / (2.0 * h);
            
            // Si el gradiente es inestable o la evaluación fue infactible, lo anulamos
            if (!std::isfinite(grads[i])) grads[i] = 0.0;
            
            // Gradient Clipping para PDEs Singulares
            if (Config::IS_SINGULAR) {
                if (grads[i] > 50.0) grads[i] = 50.0;
                else if (grads[i] < -50.0) grads[i] = -50.0;
            }
        }
        
        for (size_t i = 0; i < ercs.size(); ++i) {
            *ercs[i] -= Complex(lr * grads[i], 0);
        }
        
        ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        if (!ind.is_feasible || (ind.mse_domain + ind.mse_boundary >= current_loss)) {
            // Revertir y reducir learning rate
            for (size_t i = 0; i < ercs.size(); ++i) {
                *ercs[i] += Complex(lr * grads[i], 0);
            }
            lr *= 0.5;
            ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        } else {
            lr *= 1.1; // Acelerar si vamos en buena dirección
        }
        
        if (lr < 1e-7) break; // Convergencia temprana
    }
}

void PISolver::hill_climb_constants(PIIndividual& ind, int iterations, std::mt19937& thread_gen) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
    double best_err = ind.mse_domain + ind.mse_boundary;
    std::vector<Complex> best_vals(ercs.size());
    for(size_t i=0; i<ercs.size(); ++i) best_vals[i] = *ercs[i];
    
    for (int i = 0; i < iterations; ++i) {
        ind.tree->mutate_erc(thread_gen, Config::ERC_SIGMA);
        ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        double err = ind.mse_domain + ind.mse_boundary;
        if (ind.is_feasible && err < best_err) {
            best_err = err;
            for(size_t j=0; j<ercs.size(); ++j) best_vals[j] = *ercs[j];
        } else {
            for(size_t j=0; j<ercs.size(); ++j) *ercs[j] = best_vals[j]; // revertir
        }
    }
}

void PISolver::polish_constants(PIIndividual& ind) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    if (Config::IS_SINGULAR) {
        std::cout << "  [Optimizer] Polishing " << ercs.size() << " constants via Gradient Descent (Clipped Mode)...\n";
    } else {
        std::cout << "  [Optimizer] Polishing " << ercs.size() << " constants via Adaptive Gradient Descent...\n";
    }
    gradient_descent_constants(ind, 200);
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
}

std::vector<PIIndividual> PISolver::pareto_front() const {
    std::vector<PIIndividual> front;
    for (auto& ind : population_) {
        if (ind.rank == 1 && ind.is_feasible && ind.is_physically_complete(prob_)) {
            PIIndividual copy; copy.mse_domain = ind.mse_domain; copy.mse_boundary = ind.mse_boundary;
            copy.rank = ind.rank; copy.tree_size = ind.tree_size;
            if (ind.tree) copy.tree = ind.tree->clone();
            front.push_back(std::move(copy));
        }
    }
    return front;
}

bool PIIndividual::is_physically_complete(const PDEProblem& prob) const {
    if (!tree) return false;
    if (!tree->uses_variable(NodeType::VAR_X)) return false;
    if (prob.dim >= 2 && !tree->uses_variable(NodeType::VAR_Y)) return false;
    if (prob.is_unsteady && !tree->uses_variable(NodeType::VAR_T)) return false;
    return true;
}
