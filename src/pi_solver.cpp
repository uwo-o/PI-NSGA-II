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
    for (int i = 0; i < Config::TOURNAMENT_SIZE; ++i) {
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

// ─── Ansatz de frontera exacta (helper compartido) ───────────────────────────
// U(x,y) = L(x,y) + B(x,y)*N(x,y), donde N es el arbol evaluado y L,B cumplen
// la condicion de frontera de forma EXACTA (B se anula en el borde, L la
// interpola) — asi el error de frontera es cero por construccion y el
// algoritmo solo tiene que aprender el residuo interior, no dos objetivos.
// Usado desde evaluate(), get_validation_mse() y el gradiente complex-step
// para que los tres midan exactamente la misma funcion U(x,y).
static bool ansatz_applies(const PDEProblem& prob, size_t bnd_size) {
    if (Config::GENERAL_MODE) return false;
    if (bnd_size < 2) return false;
    if (prob.dim == 1) return true;
    if (prob.dim == 2) return prob.type != PDE::NAVIER_STOKES && prob.type != PDE::NAVIER_STOKES_UNSTEADY;
    return false;
}

// Aplica el ansatz a un AD ya evaluado del arbol (N, con sus derivadas). Se
// preserva el valor COMPLEJO completo (sin truncar a .real()) para que la
// composición siga siendo holomorfa en las constantes ERC — lo necesita la
// diferenciación por complex-step (ver gradient_descent_constants). Para
// constantes reales (uso normal) el resultado es idéntico a antes: la parte
// imaginaria ya es 0.
static AD apply_boundary_ansatz(const PDEProblem& prob, const AD& ad, double x, double y) {
    if (prob.dim == 1) {
        double u0 = prob.bc(0.0, 0.0, 0.0).real();
        double u1 = prob.bc(1.0, 0.0, 0.0).real();
        double L = u0 + x * (u1 - u0);
        double dL = (u1 - u0);
        double B = x * (x - 1.0);
        double dB = 2.0 * x - 1.0;
        double d2B = 2.0;

        AD U;
        U.v = L + B * ad.v;
        U.dx = dL + dB * ad.v + B * ad.dx;
        U.dxx = d2B * ad.v + 2.0 * dB * ad.dx + B * ad.dxx;
        U.dy = U.dt = U.dyy = U.dtt = 0.0;
        return U;
    }

    // dim == 2: interpolación transfinita (Coons patch) sobre [0,1]². B se anula
    // en las 4 aristas, L reproduce bc() exacto en cada arista (cancelación de
    // esquinas verificada a mano). g0,g1,h0,h1 (bc a lo largo de cada arista) y
    // sus derivadas se obtienen con diferencias finitas centradas sobre bc()
    // (h=1e-4) — es una función fija y barata (no parte de la optimización), no
    // hace falta AD ahí, sólo en la parte del árbol (ad.v/ad.dx/ad.dy/...).
    double hfd = 1e-4;
    double g0 = prob.bc(0.0, y, 0.0).real(),  g1 = prob.bc(1.0, y, 0.0).real();
    double h0 = prob.bc(x, 0.0, 0.0).real(),  h1 = prob.bc(x, 1.0, 0.0).real();
    double c00 = prob.bc(0.0, 0.0, 0.0).real(), c10 = prob.bc(1.0, 0.0, 0.0).real();
    double c01 = prob.bc(0.0, 1.0, 0.0).real(), c11 = prob.bc(1.0, 1.0, 0.0).real();

    double g0_yp = prob.bc(0.0, y + hfd, 0.0).real(), g0_ym = prob.bc(0.0, y - hfd, 0.0).real();
    double g1_yp = prob.bc(1.0, y + hfd, 0.0).real(), g1_ym = prob.bc(1.0, y - hfd, 0.0).real();
    double h0_xp = prob.bc(x + hfd, 0.0, 0.0).real(), h0_xm = prob.bc(x - hfd, 0.0, 0.0).real();
    double h1_xp = prob.bc(x + hfd, 1.0, 0.0).real(), h1_xm = prob.bc(x - hfd, 1.0, 0.0).real();

    double g0_y = (g0_yp - g0_ym) / (2.0 * hfd);
    double g0_yy = (g0_yp - 2.0 * g0 + g0_ym) / (hfd * hfd);
    double g1_y = (g1_yp - g1_ym) / (2.0 * hfd);
    double g1_yy = (g1_yp - 2.0 * g1 + g1_ym) / (hfd * hfd);
    double h0_x = (h0_xp - h0_xm) / (2.0 * hfd);
    double h0_xx = (h0_xp - 2.0 * h0 + h0_xm) / (hfd * hfd);
    double h1_x = (h1_xp - h1_xm) / (2.0 * hfd);
    double h1_xx = (h1_xp - 2.0 * h1 + h1_xm) / (hfd * hfd);

    double L = (1.0 - x) * g0 + x * g1 + (1.0 - y) * h0 + y * h1
             - (1.0 - x) * (1.0 - y) * c00 - x * (1.0 - y) * c10
             - (1.0 - x) * y * c01 - x * y * c11;
    double L_x = -g0 + g1 + (1.0 - y) * h0_x + y * h1_x
                + (1.0 - y) * c00 - (1.0 - y) * c10 + y * c01 - y * c11;
    double L_xx = (1.0 - y) * h0_xx + y * h1_xx;
    double L_y = (1.0 - x) * g0_y + x * g1_y - h0 + h1
                + (1.0 - x) * c00 + x * c10 - (1.0 - x) * c01 - x * c11;
    double L_yy = (1.0 - x) * g0_yy + x * g1_yy;

    double B = x * (1.0 - x) * y * (1.0 - y);
    double B_x = (1.0 - 2.0 * x) * y * (1.0 - y);
    double B_xx = -2.0 * y * (1.0 - y);
    double B_y = x * (1.0 - x) * (1.0 - 2.0 * y);
    double B_yy = -2.0 * x * (1.0 - x);

    AD U;
    U.v = L + B * ad.v;
    U.dx = L_x + B_x * ad.v + B * ad.dx;
    U.dxx = L_xx + B_xx * ad.v + 2.0 * B_x * ad.dx + B * ad.dxx;
    U.dy = L_y + B_y * ad.v + B * ad.dy;
    U.dyy = L_yy + B_yy * ad.v + 2.0 * B_y * ad.dy + B * ad.dyy;
    U.dt = ad.dt; U.dtt = ad.dtt; // problemas 2D en el ansatz son estacionarios
    return U;
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

    bool use_ansatz = ansatz_applies(prob, bnd.size());

    // Restricción de simetría: si probe_priors detectó que TANTO el operador
    // COMO la condición de frontera son invariantes ante x<->1-x (o y<->1-y en
    // 2D) — ver PDEPriors::mirror_symmetric_x/y — la solución real está
    // matemáticamente obligada a respetar esa reflexión (unicidad). Un
    // candidato que no la respeta no puede ser la solución, sin importar qué
    // tan bajo sea su residuo — se marca infactible, mismo patrón que el resto
    // de las restricciones de esta función.
    if (prob.priors.mirror_symmetric_x || (prob.dim == 2 && prob.priors.mirror_symmetric_y)) {
        auto eval_u = [&](double x, double y) -> Complex {
            AD a = tree->ad_eval_t(x, y, 0.0, prob.dim);
            if (use_ansatz) a = apply_boundary_ansatz(prob, a, x, y);
            return a.v;
        };
        constexpr double SYM_TOL = 0.05; // 5% de discrepancia relativa tolerada
        if (prob.priors.mirror_symmetric_x) {
            Complex ua = eval_u(0.25, 0.5), ub = eval_u(0.75, 0.5);
            double scale = std::max({std::abs(ua), std::abs(ub), 1e-6});
            if (!std::isfinite(ua.real()) || !std::isfinite(ub.real()) || std::abs(ua - ub) > SYM_TOL * scale) {
                is_feasible = false;
                constraint_violation += 1.0;
            }
        }
        if (prob.dim == 2 && prob.priors.mirror_symmetric_y) {
            Complex ua = eval_u(0.5, 0.25), ub = eval_u(0.5, 0.75);
            double scale = std::max({std::abs(ua), std::abs(ub), 1e-6});
            if (!std::isfinite(ua.real()) || !std::isfinite(ub.real()) || std::abs(ua - ub) > SYM_TOL * scale) {
                is_feasible = false;
                constraint_violation += 1.0;
            }
        }
    }

    for (auto& p : dom) {
        AD ad = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);

        if (use_ansatz) ad = apply_boundary_ansatz(prob, ad, p.x, p.y);

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
        // Recorte universal del residuo por punto (antes sólo para PDEs marcados
        // IS_SINGULAR, ver Lane-Emden/Thomas-Fermi: un único punto con residuo
        // finito pero astronómico cerca de una singularidad podía dominar el
        // promedio y hacer saltar mse_domain ordenes de magnitud sin relación con
        // la calidad real del resto del dominio). El mismo riesgo — un outlier
        // numérico arruinando el promedio — existe en cualquier PDE, no sólo los
        // marcados como singulares, así que el cap aplica siempre. Para
        // individuos bien comportados (la inmensa mayoría) el residuo normal
        // está muy por debajo de 1e4, así que esto no cambia nada en la práctica
        // salvo proteger contra outliers.
        double point_err = std::min(std::norm(res), 1e4);
        pde_mse += point_err;
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
    } else if (use_ansatz) {
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
    // Mismo ansatz U(x,y) = L+B·N que aplica evaluate() durante el entrenamiento
    // (apply_boundary_ansatz, definido más arriba) — una sola fuente de verdad
    // para las tres rutas (numérica 1D/2D, no-numérica) en vez de reimplementar
    // el desenvuelto en cada una por separado. Antes, cada rama tenía su propia
    // copia (o ninguna, para el caso numérico 2D) y podían divergir en silencio
    // — exactamente la causa de que val_mse quedara pegado en Airy/Fisher_2D/etc
    // pese a mejoras reales en el resto de la población.
    bool use_ansatz = ansatz_applies(prob, val_bnd.size());

    if (prob.is_numerical && !prob.numerical_truth.empty()) {
        if (prob.dim == 1) {
            for (auto& pt : val_dom) {
                AD ad = tree->ad_eval_t(pt.x, pt.y, pt.t, prob.dim);
                if (use_ansatz) ad = apply_boundary_ansatz(prob, ad, pt.x, pt.y);
                Complex u_exact = prob.numerical_exact(pt.x, pt.y, pt.t);
                if (!std::isfinite(ad.v.real())) continue;
                sum_sq_err += std::norm(ad.v - u_exact); ++n_pts;
            }
        } else {
            int N = (int)std::sqrt(prob.numerical_truth.size());
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double x = i / (double)(N - 1), y = j / (double)(N - 1);
                    AD ad = tree->ad_eval_t(x, y, 0.0, prob.dim);
                    if (use_ansatz) ad = apply_boundary_ansatz(prob, ad, x, y);
                    if (!std::isfinite(ad.v.real())) continue;
                    sum_sq_err += std::norm(ad.v - prob.numerical_truth[i * N + j]); ++n_pts;
                }
            }
        }
    } else {
        for (auto& pt : val_dom) {
            AD ad = tree->ad_eval_t(pt.x, pt.y, pt.t, prob.dim);
            if (use_ansatz) ad = apply_boundary_ansatz(prob, ad, pt.x, pt.y);
            Complex u_exact = prob.exact(pt.x, pt.y, pt.t);
            if (!std::isfinite(ad.v.real())) continue;
            sum_sq_err += std::norm(ad.v - u_exact); ++n_pts;
        }
    }
    return (n_pts > 0) ? sum_sq_err / n_pts : 1e18;
}

PISolver::PISolver(const PDEProblem& prob, unsigned seed) : prob_(prob), gen_(seed) {
    priors_ = probe_priors(prob_);
    prob_.priors = priors_; // cacheado en el propio PDEProblem para que evaluate()
                             // pueda exigir las simetrías detectadas sin necesitar
                             // un parámetro nuevo en cada call site.
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
            std::cout << "[INFO] Singularidad detectada en el PDE. Se usara Gradient Descent (Clipped Mode)." << std::endl;
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
    // Enfriamiento (annealing): exploración agresiva al inicio de la corrida,
    // refinamiento más conservador cerca del final para no destruir por azar
    // estructura físicamente correcta ya encontrada. Con piso: si la
    // agresividad cae a 0, la población pierde toda capacidad de escapar de
    // un óptimo local mediocre (visto empíricamente: val_mse quedándose fijo
    // desde generación 0 en Airy/Duffing incluso con cientos de generaciones
    // restantes) — MIN_AGGRESSIVENESS mantiene siempre una vía de escape.
    double progress = max_gen_ > 0 ? std::clamp((double)current_gen_ / max_gen_, 0.0, 1.0) : 0.0;
    constexpr double MIN_AGGRESSIVENESS = 0.3;
    double aggressiveness = MIN_AGGRESSIVENESS + (1.0 - MIN_AGGRESSIVENESS) * (1.0 - progress);
    if (p_dist(gen_) < Config::CROSSOVER_PROB) {
        auto [c1, c2] = tree_crossover(a.tree, b.tree, gen_);
        child.tree = std::move(c1);
    } else {
        child.tree = a.tree->clone();
    }
    if (is_elite) {
        if (p_dist(gen_) < 0.7) child.tree->mutate_erc(gen_, Config::ERC_SIGMA * 0.5);
        else child.tree = tree_mutate(child.tree, gen_, prob_, aggressiveness);
    } else {
        if (p_dist(gen_) < Config::MUTATION_PROB) child.tree = tree_mutate(child.tree, gen_, prob_, aggressiveness);
        if (p_dist(gen_) < 0.4) child.tree->mutate_erc(gen_, Config::ERC_SIGMA);
    }
    return child;
}

void PISolver::update_hall_of_fame() {
    // Paso 1 (barato): entre los candidatos de rank 1 de ESTA generacion, quedarnos
    // con el mejor segun el mini-batch de entrenamiento ya evaluado — no cuesta
    // evaluaciones extra, solo un min() sobre lo que ya se calculo.
    PIIndividual* best_candidate = nullptr;
    double best_train_err = 1e18;
    for (auto& ind : population_) {
        if (ind.rank == 1 && ind.tree_size >= 1 && ind.is_feasible && ind.is_physically_complete(prob_)) {
            double train_err = ind.mse_domain + ind.mse_boundary;
            if (train_err < best_train_err) { best_train_err = train_err; best_candidate = &ind; }
        }
    }
    if (!best_candidate) return;

    // Paso 2: decidir si reemplaza al campeon comparando AMBOS en la misma grilla
    // fija de validacion (get_validation_mse), no el mini-batch ruidoso y distinto
    // de cada generacion. Antes se comparaba ind.mse_domain (batch de esta gen)
    // contra best_ever_.mse_domain (quedaba stale, del batch de otra generacion) —
    // apples-to-oranges que podia bloquear al campeon real o admitir uno con
    // suerte de batch facil.
    double candidate_val = best_candidate->get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_);
    double incumbent_val = has_best_ever_ ? best_ever_.get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_) : 1e18;

    if (!has_best_ever_ || candidate_val < incumbent_val) {
        best_ever_.mse_domain = best_candidate->mse_domain; best_ever_.mse_boundary = best_candidate->mse_boundary;
        best_ever_.rank = best_candidate->rank; best_ever_.crowding = best_candidate->crowding;
        best_ever_.tree_size = best_candidate->tree_size; best_ever_.root_type = best_candidate->root_type;
        if (best_candidate->tree) best_ever_.tree = best_candidate->tree->clone();
        has_best_ever_ = true;
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
    int pool_size = (prob_.dim == 1) ? Config::RAR_CANDIDATES : 2000;
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
    int n_total = (prob_.dim == 1) ? Config::N_DOMAIN : 800;
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
    max_gen_ = max_gen;
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
    for (int i = 0; i < pop_size; ++i) {
        PIIndividual ind;
        if (i < pop_size/2) ind.tree = random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_, priors_);
        else ind.tree = random_tree(Config::MAX_TREE_DEPTH, gen_, prob_);
        if (ind.tree) ind.tree = ind.tree->simplify();
        ind.evaluate(prob_, dom_pts_, bnd_pts_, 0);
        population_.push_back(std::move(ind));
    }
    fast_non_dominated_sort(population_);
    update_hall_of_fame();
    int n_threads = omp_get_max_threads();

    // Viven fuera del loop: el batch ahora persiste entre generaciones (se
    // resamplea cada RESAMPLE_INTERVAL, no en cada una — ver comentario abajo).
    std::vector<Point> curr_dom = dom_pts_;
    std::vector<Point> curr_bnd = bnd_pts_;

    for (int g = 0; g < max_gen; ++g) {
        current_gen_ = g;

        // Stochastic Mini-Batching
        // Batch ~80% del pool disponible (antes 8-16%, luego 30-40%). Diagnostico
        // empirico (ver update_hall_of_fame): con batches chicos, una funcion casi
        // trivial (ej. N(x)~=0 en el ansatz L(x)+B(x)*N(x)) puede lograr residuo de
        // entrenamiento carisimamente bajo por pura coincidencia de muestreo, sin
        // parecerse en nada a la solucion real — se vio directamente en Airy_1D:
        // "0.074*x" con train_err=0.00095 pero val_mse=0.82 (55x peor que el
        // campeon real). Un batch mucho mas grande hace estadisticamente mucho mas
        // dificil que una funcion degenerada tenga residuo bajo en TANTOS puntos
        // por casualidad. Con la build en Release (-O3 -march=native) hay margen
        // de sobra de velocidad para pagar el costo extra.
        //
        // Ademas, el batch se resamplea cada RESAMPLE_INTERVAL generaciones, no en
        // cada una. Antes se resampleaba siempre: la seleccion (offspring vs.
        // resto de la poblacion) comparaba contra un blanco que cambiaba a cada
        // paso, asi que una "mejora" de una generacion a la siguiente podia ser pura casualidad
        // de que tocaron puntos mas faciles, no progreso real. Con un blanco
        // estable durante varias generaciones, la presion de seleccion tiene
        // tiempo de premiar mejoras genuinas en vez de ruido de muestreo.
        constexpr int RESAMPLE_INTERVAL = 8;
        auto resample_batch = [&]() {
            int n_dom = (int)(dom_pts_.size() * 0.8);
            int n_bnd = (int)(bnd_pts_.size() * 0.8);
            curr_dom.clear(); curr_bnd.clear();
            std::sample(dom_pts_.begin(), dom_pts_.end(), std::back_inserter(curr_dom), n_dom, gen_);
            std::sample(bnd_pts_.begin(), bnd_pts_.end(), std::back_inserter(curr_bnd), n_bnd, gen_);
        };

        if (g < max_gen - 50) {
            if (g % RESAMPLE_INTERVAL == 0) resample_batch();
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) {
                population_[i].evaluate(prob_, curr_dom, curr_bnd, current_gen_);
            }
        } else if (g == max_gen - 50) {
            // Re-evaluate on full grid when switching
            curr_dom = dom_pts_;
            curr_bnd = bnd_pts_;
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) {
                population_[i].evaluate(prob_, curr_dom, curr_bnd, current_gen_);
            }
        }

        if (g > 0 && g % Config::RAR_INTERVAL == 0) {
            apply_committee_rar();
            // dom_pts_/bnd_pts_ cambiaron (RAR inyecta puntos de residuo alto):
            // reflejarlo en el batch YA, sin esperar al proximo multiplo de
            // RESAMPLE_INTERVAL.
            if (g >= max_gen - 50) {
                curr_dom = dom_pts_;
                curr_bnd = bnd_pts_;
            } else {
                resample_batch();
            }
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) population_[i].evaluate(prob_, curr_dom, curr_bnd, current_gen_);
        }

        // Re-sort and update hall of fame based on current batch evaluation
        fast_non_dominated_sort(population_);
        update_hall_of_fame();

        if (has_best_ever_) {
            best_ever_.evaluate(prob_, curr_dom, curr_bnd, current_gen_);
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

        auto clone_ind = [](const PIIndividual& ind) {
            PIIndividual c;
            c.mse_domain = ind.mse_domain; c.mse_boundary = ind.mse_boundary;
            c.rank = ind.rank; c.crowding = ind.crowding; c.tree_size = ind.tree_size;
            c.is_feasible = ind.is_feasible; c.constraint_violation = ind.constraint_violation;
            if (ind.tree) c.tree = ind.tree->clone();
            return c;
        };

        // ─── NSGA-II: torneo binario/4-ario + combinar padres+hijos + orden
        // no-dominado + crowding distance (con bono de diversidad estructural
        // por tipo de nodo raíz en calculate_crowding_distance).
        std::vector<PIIndividual> offsprings(pop_size);
        for (int i = 0; i < pop_size; ++i) {
            int r1 = tournament_select(population_, gen_);
            int r2 = tournament_select(population_, gen_);
            offsprings[i] = make_offspring(population_[r1], population_[r2]);
            if (offsprings[i].tree) offsprings[i].tree = offsprings[i].tree->simplify();
        }

        #pragma omp parallel for schedule(dynamic) num_threads(Config::CORES > 0 ? Config::CORES : omp_get_max_threads())
        for (int i = 0; i < pop_size; ++i) {
            offsprings[i].evaluate(prob_, curr_dom, curr_bnd, current_gen_);
        }

        std::vector<PIIndividual> combined;
        combined.reserve(pop_size * 2);
        for (const auto& ind : population_) combined.push_back(clone_ind(ind));
        for (auto& ind : offsprings) combined.push_back(std::move(ind));

        population_ = nsga2_select_next(std::move(combined), pop_size);
        fast_non_dominated_sort(population_);

        // Pulido de constantes para TODA la población, no sólo rank==1. Antes una
        // estructura genuinamente buena con constantes sin afinar podía perder la
        // comparación de Pareto contra estructuras peores pero con mejor suerte de
        // inicialización, y quedaba descartada antes de tener oportunidad de
        // demostrar su valor. Prioridad no es velocidad sino calidad del resultado
        // final; ya tenemos margen de sobra desde el build en Release.
        unsigned int g_seed = gen_();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)population_.size(); ++i) {
            std::mt19937 local_gen(g_seed + i);
            gradient_descent_constants(population_[i], (g % 10 == 0 ? 30 : 10));
        }
        {
            double b_dom = 1e18, b_bnd = 1e18;
            for (auto& ind : population_) if (ind.rank == 1) { b_dom = std::min(b_dom, ind.mse_domain); b_bnd = std::min(b_bnd, ind.mse_boundary); }
            // MSE real del Hall of Fame contra la solucion exacta en grilla fija —
            // a diferencia de b_dom/b_bnd (mini-batch estocastico, distinto cada
            // generacion), esta si es comparable generacion a generacion.
            double val_mse = has_best_ever_ ? best_ever_.get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_) : 1e18;
            history_.push_back({g, b_dom, b_bnd, b_dom + b_bnd, val_mse});
            if (g % 25 == 0) {
                int n_infeasible = 0, front1 = 0;
                for (auto& ind : population_) { if (!ind.is_feasible) n_infeasible++; if (ind.rank == 1) front1++; }
                std::cout << "  [PI/" << prob_.name() << "] gen=" << g
                          << "  best_dom(batch)=" << std::scientific << b_dom
                          << "  best_bnd(batch)=" << b_bnd
                          << "  val_mse(hof)=" << val_mse << std::defaultfloat
                          << "  infeasible=" << n_infeasible << "/" << population_.size()
                          << "  front1=" << front1 << "\n";
            }
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

// Gradiente de mse_domain respecto a cada constante ERC por differenciacion de
// paso complejo (complex-step): perturbar la constante en la direccion
// imaginaria (c + i*h) y evaluar UNA vez da la derivada del residuo con
// precision ~exacta (sin cancelacion por resta ni error de truncamiento de
// diferencias finitas), la mitad de evaluaciones que el esquema anterior
// (1 base + K perturbadas, contra 2K+1). Válido porque el arbol se evalúa en
// Complex de punta a punta y el residuo es holomorfo en la constante para
// todos los PDEs activos salvo Thomas-Fermi (usa abs(.real()) a propósito, ver
// pde_residual_ad) — ese sigue con diferencias finitas (ver caller).
// Sólo se usa cuando el ansatz de frontera aplica (mse_boundary ≡ 0), así el
// residuo de dominio es la única pieza a diferenciar — ver ansatz_applies().
static void compute_domain_gradient_complex_step(const PDEProblem& prob,
                                                   const std::vector<Point>& dom,
                                                   const Node* tree,
                                                   const std::vector<Complex*>& ercs,
                                                   double h,
                                                   std::vector<double>& grads_out)
{
    size_t N = dom.size();
    std::vector<double> res_base(N, 0.0);
    for (size_t p = 0; p < N; ++p) {
        AD ad = tree->ad_eval_t(dom[p].x, dom[p].y, dom[p].t, prob.dim);
        ad = apply_boundary_ansatz(prob, ad, dom[p].x, dom[p].y);
        res_base[p] = prob.pde_residual_ad(ad, dom[p].x, dom[p].y, dom[p].t).real();
    }
    for (size_t k = 0; k < ercs.size(); ++k) {
        Complex orig = *ercs[k];
        *ercs[k] = orig + Complex(0.0, h);
        double grad_sum = 0.0;
        for (size_t p = 0; p < N; ++p) {
            double r = res_base[p];
            if (r * r > 1e4) continue; // punto recortado (item 3): región plana, sin gradiente
            AD ad = tree->ad_eval_t(dom[p].x, dom[p].y, dom[p].t, prob.dim);
            ad = apply_boundary_ansatz(prob, ad, dom[p].x, dom[p].y);
            double res_im = prob.pde_residual_ad(ad, dom[p].x, dom[p].y, dom[p].t).imag();
            double dres_dc = res_im / h;
            grad_sum += r * dres_dc;
        }
        *ercs[k] = orig;
        grads_out[k] = N > 0 ? (2.0 * grad_sum / (double)N) : 0.0;
    }
}

void PISolver::gradient_descent_constants(PIIndividual& ind, int iterations,
                                           const std::vector<Point>* dom_arg,
                                           const std::vector<Point>* bnd_arg) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    // Por defecto opera sobre el mini-batch de entrenamiento (uso in-loop,
    // pulido de toda la poblacion cada generacion). polish_constants() del
    // campeon final pasa val_dom_pts_/val_bnd_pts_ explicitamente para no
    // reintroducir el overfitting al ultimo mini-batch que el Hall of Fame
    // (grilla fija) esta diseñado a evitar.
    const std::vector<Point>& dom = dom_arg ? *dom_arg : dom_pts_;
    const std::vector<Point>& bnd = bnd_arg ? *bnd_arg : bnd_pts_;

    bool use_complex_step = (prob_.type != PDE::THOMAS_FERMI) && ansatz_applies(prob_, bnd.size());

    double lr = 0.05;
    double h = use_complex_step ? 1e-8 : 1e-6;
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> grads(ercs.size(), 0.0);
        ind.evaluate(prob_, dom, bnd, current_gen_);
        if (!ind.is_feasible) return;
        double current_loss = ind.mse_domain + ind.mse_boundary;

        if (use_complex_step) {
            compute_domain_gradient_complex_step(prob_, dom, ind.tree.get(), ercs, h, grads);
            for (size_t i = 0; i < ercs.size(); ++i) {
                if (!std::isfinite(grads[i])) grads[i] = 0.0;
            }
        } else {
            for (size_t i = 0; i < ercs.size(); ++i) {
                double orig = ercs[i]->real();
                *ercs[i] = Complex(orig + h, ercs[i]->imag());
                ind.evaluate(prob_, dom, bnd, current_gen_);
                double loss_plus = ind.mse_domain + ind.mse_boundary;

                *ercs[i] = Complex(orig - h, ercs[i]->imag());
                ind.evaluate(prob_, dom, bnd, current_gen_);
                double loss_minus = ind.mse_domain + ind.mse_boundary;

                *ercs[i] = Complex(orig, ercs[i]->imag());
                grads[i] = (loss_plus - loss_minus) / (2.0 * h);

                // Si el gradiente es inestable o la evaluación fue infactible, lo anulamos
                if (!std::isfinite(grads[i])) grads[i] = 0.0;
            }
        }

        // Gradient Clipping para PDEs Singulares
        if (Config::IS_SINGULAR) {
            for (size_t i = 0; i < ercs.size(); ++i) {
                if (grads[i] > 50.0) grads[i] = 50.0;
                else if (grads[i] < -50.0) grads[i] = -50.0;
            }
        }

        for (size_t i = 0; i < ercs.size(); ++i) {
            *ercs[i] -= Complex(lr * grads[i], 0);
        }
        
        ind.evaluate(prob_, dom, bnd, current_gen_);
        if (!ind.is_feasible || (ind.mse_domain + ind.mse_boundary >= current_loss)) {
            // Revertir y reducir learning rate
            for (size_t i = 0; i < ercs.size(); ++i) {
                *ercs[i] += Complex(lr * grads[i], 0);
            }
            lr *= 0.5;
            ind.evaluate(prob_, dom, bnd, current_gen_);
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
    // Se pule contra la grilla FIJA de validacion (val_dom_pts_/val_bnd_pts_),
    // no el mini-batch de entrenamiento — este metodo se llama una sola vez
    // sobre el campeon del Hall of Fame, ya elegido por su val_mse; pulir
    // contra el mini-batch (que puede ser un subconjunto mas facil o mas
    // dificil que el promedio) puede sobreajustar y empeorar el resultado
    // real reportado, deshaciendo la proteccion que da el Hall of Fame.
    gradient_descent_constants(ind, 200, &val_dom_pts_, &val_bnd_pts_);
    ind.evaluate(prob_, val_dom_pts_, val_bnd_pts_, current_gen_);
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
