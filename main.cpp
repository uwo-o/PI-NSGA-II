// =============================================================================
// main.cpp  —  Orquestador: PISR-NSGA-II (Solo nuestro Algoritmo)
//   Modos:
//     ./build/pi_nsga2              → 1 corrida (comportamiento estándar)
//     ./build/pi_nsga2 --runs N     → N corridas independientes (análisis estadístico)
//
//   Genera CSVs de frentes de Pareto para análisis comparativo.
//   Ecuaciones: Laplace, Poisson, Helmholtz, Liouville, Sine-Gordon.
// =============================================================================

#include "pi_solver.hpp"
int last_special_choice = -1;
#include "numerical_solver.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <omp.h>

#include "common.hpp"
#include "pde_problems.hpp"
#include "pi_solver.hpp"

namespace fs = std::filesystem;

// ─── Guardar frente de Pareto como CSV ───────────────────────────────────────
template<typename Ind>
void save_pareto_csv(const std::vector<Ind>& pop,
                     const std::string& path,
                     const std::string& method,
                     const std::string& pde_label,
                     int dim)
{
    std::ofstream f(path);
    f << std::fixed << std::setprecision(10);
    f << "method,pde,dim,mse_domain,mse_boundary,tree_size,rank\n";
    for (auto& ind : pop) {
        f << method << "," << pde_label << "," << dim << ","
          << ind.mse_domain << "," << ind.mse_boundary << ","
          << ind.tree_size << "," << ind.rank << "\n";
    }
}

// ─── Guardar Historial de Convergencia ────────────────────────────────────────
void save_convergence_csv(const std::vector<ConvergenceStats>& history,
                           const std::string& path)
{
    std::ofstream f(path);
    f << "gen,best_mse_domain,best_mse_boundary,best_total_mse,best_val_mse\n";
    for (auto& s : history) {
        f << s.gen << "," << s.best_mse_domain << ","
          << s.best_mse_boundary << "," << s.best_total_mse << ","
          << s.best_val_mse << "\n";
    }
}

// ─── Guardar Mejor Expresión en LaTeX ────────────────────────────────────────
// Recibe directamente al campeon del Hall of Fame (ya validado contra la
// grilla fija dentro de PISolver) en vez de re-escanear la poblacion final
// por mse de entrenamiento — ese escaneo podia elegir un individuo que
// simplemente sobreajustaba el mini-batch de la ultima generacion.
template<typename Ind>
void save_best_expression(const Ind& champion, const std::string& path, int dim = 2) {
    if (!champion.tree) return;
    std::ofstream f(path);
    f << (dim == 1 ? "$$ \\hat{u}(x) = " : "$$ \\hat{u}(x,y) = ");
    auto final_tree = champion.tree->simplify(); // Limpieza simbólica
    final_tree->round_constants(0.05);           // Redondeo inteligente (1.01 -> 1)
    final_tree->print_formal(f, 0);              // Impresión formal sin paréntesis redundantes
    f << " $$" << std::endl;
}

// ─── Guardar Rejilla de Evaluación (para plots 3D/1D) ─────────────────────────
template<typename Ind>
void save_best_grid(const Ind& champion, const PDEProblem& prob, const std::string& path) {
    if (!champion.tree) return;
    std::ofstream f(path);
    if (prob.dim == 1) {
        // El árbol representa N(x) dentro del ansatz U(x) = L(x) + B(x)*N(x)
        // (mismo que usa PIIndividual::evaluate durante el entrenamiento) — hay
        // que deshacerlo acá también, si no el grid grafica la corrección cruda
        // en vez de la solución real.
        double u0 = prob.bc(0.0, 0.0, 0.0).real();
        double u1 = prob.bc(1.0, 0.0, 0.0).real();
        f << "x,u_exact,u_approx\n";
        for (int i = 0; i <= 100; ++i) {
            double x = (double)i / 100.0;
            double N = std::real(champion.tree->eval(x, 0));
            double L = u0 + x * (u1 - u0);
            double B = x * (x - 1.0);
            double u_approx = L + B * N;
            f << x << "," << std::real(prob.numerical_exact(x, 0, 0)) << "," << u_approx << "\n";
        }
    } else {
        // Igual que en 1D: si el ansatz de frontera exacta aplica en 2D (Dirichlet
        // simple en las 4 aristas — no Navier-Stokes, que usa condiciones sobre
        // función de corriente/velocidad), el árbol representa N(x,y) dentro de
        // U(x,y) = L(x,y) + B(x,y)*N(x,y) (interpolación transfinita — mismas
        // fórmulas que PIIndividual::evaluate/apply_boundary_ansatz en pi_solver.cpp).
        bool use_ansatz_2d = (prob.type != PDE::NAVIER_STOKES && prob.type != PDE::NAVIER_STOKES_UNSTEADY);
        f << "x,y,u_exact,u_approx\n";
        for (int i = 0; i <= 30; ++i) {
            for (int j = 0; j <= 30; ++j) {
                double x = (double)i / 30.0;
                double y = (double)j / 30.0;
                double N = std::real(champion.tree->eval(x, y));
                double u_approx = N;
                if (use_ansatz_2d) {
                    double g0 = prob.bc(0.0, y, 0.0).real(), g1 = prob.bc(1.0, y, 0.0).real();
                    double h0 = prob.bc(x, 0.0, 0.0).real(), h1 = prob.bc(x, 1.0, 0.0).real();
                    double c00 = prob.bc(0.0, 0.0, 0.0).real(), c10 = prob.bc(1.0, 0.0, 0.0).real();
                    double c01 = prob.bc(0.0, 1.0, 0.0).real(), c11 = prob.bc(1.0, 1.0, 0.0).real();
                    double L = (1.0 - x) * g0 + x * g1 + (1.0 - y) * h0 + y * h1
                             - (1.0 - x) * (1.0 - y) * c00 - x * (1.0 - y) * c10
                             - (1.0 - x) * y * c01 - x * y * c11;
                    double B = x * (1.0 - x) * y * (1.0 - y);
                    u_approx = L + B * N;
                }
                f << x << "," << y << "," << std::real(prob.numerical_exact(x, y, 0)) << "," << u_approx << "\n";
            }
        }
    }
}

// ─── Estadísticas de una corrida ─────────────────────────────────────────────
struct Stats {
    std::string method, pde;
    int    front_size  = 0;
    double best_domain = 1e18;
    double best_bnd    = 1e18;
    double mean_domain = 0.0;
    double mean_bnd    = 0.0;
    double runtime_s   = 0.0;
    double hypervolume = 0.0;
    double min_bic     = 1e18;
    double infeasible_ratio = 0.0;
    double bloat_factor = 0.0;
    // Error de SOLUCION del campeon (vs. verdad exacta/numerica, en la grilla
    // fija de validacion) — distinto de best_domain (residuo de la EDP, no
    // supervisado). -1 si no hay campeon. Esta es la metrica comparable con
    // el MSE que reportan PySR/PySINDy (que sí entrenan contra la solucion).
    double solution_mse = -1.0;
};

// ─── Hipervolumen 3D: mse_domain × mse_boundary × tree_size (normalizado) ─────
// Algoritmo: slice-by-slice sweep sobre el tercer objetivo (tree_size).
// Complejidad: O(n² log n), exacto para 3 objetivos.
// Referencia: Emmerich et al. (2006), WFG algorithm.
//
// Objetivos (todos a minimizar):
//   f1 = mse_domain     (ref: ref_dom)
//   f2 = mse_boundary   (ref: ref_bnd)
//   f3 = tree_size_norm (ref: 1.0, normalizado al máximo permitido)
// ─────────────────────────────────────────────────────────────────────────────

// Calcula HV 2D del conjunto de puntos 2D respecto al punto de referencia (rx, ry).
// Los puntos deben estar en el espacio (minimización); solo se consideran los que
// dominan al punto de referencia (xi < rx AND yi < ry).
static double hv2d(std::vector<std::pair<double,double>> pts, double rx, double ry)
{
    // Filtrar puntos que no dominan la referencia
    pts.erase(std::remove_if(pts.begin(), pts.end(),
        [rx, ry](const std::pair<double,double>& p){
            return p.first >= rx || p.second >= ry;
        }), pts.end());

    if (pts.empty()) return 0.0;

    // Ordenar por primer objetivo (ascendente)
    std::sort(pts.begin(), pts.end(),
              [](const std::pair<double,double>& a, const std::pair<double,double>& b){
                  return a.first < b.first;
              });

    // Sweepline: acumular área
    double hv = 0.0;
    double prev_x = pts[0].first;
    double cur_min_y = pts[0].second;

    for (size_t i = 1; i < pts.size(); ++i) {
        double x = pts[i].first;
        if (cur_min_y < ry)                         // solo si contribuye
            hv += (x - prev_x) * (ry - cur_min_y);
        cur_min_y = std::min(cur_min_y, pts[i].second);
        prev_x = x;
    }
    if (cur_min_y < ry)
        hv += (rx - prev_x) * (ry - cur_min_y);

    return hv;
}

template<typename Ind>
double compute_hypervolume(const std::vector<Ind>& pop,
                           const PDEProblem& prob,
                           double ref_dom  = 1e4,
                           double ref_bnd  = 1e4,
                           double ref_size = 1.2)   // tree_size normalizado
{
    // ── Recopilar puntos del frente de Pareto (rank == 1) ──────────────────
    struct Pt3 { double f1, f2, f3; };
    std::vector<Pt3> pts;

    // Determinar max tree_size para normalización
    double max_ts = 1.0;
    for (auto& ind : pop)
        if (ind.rank == 1 && ind.is_physically_complete(prob))
            max_ts = std::max(max_ts, (double)ind.tree_size);

    for (auto& ind : pop) {
        if (ind.rank != 1 || !ind.is_physically_complete(prob)) continue;
        double f1 = ind.mse_domain;
        double f2 = ind.mse_boundary;
        double f3 = (double)ind.tree_size / max_ts;   // normalizar a [0,1]

        // Filtrar puntos fuera de la caja de referencia
        if (f1 >= ref_dom || f2 >= ref_bnd || f3 >= ref_size) continue;
        pts.push_back({f1, f2, f3});
    }

    if (pts.empty()) return 0.0;

    // ── Slice-by-slice sobre f3 (ascendente = menor tree_size primero) ─────
    // Ordenar por f3 ascendente
    std::sort(pts.begin(), pts.end(),
              [](const Pt3& a, const Pt3& b){ return a.f3 < b.f3; });

    double hv3 = 0.0;
    double prev_f3 = pts[0].f3;

    // Para cada "slice" entre f3[i-1] y f3[i], calcular el HV 2D del
    // conjunto acumulado de puntos proyectados sobre (f1, f2).
    std::vector<std::pair<double,double>> slice_pts;

    for (size_t i = 0; i < pts.size(); ++i) {
        slice_pts.push_back({pts[i].f1, pts[i].f2});

        // Grosor del slice en la dimensión f3
        double next_f3 = (i + 1 < pts.size()) ? pts[i + 1].f3 : ref_size;
        double thickness = next_f3 - prev_f3;

        if (thickness > 0.0) {
            double area = hv2d(slice_pts, ref_dom, ref_bnd);
            hv3 += area * thickness;
        }
        prev_f3 = next_f3;
    }

    // Normalizar por el volumen total de la caja de referencia
    return hv3 / (ref_dom * ref_bnd * ref_size);
}

// `champion`, si se entrega, es el individuo del Hall of Fame (ya validado en
// grilla fija) del que se toman best_domain/best_bnd/min_bic — el resto de las
// estadisticas (front_size, medias, infeasible_ratio, bloat_factor, hipervolumen)
// siguen viniendo del escaneo de toda la poblacion final, que es diversidad
// real y no necesita el campeon validado.
template<typename Ind>
Stats compute_stats(const std::vector<Ind>& pop,
                    const PDEProblem& prob,
                    const std::string& method,
                    const std::string& pde,
                    double rt,
                    const Ind* champion = nullptr,
                    double champion_solution_mse = -1.0)
{
    Stats s;
    s.method = method; s.pde = pde; s.runtime_s = rt;
    int infeasible = 0;
    int total_nodes = 0;
    for (auto& ind : pop) {
        if (!ind.is_feasible) {
            infeasible++;
        }
        total_nodes += ind.tree_size;

        // Análisis del Frente de Pareto (Individuos de Rank 1 y Físicamente Completos)
        if (ind.rank == 1 && ind.is_feasible && ind.is_physically_complete(prob)) {
            s.front_size++;
            s.mean_domain += ind.mse_domain;
            s.mean_bnd += ind.mse_boundary;
        }
    }
    if (champion && champion->tree) {
        double n_samples = 400.0; // aprox n_domain + n_boundary
        double current_total = champion->mse_domain + champion->mse_boundary;
        s.best_domain = champion->mse_domain;
        s.best_bnd = champion->mse_boundary;
        s.min_bic = n_samples * std::log(std::max(current_total, 1e-16)) + champion->tree_size * std::log(n_samples);
        s.solution_mse = champion_solution_mse;
    }
    if (!pop.empty()) {
        s.infeasible_ratio = 100.0 * infeasible / pop.size();
        s.bloat_factor = (double)total_nodes / pop.size();
    }
    if (s.front_size > 0) {
        s.mean_domain /= s.front_size;
        s.mean_bnd    /= s.front_size;
    }
    s.hypervolume = compute_hypervolume(pop, prob);
    return s;
}

// ─── Guardar resumen comparativo ─────────────────────────────────────────────
void save_summary(const std::vector<Stats>& stats, const std::string& path) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(10);
    f << "method,pde,pareto_size,best_mse_domain,best_mse_boundary,"
         "mean_mse_domain,mean_mse_boundary,best_bic,runtime_s,solution_mse\n";
    for (auto& s : stats) {
        f << s.method << "," << s.pde << "," << s.front_size << ","
          << s.best_domain << "," << s.best_bnd << ","
          << s.mean_domain << "," << s.mean_bnd << ","
          << s.min_bic << "," << s.runtime_s << "," << s.solution_mse << "\n";
    }
}

// ─── Tabla en consola (Solo PISR-NSGA-II) ──────────────────────────────────────
void print_table(const std::string& lbl, const Stats& p) {
    std::cout << "\n+------------------+-------------+-------------+----------+-------------+-------------+----------+\n";
    std::cout << "| " << std::left << std::setw(92) << ("  Ecuacion: " + lbl) << "|\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+-------------+----------+\n";
    std::cout << "| Metodo           | MSE Dom.    | MSE Bnd.    | Pareto   | Best BIC    | MSE Solucion| Tiempo   |\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+-------------+----------+\n";
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "| PISR-NSGA-II     | " << std::setw(11) << p.best_domain
              << " | " << std::setw(11) << p.best_bnd
              << std::defaultfloat << std::setprecision(4)
              << " | " << std::setw(8)  << p.front_size
              << std::scientific << std::setprecision(4)
              << " | " << std::setw(11) << p.min_bic
              << " | " << std::setw(11) << p.solution_mse
              << std::defaultfloat << std::setprecision(4)
              << " | " << std::setw(7)  << p.runtime_s << "s |\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+-------------+----------+\n";
    std::cout << "| Health Analytics | Infeasible: " << std::setw(5) << std::fixed << std::setprecision(1) << p.infeasible_ratio << "% | Bloat Factor (Avg nodes): " << std::setw(5) << p.bloat_factor << "                       |\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+-------------+----------+\n";
}

// ─── Una corrida completa (Solo PISR-NSGA-II) ──────────────────────────────────
std::vector<Stats> run_once(int run_id, const std::string& out_dir, bool verbose, bool is_test, const std::string& only_pde, bool replicable) {
    unsigned seed_base = replicable ? (1000u * (unsigned)(run_id + 1)) : std::random_device{}();
    std::vector<PDEProblem> all_problems;
    
    // 1D y 2D Hardcore Equations
    for (int d : {1, 2}) {
        all_problems.push_back(make_airy(d));
        all_problems.push_back(make_fisher(d));
        all_problems.push_back(make_duffing(d));
        all_problems.push_back(make_thomas_fermi(d));
    }

    // Navier-Stokes (The "Boss" Analyticals)
    all_problems.push_back(make_navier_stokes());
    all_problems.push_back(make_navier_stokes_unsteady());
    
    // 1D Only Hardcore Numerical Equations
    all_problems.push_back(make_lane_emden());
    all_problems.push_back(make_troesch());
    all_problems.push_back(make_ginzburg_landau());
    all_problems.push_back(make_painleve1());

    std::vector<PDEProblem> problems;
    if (!only_pde.empty()) {
        for (auto& p : all_problems) {
            std::string lbl = p.name() + (p.dim == 1 ? "_1D" : "_2D");
            if (lbl == only_pde) {
                problems.push_back(p);
            }
        }
        if (problems.empty()) {
            std::cerr << "[ERROR] PDE '" << only_pde << "' no encontrada.\n";
        }
    } else {
        problems = all_problems;
    }

    std::vector<Stats> all_stats;


    for (auto& prob : problems) {
        std::string lbl = prob.name() + (prob.dim == 1 ? "_1D" : "_2D");
        
        if (verbose) std::cout << "\n[Run] PISR-NSGA-II en " << lbl << "\n";
        if (prob.is_numerical) {
            prob.numerical_truth = NumericalSolver::solve(prob, 50);
        }

        // Run Numerical Solver (RK4/FDM) baseline for comparison
        bool has_numerical = false;
        double num_rt = 0.0;
        double num_mse_dom = 0.0;
        double num_mse_bnd = 0.0;

        if (prob.type != PDE::NAVIER_STOKES && prob.type != PDE::NAVIER_STOKES_UNSTEADY) {
            has_numerical = true;
            auto t_num0 = std::chrono::steady_clock::now();
            auto num_sol = NumericalSolver::solve(prob, 50);
            num_rt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_num0).count();

            if (prob.is_numerical) {
                // For numerical equations, the numerical solution is the ground truth
                num_mse_dom = 0.0;
                num_mse_bnd = 0.0;
            } else {
                double sum_sq_dom = 0.0;
                if (prob.dim == 1) {
                    int N = num_sol.size();
                    double h = 1.0 / (N - 1);
                    for (int i = 0; i < N; ++i) {
                        double x = i * h;
                        double diff = std::abs(num_sol[i] - prob.exact(x, 0.0));
                        sum_sq_dom += diff * diff;
                    }
                    num_mse_dom = sum_sq_dom / N;
                    num_mse_bnd = 0.0;
                } else {
                    int N = std::sqrt(num_sol.size());
                    double h = 1.0 / (N - 1);
                    for (int i = 0; i < N; ++i) {
                        for (int j = 0; j < N; ++j) {
                            double x = i * h;
                            double y = j * h;
                            double diff = std::abs(num_sol[i * N + j] - prob.exact(x, y));
                            sum_sq_dom += diff * diff;
                        }
                    }
                    num_mse_dom = sum_sq_dom / (N * N);
                    num_mse_bnd = 0.0;
                }
            }
        }

        {
            auto t0 = std::chrono::steady_clock::now();
            PISolver pi(prob, seed_base + 500u);
            int pop = is_test ? 10 : Config::POP_SIZE;
            int gen = is_test ? 5 : Config::MAX_GEN;
            auto pi_pop = pi.run(pop, gen);
            
            // ─── Post-Evolution Polishing ───
            // Pulimos al campeón del Hall of Fame (ya validado en grilla fija dentro
            // de PISolver, ver update_hall_of_fame) en vez de re-elegir "el mejor" acá
            // escaneando mse de entrenamiento de la población final — ese segundo
            // escaneo podía terminar eligiendo (y puliendo, y exportando) un individuo
            // que sólo sobreajustaba el mini-batch de la última generación, no la
            // solución real. is_physically_complete (chequeado antes de admitir al
            // campeón) ya cubre el filtro de variables que se hacía acá.
            PIIndividual champion;
            bool has_champion = pi.has_champion();
            double champion_sol_mse = -1.0;
            if (has_champion) {
                const PIIndividual& hof = pi.champion();
                champion.mse_domain = hof.mse_domain; champion.mse_boundary = hof.mse_boundary;
                champion.rank = hof.rank; champion.crowding = hof.crowding;
                champion.tree_size = hof.tree_size; champion.root_type = hof.root_type;
                champion.is_feasible = hof.is_feasible; champion.constraint_violation = hof.constraint_violation;
                if (hof.tree) champion.tree = hof.tree->clone();
                pi.polish_constants(champion);
                // Error de solucion real (vs. verdad), no el residuo de la EDP —
                // ver comentario en Stats::solution_mse.
                champion_sol_mse = pi.solution_mse(champion);
            }

            double pi_rt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            save_pareto_csv(pi_pop, out_dir + "/" + lbl + "_pi_gn_pareto.csv", "PISR-NSGA-II", prob.name(), prob.dim);
            save_convergence_csv(pi.history(), out_dir + "/" + lbl + "_pi_gn_convergence.csv");
            if (has_champion) {
                save_best_expression(champion, out_dir + "/expr_" + lbl + "_PISR-EMOAD.tex", prob.dim);
                save_best_grid(champion, prob, out_dir + "/grid_" + lbl + "_PISR-EMOAD.csv");
            }

            Stats ps = compute_stats(pi_pop, prob, "PISR-NSGA-II", lbl, pi_rt, has_champion ? &champion : nullptr, champion_sol_mse);
            if (verbose) print_table(lbl, ps);
            all_stats.push_back(ps);
        }

        if (has_numerical) {
            Stats ns;
            ns.method = "RK4/FDM";
            ns.pde = lbl;
            ns.front_size = 1;
            ns.best_domain = num_mse_dom;
            ns.best_bnd = num_mse_bnd;
            ns.mean_domain = num_mse_dom;
            ns.mean_bnd = num_mse_bnd;
            ns.runtime_s = num_rt;
            ns.hypervolume = 0.0;
            ns.solution_mse = num_mse_dom; // ya es error contra prob.exact(), misma metrica
            all_stats.push_back(ns);
        }
    }
    return all_stats;
}

void print_usage(char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --runs N          Number of independent runs (default: 1)\n"
              << "  --test            Fast test mode (small pop/gen)\n"
              << "  --replicable      Use deterministic seeds for reproducibility\n"
              << "  --only NAME       Run only a specific PDE (e.g., Airy_1D)\n"
              << "  --pop N           Population size (default: " << Config::POP_SIZE << ")\n"
              << "  --gen N           Max generations (default: " << Config::MAX_GEN << ")\n"
              << "  --domain N        Number of domain points (default: " << Config::N_DOMAIN << ")\n"
              << "  --boundary N      Number of boundary points (default: " << Config::N_BOUNDARY << ")\n"
              << "  --depth N         Max tree depth (default: " << Config::MAX_TREE_DEPTH << ")\n"
              << "  --sigma F         ERC mutation sigma (default: " << Config::ERC_SIGMA << ")\n"
              << "  --stop F          Convergence threshold (default: " << Config::STOP_THRESHOLD << ")\n"
              << "  --cores N         Number of CPU threads (default: 1)\n"
              << "  --help            Show this help\n";
}

int main(int argc, char* argv[]) {
    int n_runs = 1;
    bool is_test = false;
    bool replicable = false;
    std::string only_pde = "";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i+1 < argc) n_runs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--test") == 0) is_test = true;
        else if (std::strcmp(argv[i], "--replicable") == 0) replicable = true;
        else if (std::strcmp(argv[i], "--only") == 0 && i+1 < argc) only_pde = argv[++i];
        else if (std::strcmp(argv[i], "--pop") == 0 && i+1 < argc) Config::POP_SIZE = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--gen") == 0 && i+1 < argc) Config::MAX_GEN = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--domain") == 0 && i+1 < argc) Config::N_DOMAIN = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--boundary") == 0 && i+1 < argc) Config::N_BOUNDARY = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--depth") == 0 && i+1 < argc) Config::MAX_TREE_DEPTH = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sigma") == 0 && i+1 < argc) Config::ERC_SIGMA = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--stop") == 0 && i+1 < argc) Config::STOP_THRESHOLD = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--general") == 0) Config::GENERAL_MODE = true;
        else if (std::strcmp(argv[i], "--cores") == 0 && i+1 < argc) Config::CORES = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); return 0; }
    }

    // Aplicar configuración de paralelismo
    if (Config::CORES > 0) {
        omp_set_num_threads(Config::CORES);
    } else {
        Config::CORES = omp_get_max_threads();
        omp_set_num_threads(Config::CORES);
    }

    if (is_test) {
        int max_threads = omp_get_max_threads();
        int threads_to_use = std::max(1, max_threads - 2);
        omp_set_num_threads(threads_to_use);
        std::cout << "[INFO] Modo --test activado. Usando " << threads_to_use << " nucleos.\n";
        Config::POP_SIZE = 10;
        Config::MAX_GEN = 5;
    }

    std::cout << "=============================================================\n";
    std::cout << "  PISR-NSGA-II --- Orquestador de Ecuaciones PDE\n";
    std::cout << "  Pop=" << Config::POP_SIZE << "  Gen=" << Config::MAX_GEN << "  Runs=" << n_runs << "\n";
    if (replicable) std::cout << "  [!] Modo --replicable ACTIVADO (Semillas Deterministas)\n";
    std::cout << "=============================================================\n\n";

    fs::create_directories("results");
    bool verbose = true; 
    std::vector<std::vector<Stats>> all_runs;

    auto t0_all = std::chrono::steady_clock::now();
    for (int r = 0; r < n_runs; ++r) {
        if (n_runs > 1) {
            std::cout << "\n\033[1;34m" << "#############################################################" << "\033[0m\n";
            std::cout << "\033[1;34m" << "  INICIANDO RUN " << r + 1 << " / " << n_runs << "\033[0m\n";
            std::cout << "\033[1;34m" << "#############################################################" << "\033[0m\n";
        }
        
        auto t0_run = std::chrono::steady_clock::now();
        std::string out_dir = (n_runs == 1) ? "results" : "results/run_" + std::to_string(r);
        if (n_runs > 1) fs::create_directories(out_dir);
        
        auto stats = run_once(r, out_dir, verbose, is_test, only_pde, replicable);
        all_runs.push_back(stats);
        save_summary(stats, out_dir + "/comparison_summary.csv");

        if (!verbose) {
            std::cout << "  Run " << r << " completed.\n";
        }
    }

    std::ofstream f_all("results/all_runs_summary.csv");
    f_all << "run,method,pde,pareto_size,best_mse_domain,best_mse_boundary,mean_mse_domain,mean_mse_boundary,hypervolume,runtime_s,solution_mse\n";
    for (int r = 0; r < n_runs; ++r) {
        for (auto& s : all_runs[r]) {
            f_all << r << "," << s.method << "," << s.pde << "," << s.front_size << "," << s.best_domain << ","
                  << s.best_bnd << "," << s.mean_domain << "," << s.mean_bnd << "," << s.hypervolume << "," << s.runtime_s << "," << s.solution_mse << "\n";
        }
    }

    auto t_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_all).count();
    std::cout << "\n  Benchmark completado en " << t_total << "s. Resultados en ./results/\n";
    return 0;
}
