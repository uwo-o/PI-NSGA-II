// =============================================================================
// main.cpp  —  Orquestador: PI-NSGA-II (Solo nuestro Algoritmo)
//   Modos:
//     ./build/pi_nsga2              → 1 corrida (comportamiento estándar)
//     ./build/pi_nsga2 --runs N     → N corridas independientes (análisis estadístico)
//
//   Genera CSVs de frentes de Pareto para análisis comparativo.
//   Ecuaciones: Laplace, Poisson, Helmholtz, Liouville, Sine-Gordon.
// =============================================================================

#include "pi_solver.hpp"
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
        if (ind.rank == 1)
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
    f << "gen,best_mse_domain,best_mse_boundary,best_total_mse\n";
    for (auto& s : history) {
        f << s.gen << "," << s.best_mse_domain << ","
          << s.best_mse_boundary << "," << s.best_total_mse << "\n";
    }
}

// ─── Guardar Mejor Expresión en LaTeX ────────────────────────────────────────
template<typename Ind>
void save_best_expression(const std::vector<Ind>& pop, const std::string& path) {
    const Ind* best = nullptr;
    double min_err = 1e18;
    for (auto& ind : pop) {
        if (ind.rank == 1) {
            double err = ind.mse_domain + ind.mse_boundary;
            if (err < min_err) { min_err = err; best = &ind; }
        }
    }
    if (best) {
        std::ofstream f(path);
        f << "$$ \\hat{u}(x,y) = ";
        auto simple_tree = best->tree->simplify(); // Limpieza antes de exportar
        simple_tree->print_latex(f);
        f << " $$" << std::endl;
    }
}

// ─── Guardar Rejilla de Evaluación (para plots 3D/1D) ─────────────────────────
template<typename Ind>
void save_best_grid(const std::vector<Ind>& pop, const PDEProblem& prob, const std::string& path) {
    const Ind* best = nullptr;
    double min_err = 1e18;
    for (auto& ind : pop) {
        if (ind.rank == 1) {
            double err = ind.mse_domain + ind.mse_boundary;
            if (err < min_err) { min_err = err; best = &ind; }
        }
    }
    if (best) {
        std::ofstream f(path);
        if (prob.dim == 1) {
            f << "x,u_exact,u_approx\n";
            for (int i = 0; i <= 100; ++i) {
                double x = (double)i / 100.0;
                f << x << "," << std::real(prob.exact(x, 0)) << "," << std::real(best->tree->eval(x, 0)) << "\n";
            }
        } else {
            f << "x,y,u_exact,u_approx\n";
            int N = 50;
            for (int i = 0; i <= N; ++i) {
                for (int j = 0; j <= N; ++j) {
                    double x = (double)i / N;
                    double y = (double)j / N;
                    f << x << "," << y << "," << std::real(prob.exact(x, y)) << "," << std::real(best->tree->eval(x, y)) << "\n";
                }
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
};

// ─── Cálculo de hipervolumen 2D ──────────────────────────────────────────────
template<typename Ind>
double compute_hypervolume(const std::vector<Ind>& pop,
                           double ref_dom = 1e4,
                           double ref_bnd = 1e4)
{
    std::vector<std::pair<double,double>> pts;
    for (auto& ind : pop)
        if (ind.rank == 1 && ind.mse_domain < ref_dom && ind.mse_boundary < ref_bnd)
            pts.push_back({ind.mse_domain, ind.mse_boundary});

    if (pts.empty()) return 0.0;
    std::sort(pts.begin(), pts.end(), [](auto& a, auto& b){ return a.first < b.first; });

    double hv = 0.0;
    double prev_x = pts[0].first;
    double cur_min_y = pts[0].second;

    for (size_t i = 1; i < pts.size(); ++i) {
        double x = pts[i].first;
        hv += (x - prev_x) * (ref_bnd - cur_min_y);
        cur_min_y = std::min(cur_min_y, pts[i].second);
        prev_x = x;
    }
    hv += (ref_dom - prev_x) * (ref_bnd - cur_min_y);
    return hv / (ref_dom * ref_bnd);
}

template<typename Ind>
Stats compute_stats(const std::vector<Ind>& pop,
                    const std::string& method,
                    const std::string& pde,
                    double rt)
{
    Stats s;
    s.method = method; s.pde = pde; s.runtime_s = rt;
    for (auto& ind : pop) {
        if (ind.rank == 1) {
            s.front_size++;
            s.best_domain = std::min(s.best_domain, ind.mse_domain);
            s.best_bnd    = std::min(s.best_bnd,    ind.mse_boundary);
            s.mean_domain += ind.mse_domain;
            s.mean_bnd    += ind.mse_boundary;
        }
    }
    if (s.front_size > 0) {
        s.mean_domain /= s.front_size;
        s.mean_bnd    /= s.front_size;
    }
    s.hypervolume = compute_hypervolume(pop);
    return s;
}

// ─── Guardar resumen comparativo ─────────────────────────────────────────────
void save_summary(const std::vector<Stats>& stats, const std::string& path) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(10);
    f << "method,pde,pareto_size,best_mse_domain,best_mse_boundary,"
         "mean_mse_domain,mean_mse_boundary,hypervolume,runtime_s\n";
    for (auto& s : stats) {
        f << s.method << "," << s.pde << "," << s.front_size << ","
          << s.best_domain << "," << s.best_bnd << ","
          << s.mean_domain << "," << s.mean_bnd << ","
          << s.hypervolume << "," << s.runtime_s << "\n";
    }
}

// ─── Tabla en consola (Solo PI-NSGA-II) ──────────────────────────────────────
void print_table(const std::string& lbl, const Stats& p) {
    std::cout << "\n+------------------+-------------+-------------+----------+-------------+----------+\n";
    std::cout << "| " << std::left << std::setw(78) << ("  Ecuacion: " + lbl) << "|\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+----------+\n";
    std::cout << "| Metodo           | MSE Dom.    | MSE Bnd.    | Pareto   | Hipervolumen| Tiempo   |\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+----------+\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "| PI-NSGA-II       | " << std::setw(11) << p.best_domain
              << " | " << std::setw(11) << p.best_bnd
              << " | " << std::setw(8)  << p.front_size
              << " | " << std::setw(11) << p.hypervolume
              << " | " << std::setw(7)  << p.runtime_s << "s |\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+----------+\n";
}

// ─── Una corrida completa (Solo PI-NSGA-II) ──────────────────────────────────
std::vector<Stats> run_once(int run_id, const std::string& out_dir, bool verbose) {
    unsigned seed_base = 1000u * (unsigned)(run_id + 1);
    std::vector<PDEProblem> problems;
    for (int d : {1, 2}) {
        problems.push_back(make_laplace(d));
        problems.push_back(make_poisson(d));
        problems.push_back(make_helmholtz(d, 1.0));
        problems.push_back(make_schrodinger(d));
        problems.push_back(make_harmonic_oscillator(d));
    }
    problems.push_back(make_airy());
    problems.push_back(make_nonlinear_poisson());
    problems.push_back(make_liouville());
    problems.push_back(make_sine_gordon());

    std::vector<Stats> all_stats;

    for (auto& prob : problems) {
        std::string lbl = prob.name() + (prob.dim == 1 ? "_1D" : "_2D");
        
        if (verbose) std::cout << "\n[Run] PI-NSGA-II en " << lbl << "\n";
        if (prob.is_numerical) {
            prob.numerical_truth = NumericalSolver::solve(prob, 50);
        }

        auto t0 = std::chrono::steady_clock::now();
        PISolver pi(prob, seed_base + 500u);
        auto pi_pop = pi.run(Config::POP_SIZE, Config::MAX_GEN);
        double pi_rt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        
        save_pareto_csv(pi_pop, out_dir + "/" + lbl + "_pi_pareto.csv", "PI-NSGA-II", prob.name(), prob.dim);
        save_convergence_csv(pi.history(), out_dir + "/" + lbl + "_pi_convergence.csv");
        save_best_expression(pi_pop, out_dir + "/expr_" + lbl + "_PI-NSGA-II.tex");
        save_best_grid(pi_pop, prob, out_dir + "/grid_" + lbl + "_PI-NSGA-II.csv");
        
        Stats ps = compute_stats(pi_pop, "PI-NSGA-II", lbl, pi_rt);
        if (verbose) print_table(lbl, ps);
        all_stats.push_back(ps);
    }
    return all_stats;
}

int main(int argc, char* argv[]) {
    int n_runs = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i+1 < argc) n_runs = std::atoi(argv[i+1]);
    }

    std::cout << "=============================================================\n";
    std::cout << "  PI-NSGA-II --- Orquestador de Ecuaciones PDE\n";
    std::cout << "  Pop=" << Config::POP_SIZE << "  Gen=" << Config::MAX_GEN << "  Runs=" << n_runs << "\n";
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
        
        auto stats = run_once(r, out_dir, verbose);
        all_runs.push_back(stats);
        save_summary(stats, out_dir + "/comparison_summary.csv");

        if (!verbose) {
            std::cout << "  Run " << r << " completed.\n";
        }
    }

    if (n_runs > 1) {
        std::ofstream f_all("results/all_runs_summary.csv");
        f_all << "run,method,pde,pareto_size,best_mse_domain,best_mse_boundary,mean_mse_domain,mean_mse_boundary,hypervolume,runtime_s\n";
        for (int r = 0; r < n_runs; ++r) {
            for (auto& s : all_runs[r]) {
                f_all << r << "," << s.method << "," << s.pde << "," << s.front_size << "," << s.best_domain << ","
                      << s.best_bnd << "," << s.mean_domain << "," << s.mean_bnd << "," << s.hypervolume << "," << s.runtime_s << "\n";
            }
        }
    }

    auto t_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_all).count();
    std::cout << "\n  Benchmark completado en " << t_total << "s. Resultados en ./results/\n";
    return 0;
}
