// =============================================================================
// main.cpp  —  Orquestador: PI-NSGA-II vs Koza-BNF
//   Modos:
//     ./build/pi_nsga2              → 1 corrida (comportamiento estándar)
//     ./build/pi_nsga2 --runs N     → N corridas independientes (análisis estadístico)
//
//   Genera CSVs de frentes de Pareto para análisis comparativo.
//   Ecuaciones: Laplace, Poisson, Helmholtz en Ω = [0,1]²
// =============================================================================

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
#include "koza_bnf.hpp"
#include "pi_solver.hpp"

namespace fs = std::filesystem;

// ─── Guardar frente de Pareto como CSV ───────────────────────────────────────
template<typename Ind>
void save_pareto_csv(const std::vector<Ind>& pop,
                     const std::string& path,
                     const std::string& method,
                     const std::string& pde_label)
{
    std::ofstream f(path);
    f << std::fixed << std::setprecision(10);
    f << "method,pde,mse_domain,mse_boundary,rank\n";
    for (auto& ind : pop) {
        if (ind.rank == 1)
            f << method << "," << pde_label << ","
              << ind.mse_domain << "," << ind.mse_boundary << ","
              << ind.rank << "\n";
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
    double hypervolume = 0.0;   // HV relativo al punto de referencia (1e4, 1e4)
};

// ─── Cálculo de hipervolumen 2D (WFG / sweep) ────────────────────────────────
// Punto de referencia: (ref_dom, ref_bnd) — debe dominar a todos los puntos del frente.
template<typename Ind>
double compute_hypervolume(const std::vector<Ind>& pop,
                           double ref_dom = 1e4,
                           double ref_bnd = 1e4)
{
    // Recolectar frente Pareto (rank==1) y ordenar por mse_domain ascendente
    std::vector<std::pair<double,double>> pts;
    for (auto& ind : pop)
        if (ind.rank == 1 &&
            ind.mse_domain < ref_dom &&
            ind.mse_boundary < ref_bnd)
            pts.push_back({ind.mse_domain, ind.mse_boundary});

    if (pts.empty()) return 0.0;

    // Ordenar por primera coordenada ascendente
    std::sort(pts.begin(), pts.end(),
              [](auto& a, auto& b){ return a.first < b.first; });

    // Sweep: HV = suma de rectángulos entre puntos consecutivos
    double hv = 0.0;
    double prev_x = pts[0].first;
    double cur_min_y = pts[0].second;

    for (size_t i = 1; i < pts.size(); ++i) {
        double x = pts[i].first;
        hv += (x - prev_x) * (ref_bnd - cur_min_y);
        cur_min_y = std::min(cur_min_y, pts[i].second);
        prev_x = x;
    }
    // Último rectángulo hasta ref_dom
    hv += (ref_dom - prev_x) * (ref_bnd - cur_min_y);
    return hv;
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

// ─── Validación con solución exacta ──────────────────────────────────────────
void validate_exact(const PDEProblem& prob) {
    auto dom = prob.domain_points(Config::N_DOMAIN);
    double mse = 0.0;
    for (auto& p : dom) {
        double u = prob.exact(p.x, p.y);
        double h   = 1e-5;
        auto fn = [&](double x, double y){ return prob.exact(x, y); };
        double lap = (fn(p.x+h,p.y) - 2*fn(p.x,p.y) + fn(p.x-h,p.y))/(h*h)
                   + (fn(p.x,p.y+h) - 2*fn(p.x,p.y) + fn(p.x,p.y-h))/(h*h);
        
        double res;
        if (prob.type == PDE::SCHRODINGER) {
            double r2 = (p.x - 0.5) * (p.x - 0.5) + (p.y - 0.5) * (p.y - 0.5);
            double V = 4.0 * r2;
            double E = 4.0;
            res = lap + (E - V) * u;
        } else {
            res = lap + prob.k2 * u - prob.source(p.x, p.y);
        }
        mse += res * res;
    }
    mse /= (double)dom.size();
    std::cout << "  [Validacion] Residuo PDE de u* (" << prob.name()
              << "): " << std::scientific << mse << " (esperado ~0)\n";
}

// ─── Extracción de la Mejor Solución Simbólica ───────────────────────────────
template<typename Ind>
const Ind* get_best_individual(const std::vector<Ind>& pop) {
    const Ind* best = nullptr;
    double min_dist = 1e18;
    for (auto& ind : pop) {
        if (ind.rank == 1) {
            // Distancia euclidiana al ideal (0,0) asumiendo misma escala
            double dist = std::sqrt(ind.mse_domain * ind.mse_domain + ind.mse_boundary * ind.mse_boundary);
            if (dist < min_dist) {
                min_dist = dist;
                best = &ind;
            }
        }
    }
    return best;
}

template<typename Ind>
void save_solution_grid(const Ind* best, const PDEProblem& prob,
                        const std::string& path_csv,
                        const std::string& path_tex)
{
    if (!best) return;

    // Guardar expresión LaTeX
    std::ofstream ftex(path_tex);
    ftex << "% Expresion matematica para: " << path_tex << "\n";
    ftex << "$$ \\hat{u}(x,y) = ";
    best->tree->print_latex(ftex);
    ftex << " $$" << std::endl;

    // Guardar malla (grid)
    std::ofstream fcsv(path_csv);
    fcsv << std::fixed << std::setprecision(6);
    fcsv << "x,y,u_exact,u_approx\n";
    
    int N = 50; // Resolución del grid
    for (int i = 0; i < N; ++i) {
        double x = (double)i / (N - 1);
        for (int j = 0; j < N; ++j) {
            double y = (double)j / (N - 1);
            double u_ex = prob.exact(x, y);
            double u_ap = best->tree->eval(x, y);
            fcsv << x << "," << y << "," << u_ex << "," << u_ap << "\n";
        }
    }
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

// ─── Tabla en consola ─────────────────────────────────────────────────────────
void print_table(const std::string& lbl, const Stats& k, const Stats& p) {
    std::cout << "\n+------------------+-------------+-------------+----------+-------------+----------+\n";
    std::cout << "| " << std::left << std::setw(78) << ("  Ecuacion: " + lbl) << "|\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+----------+\n";
    std::cout << "| Metodo           | MSE Dom.    | MSE Bnd.    | Pareto   | Hipervolumen| Tiempo   |\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+----------+\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "| Koza (BNF)       | " << std::setw(11) << k.best_domain
              << " | " << std::setw(11) << k.best_bnd
              << " | " << std::setw(8)  << k.front_size
              << " | " << std::setw(11) << k.hypervolume
              << " | " << std::setw(7)  << k.runtime_s << "s |\n";
    std::cout << "| PI-NSGA-II       | " << std::setw(11) << p.best_domain
              << " | " << std::setw(11) << p.best_bnd
              << " | " << std::setw(8)  << p.front_size
              << " | " << std::setw(11) << p.hypervolume
              << " | " << std::setw(7)  << p.runtime_s << "s |\n";
    std::cout << "+------------------+-------------+-------------+----------+-------------+----------+\n";
}

// ─── Una corrida completa (3 PDEs × 2 métodos) ───────────────────────────────
std::vector<Stats> run_once(int run_id,
                            const std::string& out_dir,
                            bool verbose)
{
    unsigned seed_base = 1000u * (unsigned)(run_id + 1);

    std::vector<PDEProblem> problems = {
        make_laplace(),
        make_poisson(),
        make_helmholtz(1.0),
        make_schrodinger()
    };

    std::vector<Stats> all_stats;

    for (auto& prob : problems) {
        std::string lbl = prob.name();
        if (verbose) {
            std::cout << "\n=====================================================\n";
            std::cout << "  " << lbl << "\n";
            std::cout << "=====================================================\n";
        }

        // ── Koza ─────────────────────────────────────────────────────────────
        if (verbose) std::cout << "\n[1/2] Koza (BNF + Diferencias Finitas)\n";
        auto t0 = std::chrono::steady_clock::now();
        KozaSolver koza(prob, seed_base);
        auto koza_pop = koza.run(Config::POP_SIZE, Config::MAX_GEN);
        double koza_rt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        save_pareto_csv(koza_pop, out_dir + "/" + lbl + "_koza_pareto.csv",
                        "Koza", lbl);
        Stats ks = compute_stats(koza_pop, "Koza", lbl, koza_rt);
        if (verbose) {
            const auto* best_koza = get_best_individual(koza_pop);
            save_solution_grid(best_koza, prob,
                               out_dir + "/grid_" + lbl + "_Koza.csv",
                               out_dir + "/expr_" + lbl + "_Koza.tex");
        }

        // ── PI-NSGA-II ────────────────────────────────────────────────────────
        if (verbose) std::cout << "\n[2/2] PI-NSGA-II (AD Simbolico + ERCs)\n";
        t0 = std::chrono::steady_clock::now();
        PISolver pi(prob, seed_base + 500u);
        auto pi_pop = pi.run(Config::POP_SIZE, Config::MAX_GEN);
        double pi_rt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        save_pareto_csv(pi_pop, out_dir + "/" + lbl + "_pi_pareto.csv",
                        "PI-NSGA-II", lbl);
        Stats ps = compute_stats(pi_pop, "PI-NSGA-II", lbl, pi_rt);
        if (verbose) {
            const auto* best_pi = get_best_individual(pi_pop);
            save_solution_grid(best_pi, prob,
                               out_dir + "/grid_" + lbl + "_PI-NSGA-II.csv",
                               out_dir + "/expr_" + lbl + "_PI-NSGA-II.tex");
        }

        if (verbose) print_table(lbl, ks, ps);
        all_stats.push_back(ks);
        all_stats.push_back(ps);
    }
    return all_stats;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Parsear argumentos
    int n_runs = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i+1 < argc)
            n_runs = std::atoi(argv[i+1]);
    }

    std::cout << "=============================================================\n";
    std::cout << "  PI-NSGA-II vs Koza BNF --- Benchmark de Ecuaciones PDE\n";
    std::cout << "  Laplace | Poisson | Helmholtz  en  Omega = [0,1]^2\n";
    std::cout << "  Pop=" << Config::POP_SIZE
              << "  Gen=" << Config::MAX_GEN
              << "  Pts_dom=" << Config::N_DOMAIN
              << "  Pts_bnd=" << Config::N_BOUNDARY
              << "  Runs=" << n_runs << "\n";
    std::cout << "  Operadores: sin,cos,sinh,cosh,tanh,exp,sqrt,log,atan\n";
    std::cout << "=============================================================\n\n";

    fs::create_directories("results");

    // Validar soluciones exactas (solo 1 vez)
    if (n_runs == 1) {
        std::cout << "--- Validacion de soluciones exactas ---\n";
        for (auto& p : {make_laplace(), make_poisson(), make_helmholtz(1.0), make_schrodinger()})
            validate_exact(p);
        std::cout << "\n";
    }

    bool verbose = (n_runs == 1);
    std::vector<std::vector<Stats>> all_runs;

    for (int r = 0; r < n_runs; ++r) {
        std::string out_dir;
        if (n_runs == 1) {
            out_dir = "results";
        } else {
            out_dir = "results/run_" + std::to_string(r);
            fs::create_directories(out_dir);
            std::cout << "\n[RUN " << (r+1) << "/" << n_runs << "]  dir=" << out_dir << "\n";
        }

        auto stats = run_once(r, out_dir, verbose);
        all_runs.push_back(stats);
        save_summary(stats, out_dir + "/comparison_summary.csv");

        // Progreso compacto en modo multi-run
        if (!verbose) {
            std::cout << "  ";
            for (auto& s : stats)
                std::cout << s.method << "/" << s.pde
                          << " HV=" << std::fixed << std::setprecision(2)
                          << s.hypervolume << "  ";
            std::cout << "\n";
        }
    }

    // Resumen global multi-run (si >1)
    if (n_runs > 1) {
        // Guardar resumen de todas las corridas concatenado
        std::ofstream f_all("results/all_runs_summary.csv");
        f_all << std::fixed << std::setprecision(10);
        f_all << "run,method,pde,pareto_size,best_mse_domain,best_mse_boundary,"
                 "mean_mse_domain,mean_mse_boundary,hypervolume,runtime_s\n";
        for (int r = 0; r < n_runs; ++r) {
            for (auto& s : all_runs[r]) {
                f_all << r << "," << s.method << "," << s.pde << ","
                      << s.front_size << "," << s.best_domain << ","
                      << s.best_bnd << "," << s.mean_domain << ","
                      << s.mean_bnd << "," << s.hypervolume << ","
                      << s.runtime_s << "\n";
            }
        }
        std::cout << "\n  -> Guardado: results/all_runs_summary.csv\n";
    }

    std::cout << "\n=============================================================\n";
    std::cout << "  Listo. Archivos CSV en ./results/\n";
    if (n_runs > 1) {
        std::cout << "  Para estadisticas:  python3 stats_analysis.py\n";
    } else {
        std::cout << "  Para graficar:      python3 plot_pareto.py\n";
    }
    std::cout << "  Para reporte paper: python3 report/generate_report.py\n";
    std::cout << "=============================================================\n";
    return 0;
}
