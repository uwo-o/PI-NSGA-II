#include "pi_solver.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

struct DEPoint {
    std::vector<double> params;
    double score;
};

void PISolver::differential_evolution_polish(PIIndividual& ind, int max_iter) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs;
    ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    int n = ercs.size();
    auto get_score = [&](const std::vector<double>& p) {
        for(int i=0; i<n; ++i) {
            if (std::abs(p[i]) > 1e4) return 1e18; // Límite blando anti-divergencia
            *ercs[i] = Complex(p[i], ercs[i]->imag());
        }
        ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        if (!ind.is_feasible) return 1e18;
        return ind.mse_domain + ind.mse_boundary;
    };

    int NP = std::max(10, n * 5); // Tamaño de población para micro-DE
    std::vector<DEPoint> pop(NP);
    
    // Punto inicial
    std::vector<double> start_p(n);
    for(int i=0; i<n; ++i) start_p[i] = ercs[i]->real();
    pop[0] = {start_p, get_score(start_p)};
    
    std::uniform_real_distribution<double> dist_u(-0.2, 0.2);
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);
    
    // Inicializar población con variaciones del punto inicial
    for (int i = 1; i < NP; ++i) {
        std::vector<double> p = start_p;
        for (int j = 0; j < n; ++j) {
            double noise = dist_u(gen_);
            p[j] += p[j] * noise + dist_u(gen_) * 0.1; // Ruido multiplicativo y aditivo
        }
        pop[i] = {p, get_score(p)};
    }
    
    double F = 0.8; // Factor de mutación
    double CR = 0.9; // Probabilidad de cruza
    
    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < NP; ++i) {
            // Seleccionar 3 distintos a, b, c que no sean i
            int a, b, c;
            do { a = std::uniform_int_distribution<int>(0, NP - 1)(gen_); } while (a == i);
            do { b = std::uniform_int_distribution<int>(0, NP - 1)(gen_); } while (b == i || b == a);
            do { c = std::uniform_int_distribution<int>(0, NP - 1)(gen_); } while (c == i || c == a || c == b);
            
            std::vector<double> u(n);
            int R = std::uniform_int_distribution<int>(0, n - 1)(gen_); // Asegurar al menos una dimensión mutada
            for (int j = 0; j < n; ++j) {
                if (dist_01(gen_) < CR || j == R) {
                    u[j] = pop[a].params[j] + F * (pop[b].params[j] - pop[c].params[j]);
                } else {
                    u[j] = pop[i].params[j];
                }
            }
            
            double score_u = get_score(u);
            // Reemplazo si es mejor o igual
            if (score_u <= pop[i].score) {
                pop[i] = {u, score_u};
            }
        }
    }
    
    // Encontrar el mejor de la población final
    int best_idx = 0;
    for (int i = 1; i < NP; ++i) {
        if (pop[i].score < pop[best_idx].score) {
            best_idx = i;
        }
    }
    
    // Aplicar el mejor resultado
    for(int i=0; i<n; ++i) *ercs[i] = Complex(pop[best_idx].params[i], ercs[i]->imag());
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
}
