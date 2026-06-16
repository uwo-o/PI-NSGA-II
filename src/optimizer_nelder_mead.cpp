#include "pi_solver.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

struct SimplexPoint {
    std::vector<double> params;
    double score;
};

void PISolver::nelder_mead_polish(PIIndividual& ind, int max_iter) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs;
    ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    // Solo optimizamos la parte real de las constantes para este benchmark físico
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

    std::vector<SimplexPoint> simplex(n + 1);
    
    // Punto inicial
    std::vector<double> start_p(n);
    for(int i=0; i<n; ++i) start_p[i] = ercs[i]->real();
    simplex[0] = {start_p, get_score(start_p)};

    // Generar simplex inicial con pequeñas perturbaciones
    for (int i = 1; i <= n; ++i) {
        std::vector<double> p = start_p;
        p[i-1] += (std::abs(p[i-1]) < 1e-6) ? 0.001 : p[i-1] * 0.05;
        simplex[i] = {p, get_score(p)};
    }

    const double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;

    for (int iter = 0; iter < max_iter; ++iter) {
        // 1. Ordenar (menor error primero)
        std::sort(simplex.begin(), simplex.end(), [](const SimplexPoint& a, const SimplexPoint& b) {
            return a.score < b.score;
        });

        if (simplex[n].score - simplex[0].score < 1e-15) break;

        // 2. Centroide de los n mejores
        std::vector<double> x0(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) x0[j] += simplex[i].params[j] / n;

        // 3. Reflexión
        std::vector<double> xr(n);
        for (int j = 0; j < n; ++j) xr[j] = x0[j] + alpha * (x0[j] - simplex[n].params[j]);
        double fr = get_score(xr);

        if (fr >= simplex[0].score && fr < simplex[n-1].score) {
            simplex[n] = {xr, fr};
        } else if (fr < simplex[0].score) {
            // 4. Expansión
            std::vector<double> xe(n);
            for (int j = 0; j < n; ++j) xe[j] = x0[j] + gamma * (xr[j] - x0[j]);
            double fe = get_score(xe);
            if (fe < fr) simplex[n] = {xe, fe};
            else simplex[n] = {xr, fr};
        } else {
            // 5. Contracción
            std::vector<double> xc(n);
            if (fr < simplex[n].score) {
                for (int j = 0; j < n; ++j) xc[j] = x0[j] + rho * (xr[j] - x0[j]);
                double fc = get_score(xc);
                if (fc < fr) { simplex[n] = {xc, fc}; continue; }
            } else {
                for (int j = 0; j < n; ++j) xc[j] = x0[j] + rho * (simplex[n].params[j] - x0[j]);
                double fc = get_score(xc);
                if (fc < simplex[n].score) { simplex[n] = {xc, fc}; continue; }
            }
            // 6. Encoger (Shrink)
            for (int i = 1; i <= n; ++i) {
                for (int j = 0; j < n; ++j)
                    simplex[i].params[j] = simplex[0].params[j] + sigma * (simplex[i].params[j] - simplex[0].params[j]);
                simplex[i].score = get_score(simplex[i].params);
            }
        }
    }

    // Aplicar el mejor resultado
    for(int i=0; i<n; ++i) *ercs[i] = Complex(simplex[0].params[i], ercs[i]->imag());
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
}
