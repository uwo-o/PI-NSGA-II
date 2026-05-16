#pragma once
// =============================================================================
// nsga2.hpp  —  Algoritmo NSGA-II genérico
//   Trabaja con cualquier vector de Individual (o derivado) que tenga
//   mse_domain, mse_boundary, rank, crowding.
// =============================================================================

#include "common.hpp"
#include <vector>
#include <algorithm>
#include <map>

// ─── Clasificación no dominada ────────────────────────────────────────────────
// Asigna rank y devuelve los frentes F1, F2, …
// Cada frente es un vector de índices en `pop`.
template<typename Ind>
std::vector<std::vector<int>> fast_non_dominated_sort(std::vector<Ind>& pop) {
    int n = (int)pop.size();
    std::vector<int>              domination_count(n, 0);
    std::vector<std::vector<int>> dominated_by(n);
    std::vector<std::vector<int>> fronts(1);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            // 3 Objetivos para el frente de Pareto: Domain MSE, Boundary MSE y Tree Size (Complejidad)
            bool i_dom_j = (pop[i].mse_domain   <= pop[j].mse_domain   &&
                            pop[i].mse_boundary <= pop[j].mse_boundary &&
                            pop[i].tree_size    <= pop[j].tree_size)      &&
                           (pop[i].mse_domain   <  pop[j].mse_domain   ||
                            pop[i].mse_boundary <  pop[j].mse_boundary ||
                            pop[i].tree_size    <  pop[j].tree_size);

            bool j_dom_i = (pop[j].mse_domain   <= pop[i].mse_domain   &&
                            pop[j].mse_boundary <= pop[i].mse_boundary &&
                            pop[j].tree_size    <= pop[i].tree_size)      &&
                           (pop[j].mse_domain   <  pop[i].mse_domain   ||
                            pop[j].mse_boundary <  pop[i].mse_boundary ||
                            pop[j].tree_size    <  pop[i].tree_size);

            if (i_dom_j) dominated_by[i].push_back(j);
            else if (j_dom_i) domination_count[i]++;
        }
        if (domination_count[i] == 0) {
            pop[i].rank = 1;
            fronts[0].push_back(i);
        }
    }
    int k = 0;
    while (!fronts[k].empty()) {
        std::vector<int> next_front;
        for (int i : fronts[k]) {
            for (int j : dominated_by[i]) {
                if (--domination_count[j] == 0) {
                    pop[j].rank = k + 2;
                    next_front.push_back(j);
                }
            }
        }
        fronts.push_back(next_front);
        ++k;
    }
    fronts.pop_back(); // último frente vacío
    return fronts;
}

// ─── Distancia de hacinamiento (crowding distance) ────────────────────────────
template<typename Ind>
void crowding_distance_assignment(std::vector<Ind>& pop,
                                  const std::vector<int>& front) {
    int sz = (int)front.size();
    if (sz <= 2) {
        for (int i : front) pop[i].crowding = 1e18;
        return;
    }
    // Reset
    for (int i : front) pop[i].crowding = 0.0;

    // Para cada objetivo (Domain MSE, Boundary MSE, Tree Size)
    auto assign_obj = [&](auto getter) {
        std::vector<int> sorted = front;
        std::sort(sorted.begin(), sorted.end(),
                  [&](int a, int b){ return getter(pop[a]) < getter(pop[b]); });
        pop[sorted.front()].crowding = 1e18;
        pop[sorted.back()].crowding  = 1e18;
        double range = getter(pop[sorted.back()]) - getter(pop[sorted.front()]);
        if (range < 1e-12) return;
        for (int k = 1; k < sz - 1; ++k)
            pop[sorted[k]].crowding +=
                (getter(pop[sorted[k+1]]) - getter(pop[sorted[k-1]])) / range;
    };
    assign_obj([](const Ind& i){ return i.mse_domain;   });
    assign_obj([](const Ind& i){ return i.mse_boundary; });
    assign_obj([](const Ind& i){ return (double)i.tree_size; });

    // ─── Bono de Diversidad Estructural (Niching) ───
    // Calculamos la frecuencia de cada tipo de raíz en este frente
    std::map<NodeType, int> counts;
    for (int i : front) counts[pop[i].root_type]++;
    
    for (int i : front) {
        if (pop[i].crowding < 1e15) { // No tocar los bordes del frente
            double frequency = (double)counts[pop[i].root_type] / sz;
            // Bonus Agresivo: penaliza fuertemente la uniformidad estructural
            pop[i].crowding *= (1.0 + 2.0 / (frequency + 0.05));
        }
    }
}

// ─── Comparador NSGA-II (rank y crowding) ────────────────────────────────────
template<typename Ind>
bool nsga2_dominates(const Ind& a, const Ind& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.crowding > b.crowding;
}

// ─── Selección por torneo binario ─────────────────────────────────────────────
template<typename Ind>
int tournament_select(const std::vector<Ind>& pop, std::mt19937& gen) {
    std::uniform_int_distribution<int> dist(0, (int)pop.size() - 1);
    int best = dist(gen);
    for (int i = 1; i < Config::TOURNAMENT_SIZE; ++i) {
        int candidate = dist(gen);
        if (nsga2_dominates(pop[candidate], pop[best])) {
            best = candidate;
        }
    }
    return best;
}

// ─── Selección de la siguiente generación ─────────────────────────────────────
// Combina padres + hijos (2N), aplica sorting + crowding, retiene los N mejores.
template<typename Ind>
std::vector<Ind> nsga2_select_next(std::vector<Ind> combined, int pop_size) {
    auto fronts = fast_non_dominated_sort(combined);
    for (auto& f : fronts)
        crowding_distance_assignment(combined, f);

    std::vector<Ind> next;
    next.reserve(pop_size);
    for (auto& front : fronts) {
        if ((int)(next.size() + front.size()) <= pop_size) {
            for (int i : front) next.push_back(std::move(combined[i]));
        } else {
            std::sort(front.begin(), front.end(),
                      [&](int a, int b){ return nsga2_dominates(combined[a], combined[b]); });
            int remaining = pop_size - (int)next.size();
            for (int k = 0; k < remaining; ++k)
                next.push_back(std::move(combined[front[k]]));
            break;
        }
    }
    return next;
}
