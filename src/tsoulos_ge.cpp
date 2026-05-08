// =============================================================================
// tsoulos_ge.cpp  —  Método Koza con gramática BNF y diferencias finitas
// =============================================================================

#include "tsoulos_ge.hpp"
#include "nsga2.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>

// ─── Gramática BNF ────────────────────────────────────────────────────────────
NodePtr bnf_to_tree(const std::vector<int>& codons,
                    int& idx, int depth, int max_depth)
{
    if (codons.empty()) return make_erc(1.0);
    auto next_codon = [&]() -> int {
        int c = codons[idx % codons.size()];
        idx++;
        return c;
    };
    static const NodeType UNARY_OPS[] = {
        NodeType::SIN,  NodeType::COS,
        NodeType::SINH, NodeType::COSH,
        NodeType::EXP,  NodeType::SQR
    };
    static constexpr int N_UNARY = 6;

    if (depth >= max_depth) {
        int c = next_codon();
        if (c % 3 == 0) return make_var('x');
        if (c % 3 == 1) return make_var('y');
        return make_erc((double)(next_codon() % 10));
    }

    int rule = next_codon() % 4;
    if (rule == 0) {
        NodePtr L = bnf_to_tree(codons, idx, depth + 1, max_depth);
        int op_code = next_codon() % 4;
        NodePtr R = bnf_to_tree(codons, idx, depth + 1, max_depth);
        NodeType op = (op_code == 0) ? NodeType::ADD
                    : (op_code == 1) ? NodeType::SUB
                    : (op_code == 2) ? NodeType::MUL
                    :                  NodeType::DIV;
        return make_binary(op, std::move(L), std::move(R));
    }
    else if (rule == 1) {
        NodeType op = UNARY_OPS[next_codon() % N_UNARY];
        NodePtr child = bnf_to_tree(codons, idx, depth + 1, max_depth);
        return make_unary(op, std::move(child));
    }
    else if (rule == 2) {
        return (next_codon() % 2 == 0) ? make_var('x') : make_var('y');
    }
    else {
        return make_erc((double)(next_codon() % 10));
    }
}

void TsoulosIndividual::decode() {
    int idx = 0;
    tree = bnf_to_tree(codons, idx, 0, Config::MAX_TREE_DEPTH);
}

void TsoulosIndividual::evaluate(const PDEProblem& prob,
                              const std::vector<Point>& dom,
                              const std::vector<Point>& bnd)
{
    if (!tree) decode();
    if (!tree) { mse_domain = 1e10; mse_boundary = 1e10; tree_size = 0; root_type = NodeType::UNKNOWN; return; }
    tree_size = tree->count_nodes();
    root_type = tree->get_type();

    double sum_dom = 0.0;
    double total_w = 0.0;
    for (auto& p : dom) {
        double lap = fd_laplacian(tree, p.x, p.y);
        double u   = tree->eval(p.x, p.y);
        double res;
        if (prob.type == PDE::SCHRODINGER) {
            double r2 = (p.x - 0.5) * (p.x - 0.5) + (p.y - 0.5) * (p.y - 0.5);
            double V = 4.0 * r2;
            double E = 4.0;
            res = lap + (E - V) * u;
        } else {
            res = lap + prob.k2 * u - prob.source(p.x, p.y);
        }
        if (!std::isfinite(res)) { mse_domain = 1e10; mse_boundary = 1e10; return; }
        double dist_x = std::min(p.x, 1.0 - p.x);
        double dist_y = std::min(p.y, 1.0 - p.y);
        double min_dist = std::min(dist_x, dist_y);
        double weight = 1.0 / (min_dist + 0.1); 
        sum_dom += weight * res * res;
        total_w += weight;
    }
    mse_domain = (sum_dom / total_w);

    double sum_bnd = 0.0;
    for (auto& p : bnd) {
        double u    = tree->eval(p.x, p.y);
        double diff = u - prob.bc(p.x, p.y);
        if (!std::isfinite(diff)) { mse_boundary = 1e10; return; }
        sum_bnd += diff * diff;
    }
    
    double raw_bc_mse = sum_bnd / (double)bnd.size();
    mse_boundary = raw_bc_mse * Config::BC_WEIGHT;
    
    // Penalización continua también en Tsoulos para evitar el "engaño" de la solución nula
    mse_domain += 100.0 * raw_bc_mse;
}

TsoulosSolver::TsoulosSolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed)
{
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
}

TsoulosIndividual TsoulosSolver::random_individual() {
    TsoulosIndividual ind;
    ind.codons.resize(Config::CODON_LENGTH);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& c : ind.codons) c = d(gen_);
    ind.decode();
    ind.evaluate(prob_, dom_pts_, bnd_pts_);
    return ind;
}

TsoulosIndividual TsoulosSolver::crossover(const TsoulosIndividual& a, const TsoulosIndividual& b) {
    TsoulosIndividual child;
    child.codons.resize(Config::CODON_LENGTH);
    std::uniform_int_distribution<int> cut(1, Config::CODON_LENGTH - 1);
    int c = cut(gen_);
    for (int i = 0; i < Config::CODON_LENGTH; ++i)
        child.codons[i] = (i < c) ? a.codons[i] : b.codons[i];
    child.decode();
    return child;
}

void TsoulosSolver::mutate(TsoulosIndividual& ind) {
    std::uniform_int_distribution<int> pos(0, Config::CODON_LENGTH - 1);
    std::uniform_int_distribution<int> val(0, 255);
    int n_mut = std::max(1, Config::CODON_LENGTH / 10);
    for (int i = 0; i < n_mut; ++i)
        ind.codons[pos(gen_)] = val(gen_);
    ind.decode();
}

std::vector<TsoulosIndividual> TsoulosSolver::run(int pop_size, int max_gen) {
    population_.clear();
    population_.reserve(pop_size);
    for (int i = 0; i < pop_size; ++i)
        population_.push_back(random_individual());
    history_.clear();
    for (int gen = 0; gen < max_gen; ++gen) {
        std::vector<TsoulosIndividual> offspring;
        offspring.reserve(pop_size);
        while ((int)offspring.size() < pop_size) {
            int p1 = tournament_select(population_, gen_);
            int p2 = tournament_select(population_, gen_);
            TsoulosIndividual child = crossover(population_[p1], population_[p2]);
            mutate(child);
            child.decode();
            child.evaluate(prob_, dom_pts_, bnd_pts_);
            offspring.push_back(std::move(child));
        }
        std::vector<TsoulosIndividual> combined;
        combined.reserve(pop_size * 2);
        for (auto& x : population_) combined.push_back(std::move(x)); 
        for (auto& x : offspring)   combined.push_back(std::move(x));
        population_ = nsga2_select_next(std::move(combined), pop_size);

        double b_dom = 1e18, b_bnd = 1e18, b_tot = 1e18;
        for (auto& ind : population_) {
            if (ind.rank == 1) {
                b_dom = std::min(b_dom, ind.mse_domain);
                b_bnd = std::min(b_bnd, ind.mse_boundary);
                b_tot = std::min(b_tot, ind.mse_domain + ind.mse_boundary);
            }
        }
        history_.push_back({gen, b_dom, b_bnd, b_tot});
        if (gen % 25 == 0) {
            std::cout << "  [Tsoulos/" << prob_.name() << "] gen=" << gen
                      << "  best_dom=" << std::scientific << std::setprecision(3) << b_dom
                      << "  best_bnd=" << b_bnd << std::defaultfloat << "\n";
        }
        if (b_tot < Config::STOP_THRESHOLD) {
            std::cout << "  [Tsoulos/" << prob_.name() << "] Convergencia alcanzada en gen " << gen 
                      << " (error " << std::scientific << std::setprecision(2) << b_tot 
                      << " < " << Config::STOP_THRESHOLD << ")\n" << std::defaultfloat;
            break;
        }
    }
    return population_;
}

std::vector<TsoulosIndividual> TsoulosSolver::pareto_front() const {
    std::vector<TsoulosIndividual> front;
    for (auto& ind : population_)
        if (ind.rank == 1) front.push_back(ind);
    return front;
}
