// =============================================================================
// tsoulos_ge.cpp  —  Baseline: Grammatical Evolution (Tsoulos)
// =============================================================================

#include "tsoulos_ge.hpp"
#include "nsga2.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// ─── TsoulosIndividual ───────────────────────────────────────────────────────
void TsoulosIndividual::decode() {
    int idx = 0;
    tree = bnf_to_tree(codons, idx, 0, 5); // Profundidad max para Tsoulos
}

void TsoulosIndividual::evaluate(const PDEProblem& prob,
                                 const std::vector<Point>& dom,
                                 const std::vector<Point>& bnd)
{
    if (!tree) { mse_domain = 1e10; mse_boundary = 1e10; tree_size = 0; root_type = NodeType::UNKNOWN; return; }
    tree_size = tree->count_nodes();
    root_type = tree->get_type();

    double sum_dom = 0.0, total_w = 0.0;
    for (auto& p : dom) {
        double lap = fd_laplacian(tree, p.x, p.y, prob.dim);
        double u   = tree->eval(p.x, p.y);
        double res;
        if (prob.type == PDE::SCHRODINGER) {
            double r2 = (p.x - 0.5) * (p.x - 0.5) + (p.y - 0.5) * (p.y - 0.5);
            double V = 4.0 * r2;
            double E = (prob.dim == 1) ? 2.0 : 4.0;
            res = lap + (E - V) * u;
        } else {
            res = lap + prob.k2 * u - prob.source(p.x, p.y);
        }
        
        if (!std::isfinite(res)) { mse_domain = 1e10; mse_boundary = 1e10; return; }
        double weight = 1.0 / (std::min({p.x, 1.0-p.x, p.y, 1.0-p.y}) + 0.1);
        sum_dom += weight * res * res;
        total_w += weight;
    }
    mse_domain = (total_w > 0) ? (sum_dom / total_w) : 1e10;

    double sum_bnd = 0.0;
    for (auto& p : bnd) {
        double diff = tree->eval(p.x, p.y) - prob.bc(p.x, p.y);
        if (!std::isfinite(diff)) { mse_boundary = 1e10; return; }
        sum_bnd += diff * diff;
    }
    double raw_bc_mse = sum_bnd / (double)bnd.size();
    
    double alpha = Config::PI_ALPHA;
    double beta  = 1.0 - alpha;

    mse_domain   = beta * mse_domain + alpha * raw_bc_mse; 
    mse_boundary = raw_bc_mse;
}

// ─── TsoulosSolver ────────────────────────────────────────────────────────────
TsoulosSolver::TsoulosSolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed) 
{
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
}

TsoulosIndividual TsoulosSolver::random_individual() {
    TsoulosIndividual ind;
    ind.codons.resize(Config::CODON_LENGTH);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int& x : ind.codons) x = dist(gen_);
    ind.decode();
    ind.evaluate(prob_, dom_pts_, bnd_pts_);
    return ind;
}

TsoulosIndividual TsoulosSolver::crossover(const TsoulosIndividual& a, const TsoulosIndividual& b) {
    TsoulosIndividual child;
    child.codons.resize(a.codons.size());
    std::uniform_int_distribution<int> cut(0, a.codons.size() - 1);
    int cp = cut(gen_);
    for (int i = 0; i < (int)a.codons.size(); ++i)
        child.codons[i] = (i < cp) ? a.codons[i] : b.codons[i];
    return child;
}

void TsoulosSolver::mutate(TsoulosIndividual& ind) {
    std::uniform_real_distribution<double> p(0.0, 1.0);
    std::uniform_int_distribution<int> val(0, 255);
    for (int& x : ind.codons) {
        if (p(gen_) < 0.05) x = val(gen_);
    }
}

std::vector<TsoulosIndividual> TsoulosSolver::run(int pop_size, int max_gen) {
    population_.clear();
    for (int i = 0; i < pop_size; ++i) population_.push_back(random_individual());
    history_.clear();

    for (int gen = 0; gen < max_gen; ++gen) {
        std::vector<TsoulosIndividual> offspring;
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
        if (b_tot < Config::STOP_THRESHOLD) break;
    }
    return population_;
}

std::vector<TsoulosIndividual> TsoulosSolver::pareto_front() const {
    std::vector<TsoulosIndividual> front;
    for (auto& ind : population_)
        if (ind.rank == 1) front.push_back(ind);
    return front;
}

NodePtr bnf_to_tree(const std::vector<int>& codons, int& idx, int depth, int max_depth) {
    auto consume = [&]() {
        int val = codons[idx % codons.size()];
        idx++;
        return val;
    };

    if (depth >= max_depth) {
        int r = consume() % 3;
        if (r == 0) return std::make_unique<TerminalNode>(NodeType::VAR_X);
        if (r == 1) return std::make_unique<TerminalNode>(NodeType::VAR_Y);
        return std::make_unique<TerminalNode>(NodeType::ERC, (double)(consume() % 100) / 10.0 - 5.0);
    }

    int choice = consume() % 10;
    if (choice < 4) { // Binario
        NodeType type = (NodeType)(choice);
        auto l = bnf_to_tree(codons, idx, depth + 1, max_depth);
        auto r = bnf_to_tree(codons, idx, depth + 1, max_depth);
        return std::make_unique<BinaryNode>(type, std::move(l), std::move(r));
    } else if (choice < 8) { // Unario
        NodeType types[] = {NodeType::SIN, NodeType::COS, NodeType::EXP, NodeType::SQR};
        auto c = bnf_to_tree(codons, idx, depth + 1, max_depth);
        return std::make_unique<UnaryNode>(types[choice - 4], std::move(c));
    } else { // Terminal
        int r = consume() % 3;
        if (r == 0) return std::make_unique<TerminalNode>(NodeType::VAR_X);
        if (r == 1) return std::make_unique<TerminalNode>(NodeType::VAR_Y);
        return std::make_unique<TerminalNode>(NodeType::ERC, (double)(consume() % 100) / 10.0 - 5.0);
    }
}
