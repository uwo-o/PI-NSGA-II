#pragma once
// =============================================================================
// tree_node.hpp  —  Árbol de expresión simbólica con AD exacto (Polimórfico)
// =============================================================================

#include "common.hpp"
#include <memory>
#include <random>
#include <ostream>

inline bool is_binary(NodeType t) {
    return t == NodeType::ADD || t == NodeType::SUB || t == NodeType::MUL;
}
inline bool is_unary(NodeType t) {
    return t == NodeType::SIN  || t == NodeType::COS  ||
           t == NodeType::SINH || t == NodeType::COSH ||
           t == NodeType::EXP  || t == NodeType::SQR;
}
inline bool is_terminal(NodeType t) {
    return t == NodeType::VAR_X || t == NodeType::VAR_Y || t == NodeType::ERC;
}

class PDEProblem;
class Node;
using NodePtr = std::unique_ptr<Node>;

class Node {
public:
    virtual ~Node() = default;

    virtual NodeType get_type() const = 0;
    virtual AD ad_eval(double x, double y) const = 0;
    virtual double eval(double x, double y) const = 0;
    virtual NodePtr clone() const = 0;
    virtual int count_nodes() const = 0;
    
    virtual void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) {}

    virtual void print(std::ostream& os) const = 0;
    virtual void print_latex(std::ostream& os) const = 0;
    virtual NodePtr simplify() const = 0;
    virtual NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) = 0;
    
    // Para Hill Climbing: recolecta punteros a todas las constantes del árbol
    virtual void collect_ercs(std::vector<double*>& ptrs) = 0;

    // Verifica si el árbol usa una variable específica (VAR_X o VAR_Y)
    virtual bool uses_variable(NodeType var_type) const = 0;
};

// ─── Clases derivadas ─────────────────────────────────────────────────────────

class TerminalNode : public Node {
public:
    NodeType type;
    double erc_val;

    TerminalNode(NodeType t, double val = 0.0) : type(t), erc_val(val) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y) const override;
    double eval(double x, double y) const override;
    NodePtr clone() const override { return std::make_unique<TerminalNode>(type, erc_val); }
    int count_nodes() const override { return 1; }
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override;
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<double*>& ptrs) override {
        if (type == NodeType::ERC) ptrs.push_back(&erc_val);
    }
    bool uses_variable(NodeType var_type) const override {
        return type == var_type;
    }
};

class UnaryNode : public Node {
public:
    NodeType type;
    NodePtr child;

    UnaryNode(NodeType t, NodePtr c) : type(t), child(std::move(c)) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y) const override;
    double eval(double x, double y) const override;
    NodePtr clone() const override { return std::make_unique<UnaryNode>(type, child->clone()); }
    int count_nodes() const override { return 1 + child->count_nodes(); }
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override { child->mutate_erc(gen, sigma); }
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<double*>& ptrs) override {
        child->collect_ercs(ptrs);
    }
    bool uses_variable(NodeType var_type) const override {
        return child->uses_variable(var_type);
    }
};

class BinaryNode : public Node {
public:
    NodeType type;
    NodePtr left;
    NodePtr right;

    BinaryNode(NodeType t, NodePtr l, NodePtr r) : type(t), left(std::move(l)), right(std::move(r)) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y) const override;
    double eval(double x, double y) const override;
    NodePtr clone() const override { return std::make_unique<BinaryNode>(type, left->clone(), right->clone()); }
    int count_nodes() const override { return 1 + left->count_nodes() + right->count_nodes(); }
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override {
        left->mutate_erc(gen, sigma);
        right->mutate_erc(gen, sigma);
    }
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<double*>& ptrs) override {
        left->collect_ercs(ptrs);
        right->collect_ercs(ptrs);
    }
    bool uses_variable(NodeType var_type) const override {
        return left->uses_variable(var_type) || right->uses_variable(var_type);
    }
};

// ─── Constructores de nodos ───────────────────────────────────────────────────
NodePtr make_var(char v);
NodePtr make_erc(double val);
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r);
NodePtr make_unary(NodeType op, NodePtr child);

// ─── Generación aleatoria de árbol ────────────────────────────────────────────
NodePtr random_tree(int max_depth, std::mt19937& gen, bool force_terminal = false);
NodePtr random_tree_special(int max_depth, std::mt19937& gen, int dim);

// ─── Operadores evolutivos ────────────────────────────────────────────────────
// Cruce homólogo estructural
std::pair<NodePtr, NodePtr> tree_crossover(const NodePtr& p1, const NodePtr& p2, std::mt19937& gen);

// Mutación: reemplaza un subárbol aleatorio con uno nuevo
NodePtr tree_mutate(const NodePtr& tree, std::mt19937& gen, int max_depth = 3);
void replace_node_at(NodePtr& current, int& target_idx, NodePtr& replacement);

// ─── Laplaciano via diferencias finitas (para método Koza) ───────────────────
double fd_laplacian(const NodePtr& tree, double x, double y, double h = 1e-5);
