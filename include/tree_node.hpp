#pragma once
// =============================================================================
// tree_node.hpp  —  Árbol de expresión simbólica con AD exacto (Polimórfico)
// =============================================================================

#include "common.hpp"
#include "dimensions.hpp"
#include "pde_problems.hpp"
#include <memory>
#include <random>
#include <ostream>
#include <optional>
#include <sstream>

inline bool is_binary(NodeType t) {
    return t == NodeType::ADD || t == NodeType::SUB || t == NodeType::MUL || t == NodeType::DIV || 
           t == NodeType::LEGENDRE || t == NodeType::HERMITE || t == NodeType::CHEBYSHEV || t == NodeType::LAGUERRE ||
           t == NodeType::POW;
}
inline bool is_trig(NodeType t) {
    return t == NodeType::SIN || t == NodeType::COS || t == NodeType::SINH || t == NodeType::COSH || t == NodeType::TANH;
}

inline bool is_polynomial(NodeType t) {
    return t == NodeType::LEGENDRE || t == NodeType::HERMITE || t == NodeType::CHEBYSHEV || t == NodeType::LAGUERRE;
}

inline bool is_unary(NodeType t) {
    return t == NodeType::SIN  || t == NodeType::COS  ||
           t == NodeType::SINH || t == NodeType::COSH ||
           t == NodeType::EXP  || t == NodeType::SQR  ||
           t == NodeType::LOG  || t == NodeType::TANH ||
           t == NodeType::GAUSSIAN;
}
inline bool is_constant(NodeType t) {
    return t == NodeType::ERC || t == NodeType::CONST_I || t == NodeType::CONST_PI || t == NodeType::CONST_E ||
           (t >= NodeType::CONST_G && t <= NodeType::CONST_EPS0);
}
inline bool is_terminal(NodeType t) {
    return t == NodeType::VAR_X || t == NodeType::VAR_Y || t == NodeType::VAR_T || t == NodeType::VAR_N || t == NodeType::VAR_Z ||
           is_constant(t);
}

class PDEProblem;
class Node;
using NodePtr = std::unique_ptr<Node>;

class Node {
public:
    virtual ~Node() = default;

    virtual NodeType get_type() const = 0;
    virtual AD ad_eval(double x, double y, int dim = 2) const = 0;
    virtual AD ad_eval_t(double x, double y, double t, int dim = 2) const = 0;
    virtual Complex eval(double x, double y) const = 0;
    virtual Complex eval_t(double x, double y, double t) const = 0;
    
    virtual std::optional<Dimension> get_dimension(const PDEProblem& prob) const = 0;
    virtual bool is_unit_flexible() const = 0;
    virtual bool contains_erc() const = 0;
    virtual bool contains_variables() const = 0;
    virtual int get_unary_depth() const = 0;
    virtual bool has_nested_trig() const = 0;
    virtual bool contains_trig() const = 0;
    virtual bool is_strictly_affine() const = 0;
    virtual bool has_non_affine_trig_arg() const = 0;
    virtual bool has_nested_polynomial() const = 0;
    virtual bool contains_polynomial() const = 0;
    virtual bool has_nested_exp() const = 0;
    virtual bool contains_exp() const = 0;
    virtual bool has_invalid_polynomial_degree() const = 0;
    
    bool is_consistent(const PDEProblem& prob) const {
        return get_dimension(prob).has_value();
    }

    virtual NodePtr clone() const = 0;
    virtual int count_nodes() const = 0;
    virtual int get_depth() const = 0;
    
    virtual void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) {}
    virtual void print(std::ostream& os) const = 0;
    virtual void print_latex(std::ostream& os) const = 0;
    virtual void print_formal(std::ostream& os, int parent_prec = 0) const = 0;
    virtual void round_constants(double epsilon = 0.05) = 0;
    
    std::string print_str() const {
        std::stringstream ss;
        print(ss);
        return ss.str();
    }

    virtual NodePtr simplify() const = 0;
    virtual NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) = 0;
    virtual void collect_ercs(std::vector<Complex*>& ptrs) = 0;
    virtual bool uses_variable(NodeType var_type) const = 0;
};

class TerminalNode final : public Node {
public:
    NodeType type;
    Complex erc_val;
    TerminalNode(NodeType t, Complex val = 0.0) : type(t), erc_val(val) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y, int dim = 2) const override;
    AD ad_eval_t(double x, double y, double t, int dim = 2) const override;
    Complex eval(double x, double y) const override;
    Complex eval_t(double x, double y, double t) const override;
    std::optional<Dimension> get_dimension(const PDEProblem& prob) const override;
    bool is_unit_flexible() const override;
    bool contains_erc() const override;
    bool contains_variables() const override;
    int get_unary_depth() const override { return 0; }
    bool has_nested_trig() const override { return false; }
    bool contains_trig() const override { return is_trig(type); }
    bool is_strictly_affine() const override { return true; }
    bool has_non_affine_trig_arg() const override { return false; }
    bool has_nested_polynomial() const override { return false; }
    bool contains_polynomial() const override { return is_polynomial(type); }
    bool has_nested_exp() const override { return false; }
    bool contains_exp() const override { return type == NodeType::EXP; }
    bool has_invalid_polynomial_degree() const override { return false; }
    NodePtr clone() const override;
    int count_nodes() const override;
    int get_depth() const override;
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override;
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    void print_formal(std::ostream& os, int parent_prec = 0) const override;
    void round_constants(double epsilon = 0.05) override;

    NodePtr simplify() const override { return clone(); }
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override { return clone(); }
    void collect_ercs(std::vector<Complex*>& ptrs) override { if (type == NodeType::ERC) ptrs.push_back(&erc_val); }
    bool uses_variable(NodeType var_type) const override { return type == var_type; }
};

class UnaryNode final : public Node {
public:
    NodeType type;
    NodePtr child;
    UnaryNode(NodeType t, NodePtr c) : type(t), child(std::move(c)) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y, int dim = 2) const override;
    AD ad_eval_t(double x, double y, double t, int dim = 2) const override;
    Complex eval(double x, double y) const override;
    Complex eval_t(double x, double y, double t) const override;
    std::optional<Dimension> get_dimension(const PDEProblem& prob) const override;
    bool is_unit_flexible() const override;
    bool contains_erc() const override;
    bool contains_variables() const override;
    int get_unary_depth() const override;
    bool has_nested_trig() const override;
    bool contains_trig() const override;
    bool has_nested_polynomial() const override;
    bool contains_polynomial() const override;
    bool has_nested_exp() const override;
    bool contains_exp() const override;
    bool has_invalid_polynomial_degree() const override;
    bool is_strictly_affine() const override;
    bool has_non_affine_trig_arg() const override;
    NodePtr clone() const override;
    int count_nodes() const override;
    int get_depth() const override;
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override { if(child) child->mutate_erc(gen, sigma); }
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    void print_formal(std::ostream& os, int parent_prec = 0) const override;
    void round_constants(double epsilon = 0.05) override;

    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<Complex*>& ptrs) override { if (child) child->collect_ercs(ptrs); }
    bool uses_variable(NodeType var_type) const override { return child && child->uses_variable(var_type); }
};

class BinaryNode final : public Node {
public:
    NodeType type;
    NodePtr left;
    NodePtr right;
    BinaryNode(NodeType t, NodePtr l, NodePtr r) : type(t), left(std::move(l)), right(std::move(r)) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y, int dim = 2) const override;
    AD ad_eval_t(double x, double y, double t, int dim = 2) const override;
    Complex eval(double x, double y) const override;
    Complex eval_t(double x, double y, double t) const override;
    std::optional<Dimension> get_dimension(const PDEProblem& prob) const override;
    bool is_unit_flexible() const override;
    bool contains_erc() const override;
    bool contains_variables() const override;
    int get_unary_depth() const override;
    bool has_nested_trig() const override;
    bool contains_trig() const override;
    bool has_nested_polynomial() const override;
    bool contains_polynomial() const override;
    bool has_nested_exp() const override;
    bool contains_exp() const override;
    bool has_invalid_polynomial_degree() const override;
    bool is_strictly_affine() const override;
    bool has_non_affine_trig_arg() const override;
    NodePtr clone() const override;
    int count_nodes() const override;
    int get_depth() const override;
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override {
        if (left) left->mutate_erc(gen, sigma);
        if (right) right->mutate_erc(gen, sigma);
    }
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    void print_formal(std::ostream& os, int parent_prec = 0) const override;
    void round_constants(double epsilon = 0.05) override;

    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<Complex*>& ptrs) override { if (left) left->collect_ercs(ptrs); if (right) right->collect_ercs(ptrs); }
    bool uses_variable(NodeType var_type) const override { return (left && left->uses_variable(var_type)) || (right && right->uses_variable(var_type)); }
};

class SeriesNode final : public Node {
public:
    int n_terms;
    std::vector<Complex> coeffs;
    NodePtr child;
    SeriesNode(int n, NodePtr c) : n_terms(n), child(std::move(c)) { coeffs.resize(n, 1.0); }
    NodeType get_type() const override { return NodeType::SERIES; }
    AD ad_eval(double x, double y, int dim = 2) const override;
    AD ad_eval_t(double x, double y, double t, int dim = 2) const override;
    Complex eval(double x, double y) const override;
    Complex eval_t(double x, double y, double t) const override;
    std::optional<Dimension> get_dimension(const PDEProblem& prob) const override;
    bool is_unit_flexible() const override;
    bool contains_erc() const override;
    bool contains_variables() const override;
    int get_unary_depth() const override;
    bool has_nested_trig() const override;
    bool contains_trig() const override;
    bool has_nested_polynomial() const override;
    bool contains_polynomial() const override;
    bool has_nested_exp() const override;
    bool contains_exp() const override;
    bool has_invalid_polynomial_degree() const override;
    bool is_strictly_affine() const override;
    bool has_non_affine_trig_arg() const override;
    NodePtr clone() const override;
    int count_nodes() const override;
    int get_depth() const override;
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override;
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    void print_formal(std::ostream& os, int parent_prec = 0) const override;
    void round_constants(double epsilon = 0.05) override;

    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<Complex*>& ptrs) override;
    bool uses_variable(NodeType var_type) const override { return child && child->uses_variable(var_type); }
};

NodePtr make_var(char v);
NodePtr make_var_n();
NodePtr make_erc(Complex val);
NodePtr make_const_i();
NodePtr make_const_pi();
NodePtr make_const_e();
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r);
NodePtr make_unary(NodeType op, NodePtr child);
NodePtr random_tree(int max_depth, std::mt19937& gen, const PDEProblem& prob, bool force_terminal = false);
NodePtr random_tree_special(int max_depth, std::mt19937& gen, const PDEProblem& prob, const PDEPriors& priors);
std::pair<NodePtr, NodePtr> tree_crossover(const NodePtr& p1, const NodePtr& p2, std::mt19937& gen);
NodePtr tree_mutate(const NodePtr& tree, std::mt19937& gen, const PDEProblem& prob, double aggressiveness = 1.0);
NodePtr tree_mutate_point(const NodePtr& tree, std::mt19937& gen, const PDEProblem& prob);
void replace_node_at(NodePtr& current, int& target_idx, NodePtr& replacement);
