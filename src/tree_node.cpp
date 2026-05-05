// =============================================================================
// tree_node.cpp  —  Árbol de expresión con AD exacto (Polimórfico)
// =============================================================================

#include "tree_node.hpp"
#include <cmath>
#include <stdexcept>
#include <sstream>

static constexpr double LOG_EPS  = 1e-8;
static constexpr double SQRT_EPS = 1e-8;
static constexpr double EXP_CLAMP = 30.0;

// ─── TerminalNode ────────────────────────────────────────────────────────────
AD TerminalNode::ad_eval(double x, double y) const {
    AD r;
    if (type == NodeType::VAR_X) {
        r.v = x; r.dx = 1.0; r.dy = 0.0; r.dxx = 0.0; r.dyy = 0.0;
    } else if (type == NodeType::VAR_Y) {
        r.v = y; r.dx = 0.0; r.dy = 1.0; r.dxx = 0.0; r.dyy = 0.0;
    } else {
        r.v = erc_val;
    }
    return r;
}
double TerminalNode::eval(double x, double y) const { return ad_eval(x,y).v; }
void TerminalNode::mutate_erc(std::mt19937& gen, double sigma) {
    if (type == NodeType::ERC) {
        std::normal_distribution<double> d(0.0, sigma);
        erc_val += d(gen);
    }
}
void TerminalNode::print(std::ostream& os) const {
    if (type == NodeType::VAR_X) os << "x";
    else if (type == NodeType::VAR_Y) os << "y";
    else os << erc_val;
}
void TerminalNode::print_latex(std::ostream& os) const {
    if (type == NodeType::VAR_X) os << "x";
    else if (type == NodeType::VAR_Y) os << "y";
    else os << erc_val;
}

// ─── UnaryNode ───────────────────────────────────────────────────────────────
AD UnaryNode::ad_eval(double x, double y) const {
    AD r;
    AD C = child->ad_eval(x, y);
    if (type == NodeType::SIN) {
        double s = std::sin(C.v), c = std::cos(C.v);
        r.v = s; r.dx = c * C.dx; r.dy = c * C.dy;
        r.dxx = c*C.dxx - s*C.dx*C.dx; r.dyy = c*C.dyy - s*C.dy*C.dy;
    } else if (type == NodeType::COS) {
        double s = std::sin(C.v), c = std::cos(C.v);
        r.v = c; r.dx = -s * C.dx; r.dy = -s * C.dy;
        r.dxx = -s*C.dxx - c*C.dx*C.dx; r.dyy = -s*C.dyy - c*C.dy*C.dy;
    } else if (type == NodeType::SINH) {
        double sh = std::sinh(C.v), ch = std::cosh(C.v);
        r.v = sh; r.dx = ch * C.dx; r.dy = ch * C.dy;
        r.dxx = ch*C.dxx + sh*C.dx*C.dx; r.dyy = ch*C.dyy + sh*C.dy*C.dy;
    } else if (type == NodeType::COSH) {
        double sh = std::sinh(C.v), ch = std::cosh(C.v);
        r.v = ch; r.dx = sh * C.dx; r.dy = sh * C.dy;
        r.dxx = sh*C.dxx + ch*C.dx*C.dx; r.dyy = sh*C.dyy + ch*C.dy*C.dy;
    } else if (type == NodeType::TANH) {
        double th = std::tanh(C.v);
        double sech2 = 1.0 - th*th;
        r.v = th; r.dx = sech2 * C.dx; r.dy = sech2 * C.dy;
        r.dxx = sech2*C.dxx - 2.0*th*sech2*C.dx*C.dx;
        r.dyy = sech2*C.dyy - 2.0*th*sech2*C.dy*C.dy;
    } else if (type == NodeType::EXP) {
        double ev = std::exp(std::min(C.v, EXP_CLAMP));
        r.v = ev; r.dx = ev * C.dx; r.dy = ev * C.dy;
        r.dxx = ev * (C.dxx + C.dx*C.dx); r.dyy = ev * (C.dyy + C.dy*C.dy);
    } else if (type == NodeType::SQRT) {
        double arg = std::abs(C.v) + SQRT_EPS;
        double sq = std::sqrt(arg);
        double sign_v = (C.v >= 0.0) ? 1.0 : -1.0;
        double gp = sign_v / (2.0 * sq);
        double gpp = -1.0 / (4.0 * sq * arg);
        r.v = sq; r.dx = gp * C.dx; r.dy = gp * C.dy;
        r.dxx = gpp*C.dx*C.dx + gp*C.dxx; r.dyy = gpp*C.dy*C.dy + gp*C.dyy;
    } else if (type == NodeType::LOG) {
        double arg = std::abs(C.v) + LOG_EPS;
        double sign_v = (C.v >= 0.0) ? 1.0 : -1.0;
        double gp = sign_v / arg;
        double gpp = -1.0 / (arg * arg);
        r.v = std::log(arg); r.dx = gp * C.dx; r.dy = gp * C.dy;
        r.dxx = gpp*C.dx*C.dx + gp*C.dxx; r.dyy = gpp*C.dy*C.dy + gp*C.dyy;
    } else if (type == NodeType::ATAN) {
        double denom = 1.0 + C.v*C.v;
        double gp = 1.0 / denom;
        double gpp = -2.0*C.v / (denom*denom);
        r.v = std::atan(C.v); r.dx = gp * C.dx; r.dy = gp * C.dy;
        r.dxx = gpp*C.dx*C.dx + gp*C.dxx; r.dyy = gpp*C.dy*C.dy + gp*C.dyy;
    }
    return r;
}
double UnaryNode::eval(double x, double y) const { return ad_eval(x, y).v; }
void UnaryNode::print(std::ostream& os) const {
    if (type == NodeType::SIN) os << "sin(";
    else if (type == NodeType::COS) os << "cos(";
    else if (type == NodeType::SINH) os << "sinh(";
    else if (type == NodeType::COSH) os << "cosh(";
    else if (type == NodeType::TANH) os << "tanh(";
    else if (type == NodeType::EXP) os << "exp(";
    else if (type == NodeType::SQRT) os << "sqrt(";
    else if (type == NodeType::LOG) os << "log(";
    else if (type == NodeType::ATAN) os << "atan(";
    child->print(os);
    os << ")";
}
void UnaryNode::print_latex(std::ostream& os) const {
    if (type == NodeType::SIN) os << "\\sin(";
    else if (type == NodeType::COS) os << "\\cos(";
    else if (type == NodeType::SINH) os << "\\sinh(";
    else if (type == NodeType::COSH) os << "\\cosh(";
    else if (type == NodeType::TANH) os << "\\tanh(";
    else if (type == NodeType::EXP) os << "\\exp(";
    else if (type == NodeType::SQRT) os << "\\sqrt{";
    else if (type == NodeType::LOG) os << "\\log(";
    else if (type == NodeType::ATAN) os << "\\arctan(";
    child->print_latex(os);
    if (type == NodeType::SQRT) os << "}";
    else os << ")";
}

// ─── BinaryNode ──────────────────────────────────────────────────────────────
AD BinaryNode::ad_eval(double x, double y) const {
    AD r;
    AD L = left->ad_eval(x, y);
    AD R = right->ad_eval(x, y);
    if (type == NodeType::ADD) {
        r.v = L.v + R.v; r.dx = L.dx + R.dx; r.dy = L.dy + R.dy;
        r.dxx = L.dxx + R.dxx; r.dyy = L.dyy + R.dyy;
    } else if (type == NodeType::SUB) {
        r.v = L.v - R.v; r.dx = L.dx - R.dx; r.dy = L.dy - R.dy;
        r.dxx = L.dxx - R.dxx; r.dyy = L.dyy - R.dyy;
    } else if (type == NodeType::MUL) {
        r.v = L.v * R.v;
        r.dx = L.dx*R.v + L.v*R.dx; r.dy = L.dy*R.v + L.v*R.dy;
        r.dxx = L.dxx*R.v + 2.0*L.dx*R.dx + L.v*R.dxx;
        r.dyy = L.dyy*R.v + 2.0*L.dy*R.dy + L.v*R.dyy;
    } else if (type == NodeType::DIV) {
        if (std::abs(R.v) < 1e-5) {
            r.v = 1.0; r.dx = 0; r.dy = 0; r.dxx = 0; r.dyy = 0;
        } else {
            double R2 = R.v * R.v;
            double R3 = R2 * R.v;
            r.v   = L.v / R.v;
            r.dx  = (L.dx * R.v - L.v * R.dx) / R2;
            r.dy  = (L.dy * R.v - L.v * R.dy) / R2;
            r.dxx = (L.dxx * R.v - L.v * R.dxx) / R2 - 2.0 * R.dx * (L.dx * R.v - L.v * R.dx) / R3;
            r.dyy = (L.dyy * R.v - L.v * R.dyy) / R2 - 2.0 * R.dy * (L.dy * R.v - L.v * R.dy) / R3;
        }
    }
    return r;
}
double BinaryNode::eval(double x, double y) const { 
    if (type == NodeType::DIV) {
        double denom = right->eval(x, y);
        if (std::abs(denom) < 1e-5) return 1.0;
        return left->eval(x, y) / denom;
    }
    return ad_eval(x, y).v; 
}
void BinaryNode::print(std::ostream& os) const {
    os << "(";
    left->print(os);
    if (type == NodeType::ADD) os << " + ";
    else if (type == NodeType::SUB) os << " - ";
    else if (type == NodeType::MUL) os << " * ";
    else if (type == NodeType::DIV) os << " / ";
    right->print(os);
    os << ")";
}
void BinaryNode::print_latex(std::ostream& os) const {
    if (type == NodeType::DIV) {
        os << "\\frac{"; left->print_latex(os); os << "}{"; right->print_latex(os); os << "}";
    } else {
        os << "(";
        left->print_latex(os);
        if (type == NodeType::ADD) os << " + ";
        else if (type == NodeType::SUB) os << " - ";
        else if (type == NodeType::MUL) os << " \\cdot ";
        right->print_latex(os);
        os << ")";
    }
}

// ─── Constructores ───────────────────────────────────────────────────────────
NodePtr make_var(char v) { return std::make_unique<TerminalNode>(v == 'x' ? NodeType::VAR_X : NodeType::VAR_Y); }
NodePtr make_erc(double val) { return std::make_unique<TerminalNode>(NodeType::ERC, val); }
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r) { return std::make_unique<BinaryNode>(op, std::move(l), std::move(r)); }
NodePtr make_unary(NodeType op, NodePtr c) { return std::make_unique<UnaryNode>(op, std::move(c)); }

// ─── Generación Aleatoria ────────────────────────────────────────────────────
NodePtr random_tree(int max_depth, std::mt19937& gen, bool force_terminal) {
    static const std::vector<NodeType> binaries = { NodeType::ADD, NodeType::SUB, NodeType::MUL, NodeType::DIV };
    static const std::vector<NodeType> unaries = {
        NodeType::SIN, NodeType::COS, NodeType::SINH, NodeType::COSH, NodeType::TANH,
        NodeType::EXP, NodeType::SQRT, NodeType::LOG, NodeType::ATAN
    };
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::uniform_real_distribution<double> erc_dist(-3.0, 3.0);

    bool terminal = force_terminal || (max_depth <= 0) || (ud(gen) < 0.30);
    if (terminal) {
        std::uniform_int_distribution<int> ti(0, 2);
        int choice = ti(gen);
        if (choice == 0) return make_var('x');
        if (choice == 1) return make_var('y');
        return make_erc(erc_dist(gen));
    }

    if (ud(gen) < 0.55) {
        std::uniform_int_distribution<int> bi(0, (int)binaries.size() - 1);
        NodeType op = binaries[bi(gen)];
        return make_binary(op, random_tree(max_depth - 1, gen), random_tree(max_depth - 1, gen));
    } else {
        std::uniform_int_distribution<int> ui(0, (int)unaries.size() - 1);
        NodeType op = unaries[ui(gen)];
        return make_unary(op, random_tree(max_depth - 1, gen));
    }
}

// ─── Homologous Crossover (Cruce Homólogo Estructural) ───────────────────────
std::pair<NodePtr, NodePtr> tree_crossover(const NodePtr& p1, const NodePtr& p2, std::mt19937& gen) {
    if (!p1 || !p2) return {nullptr, nullptr};
    // Si las estructuras coinciden, bajamos de forma homóloga
    if (p1->get_type() == p2->get_type()) {
        if (auto* un1 = dynamic_cast<UnaryNode*>(p1.get())) {
            auto* un2 = dynamic_cast<UnaryNode*>(p2.get());
            auto children = tree_crossover(un1->child, un2->child, gen);
            return {std::make_unique<UnaryNode>(un1->type, std::move(children.first)),
                    std::make_unique<UnaryNode>(un2->type, std::move(children.second))};
        } else if (auto* bn1 = dynamic_cast<BinaryNode*>(p1.get())) {
            auto* bn2 = dynamic_cast<BinaryNode*>(p2.get());
            auto lefts = tree_crossover(bn1->left, bn2->left, gen);
            auto rights = tree_crossover(bn1->right, bn2->right, gen);
            return {std::make_unique<BinaryNode>(bn1->type, std::move(lefts.first), std::move(rights.first)),
                    std::make_unique<BinaryNode>(bn2->type, std::move(lefts.second), std::move(rights.second))};
        } else {
            return {p1->clone(), p2->clone()};
        }
    } else {
        // Estructuras difieren. Puntos de divergencia: cruzamos con 50% de probabilidad
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(gen) < 0.5) return {p2->clone(), p1->clone()};
        else return {p1->clone(), p2->clone()};
    }
}

// ─── Mutación Estructural (Reemplazo) ────────────────────────────────────────
void replace_node_at(NodePtr& current, int& target_idx, NodePtr& replacement) {
    if (target_idx == 0) {
        current = std::move(replacement);
        target_idx--;
        return;
    }
    target_idx--;
    if (auto* un = dynamic_cast<UnaryNode*>(current.get())) {
        if (target_idx >= 0) replace_node_at(un->child, target_idx, replacement);
    } else if (auto* bn = dynamic_cast<BinaryNode*>(current.get())) {
        if (target_idx >= 0) replace_node_at(bn->left, target_idx, replacement);
        if (target_idx >= 0) replace_node_at(bn->right, target_idx, replacement);
    }
}

NodePtr tree_mutate(const NodePtr& tree, std::mt19937& gen) {
    NodePtr t = tree->clone();
    int sz = t->count_nodes();
    std::uniform_int_distribution<int> d(0, sz - 1);
    int target = d(gen);
    NodePtr new_sub = random_tree(3, gen); // Max depth 3 for subtrees
    replace_node_at(t, target, new_sub);
    return t;
}

// ─── Laplaciano por diferencias finitas (Koza) ────────────────────────────────
double fd_laplacian(const NodePtr& tree, double x, double y, double h) {
    double v   = tree->eval(x,   y);
    double vxp = tree->eval(x+h, y);
    double vxm = tree->eval(x-h, y);
    double vyp = tree->eval(x,   y+h);
    double vym = tree->eval(x,   y-h);
    double dxx = (vxp - 2.0*v + vxm) / (h*h);
    double dyy = (vyp - 2.0*v + vym) / (h*h);
    return dxx + dyy;
}
