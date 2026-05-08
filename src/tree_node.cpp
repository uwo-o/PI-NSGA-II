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
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        if (prob(gen) < 0.15) { // 15% chance to "snap" to a canonical constant
            static const double PI = std::acos(-1.0);
            static const double E  = std::exp(1.0);
            
            // Candidates: Nearest integer, half-integer, PI, and e
            double v = erc_val;
            std::vector<double> candidates = {
                std::round(v),           // Integer
                std::round(v * 2.0)/2.0, // Half-integer
                PI, -PI, PI/2.0, -PI/2.0,
                E,  -E
            };
            
            double best_c = candidates[0];
            double min_dist = std::abs(v - best_c);
            for (double c : candidates) {
                if (std::abs(v - c) < min_dist) {
                    min_dist = std::abs(v - c);
                    best_c = c;
                }
            }
            erc_val = best_c;
        } else {
            // Normal Gaussian mutation
            std::normal_distribution<double> d(0.0, sigma);
            erc_val += d(gen);
        }
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
    else {
        static const double PI = std::acos(-1.0);
        static const double E  = std::exp(1.0);
        double v = erc_val;
        double eps = 1e-6;
        
        if (std::abs(v - PI) < eps) os << "\\pi";
        else if (std::abs(v + PI) < eps) os << "-\\pi";
        else if (std::abs(v - PI/2.0) < eps) os << "\\frac{\\pi}{2}";
        else if (std::abs(v + PI/2.0) < eps) os << "-\\frac{\\pi}{2}";
        else if (std::abs(v - E) < eps) os << "e";
        else if (std::abs(v + E) < eps) os << "-e";
        else os << erc_val;
    }
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
    } else if (type == NodeType::EXP) {
        double ev = std::exp(std::min(C.v, EXP_CLAMP));
        r.v = ev; r.dx = ev * C.dx; r.dy = ev * C.dy;
        r.dxx = ev * (C.dxx + C.dx*C.dx); r.dyy = ev * (C.dyy + C.dy*C.dy);
    } else if (type == NodeType::SQR) {
        r.v = C.v * C.v;
        r.dx = 2.0 * C.v * C.dx;
        r.dy = 2.0 * C.v * C.dy;
        r.dxx = 2.0 * (C.dx * C.dx + C.v * C.dxx);
        r.dyy = 2.0 * (C.dy * C.dy + C.v * C.dyy);
    }
    return r;
}
double UnaryNode::eval(double x, double y) const { return ad_eval(x, y).v; }
void UnaryNode::print(std::ostream& os) const {
    if (type == NodeType::SIN) os << "sin(";
    else if (type == NodeType::COS) os << "cos(";
    else if (type == NodeType::SINH) os << "sinh(";
    else if (type == NodeType::COSH) os << "cosh(";
    else if (type == NodeType::EXP) os << "exp(";
    else if (type == NodeType::SQR) os << "sqr(";
    child->print(os);
    os << ")";
}
void UnaryNode::print_latex(std::ostream& os) const {
    if (type == NodeType::SIN) os << "\\sin(";
    else if (type == NodeType::COS) os << "\\cos(";
    else if (type == NodeType::SINH) os << "\\sinh(";
    else if (type == NodeType::COSH) os << "\\cosh(";
    else if (type == NodeType::EXP) os << "\\exp(";
    else if (type == NodeType::SQR) os << "{(";
    child->print_latex(os);
    if (type == NodeType::SQR) os << ")}^2";
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
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r) { 
    auto node = std::make_unique<BinaryNode>(op, std::move(l), std::move(r));
    return node->simplify();
}
NodePtr make_unary(NodeType op, NodePtr c) { 
    auto node = std::make_unique<UnaryNode>(op, std::move(c));
    return node->simplify();
}

// ─── Generación Aleatoria ────────────────────────────────────────────────────
NodePtr random_tree(int max_depth, std::mt19937& gen, bool force_terminal) {
    static const std::vector<NodeType> binaries = { NodeType::ADD, NodeType::SUB, NodeType::MUL, NodeType::DIV };
    static const std::vector<NodeType> unaries = {
        NodeType::SIN, NodeType::COS, NodeType::SINH, NodeType::COSH, NodeType::EXP, NodeType::SQR
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

// ─── Simplificación Simbólica ────────────────────────────────────────────────
bool is_structurally_equal(const Node* n1, const Node* n2) {
    if (!n1 || !n2) return n1 == n2;
    if (n1->get_type() != n2->get_type()) return false;
    
    if (auto* t1 = dynamic_cast<const TerminalNode*>(n1)) {
        auto* t2 = dynamic_cast<const TerminalNode*>(n2);
        if (t1->type == NodeType::ERC) return std::abs(t1->erc_val - t2->erc_val) < 1e-9;
        return true; 
    }
    if (auto* u1 = dynamic_cast<const UnaryNode*>(n1)) {
        auto* u2 = dynamic_cast<const UnaryNode*>(n2);
        return is_structurally_equal(u1->child.get(), u2->child.get());
    }
    if (auto* b1 = dynamic_cast<const BinaryNode*>(n1)) {
        auto* b2 = dynamic_cast<const BinaryNode*>(n2);
        return is_structurally_equal(b1->left.get(), b2->left.get()) &&
               is_structurally_equal(b1->right.get(), b2->right.get());
    }
    return false;
}

// Helper para detectar f(x)^2 representado como MUL(f(x), f(x))
const Node* get_arg_of_square(const Node* n, NodeType func) {
    if (!n || n->get_type() != NodeType::MUL) return nullptr;
    auto* b = dynamic_cast<const BinaryNode*>(n);
    if (b->left->get_type() == func && b->right->get_type() == func) {
        auto* uL = dynamic_cast<const UnaryNode*>(b->left.get());
        auto* uR = dynamic_cast<const UnaryNode*>(b->right.get());
        if (is_structurally_equal(uL->child.get(), uR->child.get())) {
            return uL->child.get();
        }
    }
    return nullptr;
}

NodePtr TerminalNode::simplify() const {
    return clone();
}

NodePtr UnaryNode::simplify() const {
    NodePtr s_child = child->simplify();
    
    // Constant folding
    if (s_child->get_type() == NodeType::ERC) {
        double val = eval(0, 0); // dummy eval
        return make_erc(val);
    }

    return std::make_unique<UnaryNode>(type, std::move(s_child));
}

NodePtr BinaryNode::simplify() const {
    NodePtr s_l = left->simplify();
    NodePtr s_r = right->simplify();
    
    bool l_erc = (s_l->get_type() == NodeType::ERC);
    bool r_erc = (s_r->get_type() == NodeType::ERC);
    double lv = l_erc ? dynamic_cast<TerminalNode*>(s_l.get())->erc_val : 0;
    double rv = r_erc ? dynamic_cast<TerminalNode*>(s_r.get())->erc_val : 0;

    // Constant folding
    if (l_erc && r_erc) {
        // Evaluate logic (copied from eval but simplified)
        if (type == NodeType::ADD) return make_erc(lv + rv);
        if (type == NodeType::SUB) return make_erc(lv - rv);
        if (type == NodeType::MUL) return make_erc(lv * rv);
        if (type == NodeType::DIV) return make_erc(std::abs(rv) < 1e-5 ? 1.0 : lv / rv);
    }
    
    // Identities
    if (type == NodeType::ADD) {
        if (l_erc && lv == 0) return s_r;
        if (r_erc && rv == 0) return s_l;
        
        // Identity: sin^2(x) + cos^2(x) = 1
        const Node* arg_sin_l = get_arg_of_square(s_l.get(), NodeType::SIN);
        const Node* arg_cos_r = get_arg_of_square(s_r.get(), NodeType::COS);
        if (arg_sin_l && arg_cos_r && is_structurally_equal(arg_sin_l, arg_cos_r)) return make_erc(1.0);

        const Node* arg_cos_l = get_arg_of_square(s_l.get(), NodeType::COS);
        const Node* arg_sin_r = get_arg_of_square(s_r.get(), NodeType::SIN);
        if (arg_cos_l && arg_sin_r && is_structurally_equal(arg_cos_l, arg_sin_r)) return make_erc(1.0);
    }
    if (type == NodeType::SUB) {
        if (r_erc && rv == 0) return s_l;
        if (is_structurally_equal(s_l.get(), s_r.get())) return make_erc(0);

        // Identity: cosh^2(x) - sinh^2(x) = 1
        const Node* arg_cosh_l = get_arg_of_square(s_l.get(), NodeType::COSH);
        const Node* arg_sinh_r = get_arg_of_square(s_r.get(), NodeType::SINH);
        if (arg_cosh_l && arg_sinh_r && is_structurally_equal(arg_cosh_l, arg_sinh_r)) return make_erc(1.0);
    }
    if (type == NodeType::MUL) {
        if (l_erc && lv == 0) return make_erc(0);
        if (r_erc && rv == 0) return make_erc(0);
        if (l_erc && lv == 1) return s_r;
        if (r_erc && rv == 1) return s_l;
    }
    if (type == NodeType::DIV) {
        if (r_erc && rv == 1) return s_l;
        if (is_structurally_equal(s_l.get(), s_r.get())) return make_erc(1);
    }

    return std::make_unique<BinaryNode>(type, std::move(s_l), std::move(s_r));
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
