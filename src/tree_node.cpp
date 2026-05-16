// =============================================================================
// tree_node.cpp  —  Árbol de expresión con AD exacto (Polimórfico)
// =============================================================================

#include "tree_node.hpp"
#include "pde_problems.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <sstream>

static constexpr double LOG_EPS  = 1e-8;
static constexpr double SQRT_EPS = 1e-8;
static constexpr double EXP_CLAMP = 30.0;

// ─── TerminalNode ────────────────────────────────────────────────────────────
AD TerminalNode::ad_eval(double x, double y, int dim) const {
    AD r;
    if (type == NodeType::VAR_X) {
        r.v = x; r.dx = 1.0; r.dy = 0.0; r.dxx = 0.0; r.dyy = 0.0;
    } else if (type == NodeType::VAR_Y) {
        r.v = y; r.dx = 0.0; r.dy = 1.0; r.dxx = 0.0; r.dyy = 0.0;
    } else if (type == NodeType::ERC) {
        r.v = erc_val; r.dx = 0; r.dy = 0; r.dxx = 0; r.dyy = 0;
    } else if (type == NodeType::CONST_I) {
        r.v = I_COMPLEX; r.dx = 0; r.dy = 0; r.dxx = 0; r.dyy = 0;
    }
    return r;
}
Complex TerminalNode::eval(double x, double y) const { 
    if (type == NodeType::VAR_X) return x;
    if (type == NodeType::VAR_Y) return y;
    if (type == NodeType::ERC)   return erc_val;
    if (type == NodeType::CONST_I) return I_COMPLEX;
    return 0.0;
}
void TerminalNode::mutate_erc(std::mt19937& gen, double sigma) {
    if (type == NodeType::ERC) {
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        if (prob(gen) < 0.15) { // 15% chance to "snap" to a canonical constant
            static const double PI_VAL = std::acos(-1.0);
            static const double E_VAL  = std::exp(1.0);
            
            double v = erc_val.real();
            std::vector<double> candidates = {
                std::round(v),           // Integer
                std::round(v * 2.0)/2.0, // Half-integer
                PI_VAL, -PI_VAL, PI_VAL/2.0, -PI_VAL/2.0,
                E_VAL,  -E_VAL
            };
            
            double best_c = candidates[0];
            double min_dist = std::abs(v - best_c);
            for (double c : candidates) {
                if (std::abs(v - c) < min_dist) {
                    min_dist = std::abs(v - c);
                    best_c = c;
                }
            }
            erc_val = Complex(best_c, 0.0);
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
    else if (type == NodeType::ERC) os << erc_val.real();
    else if (type == NodeType::CONST_I) os << "i";
}
void TerminalNode::print_latex(std::ostream& os) const {
    if (type == NodeType::VAR_X) os << "x";
    else if (type == NodeType::VAR_Y) os << "y";
    else if (type == NodeType::CONST_I) os << "i";
    else {
        static const double PI_VAL = std::acos(-1.0);
        static const double E_VAL  = std::exp(1.0);
        double v = erc_val.real();
        double eps = 1e-6;
        
        if (std::abs(v - PI_VAL) < eps) os << "\\pi";
        else if (std::abs(v + PI_VAL) < eps) os << "-\\pi";
        else if (std::abs(v - PI_VAL/2.0) < eps) os << "\\frac{\\pi}{2}";
        else if (std::abs(v + PI_VAL/2.0) < eps) os << "-\\frac{\\pi}{2}";
        else if (std::abs(v - E_VAL) < eps) os << "e";
        else if (std::abs(v + E_VAL) < eps) os << "-e";
        else {
            if (std::abs(erc_val.imag()) < 1e-9) os << erc_val.real();
            else os << erc_val;
        }
    }
}

// ─── UnaryNode ───────────────────────────────────────────────────────────────
AD UnaryNode::ad_eval(double x, double y, int dim) const {
    AD r;
    AD C = child->ad_eval(x, y, dim);
    if (type == NodeType::SIN) {
        Complex s = std::sin(C.v), c = std::cos(C.v);
        r.v = s; r.dx = c * C.dx; r.dxx = c*C.dxx - s*C.dx*C.dx;
        if (dim == 2) {
            r.dy = c * C.dy; r.dyy = c*C.dyy - s*C.dy*C.dy;
        }
    } else if (type == NodeType::COS) {
        Complex s = std::sin(C.v), c = std::cos(C.v);
        r.v = c; r.dx = -s * C.dx; r.dxx = -s*C.dxx - c*C.dx*C.dx;
        if (dim == 2) {
            r.dy = -s * C.dy; r.dyy = -s*C.dyy - c*C.dy*C.dy;
        }
    } else if (type == NodeType::EXP) {
        // Clamp real part to prevent overflow: exp(30) is enough for any PDE
        Complex clamped_v = C.v;
        if (clamped_v.real() > 30.0) clamped_v = Complex(30.0, clamped_v.imag());
        if (clamped_v.real() < -30.0) clamped_v = Complex(-30.0, clamped_v.imag());

        Complex ev = std::exp(clamped_v);
        r.v = ev; r.dx = ev * C.dx; r.dxx = ev * (C.dxx + C.dx*C.dx);
        if (dim == 2) {
            r.dy = ev * C.dy; r.dyy = ev * (C.dyy + C.dy*C.dy);
        }
    } else if (type == NodeType::SQR) {
        r.v = C.v * C.v;
        r.dx = 2.0 * C.v * C.dx; r.dxx = 2.0 * (C.dx * C.dx + C.v * C.dxx);
        if (dim == 2) {
            r.dy = 2.0 * C.v * C.dy; r.dyy = 2.0 * (C.dy * C.dy + C.v * C.dyy);
        }
    } else if (type == NodeType::SINH) {
        Complex sh = std::sinh(C.v), ch = std::cosh(C.v);
        r.v = sh; r.dx = ch * C.dx; r.dxx = ch*C.dxx + sh*C.dx*C.dx;
        if (dim == 2) { r.dy = ch * C.dy; r.dyy = ch*C.dyy + sh*C.dy*C.dy; }
    } else if (type == NodeType::COSH) {
        Complex sh = std::sinh(C.v), ch = std::cosh(C.v);
        r.v = ch; r.dx = sh * C.dx; r.dxx = sh*C.dxx + ch*C.dx*C.dx;
        if (dim == 2) { r.dy = sh * C.dy; r.dyy = sh*C.dyy + ch*C.dy*C.dy; }
    }
    return r;
}
Complex UnaryNode::eval(double x, double y) const { 
    Complex v = child->eval(x, y);
    if (type == NodeType::SIN) return std::sin(v);
    if (type == NodeType::COS) return std::cos(v);
    if (type == NodeType::EXP) {
        double re = std::clamp(v.real(), -30.0, 30.0);
        return std::exp(Complex(re, v.imag()));
    }
    if (type == NodeType::SQR) return v * v;
    if (type == NodeType::SINH) {
        double re = std::clamp(v.real(), -30.0, 30.0);
        return std::sinh(Complex(re, v.imag()));
    }
    if (type == NodeType::COSH) {
        double re = std::clamp(v.real(), -30.0, 30.0);
        return std::cosh(Complex(re, v.imag()));
    }
    return 0.0;
}
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
AD BinaryNode::ad_eval(double x, double y, int dim) const {
    AD r;
    AD L = left->ad_eval(x, y, dim);
    AD R = right->ad_eval(x, y, dim);
    if (type == NodeType::ADD) {
        r.v = L.v + R.v; r.dx = L.dx + R.dx; r.dxx = L.dxx + R.dxx;
        if (dim == 2) { r.dy = L.dy + R.dy; r.dyy = L.dyy + R.dyy; }
    } else if (type == NodeType::SUB) {
        r.v = L.v - R.v; r.dx = L.dx - R.dx; r.dxx = L.dxx - R.dxx;
        if (dim == 2) { r.dy = L.dy - R.dy; r.dyy = L.dyy - R.dyy; }
    } else if (type == NodeType::MUL) {
        r.v = L.v * R.v;
        r.dx = L.dx*R.v + L.v*R.dx;
        r.dxx = L.dxx*R.v + 2.0*L.dx*R.dx + L.v*R.dxx;
        if (dim == 2) {
            r.dy = L.dy*R.v + L.v*R.dy;
            r.dyy = L.dyy*R.v + 2.0*L.dy*R.dy + L.v*R.dyy;
        }
    } else if (type == NodeType::DIV) {
        if (std::abs(R.v) < 1e-6) {
            r.v = 1.0; r.dx = 0; r.dy = 0; r.dxx = 0; r.dyy = 0;
        } else {
            Complex R2 = R.v * R.v;
            r.v   = L.v / R.v;
            r.dx  = (L.dx * R.v - L.v * R.dx) / R2;
            r.dxx = (L.dxx * R.v - L.v * R.dxx) / R2 - 2.0 * R.dx * (L.dx * R.v - L.v * R.dx) / (R2 * R.v);
            if (dim == 2) {
                r.dy  = (std::abs(R.v) < 1e-8) ? 0.0 : (L.dy * R.v - L.v * R.dy) / R2;
                r.dyy = (std::abs(R.v) < 1e-8) ? 0.0 : (L.dyy * R.v - L.v * R.dyy) / R2 - 2.0 * R.dy * (L.dy * R.v - L.v * R.dy) / (R2 * R.v);
            }
        }
    }
    return r;
}
Complex BinaryNode::eval(double x, double y) const { 
    Complex lv = left->eval(x, y);
    Complex rv = right->eval(x, y);
    if (type == NodeType::ADD) return lv + rv;
    if (type == NodeType::SUB) return lv - rv;
    if (type == NodeType::MUL) return lv * rv;
    if (type == NodeType::DIV) return (std::abs(rv) < 1e-8) ? 1.0 : lv / rv;
    return 0.0;
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
NodePtr make_erc(Complex val) { return std::make_unique<TerminalNode>(NodeType::ERC, val); }
NodePtr make_const_i() { return std::make_unique<TerminalNode>(NodeType::CONST_I); }
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
        std::uniform_int_distribution<int> ti(0, 3);
        int choice = ti(gen);
        if (choice == 0) return make_var('x');
        if (choice == 1) return make_var('y');
        if (choice == 2) return make_erc(erc_dist(gen));
        return std::make_unique<TerminalNode>(NodeType::CONST_I);
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

// Helper para generar árboles restringidos a ciertas variables
static NodePtr random_tree_restricted(int max_depth, std::mt19937& gen, const std::vector<NodeType>& allowed_vars) {
    static const std::vector<NodeType> binaries = { NodeType::ADD, NodeType::SUB, NodeType::MUL, NodeType::DIV };
    static const std::vector<NodeType> unaries = { NodeType::SIN, NodeType::COS, NodeType::EXP, NodeType::SQR };
    
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::uniform_real_distribution<double> erc_dist(-2.0, 2.0);

    if (max_depth <= 0 || ud(gen) < 0.3) {
        if (allowed_vars.empty()) return make_erc(erc_dist(gen));
        std::uniform_int_distribution<int> vi(0, (int)allowed_vars.size() + 1);
        int c = vi(gen);
        if (c < (int)allowed_vars.size()) return std::make_unique<TerminalNode>(allowed_vars[c]);
        if (c == (int)allowed_vars.size()) return make_erc(erc_dist(gen));
        return std::make_unique<TerminalNode>(NodeType::CONST_I);
    }

    if (ud(gen) < 0.6) {
        NodeType op = binaries[std::uniform_int_distribution<int>(0, binaries.size()-1)(gen)];
        return make_binary(op, random_tree_restricted(max_depth-1, gen, allowed_vars), 
                               random_tree_restricted(max_depth-1, gen, allowed_vars));
    } else {
        NodeType op = unaries[std::uniform_int_distribution<int>(0, unaries.size()-1)(gen)];
        return make_unary(op, random_tree_restricted(max_depth-1, gen, allowed_vars));
    }
}

NodePtr random_tree_special(int max_depth, std::mt19937& gen, const PDEProblem& prob) {
    std::uniform_real_distribution<double> ud(0, 1);
    int dim = prob.dim;

    // Arquetipo: Gaussiana centrada P(x,y)*exp(-a*r^2)
    auto make_gaussian = [&](double sigma_val) {
        auto dx = make_binary(NodeType::SUB, make_var('x'), make_erc(0.5));
        auto r2 = make_binary(NodeType::MUL, dx->clone(), dx->clone());
        if (dim == 2) {
            auto dy = make_binary(NodeType::SUB, make_var('y'), make_erc(0.5));
            auto y2 = make_binary(NodeType::MUL, dy->clone(), dy->clone());
            r2 = make_binary(NodeType::ADD, std::move(r2), std::move(y2));
        }
        auto exponent = make_binary(NodeType::MUL, make_erc(-sigma_val), std::move(r2));
        auto gauss = make_unary(NodeType::EXP, std::move(exponent));
        auto poly = (ud(gen) < 0.5) ? make_var('x') : make_erc(1.0);
        return make_binary(NodeType::MUL, std::move(poly), std::move(gauss));
    };

    // Arquetipo: Separación de variables sin(kx)*sinh(ky)
    auto make_sep_vars = [&]() {
        auto sx = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_erc(3.1415), make_var('x')));
        if (dim == 1) return sx;
        auto sy = make_unary(NodeType::SINH, make_binary(NodeType::MUL, make_erc(1.0), make_var('y')));
        return make_binary(NodeType::MUL, std::move(sx), std::move(sy));
    };

    // Arquetipo: Polinomio de Taylor aleatorio
    auto make_poly = [&]() {
        auto x = make_var('x');
        auto x2 = make_binary(NodeType::MUL, x->clone(), x->clone());
        auto term1 = make_binary(NodeType::MUL, make_erc(ud(gen)*2-1), std::move(x));
        auto term2 = make_binary(NodeType::MUL, make_erc(ud(gen)*2-1), std::move(x2));
        return make_binary(NodeType::ADD, std::move(term1), std::move(term2));
    };

    // Arquetipo: Función Racional P(x)/Q(x)
    auto make_rational = [&]() {
        return make_binary(NodeType::DIV, make_poly(), make_binary(NodeType::ADD, make_erc(1.0), make_unary(NodeType::SQR, make_var('x'))));
    };

    // --- Inyección de Soluciones Analíticas Conocidas (Seed de alta calidad) ---
    // Las constantes son ERCs LIBRES en un rango razonable: el optimizador
    // (QR + hill-climbing) encontrará el valor exacto (π, 2π, etc.) sin
    // necesitar nodos extra en el árbol.
    static const double PI_VAL = std::acos(-1.0);
    // ERC libre en [0.5, 4π] — cubre π, 2π, 3π sin sesgo
    std::uniform_real_distribution<double> free_k(0.5, 4.0 * PI_VAL);
    auto free_erc = [&]() { return make_erc(free_k(gen)); };

    if (ud(gen) < 0.20) {
        if (prob.type == PDE::LAPLACE) {
            if (dim == 1) return make_var('x');
            // sin(πx) * sinh(πy) / sinh(π)  — constantes como ERC ajustable
            auto s_pi_x  = make_unary(NodeType::SIN,  make_binary(NodeType::MUL, free_erc(), make_var('x')));
            auto sh_pi_y = make_unary(NodeType::SINH, make_binary(NodeType::MUL, free_erc(), make_var('y')));
            return make_binary(NodeType::MUL, std::move(s_pi_x),
                               make_binary(NodeType::MUL, make_erc(1.0/11.5487), std::move(sh_pi_y)));
        }
        if (prob.type == PDE::POISSON || prob.type == PDE::HELMHOLTZ) {
            auto s_pi_x = make_unary(NodeType::SIN, make_binary(NodeType::MUL, free_erc(), make_var('x')));
            if (dim == 1) return s_pi_x;
            auto s_pi_y = make_unary(NodeType::SIN, make_binary(NodeType::MUL, free_erc(), make_var('y')));
            return make_binary(NodeType::MUL, std::move(s_pi_x), std::move(s_pi_y));
        }
        if (prob.type == PDE::SCHRODINGER) {
            if (dim == 1) {
                // exp(i * π * x)  — π como ERC ajustable
                auto i_pi_x = make_binary(NodeType::MUL, make_const_i(),
                                make_binary(NodeType::MUL, free_erc(), make_var('x')));
                return make_unary(NodeType::EXP, std::move(i_pi_x));
            } else {
                auto i_pi_xy = make_binary(NodeType::MUL, make_const_i(),
                                 make_binary(NodeType::MUL, free_erc(),
                                   make_binary(NodeType::ADD, make_var('x'), make_var('y'))));
                return make_unary(NodeType::EXP, std::move(i_pi_xy));
            }
        }
        if (prob.type == PDE::HARMONIC_OSCILLATOR) {
            // u = Hermite * Gaussiana: H_0 * exp(-x^2/2) = exp(-a*x^2)
            std::normal_distribution<double> perturb(0.0, 0.05);
            auto exp_arg = make_binary(NodeType::MUL,
                               make_erc(-0.5 + perturb(gen)),
                               make_binary(NodeType::MUL, make_var('x'), make_var('x')));
            return make_unary(NodeType::EXP, std::move(exp_arg));
        }
        if (prob.type == PDE::AIRY) {
            // Aproximación: Ai(x) ~ exp(-2/3 * x^(3/2)) para x>0
            // Aproximamos como exp(-a * x * sqrt(x))
            std::normal_distribution<double> perturb(0.0, 0.05);
            auto x_sq = make_unary(NodeType::SQR, make_var('x'));
            auto x15  = make_binary(NodeType::MUL, make_var('x'), std::move(x_sq));
            auto arg  = make_binary(NodeType::MUL, make_erc(-0.6667 + perturb(gen)), std::move(x15));
            return make_unary(NodeType::EXP, std::move(arg));
        }
    }

    // Selección de arquetipo según el problema
    if (prob.type == PDE::HARMONIC_OSCILLATOR || prob.type == PDE::SCHRODINGER) {
        if (ud(gen) < 0.7) return make_gaussian(ud(gen) * 2.0 + 0.5);
        return make_binary(NodeType::MUL, make_poly(), make_gaussian(1.0)); // P(x)*exp(-r^2)
    }
    if (prob.type == PDE::LAPLACE || prob.type == PDE::POISSON || prob.type == PDE::HELMHOLTZ) {
        if (ud(gen) < 0.5) return make_sep_vars();
        return make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_erc(3.1415), make_var('x')));
    }
    if (prob.type == PDE::AIRY) {
        if (ud(gen) < 0.5) {
            auto x_pow = make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::SQR, make_var('x')));
            return make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-0.66), std::move(x_pow)));
        }
        return make_rational();
    }

    double p = ud(gen);
    if (p < 0.25) return make_gaussian(1.0);
    if (p < 0.50) return make_sep_vars();
    if (p < 0.75) return make_poly();
    return make_rational();
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
    NodePtr new_sub = random_tree(3, gen);
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
        Complex val = dynamic_cast<TerminalNode*>(s_child.get())->erc_val;
        if (type == NodeType::SIN) return make_erc(std::sin(val));
        if (type == NodeType::COS) return make_erc(std::cos(val));
        if (type == NodeType::EXP) return make_erc(std::exp(val));
        if (type == NodeType::SQR) return make_erc(val * val);
        if (type == NodeType::SINH) return make_erc(std::sinh(val));
        if (type == NodeType::COSH) return make_erc(std::cosh(val));
    }

    // Special cases for functions at 0
    if (s_child->get_type() == NodeType::ERC) {
        Complex val = dynamic_cast<TerminalNode*>(s_child.get())->erc_val;
        if (std::abs(val) < 1e-9) {
            if (type == NodeType::SIN) return make_erc(0.0);
            if (type == NodeType::COS) return make_erc(1.0);
            if (type == NodeType::EXP) return make_erc(1.0);
            if (type == NodeType::SINH) return make_erc(0.0);
            if (type == NodeType::COSH) return make_erc(1.0);
        }
    }

    return std::make_unique<UnaryNode>(type, std::move(s_child));
}

NodePtr BinaryNode::simplify() const {
    NodePtr s_l = left->simplify();
    NodePtr s_r = right->simplify();
    
    bool l_erc = (s_l->get_type() == NodeType::ERC);
    bool r_erc = (s_r->get_type() == NodeType::ERC);
    Complex lv = l_erc ? dynamic_cast<TerminalNode*>(s_l.get())->erc_val : 0.0;
    Complex rv = r_erc ? dynamic_cast<TerminalNode*>(s_r.get())->erc_val : 0.0;

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
        if (l_erc && std::abs(lv) < 1e-9) return s_r;
        if (r_erc && std::abs(rv) < 1e-9) return s_l;
        
        // Identity: sin^2(x) + cos^2(x) = 1
        const Node* arg_sin_l = get_arg_of_square(s_l.get(), NodeType::SIN);
        const Node* arg_cos_r = get_arg_of_square(s_r.get(), NodeType::COS);
        if (arg_sin_l && arg_cos_r && is_structurally_equal(arg_sin_l, arg_cos_r)) return make_erc(1.0);

        const Node* arg_cos_l = get_arg_of_square(s_l.get(), NodeType::COS);
        const Node* arg_sin_r = get_arg_of_square(s_r.get(), NodeType::SIN);
        if (arg_cos_l && arg_sin_r && is_structurally_equal(arg_cos_l, arg_sin_r)) return make_erc(1.0);
    }
    if (type == NodeType::SUB) {
        if (r_erc && std::abs(rv) < 1e-9) return s_l;
        if (is_structurally_equal(s_l.get(), s_r.get())) return make_erc(0.0);

        // Identity: cosh^2(x) - sinh^2(x) = 1
        const Node* arg_cosh_l = get_arg_of_square(s_l.get(), NodeType::COSH);
        const Node* arg_sinh_r = get_arg_of_square(s_r.get(), NodeType::SINH);
        if (arg_cosh_l && arg_sinh_r && is_structurally_equal(arg_cosh_l, arg_sinh_r)) return make_erc(1.0);
    }
    if (type == NodeType::MUL) {
        if (l_erc && std::abs(lv) < 1e-9) return make_erc(0.0);
        if (r_erc && std::abs(rv) < 1e-9) return make_erc(0.0);
        if (l_erc && std::abs(lv - 1.0) < 1e-9) return s_r;
        if (r_erc && std::abs(rv - 1.0) < 1e-9) return s_l;
    }
    if (type == NodeType::DIV) {
        if (l_erc && std::abs(lv) < 1e-9) return make_erc(0.0);
        if (r_erc && std::abs(rv - 1.0) < 1e-9) return s_l;
        if (is_structurally_equal(s_l.get(), s_r.get())) return make_erc(1.0);
    }

    return std::make_unique<BinaryNode>(type, std::move(s_l), std::move(s_r));
}

// ─── Laplaciano por diferencias finitas (Koza) ────────────────────────────────
Complex fd_laplacian(const NodePtr& tree, double x, double y, int dim, double h) {
    Complex v   = tree->eval(x,   y);
    Complex vxp = tree->eval(x+h, y);
    Complex vxm = tree->eval(x-h, y);
    Complex dxx = (vxp - 2.0*v + vxm) / (h*h);
    if (dim == 1) return dxx;
    Complex vyp = tree->eval(x,   y+h);
    Complex vym = tree->eval(x,   y-h);
    Complex dyy = (vyp - 2.0*v + vym) / (h*h);
    return dxx + dyy;
}

// ─── Poda Recursiva ───────────────────────────────────────────────────────────
static double calculate_mse_simple(const NodePtr& tree, const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd) {
    double sum_dom = 0.0;
    for (auto& p : dom) {
        AD ad = tree->ad_eval(p.x, p.y, prob.dim);
        Complex res = prob.pde_residual_ad(ad, p.x, p.y);
        sum_dom += std::norm(res);
    }
    double sum_bnd = 0.0;
    for (auto& p : bnd) {
        Complex diff = tree->eval(p.x, p.y) - prob.bc(p.x, p.y);
        sum_bnd += std::norm(diff);
    }
    return (sum_dom / dom.size()) + (sum_bnd / bnd.size());
}

NodePtr TerminalNode::prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) {
    return clone();
}

NodePtr UnaryNode::prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) {
    auto new_child = child->prune_recursive(prob, dom, bnd, original_mse, tolerance);
    auto current_node = std::make_unique<UnaryNode>(type, std::move(new_child));
    Complex v0 = current_node->eval(0.5, 0.5);
    NodePtr const_node = std::make_unique<TerminalNode>(NodeType::ERC, v0);
    double new_mse = calculate_mse_simple(const_node, prob, dom, bnd);
    if (new_mse <= original_mse * (1.0 + tolerance)) return const_node;
    return current_node;
}

NodePtr BinaryNode::prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) {
    auto new_l = left->prune_recursive(prob, dom, bnd, original_mse, tolerance);
    auto new_r = right->prune_recursive(prob, dom, bnd, original_mse, tolerance);
    auto current_node = std::make_unique<BinaryNode>(type, std::move(new_l), std::move(new_r));
    Complex v0 = current_node->eval(0.5, 0.5);
    NodePtr const_node = std::make_unique<TerminalNode>(NodeType::ERC, v0);
    double new_mse = calculate_mse_simple(const_node, prob, dom, bnd);
    if (new_mse <= original_mse * (1.0 + tolerance)) return const_node;
    return current_node;
}
