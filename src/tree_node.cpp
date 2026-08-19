#include "tree_node.hpp"
#include <iomanip>
#include <algorithm>
#include <sstream>

extern int last_special_choice;

// ─── Polinomios Ortogonales (Motor Físico) ───────────────────────────────────
static void eval_poly_all(NodeType type, int n, double x, double& v, double& dv, double& dvv) {
    if (n < 0 || n > 10 || !std::isfinite(x)) { v = NAN; dv = NAN; dvv = NAN; return; }
    if (n == 0) { v = 1.0; dv = 0.0; dvv = 0.0; return; }
    if (n == 1) {
        if (type == NodeType::HERMITE) { v = 2.0*x; dv = 2.0; dvv = 0.0; }
        else if (type == NodeType::LAGUERRE) { v = 1.0 - x; dv = -1.0; dvv = 0.0; }
        else { v = x; dv = 1.0; dvv = 0.0; }
        return;
    }

    double p0 = 1.0, p1 = (type == NodeType::HERMITE) ? 2.0*x : ((type == NodeType::LAGUERRE) ? 1.0-x : x);
    double dp0 = 0.0, dp1 = (type == NodeType::HERMITE) ? 2.0 : ((type == NodeType::LAGUERRE) ? -1.0 : 1.0);
    double ddp0 = 0.0, ddp1 = 0.0;

    for (int k = 1; k < n; ++k) {
        double cur_v, cur_dv, cur_dvv;
        if (type == NodeType::LEGENDRE) {
            cur_v = ((2.0*k + 1.0)*x*p1 - k*p0) / (k + 1.0);
            cur_dv = ((2.0*k + 1.0)*(p1 + x*dp1) - k*dp0) / (k + 1.0);
            cur_dvv = ((2.0*k + 1.0)*(2.0*dp1 + x*ddp1) - k*ddp0) / (k + 1.0);
        } else if (type == NodeType::CHEBYSHEV) {
            cur_v = 2.0*x*p1 - p0;
            cur_dv = 2.0*(p1 + x*dp1) - dp0;
            cur_dvv = 2.0*(2.0*dp1 + x*ddp1) - ddp0;
        } else if (type == NodeType::HERMITE) {
            cur_v = 2.0*x*p1 - 2.0*k*p0;
            cur_dv = 2.0*(p1 + x*dp1) - 2.0*k*dp0;
            cur_dvv = 2.0*(2.0*dp1 + x*ddp1) - 2.0*k*ddp0;
        } else { // LAGUERRE
            cur_v = ((2.0*k + 1.0 - x)*p1 - k*p0) / (k + 1.0);
            cur_dv = ((2.0*k + 1.0 - x)*dp1 - p1 - k*dp0) / (k + 1.0);
            cur_dvv = ((2.0*k + 1.0 - x)*ddp1 - 2.0*dp1 - k*ddp0) / (k + 1.0);
        }
        p0 = p1; p1 = cur_v; dp0 = dp1; dp1 = cur_dv; ddp0 = ddp1; ddp1 = cur_dvv;
    }
    v = p1; dv = dp1; dvv = ddp1;
}

Complex apply_unary(NodeType type, Complex v) {
    if (type == NodeType::SIN) return std::sin(v); if (type == NodeType::COS) return std::cos(v);
    if (type == NodeType::EXP) return (v.real() > 100.0) ? Complex(NAN, NAN) : std::exp(v); 
    if (type == NodeType::SQR) return v*v;
    if (type == NodeType::GAUSSIAN) return std::exp(-0.5*v*v); if (type == NodeType::TANH) return std::tanh(v);
    if (type == NodeType::LOG) return (v.real() <= 0.0) ? Complex(NAN, NAN) : std::log(v);
    if (type == NodeType::SINH) return std::sinh(v); if (type == NodeType::COSH) return std::cosh(v);
    return 0.0;
}

AD apply_unary_ad(NodeType type, const AD& C) {
    AD r;
    if (type == NodeType::SIN) {
        Complex s = std::sin(C.v), c = std::cos(C.v); r.v = s; r.dx = c*C.dx; r.dy = c*C.dy; r.dt = c*C.dt;
        r.dxx = c*C.dxx - s*C.dx*C.dx; r.dyy = c*C.dyy - s*C.dy*C.dy; r.dtt = c*C.dtt - s*C.dt*C.dt;
    } else if (type == NodeType::COS) {
        Complex s = std::sin(C.v), c = std::cos(C.v); r.v = c; r.dx = -s*C.dx; r.dy = -s*C.dy; r.dt = -s*C.dt;
        r.dxx = -s*C.dxx - c*C.dx*C.dx; r.dyy = -s*C.dyy - c*C.dy*C.dy; r.dtt = -s*C.dtt - c*C.dt*C.dt;
    } else if (type == NodeType::EXP) {
        if (C.v.real() > 100.0) { r.v = NAN; return r; }
        Complex ev = std::exp(C.v); r.v = ev; r.dx = ev*C.dx; r.dy = ev*C.dy; r.dt = ev*C.dt;
        r.dxx = ev*(C.dxx + C.dx*C.dx); r.dyy = ev*(C.dyy + C.dy*C.dy); r.dtt = ev*(C.dtt + C.dt*C.dt);
    } else if (type == NodeType::SQR) {
        r.v = C.v*C.v; r.dx = 2.0*C.v*C.dx; r.dy = 2.0*C.v*C.dy; r.dt = 2.0*C.v*C.dt;
        r.dxx = 2.0*(C.dx*C.dx + C.v*C.dxx); r.dyy = 2.0*(C.dy*C.dy + C.v*C.dyy); r.dtt = 2.0*(C.dt*C.dt + C.v*C.dtt);
    } else if (type == NodeType::GAUSSIAN) {
        Complex g = std::exp(-0.5*C.v*C.v); r.v = g;
        r.dx = -C.v*g*C.dx; r.dy = -C.v*g*C.dy; r.dt = -C.v*g*C.dt;
        r.dxx = g*( (C.v*C.dx)*(C.v*C.dx) - C.dx*C.dx - C.v*C.dxx );
        r.dyy = g*( (C.v*C.dy)*(C.v*C.dy) - C.dy*C.dy - C.v*C.dyy );
    } else if (type == NodeType::TANH) {
        Complex t = std::tanh(C.v), s2 = 1.0/(std::cosh(C.v)*std::cosh(C.v));
        r.v = t; r.dx = s2*C.dx; r.dy = s2*C.dy; r.dt = s2*C.dt;
        r.dxx = s2*C.dxx - 2.0*t*s2*C.dx*C.dx; r.dyy = s2*C.dyy - 2.0*t*s2*C.dy*C.dy;
    } else if (type == NodeType::LOG) {
        if (C.v.real() <= 0.0) { r.v = NAN; return r; }
        Complex inv = 1.0/C.v; r.v = std::log(C.v); r.dx = inv*C.dx; r.dy = inv*C.dy; r.dt = inv*C.dt;
        r.dxx = inv*C.dxx - (inv*inv)*C.dx*C.dx; r.dyy = inv*C.dyy - (inv*inv)*C.dy*C.dy;
    } else if (type == NodeType::SINH) {
        Complex sh = std::sinh(C.v), ch = std::cosh(C.v); r.v = sh; r.dx = ch*C.dx; r.dy = ch*C.dy; r.dt = ch*C.dt;
        r.dxx = ch*C.dxx + sh*C.dx*C.dx; r.dyy = ch*C.dyy + sh*C.dy*C.dy; r.dtt = ch*C.dtt + sh*C.dt*C.dt;
    } else if (type == NodeType::COSH) {
        Complex sh = std::sinh(C.v), ch = std::cosh(C.v); r.v = ch; r.dx = sh*C.dx; r.dy = sh*C.dy; r.dt = sh*C.dt;
        r.dxx = ch*C.dxx + ch*C.dx*C.dx; r.dyy = ch*C.dyy + ch*C.dy*C.dy; r.dtt = ch*C.dtt + ch*C.dt*C.dt;
    }
    return r;
}

Complex apply_binary(NodeType type, Complex lv, Complex rv) {
    if (type == NodeType::ADD) return lv+rv; if (type == NodeType::SUB) return lv-rv;
    if (type == NodeType::MUL) return lv*rv; 
    if (type == NodeType::DIV) return (std::abs(rv.real()) < 1e-12) ? Complex(NAN, NAN) : lv / rv;
    if (type == NodeType::POW) return (lv.real() < 0.0) ? Complex(NAN, NAN) : std::pow(lv, rv);
    
    if (!std::isfinite(rv.real())) return Complex(NAN, NAN);
    double pv, pdv, pdvv; int n = std::clamp((int)std::round(rv.real()), 0, 10);
    eval_poly_all(type, n, lv.real(), pv, pdv, pdvv);
    return Complex(pv, 0.0);
}

AD apply_binary_ad(NodeType type, const AD& L, const AD& R) {
    AD r;
    if (type == NodeType::ADD) {
        r.v = L.v+R.v; r.dx = L.dx+R.dx; r.dy = L.dy+R.dy; r.dt = L.dt+R.dt; r.dxx = L.dxx+R.dxx; r.dyy = L.dyy+R.dyy;
    } else if (type == NodeType::SUB) {
        r.v = L.v-R.v; r.dx = L.dx-R.dx; r.dy = L.dy-R.dy; r.dt = L.dt-R.dt; r.dxx = L.dxx-R.dxx; r.dyy = L.dyy-R.dyy;
    } else if (type == NodeType::MUL) {
        r.v = L.v*R.v; r.dx = L.dx*R.v + L.v*R.dx; r.dy = L.dy*R.v + L.v*R.dy; r.dt = L.dt*R.v + L.v*R.dt;
        r.dxx = L.dxx*R.v + 2.0*L.dx*R.dx + L.v*R.dxx;
        r.dyy = L.dyy*R.v + 2.0*L.dy*R.dy + L.v*R.dyy;
    } else if (type == NodeType::DIV) {
        if (std::abs(R.v.real()) < 1e-12) { r.v = NAN; return r; }
        Complex inv = 1.0/R.v, inv2 = inv*inv, inv3 = inv2*inv;
        r.v = L.v*inv;
        r.dx = (L.dx*R.v - L.v*R.dx)*inv2; r.dy = (L.dy*R.v - L.v*R.dy)*inv2;
        r.dxx = L.dxx*inv - (2.0*L.dx*R.dx + L.v*R.dxx)*inv2 + (2.0*L.v*R.dx*R.dx)*inv3;
        r.dyy = L.dyy*inv - (2.0*L.dy*R.dy + L.v*R.dyy)*inv2 + (2.0*L.v*R.dy*R.dy)*inv3;
    } else if (type == NodeType::POW) {
        if (L.v.real() < 0.0) { r.v = NAN; return r; }
        Complex val = std::pow(L.v, R.v); r.v = val;
        Complex d_base = R.v * std::pow(L.v, R.v - 1.0);
        Complex dd_base = R.v * (R.v - 1.0) * std::pow(L.v, R.v - 2.0);
        r.dx = d_base * L.dx; r.dy = d_base * L.dy; r.dt = d_base * L.dt;
        r.dxx = dd_base * L.dx * L.dx + d_base * L.dxx;
        r.dyy = dd_base * L.dy * L.dy + d_base * L.dyy;
    } else { // Polinomios Ortogonales
        if (!std::isfinite(R.v.real())) { r.v = NAN; return r; }
        double pv, pdv, pdvv; int n = std::clamp((int)std::round(R.v.real()), 0, 10);
        eval_poly_all(type, n, L.v.real(), pv, pdv, pdvv);
        r.v = pv; r.dx = pdv*L.dx; r.dy = pdv*L.dy; r.dt = pdv*L.dt;
        r.dxx = pdvv*L.dx*L.dx + pdv*L.dxx; r.dyy = pdvv*L.dy*L.dy + pdv*L.dyy; r.dtt = pdvv*L.dt*L.dt + pdv*L.dtt;
    }
    return r;
}

// ─── TerminalNode Implementation ─────────────────────────────────────────────
thread_local int current_n = 1;
AD TerminalNode::ad_eval(double x, double y, int dim) const { return ad_eval_t(x, y, 0.0, dim); }
AD TerminalNode::ad_eval_t(double x, double y, double t, int dim) const {
    AD r; r.v = eval_t(x, y, t);
    if (type == NodeType::VAR_X) r.dx = 1.0;
    else if (type == NodeType::VAR_Y && dim >= 2) r.dy = 1.0;
    else if (type == NodeType::VAR_T) r.dt = 1.0;
    return r;
}
Complex TerminalNode::eval(double x, double y) const { return eval_t(x, y, 0.0); }
Complex TerminalNode::eval_t(double x, double y, double t) const {
    if (type == NodeType::VAR_X) return x; if (type == NodeType::VAR_Y) return y;
    if (type == NodeType::VAR_T) return t; if (type == NodeType::VAR_N) return (double)current_n; 
    if (type == NodeType::VAR_Z) return x*0.001 + y*0.001; 
    if (type == NodeType::ERC) {
        if (std::abs(erc_val.real()) > 1e10 || std::abs(erc_val.imag()) > 1e10) return Complex(NAN, NAN);
        return erc_val;
    }
    if (type == NodeType::CONST_I) return {0,1}; if (type == NodeType::CONST_PI) return 3.14159265358979;
    if (type == NodeType::CONST_E) return 2.71828182845904; 
    if (type == NodeType::CONST_G) return 6.67430e-11;
    if (type == NodeType::CONST_C) return 299792458.0;
    if (type == NodeType::CONST_HBAR) return 1.0545718e-34;
    if (type == NodeType::CONST_KB) return 1.380649e-23;
    if (type == NodeType::CONST_EPS0) return 8.8541878e-12;
    return 0.0;
}
void TerminalNode::print(std::ostream& os) const {
    if (type == NodeType::VAR_X) os << "x"; else if (type == NodeType::VAR_Y) os << "y";
    else if (type == NodeType::VAR_T) os << "t"; else if (type == NodeType::VAR_Z) os << "z";
    else if (type == NodeType::VAR_N) os << "n";
    else if (type == NodeType::ERC) os << std::fixed << std::setprecision(3) << erc_val.real();
    else if (type == NodeType::CONST_I) os << "i"; else if (type == NodeType::CONST_PI) os << "pi";
    else if (type == NodeType::CONST_E) os << "e";
}
void TerminalNode::print_latex(std::ostream& os) const {
    if (type == NodeType::VAR_X) os << "x"; else if (type == NodeType::VAR_Y) os << "y";
    else if (type == NodeType::VAR_T) os << "t"; else if (type == NodeType::VAR_Z) os << "z";
    else if (type == NodeType::VAR_N) os << "n";
    else if (type == NodeType::ERC) os << std::fixed << std::setprecision(3) << erc_val.real();
    else if (type == NodeType::CONST_I) os << "i"; else if (type == NodeType::CONST_PI) os << "\\pi";
    else if (type == NodeType::CONST_E) os << "e";
}
bool TerminalNode::contains_variables() const { 
    return type == NodeType::VAR_X || type == NodeType::VAR_Y || type == NodeType::VAR_T || type == NodeType::VAR_Z; 
}
bool TerminalNode::is_unit_flexible() const { return is_constant(type); }
bool TerminalNode::contains_erc() const { return type == NodeType::ERC; }

std::optional<Dimension> TerminalNode::get_dimension(const PDEProblem& p) const {
    if (type == NodeType::VAR_X) return p.dim_x; 
    if (type == NodeType::VAR_Y) return p.dim_y;
    if (type == NodeType::VAR_T) return p.dim_t; 
    if (type == NodeType::VAR_N) return Units::None;
    if (type == NodeType::VAR_Z) return Units::Length;
    if (type == NodeType::CONST_G) return Dimension(3, -2, -1);
    if (type == NodeType::CONST_C) return Units::Velocity;
    if (type == NodeType::CONST_HBAR) return Dimension(2, -1, 1);
    if (type == NodeType::CONST_KB) return Dimension(2, -2, 1, 0, -1);
    if (type == NodeType::CONST_EPS0) return Dimension(-3, 4, -1, 2);
    return Units::None; 
}
void TerminalNode::mutate_erc(std::mt19937& gen, double sigma) {
    if (type == NodeType::ERC) {
        std::normal_distribution<double> dist(0, sigma);
        erc_val += Complex(dist(gen), 0);
    }
}
NodePtr TerminalNode::clone() const { return std::make_unique<TerminalNode>(type, erc_val); }
int TerminalNode::count_nodes() const { return 1; }
int TerminalNode::get_depth() const { return 1; }

// ─── UnaryNode Implementation ───────────────────────────────────────────────
AD UnaryNode::ad_eval(double x, double y, int dim) const { return ad_eval_t(x, y, 0.0, dim); }
AD UnaryNode::ad_eval_t(double x, double y, double t, int dim) const {
    if (!child) return AD(0.0);
    AD C = child->ad_eval_t(x, y, t, dim);
    return apply_unary_ad(type, C);
}
Complex UnaryNode::eval(double x, double y) const { return eval_t(x, y, 0.0); }
Complex UnaryNode::eval_t(double x, double y, double t) const {
    if (!child) return 0.0;
    return apply_unary(type, child->eval_t(x, y, t));
}
std::optional<Dimension> UnaryNode::get_dimension(const PDEProblem& p) const {
    if (!child) return std::nullopt;
    auto d = child->get_dimension(p);
    if (!d) return std::nullopt;
    if (type == NodeType::SQR) return *d + *d;
    if (type == NodeType::SIN || type == NodeType::COS || type == NodeType::EXP || 
        type == NodeType::TANH || type == NodeType::LOG || type == NodeType::GAUSSIAN ||
        type == NodeType::SINH || type == NodeType::COSH) 
    {
        if (d->is_adimensional()) return Units::None;
        if (child->contains_erc()) return Units::None;
        return std::nullopt; 
    }
    return d; 
}
bool UnaryNode::is_unit_flexible() const { return child && child->is_unit_flexible(); }
bool UnaryNode::contains_erc() const { return child && child->contains_erc(); }
bool UnaryNode::contains_variables() const { return child && child->contains_variables(); }
void UnaryNode::print(std::ostream& os) const {
    if (type == NodeType::SIN) os << "sin("; else if (type == NodeType::COS) os << "cos(";
    else if (type == NodeType::EXP) os << "exp("; else if (type == NodeType::SQR) os << "sqr(";
    else if (type == NodeType::GAUSSIAN) os << "G("; else if (type == NodeType::TANH) os << "tanh(";
    else if (type == NodeType::LOG) os << "log("; else if (type == NodeType::SINH) os << "sinh(";
    else if (type == NodeType::COSH) os << "cosh("; else os << "u(";
    if (child) child->print(os); os << ")";
}
void UnaryNode::print_latex(std::ostream& os) const {
    if (type == NodeType::SIN) os << "\\sin("; else if (type == NodeType::COS) os << "\\cos(";
    else if (type == NodeType::EXP) os << "\\exp("; else if (type == NodeType::SQR) os << "(";
    else if (type == NodeType::GAUSSIAN) os << "\\exp(-0.5 "; else if (type == NodeType::TANH) os << "\\tanh(";
    else if (type == NodeType::LOG) os << "\\log("; else if (type == NodeType::SINH) os << "\\sinh(";
    else if (type == NodeType::COSH) os << "\\cosh("; else os << "u(";
    if (child) child->print_latex(os);
    if (type == NodeType::SQR) os << ")^2";
    else if (type == NodeType::GAUSSIAN) os << "^2)";
    else os << ")";
}
NodePtr UnaryNode::simplify() const {
    if (!child) return clone();
    if (!contains_variables()) { return make_erc(eval_t(0, 0, 0)); }
    auto s = child->simplify();
    if (is_constant(s->get_type())) {
        Complex val = s->eval_t(0, 0, 0);
        double vr = val.real();
        const double PI_L = 3.14159265358979323846;
        if (type == NodeType::SIN) {
            if (std::abs(vr) < 1e-7) return make_erc(0.0);
            double n_pi = vr / PI_L;
            if (std::abs(n_pi - std::round(n_pi)) < 1e-7) return make_erc(0.0);
        }
        if (type == NodeType::COS) {
            if (std::abs(vr) < 1e-7) return make_erc(1.0);
            double n_pi = vr / PI_L;
            if (std::abs(n_pi - std::round(n_pi)) < 1e-7) {
                int n = (int)std::round(n_pi);
                return make_erc((n % 2 == 0) ? 1.0 : -1.0);
            }
        }
        if (type == NodeType::LOG && std::abs(vr - 1.0) < 1e-9) return make_erc(0.0);
        if (type == NodeType::EXP && std::abs(vr) < 1e-9) return make_erc(1.0);
        if (type == NodeType::TANH && std::abs(vr) < 1e-9) return make_erc(0.0);
        return make_erc(apply_unary(type, val));
    }
    if (type == NodeType::LOG && s->get_type() == NodeType::EXP) {
        auto* en = dynamic_cast<UnaryNode*>(s.get()); return en->child->clone();
    }
    if (type == NodeType::EXP && s->get_type() == NodeType::LOG) {
        auto* ln = dynamic_cast<UnaryNode*>(s.get()); return ln->child->clone();
    }
    return std::make_unique<UnaryNode>(type, std::move(s));
}
NodePtr UnaryNode::prune_recursive(const PDEProblem& p, const std::vector<Point>& d, const std::vector<Point>& b, double o, double t) {
    if (!child) return clone();
    child = child->prune_recursive(p, d, b, o, t);
    return clone();
}
int UnaryNode::get_unary_depth() const { return 1 + (child ? child->get_unary_depth() : 0); }
int UnaryNode::get_depth() const { return 1 + (child ? child->get_depth() : 0); }
NodePtr UnaryNode::clone() const { return std::make_unique<UnaryNode>(type, child ? child->clone() : nullptr); }
int UnaryNode::count_nodes() const { return 1 + (child ? child->count_nodes() : 0); }

// ─── BinaryNode Implementation ───────────────────────────────────────────────
AD BinaryNode::ad_eval(double x, double y, int dim) const { return ad_eval_t(x, y, 0.0, dim); }
AD BinaryNode::ad_eval_t(double x, double y, double t, int dim) const {
    if (!left || !right) return AD(0.0);
    AD L = left->ad_eval_t(x, y, t, dim); AD R = right->ad_eval_t(x, y, t, dim);
    return apply_binary_ad(type, L, R);
}
Complex BinaryNode::eval(double x, double y) const { return eval_t(x, y, 0.0); }
Complex BinaryNode::eval_t(double x, double y, double t) const {
    if (!left || !right) return 0.0;
    return apply_binary(type, left->eval_t(x, y, t), right->eval_t(x, y, t));
}
std::optional<Dimension> BinaryNode::get_dimension(const PDEProblem& p) const {
    if (!left || !right) return std::nullopt;
    auto dl = left->get_dimension(p);
    auto dr = right->get_dimension(p);
    if (!dl || !dr) return std::nullopt;
    if (type == NodeType::ADD || type == NodeType::SUB) {
        if (*dl == *dr) return dl;
        if (left->contains_erc()) return dr;
        if (right->contains_erc()) return dl;
        return std::nullopt;
    }
    if (type == NodeType::MUL) return *dl + *dr;
    if (type == NodeType::DIV) return *dl - *dr;
    if (type == NodeType::POW) { if (dr->is_adimensional() || right->contains_erc()) return dl; return std::nullopt; }
    if (dl->is_adimensional() && dr->is_adimensional()) return Units::None;
    if (left->contains_erc() && right->contains_erc()) return Units::None;
    return std::nullopt;
}
bool BinaryNode::is_unit_flexible() const { return false; }
bool BinaryNode::contains_erc() const { return (left && left->contains_erc()) || (right && right->contains_erc()); }
bool BinaryNode::contains_variables() const {
    return (left && left->contains_variables()) || (right && right->contains_variables());
}
void BinaryNode::print(std::ostream& os) const {
    os << "("; if (left) left->print(os);
    if (type == NodeType::ADD) os << "+"; else if (type == NodeType::SUB) os << "-";
    else if (type == NodeType::MUL) os << "*"; else if (type == NodeType::DIV) os << "/";
    else if (type == NodeType::POW) os << "^";
    else if (type == NodeType::LEGENDRE) os << "P"; else if (type == NodeType::HERMITE) os << "H";
    else if (type == NodeType::CHEBYSHEV) os << "T"; else if (type == NodeType::LAGUERRE) os << "L";
    if (right) right->print(os); os << ")";
}
void BinaryNode::print_latex(std::ostream& os) const {
    if (type == NodeType::DIV) {
        os << "\\frac{"; if (left) left->print_latex(os); os << "}{"; if (right) right->print_latex(os); os << "}";
    } else {
        os << "("; if (left) left->print_latex(os);
        if (type == NodeType::ADD) os << "+"; else if (type == NodeType::SUB) os << "-";
        else if (type == NodeType::MUL) os << " "; else if (type == NodeType::POW) os << "^{";
        else if (type >= NodeType::LEGENDRE && type <= NodeType::LAGUERRE) os << "_";
        if (right) right->print_latex(os); if (type == NodeType::POW) os << "}"; os << ")";
    }
}
NodePtr BinaryNode::simplify() const {
    if (!left || !right) return clone();
    if (!contains_variables()) { return make_erc(eval_t(0, 0, 0)); }
    auto sl = left->simplify(), sr = right->simplify();
    NodeType lt = sl->get_type(), rt = sr->get_type();
    if (is_constant(lt) && is_constant(rt)) { return make_erc(apply_binary(type, sl->eval_t(0,0,0), sr->eval_t(0,0,0))); }
    auto is_zero = [](double v) { return std::isfinite(v) && std::abs(v) < 1e-6; };
    auto is_one = [](double v) { return std::isfinite(v) && std::abs(v - 1.0) < 1e-6; };
    double lv_v = is_constant(lt) ? sl->eval_t(0,0,0).real() : NAN;
    double rv_v = is_constant(rt) ? sr->eval_t(0,0,0).real() : NAN;
    if (type == NodeType::ADD) {
        if (is_zero(lv_v)) return sr; if (is_zero(rv_v)) return sl;
        if (sl->print_str() == sr->print_str()) return make_binary(NodeType::MUL, make_erc(2.0), std::move(sl));
    }
    if (type == NodeType::SUB) { if (is_zero(rv_v)) return sl; if (sl->print_str() == sr->print_str()) return make_erc(0.0); }
    if (type == NodeType::MUL) {
        if (is_zero(lv_v) || is_zero(rv_v)) return make_erc(0.0);
        if (is_one(lv_v)) return sr; if (is_one(rv_v)) return sl;
    }
    if (type == NodeType::DIV) { if (is_zero(lv_v)) return make_erc(0.0); if (is_one(rv_v)) return sl; if (sl->print_str() == sr->print_str()) return make_erc(1.0); }
    if (type == NodeType::POW) {
        if (is_zero(rv_v)) return make_erc(1.0);
        if (is_one(rv_v)) return sl;
        if (is_zero(lv_v)) return make_erc(0.0);
        if (is_one(lv_v)) return make_erc(1.0);
    }
    return std::make_unique<BinaryNode>(type, std::move(sl), std::move(sr));
}
NodePtr BinaryNode::prune_recursive(const PDEProblem& p, const std::vector<Point>& d, const std::vector<Point>& b, double o, double t) {
    if (!left || !right) return clone();
    left = left->prune_recursive(p, d, b, o, t); right = right->prune_recursive(p, d, b, o, t);
    return clone();
}
int BinaryNode::get_unary_depth() const { return std::max(left ? left->get_unary_depth() : 0, right ? right->get_unary_depth() : 0); }
int BinaryNode::get_depth() const { return 1 + std::max(left ? left->get_depth() : 0, right ? right->get_depth() : 0); }
NodePtr BinaryNode::clone() const { return std::make_unique<BinaryNode>(type, left ? left->clone() : nullptr, right ? right->clone() : nullptr); }
int BinaryNode::count_nodes() const { return 1 + (left ? left->count_nodes() : 0) + (right ? right->count_nodes() : 0); }

// ─── SeriesNode Implementation ───────────────────────────────────────────────
AD SeriesNode::ad_eval(double x, double y, int dim) const { return ad_eval_t(x, y, 0.0, dim); }
AD SeriesNode::ad_eval_t(double x, double y, double t, int dim) const {
    AD r; r.v = 0.0; int save_n = current_n;
    for (int n = 1; n <= n_terms; ++n) { current_n = n; AD C = child->ad_eval_t(x, y, t, dim);
        Complex coef = coeffs[n - 1]; r.v += coef * C.v; r.dx += coef * C.dx; r.dy += coef * C.dy; r.dt += coef * C.dt;
        r.dxx += coef * C.dxx; r.dyy += coef * C.dyy; r.dtt += coef * C.dtt;
    }
    current_n = save_n; return r;
}
Complex SeriesNode::eval(double x, double y) const { return eval_t(x, y, 0.0); }
Complex SeriesNode::eval_t(double x, double y, double t) const {
    Complex sum = 0.0; int save_n = current_n;
    for (int n = 1; n <= n_terms; ++n) { current_n = n; sum += coeffs[n - 1] * child->eval_t(x, y, t); }
    current_n = save_n; return sum;
}
std::optional<Dimension> SeriesNode::get_dimension(const PDEProblem& prob) const { return child ? child->get_dimension(prob) : std::nullopt; }
bool SeriesNode::is_unit_flexible() const { return false; }
bool SeriesNode::contains_erc() const { return child && child->contains_erc(); }
bool SeriesNode::contains_variables() const { return child && child->contains_variables(); }
int SeriesNode::get_unary_depth() const { return 1 + (child ? child->get_unary_depth() : 0); }
int SeriesNode::get_depth() const { return 1 + (child ? child->get_depth() : 0); }
NodePtr SeriesNode::clone() const {
    auto sn = std::make_unique<SeriesNode>(n_terms, child ? child->clone() : nullptr);
    sn->coeffs = coeffs; return sn;
}
int SeriesNode::count_nodes() const { return 1 + (child ? child->count_nodes() : 0); }
void SeriesNode::mutate_erc(std::mt19937& gen, double sigma) {
    std::normal_distribution<double> dist(0, sigma);
    for (auto& c : coeffs) c += Complex(dist(gen), 0);
    if (child) child->mutate_erc(gen, sigma);
}
void SeriesNode::print(std::ostream& os) const { os << "SUM(n=1.." << n_terms << ")[C_n*"; if (child) child->print(os); os << "]"; }
void SeriesNode::print_latex(std::ostream& os) const { os << "\\sum_{n=1}^{" << n_terms << "} C_n "; if (child) child->print_latex(os); }
NodePtr SeriesNode::simplify() const { return child ? std::make_unique<SeriesNode>(n_terms, child->simplify()) : clone(); }
NodePtr SeriesNode::prune_recursive(const PDEProblem& p, const std::vector<Point>& d, const std::vector<Point>& b, double o, double t) {
    return child ? std::make_unique<SeriesNode>(n_terms, child->prune_recursive(p, d, b, o, t)) : clone();
}
void SeriesNode::collect_ercs(std::vector<Complex*>& ptrs) { for (auto& c : coeffs) ptrs.push_back(&c); if (child) child->collect_ercs(ptrs); }

// ─── Fabricación y Evolución ────────────────────────────────────────────────
NodePtr make_var(char v) { 
    if (v == 'x') return std::make_unique<TerminalNode>(NodeType::VAR_X);
    if (v == 'y') return std::make_unique<TerminalNode>(NodeType::VAR_Y);
    if (v == 't') return std::make_unique<TerminalNode>(NodeType::VAR_T);
    return std::make_unique<TerminalNode>(NodeType::VAR_Z);
}
NodePtr make_var_n() { return std::make_unique<TerminalNode>(NodeType::VAR_N); }
NodePtr make_erc(Complex v) { return std::make_unique<TerminalNode>(NodeType::ERC, v); }
NodePtr make_const_i() { return std::make_unique<TerminalNode>(NodeType::CONST_I, Complex(0,1)); }
NodePtr make_const_pi() { return std::make_unique<TerminalNode>(NodeType::CONST_PI); }
NodePtr make_const_e() { return std::make_unique<TerminalNode>(NodeType::CONST_E); }
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r) { return std::make_unique<BinaryNode>(op, std::move(l), std::move(r)); }
NodePtr make_unary(NodeType op, NodePtr c) { return std::make_unique<UnaryNode>(op, std::move(c)); }

static NodePtr skeleton_rational(int depth, std::mt19937& gen, const PDEProblem& prob) {
    auto num = random_tree(depth-1, gen, prob); auto den = random_tree(depth-1, gen, prob);
    return make_binary(NodeType::DIV, std::move(num), std::move(den));
}
static NodePtr skeleton_spectral(int depth, std::mt19937& gen, const PDEProblem& prob) {
    auto arg = (prob.dim == 1 || std::uniform_real_distribution<double>(0,1)(gen) < 0.5) ? make_var('x') : make_var('y');
    return std::make_unique<SeriesNode>(5, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), std::move(arg))));
}
static NodePtr skeleton_frobenius(int depth, std::mt19937& gen, const PDEProblem& prob) {
    auto var = (prob.dim == 1 || std::uniform_real_distribution<double>(0,1)(gen) < 0.5) ? make_var('x') : make_var('y');
    auto pwr = make_binary(NodeType::POW, var->clone(), make_erc(0.5));
    auto series = std::make_unique<SeriesNode>(5, make_binary(NodeType::POW, std::move(var), make_var_n()));
    return make_binary(NodeType::MUL, std::move(pwr), std::move(series));
}

NodePtr random_tree(int depth, std::mt19937& gen, const PDEProblem& prob, bool force_t) {
    std::uniform_real_distribution<double> ud(0, 1);
    if (depth <= 0 || ud(gen) < 0.3) {
        // Balance: 60% variables, 40% constants/ERCs
        double p_leaf = ud(gen);
        if (p_leaf < 0.6) {
            int v = std::uniform_int_distribution<int>(0, 2)(gen);
            if (v == 0) return make_var('x');
            if (v == 1) return (prob.dim >= 2) ? make_var('y') : make_var('x');
            return (prob.is_unsteady) ? make_var('t') : make_var('x');
        } else {
            int c = std::uniform_int_distribution<int>(0, 8)(gen);
            if (c == 0) return make_const_pi(); if (c == 1) return make_const_e();
            if (c == 2) return make_const_i();
            if (c == 3) return std::make_unique<TerminalNode>(NodeType::CONST_G);
            if (c == 4) return std::make_unique<TerminalNode>(NodeType::CONST_C);
            if (c == 5) return std::make_unique<TerminalNode>(NodeType::CONST_HBAR);
            if (c == 6) return std::make_unique<TerminalNode>(NodeType::CONST_KB);
            if (c == 7) return std::make_unique<TerminalNode>(NodeType::CONST_EPS0);
            return make_erc(std::uniform_real_distribution<double>(-2.0, 2.0)(gen));
        }
    }
    double p = ud(gen);
    if (p < 0.4) {
        NodeType ops[] = {NodeType::ADD, NodeType::SUB, NodeType::MUL, NodeType::DIV};
        return make_binary(ops[std::uniform_int_distribution<int>(0,3)(gen)], random_tree(depth-1, gen, prob), random_tree(depth-1, gen, prob));
    } else if (p < 0.8) {
        NodeType u_ops[] = {NodeType::SIN, NodeType::COS, NodeType::EXP, NodeType::TANH, NodeType::SQR,
                            NodeType::SINH, NodeType::COSH, NodeType::LOG, NodeType::GAUSSIAN};
        return make_unary(u_ops[std::uniform_int_distribution<int>(0,8)(gen)], random_tree(depth-1, gen, prob));
    } else {
        NodeType p_ops[] = {NodeType::LEGENDRE, NodeType::HERMITE, NodeType::CHEBYSHEV, NodeType::LAGUERRE};
        return make_binary(p_ops[std::uniform_int_distribution<int>(0,3)(gen)], random_tree(depth-1, gen, prob), make_erc(std::uniform_int_distribution<int>(0,3)(gen)));
    }
}

// Mismo criterio que random_tree(): sustituir 'y'/'t' por 'x' cuando el
// problema no tiene esa dimensión/es estacionario, en vez de generarlas igual.
// Antes estos casos metían una 'y' (o 't') fantasma en problemas 1D/estacionarios
// — inofensiva en valor (siempre evalúa en 0), pero ensucia la fórmula exportada
// y desperdicia tamaño de árbol (BIC/parsimonia) sin aportar nada.
static NodePtr special_var2(const PDEProblem& prob) {
    return (prob.dim >= 2) ? make_var('y') : make_var('x');
}
static NodePtr special_var_t(const PDEProblem& prob) {
    return prob.is_unsteady ? make_var('t') : make_var('x');
}

NodePtr random_tree_special(int depth, std::mt19937& gen, const PDEProblem& prob, const PDEPriors& priors) {
    std::vector<int> valid_choices;
    for (int i = 0; i <= 9; ++i) valid_choices.push_back(i);
    if (priors.pole_at_origin) { for (int i = 0; i < 5; ++i) valid_choices.push_back(10); }
    if (priors.autonomous_x || priors.autonomous_y) { for (int i = 0; i < 5; ++i) valid_choices.push_back(11); }
    if (priors.scale_invariant) { for (int i = 0; i < 5; ++i) valid_choices.push_back(12); }
    int choice = valid_choices[std::uniform_int_distribution<int>(0, valid_choices.size() - 1)(gen)];
    last_special_choice = choice;
    switch (choice) {
        case 0: return make_binary(NodeType::DIV, make_erc(1.0), make_binary(NodeType::ADD, make_erc(1.0), make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, special_var2(prob)))));
        case 1: return make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, special_var2(prob)))));
        case 2: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), special_var2(prob))));
        case 3: return make_binary(NodeType::MUL, make_unary(NodeType::COS, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_unary(NodeType::COS, make_binary(NodeType::MUL, make_const_pi(), special_var2(prob))));
        case 4: return make_binary(NodeType::MUL, make_unary(NodeType::TANH, make_var('x')), make_unary(NodeType::TANH, special_var2(prob)));
        case 5: return make_binary(NodeType::MUL, make_var('x'), special_var2(prob));
        case 6: return make_binary(NodeType::ADD, make_binary(NodeType::MUL, make_erc(1.0), make_var('x')), make_binary(NodeType::MUL, make_erc(1.0), special_var2(prob)));
        case 7: return make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, special_var2(prob)));
        case 8: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), special_var_t(prob))));
        case 9: return make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_unary(NodeType::SQR, make_var('x')))));
        case 10: return skeleton_rational(depth, gen, prob);
        case 11: return skeleton_spectral(depth, gen, prob);
        case 12: return skeleton_frobenius(depth, gen, prob);
        // Inalcanzable en la práctica (valid_choices sólo puebla 0-12), pero si algún
        // día deja de serlo, degradamos a un árbol genérico en vez de filtrar la
        // solución exacta del PDE (ver auditoría: get_exact_solution_tree eliminada).
        default: return random_tree(depth, gen, prob);
    }
}

static bool is_v(const Node* n, double target) {
    if (!n || n->contains_variables()) return false;
    return std::abs(n->eval_t(0,0,0).real() - target) < 1e-6;
}

void TerminalNode::print_formal(std::ostream& os, int parent_prec) const { print_latex(os); }
void UnaryNode::print_formal(std::ostream& os, int parent_prec) const {
    if (type == NodeType::SQR) { bool need_paren = (parent_prec > 2); if (need_paren) os << "("; child->print_formal(os, 3); os << "^2"; if (need_paren) os << ")"; }
    else { print_latex(os); }
}
void BinaryNode::print_formal(std::ostream& os, int parent_prec) const {
    if (type == NodeType::ADD) { if (is_v(left.get(), 0.0)) { right->print_formal(os, parent_prec); return; } if (is_v(right.get(), 0.0)) { left->print_formal(os, parent_prec); return; } }
    if (type == NodeType::SUB) { if (is_v(right.get(), 0.0)) { left->print_formal(os, parent_prec); return; } }
    if (type == NodeType::MUL) {
        if (is_v(left.get(), 1.0)) { right->print_formal(os, parent_prec); return; } if (is_v(right.get(), 1.0)) { left->print_formal(os, parent_prec); return; }
        if (is_v(left.get(), -1.0)) { os << "-"; right->print_formal(os, 2); return; } if (is_v(right.get(), -1.0)) { os << "-"; left->print_formal(os, 2); return; }
    }
    int own_prec = (type == NodeType::MUL || type == NodeType::DIV) ? 1 : ((type == NodeType::POW) ? 2 : ((type >= NodeType::LEGENDRE && type <= NodeType::LAGUERRE) ? 3 : 0));
    bool need_paren = (own_prec < parent_prec); if (need_paren) os << "(";
    if (type == NodeType::DIV) { os << "\\frac{"; left->print_formal(os, 0); os << "}{"; right->print_formal(os, 0); os << "}"; }
    else if (type == NodeType::POW) { left->print_formal(os, 2); os << "^{"; right->print_formal(os, 0); os << "}"; }
    else if (type >= NodeType::LEGENDRE && type <= NodeType::LAGUERRE) {
        os << (type == NodeType::LEGENDRE ? "P_" : (type == NodeType::HERMITE ? "H_" : (type == NodeType::CHEBYSHEV ? "T_" : "L_")));
        right->print_formal(os, 3); os << "("; left->print_formal(os, 0); os << ")";
    } else {
        left->print_formal(os, own_prec);
        if (type == NodeType::ADD) os << " + "; else if (type == NodeType::SUB) os << " - "; else if (type == NodeType::MUL) os << " "; 
        right->print_formal(os, own_prec);
    }
    if (need_paren) os << ")";
}
void SeriesNode::print_formal(std::ostream& os, int parent_prec) const { print_latex(os); }

void TerminalNode::round_constants(double epsilon) {
    if (type == NodeType::ERC) { double r = erc_val.real(); double nearest = std::round(r); if (std::abs(r - nearest) < epsilon) erc_val = Complex(nearest, erc_val.imag()); }
}
void UnaryNode::round_constants(double epsilon) { if (child) child->round_constants(epsilon); }
void BinaryNode::round_constants(double epsilon) { 
    if (left) left->round_constants(epsilon); 
    if (right) {
        if (is_polynomial(type)) {
            // Snapping agresivo para el grado del polinomio (siempre entero)
            double r = right->eval_t(0,0,0).real();
            double nearest = std::round(r);
            // Reemplazamos el nodo derecho por un ERC entero exacto
            right = make_erc(Complex(nearest, 0.0));
        } else {
            right->round_constants(epsilon); 
        }
    }
}
void SeriesNode::round_constants(double epsilon) { for (auto& c : coeffs) { double r = c.real(); double nearest = std::round(r); if (std::abs(r - nearest) < epsilon) c = Complex(nearest, c.imag()); } if (child) child->round_constants(epsilon); }

bool UnaryNode::has_nested_trig() const { if (!child) return false; if (is_trig(type) && child->contains_trig()) return true; return child->has_nested_trig(); }
bool UnaryNode::contains_trig() const { if (is_trig(type)) return true; return child && child->contains_trig(); }
bool BinaryNode::has_nested_trig() const { return (left && left->has_nested_trig()) || (right && right->has_nested_trig()); }
bool BinaryNode::contains_trig() const { if (is_trig(type)) return true; return (left && left->contains_trig()) || (right && right->contains_trig()); }
bool SeriesNode::has_nested_trig() const { return child && child->has_nested_trig(); }
bool SeriesNode::contains_trig() const { return child && child->contains_trig(); }

static NodePtr get_node_at(const NodePtr& root, int& idx) {
    if (!root) return nullptr;
    if (idx == 0) { idx = -1; return root->clone(); }
    idx--;
    if (auto* un = dynamic_cast<UnaryNode*>(root.get())) {
        if (un->child) { auto r = get_node_at(un->child, idx); if (idx == -1) return r; }
    } else if (auto* bn = dynamic_cast<BinaryNode*>(root.get())) {
        if (bn->left) { auto r = get_node_at(bn->left, idx); if (idx == -1) return r; }
        if (bn->right) { auto r = get_node_at(bn->right, idx); if (idx == -1) return r; }
    } else if (auto* sn = dynamic_cast<SeriesNode*>(root.get())) {
        if (sn->child) { auto r = get_node_at(sn->child, idx); if (idx == -1) return r; }
    }
    return nullptr;
}

void replace_node_at(NodePtr& cur, int& idx, NodePtr& rep) {
    if (!cur || !rep) return;
    if (idx == 0) { cur = std::move(rep); idx = -1; return; }
    idx--;
    if (auto* un = dynamic_cast<UnaryNode*>(cur.get())) {
        if (un->child && idx >= 0) replace_node_at(un->child, idx, rep);
    } else if (auto* bn = dynamic_cast<BinaryNode*>(cur.get())) {
        if (bn->left && idx >= 0) replace_node_at(bn->left, idx, rep);
        if (bn->right && idx >= 0) replace_node_at(bn->right, idx, rep);
    } else if (auto* sn = dynamic_cast<SeriesNode*>(cur.get())) {
        if (sn->child && idx >= 0) replace_node_at(sn->child, idx, rep);
    }
}

std::vector<NodePtr> extract_additive_terms(const NodePtr& root) {
    std::vector<NodePtr> terms;
    if (!root) return terms;
    if (root->get_type() == NodeType::ADD) {
        auto* bn = dynamic_cast<const BinaryNode*>(root.get());
        auto left_terms = extract_additive_terms(bn->left);
        auto right_terms = extract_additive_terms(bn->right);
        for (auto& t : left_terms) terms.push_back(std::move(t));
        for (auto& t : right_terms) terms.push_back(std::move(t));
    } else if (root->get_type() == NodeType::SUB) {
        auto* bn = dynamic_cast<const BinaryNode*>(root.get());
        auto left_terms = extract_additive_terms(bn->left);
        auto right_terms = extract_additive_terms(bn->right);
        for (auto& t : left_terms) terms.push_back(std::move(t));
        for (auto& t : right_terms) {
            terms.push_back(make_binary(NodeType::MUL, make_erc(-1.0), std::move(t)));
        }
    } else {
        terms.push_back(root->clone());
    }
    return terms;
}

NodePtr build_additive_tree(const std::vector<NodePtr>& terms) {
    if (terms.empty()) return make_erc(0.0);
    NodePtr root = terms[0]->clone();
    for (size_t i = 1; i < terms.size(); ++i) {
        root = make_binary(NodeType::ADD, std::move(root), terms[i]->clone());
    }
    return root;
}

// ─── Descomposición multiplicativa (análoga a la aditiva, un nivel abajo) ───
// Muchas soluciones reales son razones, no sumas (ej. Troesch: sinh(3x)/sinh(3)).
// Sin esto, la mutación/cruzamiento sólo sabe recombinar sumandos — un término
// que es un cociente se trata como un bloque indivisible (sólo mutación puntual
// o regeneración total), sin forma de recombinar sus factores por separado.
//
// "Invertir" un factor de un DIV se representa como DIV(1, factor), NO como
// POW(factor, -1): apply_binary devuelve NaN si la base de POW es negativa
// (para evitar ramas complejas), y cualquier subárbol puede evaluar negativo —
// con POW esto reventaría constantemente. DIV no tiene esa restricción (sólo
// falla con denominador ~0).
std::vector<NodePtr> extract_multiplicative_factors(const NodePtr& root) {
    std::vector<NodePtr> factors;
    if (!root) return factors;
    if (root->get_type() == NodeType::MUL) {
        auto* bn = dynamic_cast<const BinaryNode*>(root.get());
        auto left_factors = extract_multiplicative_factors(bn->left);
        auto right_factors = extract_multiplicative_factors(bn->right);
        for (auto& f : left_factors) factors.push_back(std::move(f));
        for (auto& f : right_factors) factors.push_back(std::move(f));
    } else if (root->get_type() == NodeType::DIV) {
        auto* bn = dynamic_cast<const BinaryNode*>(root.get());
        auto left_factors = extract_multiplicative_factors(bn->left);
        auto right_factors = extract_multiplicative_factors(bn->right);
        for (auto& f : left_factors) factors.push_back(std::move(f));
        for (auto& f : right_factors) {
            factors.push_back(make_binary(NodeType::DIV, make_erc(1.0), std::move(f)));
        }
    } else {
        factors.push_back(root->clone());
    }
    return factors;
}

NodePtr build_multiplicative_tree(const std::vector<NodePtr>& factors) {
    if (factors.empty()) return make_erc(1.0);
    NodePtr root = factors[0]->clone();
    for (size_t i = 1; i < factors.size(); ++i) {
        root = make_binary(NodeType::MUL, std::move(root), factors[i]->clone());
    }
    return root;
}

// ─── Familias de vecindad funcional (para mutación suave) ───────────────────
// Grupos de NodeType matemáticamente "cercanos": intercambiar dentro de un
// mismo grupo es una adaptación suave (ej. sin<->cos<->exp vía la fórmula de
// Euler); saltar entre grupos (ej. sin->x) puede destruir estructura físicamente
// correcta y sólo debe ocurrir por otros operadores (crecimiento/reemplazo raro).
static const std::vector<NodeType>& family_of(NodeType t) {
    static const std::vector<NodeType> TRANSCENDENTAL = {
        NodeType::SIN, NodeType::COS, NodeType::SINH, NodeType::COSH,
        NodeType::TANH, NodeType::EXP, NodeType::GAUSSIAN, NodeType::LOG
    };
    static const std::vector<NodeType> POWER_LIKE = {
        NodeType::POW, NodeType::LEGENDRE, NodeType::HERMITE,
        NodeType::CHEBYSHEV, NodeType::LAGUERRE
    };
    static const std::vector<NodeType> ADD_SUB = {NodeType::ADD, NodeType::SUB};
    static const std::vector<NodeType> MUL_DIV = {NodeType::MUL, NodeType::DIV};
    static const std::vector<NodeType> VARS = {NodeType::VAR_X, NodeType::VAR_Y, NodeType::VAR_T};
    static const std::vector<NodeType> NAMED_CONSTS = {
        NodeType::CONST_PI, NodeType::CONST_E, NodeType::CONST_G, NodeType::CONST_C,
        NodeType::CONST_HBAR, NodeType::CONST_KB, NodeType::CONST_EPS0
    };
    static const std::vector<NodeType> EMPTY = {};
    auto has = [](const std::vector<NodeType>& v, NodeType x) { return std::find(v.begin(), v.end(), x) != v.end(); };
    if (has(TRANSCENDENTAL, t)) return TRANSCENDENTAL;
    if (has(POWER_LIKE, t))      return POWER_LIKE;
    if (has(ADD_SUB, t))         return ADD_SUB;
    if (has(MUL_DIV, t))         return MUL_DIV;
    if (has(VARS, t))            return VARS;
    if (has(NAMED_CONSTS, t))    return NAMED_CONSTS;
    // SQR (deliberadamente aislado: es el borde "algebraico" que no debe
    // fusionarse con el mundo transcendental — el salto sin->lineal que destruye
    // estructura), ERC, VAR_N, VAR_Z, CONST_I, SERIES: sin swap.
    return EMPTY;
}

// Recorre el árbol por índice (mismo patrón que get_node_at/replace_node_at) y,
// al llegar al nodo objetivo, cambia su NodeType por otro miembro de su misma
// familia funcional, preservando exactamente los mismos hijos. `done` indica si
// realmente hubo cambio (el nodo objetivo puede no tener familia, ej. un ERC).
static void mutate_node_type_at(NodePtr& cur, int& idx, std::mt19937& gen, const PDEProblem& prob, bool& done) {
    if (!cur || done) return;
    if (idx == 0) {
        idx = -1;
        NodeType t = cur->get_type();
        std::vector<NodeType> fam;
        if (t == NodeType::VAR_X || t == NodeType::VAR_Y || t == NodeType::VAR_T) {
            fam.push_back(NodeType::VAR_X);
            if (prob.dim >= 2) fam.push_back(NodeType::VAR_Y);
            if (prob.is_unsteady) fam.push_back(NodeType::VAR_T);
        } else {
            fam = family_of(t);
        }
        if (fam.size() > 1) {
            NodeType new_type = t;
            while (new_type == t) new_type = fam[std::uniform_int_distribution<int>(0, (int)fam.size() - 1)(gen)];
            if (auto* tn = dynamic_cast<TerminalNode*>(cur.get())) tn->type = new_type;
            else if (auto* un = dynamic_cast<UnaryNode*>(cur.get())) un->type = new_type;
            else if (auto* bn = dynamic_cast<BinaryNode*>(cur.get())) bn->type = new_type;
            done = true;
        }
        return;
    }
    idx--;
    if (auto* un = dynamic_cast<UnaryNode*>(cur.get())) {
        if (un->child && idx >= 0) mutate_node_type_at(un->child, idx, gen, prob, done);
    } else if (auto* bn = dynamic_cast<BinaryNode*>(cur.get())) {
        if (bn->left && idx >= 0) mutate_node_type_at(bn->left, idx, gen, prob, done);
        if (bn->right && idx >= 0) mutate_node_type_at(bn->right, idx, gen, prob, done);
    } else if (auto* sn = dynamic_cast<SeriesNode*>(cur.get())) {
        if (sn->child && idx >= 0) mutate_node_type_at(sn->child, idx, gen, prob, done);
    }
}

NodePtr tree_mutate_point(const NodePtr& tree, std::mt19937& gen, const PDEProblem& prob) {
    if (!tree) return nullptr;
    NodePtr result = tree->clone();
    int n = result->count_nodes();
    if (n <= 0) return result;
    // Varios intentos: no todos los nodos tienen familia (ERC, SQR, SERIES, ...),
    // así que si el índice elegido cae en uno sin familia, se reintenta.
    for (int attempt = 0; attempt < 5; ++attempt) {
        int idx = std::uniform_int_distribution<int>(0, n - 1)(gen);
        bool done = false;
        mutate_node_type_at(result, idx, gen, prob, done);
        if (done) break;
    }
    return result;
}

// Mutación a nivel de factor multiplicativo: borra/agrega/reemplaza un factor
// de un producto/razón, mismo espíritu que la mutación de términos aditivos
// pero un nivel abajo. Si el término no tiene más de un factor (no es un
// producto/razón, ej. un solo sin(x)), no hay nada que recombinar y cae de
// vuelta a la mutación puntual suave de siempre.
// Factor de rotación de fase: exp(i*theta), theta evolucionable (ERC). El motor
// ya evalúa todo en Complex de punta a punta, así que multiplicar por esto es
// una rotación genuina en el plano complejo — la operación que conecta
// directamente con Euler (sin/cos/senh/cosh vía e^{i*theta}), sin reconstruir
// nada del resto del árbol. Es holomorfa en theta (EXP y MUL lo son), así que
// el gradiente complex-step de gradient_descent_constants la afina gratis, sin
// ningún caso especial.
static NodePtr make_phase_rotation_factor(std::mt19937& gen) {
    double theta = std::uniform_real_distribution<double>(-PI_VAL, PI_VAL)(gen);
    return make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_const_i(), make_erc(theta)));
}

NodePtr tree_mutate_factor(const NodePtr& term, std::mt19937& gen, const PDEProblem& p) {
    if (!term) return nullptr;
    auto factors = extract_multiplicative_factors(term);
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    if (factors.size() <= 1) {
        // Agregar una rotación de fase es válido incluso si el término no es
        // (todavía) un producto — es la forma más barata de empezar a explorar
        // la dimensión de fase sin tocar nada de lo que ya había.
        if (ud(gen) < 0.2) {
            factors.push_back(make_phase_rotation_factor(gen));
            return build_multiplicative_tree(factors)->simplify();
        }
        return tree_mutate_point(term, gen, p);
    }

    int type_mut = std::uniform_int_distribution<int>(0, 3)(gen);
    if (type_mut == 0 && factors.size() > 1) {
        int idx = std::uniform_int_distribution<int>(0, factors.size() - 1)(gen);
        factors.erase(factors.begin() + idx);
    } else if (type_mut == 1) {
        factors.push_back(random_tree(2, gen, p));
    } else if (type_mut == 3) {
        factors.push_back(make_phase_rotation_factor(gen));
    } else {
        int idx = std::uniform_int_distribution<int>(0, factors.size() - 1)(gen);
        // Igual que en tree_mutate: casi siempre mutación puntual suave dentro
        // del factor elegido; regeneración total del factor sólo raramente.
        if (ud(gen) < 0.15) {
            factors[idx] = random_tree(2, gen, p);
        } else {
            factors[idx] = tree_mutate_point(factors[idx], gen, p);
        }
    }
    return build_multiplicative_tree(factors)->simplify();
}

NodePtr tree_mutate(const NodePtr& t, std::mt19937& gen, const PDEProblem& p, double aggressiveness) {
    if (!t) return make_binary(NodeType::MUL, make_erc(1.0), random_tree(2, gen, p));
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    // La mayoría de las mutaciones son un swap de NodeType dentro de la misma
    // familia funcional (adaptación suave), no una regeneración total del árbol.
    if (ud(gen) < 0.55) {
        return tree_mutate_point(t, gen, p);
    }

    auto terms = extract_additive_terms(t);
    if (!terms.empty()) {
        int type_mut = std::uniform_int_distribution<int>(0, 2)(gen);
        if (type_mut == 0 && terms.size() > 1) {
            int idx = std::uniform_int_distribution<int>(0, terms.size() - 1)(gen);
            terms.erase(terms.begin() + idx);
        } else if (type_mut == 1) {
            terms.push_back(make_binary(NodeType::MUL, make_erc(1.0), random_tree(2, gen, p)));
        } else {
            int idx = std::uniform_int_distribution<int>(0, terms.size() - 1)(gen);
            // Reemplazo de término: la mayoría de las veces una mutación puntual
            // suave dentro del propio término; con probabilidad escalada por
            // `aggressiveness` (que baja con las generaciones pero nunca a 0 — ver
            // MIN_AGGRESSIVENESS en make_offspring) se regenera el término entero
            // desde cero. Es el único movimiento capaz de sacar a la población de
            // una estructura mediocre encontrada temprano; con el multiplicador
            // viejo (0.15) la probabilidad efectiva real era ~2%, insuficiente en
            // la práctica (val_mse quedaba fijo desde gen 0 en varios PDEs).
            double full_replace_prob = 0.6 * std::clamp(aggressiveness, 0.0, 1.0);
            if (ud(gen) < full_replace_prob) {
                terms[idx] = make_binary(NodeType::MUL, make_erc(1.0), random_tree(2, gen, p));
            } else {
                // tree_mutate_factor recombina a nivel de factor si el término es un
                // producto/razón (ej. sinh(3x)/sinh(3)) y cae a mutación puntual si
                // no hay nada que decomponer — generaliza el caso anterior sin
                // perder ningún comportamiento existente.
                terms[idx] = tree_mutate_factor(terms[idx], gen, p);
            }
        }
        return build_additive_tree(terms)->simplify();
    }
    return make_binary(NodeType::MUL, make_erc(1.0), random_tree(2, gen, p));
}

std::pair<NodePtr, NodePtr> tree_crossover(const NodePtr& p1, const NodePtr& p2, std::mt19937& gen) {
    if (!p1 || !p2) return {p1 ? p1->clone() : nullptr, p2 ? p2->clone() : nullptr};
    auto terms1 = extract_additive_terms(p1);
    auto terms2 = extract_additive_terms(p2);
    if (!terms1.empty() && !terms2.empty()) {
        int idx1 = std::uniform_int_distribution<int>(0, terms1.size() - 1)(gen);
        int idx2 = std::uniform_int_distribution<int>(0, terms2.size() - 1)(gen);

        // Si ambos términos elegidos son productos/razones (ej. sinh(3x)/sinh(3)),
        // con alta probabilidad recombinar a nivel de FACTOR en vez de intercambiar
        // el término entero — mismo espíritu que la mutación: no perder toda la
        // estructura de un término que funcionaba por cambiarlo entero por uno del
        // otro padre que puede no encajar en absoluto.
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        auto factors1 = extract_multiplicative_factors(terms1[idx1]);
        auto factors2 = extract_multiplicative_factors(terms2[idx2]);
        if (factors1.size() > 1 && factors2.size() > 1 && ud(gen) < 0.5) {
            int fidx1 = std::uniform_int_distribution<int>(0, factors1.size() - 1)(gen);
            int fidx2 = std::uniform_int_distribution<int>(0, factors2.size() - 1)(gen);
            std::swap(factors1[fidx1], factors2[fidx2]);
            terms1[idx1] = build_multiplicative_tree(factors1)->simplify();
            terms2[idx2] = build_multiplicative_tree(factors2)->simplify();
        } else {
            NodePtr temp = std::move(terms1[idx1]);
            terms1[idx1] = std::move(terms2[idx2]);
            terms2[idx2] = std::move(temp);
        }
        return {build_additive_tree(terms1)->simplify(), build_additive_tree(terms2)->simplify()};
    }
    return {p1->clone(), p2->clone()};
}

bool UnaryNode::has_nested_polynomial() const {
    if (!child) return false;
    // Unary nodes are not polynomials, so we just recurse
    return child->has_nested_polynomial();
}

bool UnaryNode::contains_polynomial() const {
    return child && child->contains_polynomial();
}

bool BinaryNode::has_nested_polynomial() const {
    if (!left || !right) return false;
    if (is_polynomial(type)) {
        // If this is a polynomial, neither child can contain a polynomial
        if (left->contains_polynomial() || right->contains_polynomial()) return true;
    }
    return left->has_nested_polynomial() || right->has_nested_polynomial();
}

bool BinaryNode::contains_polynomial() const {
    if (is_polynomial(type)) return true;
    return (left && left->contains_polynomial()) || (right && right->contains_polynomial());
}

bool SeriesNode::has_nested_polynomial() const {
    return child && (child->has_nested_polynomial());
}

bool SeriesNode::contains_polynomial() const {
    return child && child->contains_polynomial();
}

bool UnaryNode::has_nested_exp() const {
    if (!child) return false;
    if (type == NodeType::EXP && child->contains_exp()) return true;
    return child->has_nested_exp();
}

bool UnaryNode::contains_exp() const {
    if (type == NodeType::EXP) return true;
    return child && child->contains_exp();
}

bool BinaryNode::has_nested_exp() const {
    if (!left || !right) return false;
    // Binary operators themselves don't prevent nesting, just recurse
    return left->has_nested_exp() || right->has_nested_exp();
}

bool BinaryNode::contains_exp() const {
    return (left && left->contains_exp()) || (right && right->contains_exp());
}

bool SeriesNode::has_nested_exp() const {
    return child && child->has_nested_exp();
}

bool SeriesNode::contains_exp() const {
    return child && child->contains_exp();
}

bool UnaryNode::has_invalid_polynomial_degree() const {
    return child && child->has_invalid_polynomial_degree();
}

bool BinaryNode::has_invalid_polynomial_degree() const {
    if (is_polynomial(type)) {
        if (right && right->contains_variables()) return true;
    }
    return (left && left->has_invalid_polynomial_degree()) || (right && right->has_invalid_polynomial_degree());
}

bool SeriesNode::has_invalid_polynomial_degree() const {
    return child && child->has_invalid_polynomial_degree();
}

bool UnaryNode::is_strictly_affine() const {
    return false;
}

bool BinaryNode::is_strictly_affine() const {
    if (type == NodeType::ADD || type == NodeType::SUB) {
        return left && left->is_strictly_affine() && right && right->is_strictly_affine();
    }
    if (type == NodeType::MUL) {
        bool l_const = left && !left->contains_variables();
        bool r_const = right && !right->contains_variables();
        if (l_const && r_const) return true;
        if (l_const && right) return right->is_strictly_affine();
        if (r_const && left) return left->is_strictly_affine();
        return false;
    }
    if (type == NodeType::DIV) {
        if (right && !right->contains_variables() && left) return left->is_strictly_affine();
        return false;
    }
    return false;
}

bool SeriesNode::is_strictly_affine() const {
    return false;
}

bool UnaryNode::has_non_affine_trig_arg() const {
    if ((type == NodeType::SIN || type == NodeType::COS) && child) {
        if (!child->is_strictly_affine()) return true;
    }
    return child && child->has_non_affine_trig_arg();
}

bool BinaryNode::has_non_affine_trig_arg() const {
    return (left && left->has_non_affine_trig_arg()) || (right && right->has_non_affine_trig_arg());
}

bool SeriesNode::has_non_affine_trig_arg() const {
    return child && child->has_non_affine_trig_arg();
}
