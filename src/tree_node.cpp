#include "tree_node.hpp"
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <numeric>

extern int last_special_choice;

// ─── Polinomios Ortogonales (Motor Físico) ───────────────────────────────────
static void eval_poly_all(NodeType type, int n, double x, double& v, double& dv, double& dvv, double& dvvv, double& dvvvv);
// Sobrecarga original (value + 1er/2do derivada) — se mantiene intacta para no
// tocar ningun call site existente; delega en la version extendida de abajo.
static void eval_poly_all(NodeType type, int n, double x, double& v, double& dv, double& dvv) {
    double dvvv, dvvvv;
    eval_poly_all(type, n, x, v, dv, dvv, dvvv, dvvvv);
}
// Version extendida (value + 1ro..4to derivada w.r.t. su argumento escalar x),
// necesaria para propagar dxxx/dxxy/.../dxxyy a traves de un nodo LEGENDRE/
// HERMITE/CHEBYSHEV/LAGUERRE via apply_composition (ver mas abajo). Se obtiene
// diferenciando la MISMA recurrencia de 3 terminos una vez mas en cada caso —
// el patron general es: si P_n(x) = A(x)*P_{n-1}(x) + B*P_{n-2}(x) con A lineal
// en x (o constante), entonces d^m/dx^m[A(x)*P_{n-1}] = A(x)*P_{n-1}^(m) +
// m*A'(x)*P_{n-1}^(m-1) (Leibniz truncado: A'' = 0 para todas las familias
// aca, A es a lo sumo lineal en x), y B*P_{n-2} se diferencia trivial (B es
// constante en x). Verificado a mano contra la recurrencia existente para
// m=1,2 antes de extender a m=3,4 (coincide termino a termino).
static void eval_poly_all(NodeType type, int n, double x, double& v, double& dv, double& dvv, double& dvvv, double& dvvvv) {
    if (n < 0 || n > 10 || !std::isfinite(x)) { v = dv = dvv = dvvv = dvvvv = NAN; return; }
    if (n == 0) { v = 1.0; dv = dvv = dvvv = dvvvv = 0.0; return; }
    if (n == 1) {
        if (type == NodeType::HERMITE) { v = 2.0*x; dv = 2.0; }
        else if (type == NodeType::LAGUERRE) { v = 1.0 - x; dv = -1.0; }
        else { v = x; dv = 1.0; }
        dvv = dvvv = dvvvv = 0.0;
        return;
    }

    double p0 = 1.0, p1 = (type == NodeType::HERMITE) ? 2.0*x : ((type == NodeType::LAGUERRE) ? 1.0-x : x);
    double dp0 = 0.0, dp1 = (type == NodeType::HERMITE) ? 2.0 : ((type == NodeType::LAGUERRE) ? -1.0 : 1.0);
    double ddp0 = 0.0, ddp1 = 0.0;
    double dddp0 = 0.0, dddp1 = 0.0;
    double ddddp0 = 0.0, ddddp1 = 0.0;

    for (int k = 1; k < n; ++k) {
        double cur_v, cur_dv, cur_dvv, cur_dvvv, cur_dvvvv;
        if (type == NodeType::LEGENDRE) {
            cur_v    = ((2.0*k + 1.0)*x*p1 - k*p0) / (k + 1.0);
            cur_dv   = ((2.0*k + 1.0)*(p1 + x*dp1) - k*dp0) / (k + 1.0);
            cur_dvv  = ((2.0*k + 1.0)*(2.0*dp1 + x*ddp1) - k*ddp0) / (k + 1.0);
            cur_dvvv = ((2.0*k + 1.0)*(3.0*ddp1 + x*dddp1) - k*dddp0) / (k + 1.0);
            cur_dvvvv= ((2.0*k + 1.0)*(4.0*dddp1 + x*ddddp1) - k*ddddp0) / (k + 1.0);
        } else if (type == NodeType::CHEBYSHEV) {
            cur_v    = 2.0*x*p1 - p0;
            cur_dv   = 2.0*(p1 + x*dp1) - dp0;
            cur_dvv  = 2.0*(2.0*dp1 + x*ddp1) - ddp0;
            cur_dvvv = 2.0*(3.0*ddp1 + x*dddp1) - dddp0;
            cur_dvvvv= 2.0*(4.0*dddp1 + x*ddddp1) - ddddp0;
        } else if (type == NodeType::HERMITE) {
            cur_v    = 2.0*x*p1 - 2.0*k*p0;
            cur_dv   = 2.0*(p1 + x*dp1) - 2.0*k*dp0;
            cur_dvv  = 2.0*(2.0*dp1 + x*ddp1) - 2.0*k*ddp0;
            cur_dvvv = 2.0*(3.0*ddp1 + x*dddp1) - 2.0*k*dddp0;
            cur_dvvvv= 2.0*(4.0*dddp1 + x*ddddp1) - 2.0*k*ddddp0;
        } else { // LAGUERRE
            cur_v    = ((2.0*k + 1.0 - x)*p1 - k*p0) / (k + 1.0);
            cur_dv   = ((2.0*k + 1.0 - x)*dp1 - p1 - k*dp0) / (k + 1.0);
            cur_dvv  = ((2.0*k + 1.0 - x)*ddp1 - 2.0*dp1 - k*ddp0) / (k + 1.0);
            cur_dvvv = ((2.0*k + 1.0 - x)*dddp1 - 3.0*ddp1 - k*dddp0) / (k + 1.0);
            cur_dvvvv= ((2.0*k + 1.0 - x)*ddddp1 - 4.0*dddp1 - k*ddddp0) / (k + 1.0);
        }
        p0 = p1; p1 = cur_v; dp0 = dp1; dp1 = cur_dv; ddp0 = ddp1; ddp1 = cur_dvv;
        dddp0 = dddp1; dddp1 = cur_dvvv; ddddp0 = ddddp1; ddddp1 = cur_dvvvv;
    }
    v = p1; dv = dp1; dvv = ddp1; dvvv = dddp1; dvvvv = ddddp1;
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

// ─── Composicion generica h(x,y,t) = f(g(x,y,t)) via la regla de la cadena
// multivariante (Faa di Bruno) hasta 4to orden total, dados f0..f4 = f, f',
// f'', f''', f'''' evaluados en g.v. Se deriva UNA vez (ver comentario junto a
// AD en common.hpp) y se reusa para TODAS las funciones unarias de un solo
// argumento (SIN, COS, EXP, LOG, TANH, SINH, COSH, SQR, GAUSSIAN, y el
// reciproco 1/u usado por DIV, y log/exp usados por POW) en vez de repetir la
// misma algebra 9+ veces con riesgo de un error de signo/coeficiente distinto
// cada vez. Formulas (derivadas a mano, diferenciando sucesivamente):
//   hxx  = f2*gx^2 + f1*gxx                         (analogo hyy, htt, hxy, hxt, hyt)
//   hxxx = f3*gx^3 + 3*f2*gx*gxx + f1*gxxx           (analogo hyyy)
//   hxxy = f3*gx^2*gy + f2*(2*gx*gxy + gy*gxx) + f1*gxxy   (analogo hxyy con x<->y, hxxt/hyyt con y->t)
//   hxxxx = f4*gx^4 + 6*f3*gx^2*gxx + f2*(3*gxx^2+4*gx*gxxx) + f1*gxxxx  (analogo hyyyy)
//   hxxyy = f4*gx^2*gy^2 + f3*(4*gx*gy*gxy+gx^2*gyy+gy^2*gxx)
//         + f2*(2*gxy^2+2*gx*gxyy+2*gy*gxxy+gxx*gyy) + f1*gxxyy
static AD apply_composition(Complex f0, Complex f1, Complex f2, Complex f3, Complex f4, const AD& g) {
    AD r;
    r.v = f0;
    r.dx = f1*g.dx; r.dy = f1*g.dy; r.dt = f1*g.dt;
    r.dxx = f2*g.dx*g.dx + f1*g.dxx;
    r.dyy = f2*g.dy*g.dy + f1*g.dyy;
    r.dtt = f2*g.dt*g.dt + f1*g.dtt;
    r.dxy = f2*g.dx*g.dy + f1*g.dxy;
    r.dxt = f2*g.dx*g.dt + f1*g.dxt;
    r.dyt = f2*g.dy*g.dt + f1*g.dyt;
    r.dxxx = f3*g.dx*g.dx*g.dx + 3.0*f2*g.dx*g.dxx + f1*g.dxxx;
    r.dyyy = f3*g.dy*g.dy*g.dy + 3.0*f2*g.dy*g.dyy + f1*g.dyyy;
    r.dxxy = f3*g.dx*g.dx*g.dy + f2*(2.0*g.dx*g.dxy + g.dy*g.dxx) + f1*g.dxxy;
    r.dxyy = f3*g.dy*g.dy*g.dx + f2*(2.0*g.dy*g.dxy + g.dx*g.dyy) + f1*g.dxyy;
    r.dxxt = f3*g.dx*g.dx*g.dt + f2*(2.0*g.dx*g.dxt + g.dt*g.dxx) + f1*g.dxxt;
    r.dyyt = f3*g.dy*g.dy*g.dt + f2*(2.0*g.dy*g.dyt + g.dt*g.dyy) + f1*g.dyyt;
    r.dxxxx = f4*(g.dx*g.dx*g.dx*g.dx) + 6.0*f3*(g.dx*g.dx*g.dxx) + f2*(3.0*g.dxx*g.dxx + 4.0*g.dx*g.dxxx) + f1*g.dxxxx;
    r.dyyyy = f4*(g.dy*g.dy*g.dy*g.dy) + 6.0*f3*(g.dy*g.dy*g.dyy) + f2*(3.0*g.dyy*g.dyy + 4.0*g.dy*g.dyyy) + f1*g.dyyyy;
    r.dxxyy = f4*(g.dx*g.dx*g.dy*g.dy)
            + f3*(4.0*g.dx*g.dy*g.dxy + g.dx*g.dx*g.dyy + g.dy*g.dy*g.dxx)
            + f2*(2.0*g.dxy*g.dxy + 2.0*g.dx*g.dxyy + 2.0*g.dy*g.dxxy + g.dxx*g.dyy)
            + f1*g.dxxyy;
    return r;
}

// ─── Producto L*R via la regla de Leibniz multivariante hasta 4to orden total
// (derivada a mano y verificada termino a termino contra la expansion directa
// del binomio para cada multi-indice — ver comentario en common.hpp). MUL,
// DIV (=L * 1/R via apply_composition con f(u)=1/u) y POW (=exp(R*log(L)) via
// dos aplicaciones de apply_composition + este producto) se construyen sobre
// esta unica funcion en vez de derivar cada regla por separado.
static AD apply_mul_ad(const AD& L, const AD& R) {
    AD r;
    r.v = L.v*R.v;
    r.dx = L.dx*R.v + L.v*R.dx; r.dy = L.dy*R.v + L.v*R.dy; r.dt = L.dt*R.v + L.v*R.dt;
    r.dxx = L.dxx*R.v + 2.0*L.dx*R.dx + L.v*R.dxx;
    r.dyy = L.dyy*R.v + 2.0*L.dy*R.dy + L.v*R.dyy;
    r.dtt = L.dtt*R.v + 2.0*L.dt*R.dt + L.v*R.dtt;
    r.dxy = L.dxy*R.v + L.dx*R.dy + L.dy*R.dx + L.v*R.dxy;
    r.dxt = L.dxt*R.v + L.dx*R.dt + L.dt*R.dx + L.v*R.dxt;
    r.dyt = L.dyt*R.v + L.dy*R.dt + L.dt*R.dy + L.v*R.dyt;
    r.dxxx = L.dxxx*R.v + 3.0*L.dxx*R.dx + 3.0*L.dx*R.dxx + L.v*R.dxxx;
    r.dyyy = L.dyyy*R.v + 3.0*L.dyy*R.dy + 3.0*L.dy*R.dyy + L.v*R.dyyy;
    r.dxxy = L.dxxy*R.v + L.dxx*R.dy + 2.0*L.dx*R.dxy + 2.0*L.dxy*R.dx + L.dy*R.dxx + L.v*R.dxxy;
    r.dxyy = L.dxyy*R.v + L.dyy*R.dx + 2.0*L.dy*R.dxy + 2.0*L.dxy*R.dy + L.dx*R.dyy + L.v*R.dxyy;
    r.dxxt = L.dxxt*R.v + L.dxx*R.dt + 2.0*L.dx*R.dxt + 2.0*L.dxt*R.dx + L.dt*R.dxx + L.v*R.dxxt;
    r.dyyt = L.dyyt*R.v + L.dyy*R.dt + 2.0*L.dy*R.dyt + 2.0*L.dyt*R.dy + L.dt*R.dyy + L.v*R.dyyt;
    r.dxxxx = L.dxxxx*R.v + 4.0*L.dxxx*R.dx + 6.0*L.dxx*R.dxx + 4.0*L.dx*R.dxxx + L.v*R.dxxxx;
    r.dyyyy = L.dyyyy*R.v + 4.0*L.dyyy*R.dy + 6.0*L.dyy*R.dyy + 4.0*L.dy*R.dyyy + L.v*R.dyyyy;
    r.dxxyy = L.dxxyy*R.v + L.dxx*R.dyy + L.dyy*R.dxx
            + 2.0*L.dxxy*R.dy + 2.0*L.dy*R.dxxy
            + 2.0*L.dxyy*R.dx + 2.0*L.dx*R.dxyy
            + 4.0*L.dxy*R.dxy
            + L.v*R.dxxyy;
    return r;
}

// ─── Version RAPIDA (solo v,dx,dy,dt,dxx,dyy,dtt) — la que usa TODA la suite
// excepto Navier-Stokes/-Unsteady. Identica a la formula original de antes de
// agregar el AD de 4to orden: no calcula dxy/dxt/dyt/3er/4to orden en
// absoluto, para que ninguna otra EDP pague ese costo. Ver apply_unary_ad_ext
// mas abajo para la version extendida (usada solo por Navier-Stokes).
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

// ─── Version EXTENDIDA (agrega dxy/dxt/dyt/3er/4to orden via
// apply_composition) — usada UNICAMENTE por el camino ad_eval_ext_t de
// Navier-Stokes/-Unsteady (ver Node::ad_eval_ext_t y compute_residual). No la
// llama nadie mas, asi que el costo extra de calcular esos 12 campos queda
// aislado a esas dos EDPs en vez de pagarlo toda la suite en cada evaluacion.
AD apply_unary_ad_ext(NodeType type, const AD& C) {
    if (type == NodeType::SIN) {
        Complex s = std::sin(C.v), c = std::cos(C.v);
        return apply_composition(s, c, -s, -c, s, C);
    } else if (type == NodeType::COS) {
        Complex s = std::sin(C.v), c = std::cos(C.v);
        return apply_composition(c, -s, -c, s, c, C);
    } else if (type == NodeType::EXP) {
        if (C.v.real() > 100.0) { AD r; r.v = NAN; return r; }
        Complex ev = std::exp(C.v);
        return apply_composition(ev, ev, ev, ev, ev, C);
    } else if (type == NodeType::SQR) {
        // f(u)=u^2: f=u^2, f'=2u, f''=2, f'''=f''''=0
        return apply_composition(C.v*C.v, 2.0*C.v, Complex(2.0,0.0), Complex(0.0,0.0), Complex(0.0,0.0), C);
    } else if (type == NodeType::GAUSSIAN) {
        // f(u)=exp(-0.5u^2)=G. f'=-u*G, f''=(u^2-1)*G, f'''=u*(3-u^2)*G, f''''=(u^4-6u^2+3)*G
        // (derivadas a mano, diferenciando sucesivamente f'=-uG y verificando cada paso).
        Complex u = C.v, u2 = u*u;
        Complex G = std::exp(-0.5*u2);
        Complex f1 = -u*G, f2 = (u2 - 1.0)*G, f3 = u*(3.0 - u2)*G, f4 = (u2*u2 - 6.0*u2 + 3.0)*G;
        return apply_composition(G, f1, f2, f3, f4, C);
    } else if (type == NodeType::TANH) {
        // t=tanh(u). f'=1-t^2, f''=-2t(1-t^2), f'''=(6t^2-2)(1-t^2), f''''=(16t-24t^3)(1-t^2)
        // (derivadas a mano, diferenciando sucesivamente t'=1-t^2).
        Complex t = std::tanh(C.v);
        Complex one_m_t2 = 1.0 - t*t;
        Complex f1 = one_m_t2;
        Complex f2 = -2.0*t*one_m_t2;
        Complex f3 = (6.0*t*t - 2.0)*one_m_t2;
        Complex f4 = (16.0*t - 24.0*t*t*t)*one_m_t2;
        return apply_composition(t, f1, f2, f3, f4, C);
    } else if (type == NodeType::LOG) {
        if (C.v.real() <= 0.0) { AD r; r.v = NAN; return r; }
        Complex u = C.v, inv = 1.0/u;
        Complex f0 = std::log(u), f1 = inv, f2 = -inv*inv, f3 = 2.0*inv*inv*inv, f4 = -6.0*inv*inv*inv*inv;
        return apply_composition(f0, f1, f2, f3, f4, C);
    } else if (type == NodeType::SINH) {
        Complex sh = std::sinh(C.v), ch = std::cosh(C.v);
        return apply_composition(sh, ch, sh, ch, sh, C);
    } else if (type == NodeType::COSH) {
        Complex sh = std::sinh(C.v), ch = std::cosh(C.v);
        return apply_composition(ch, sh, ch, sh, ch, C);
    }
    return AD();
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

// ─── Version RAPIDA (solo v,dx,dy,dt,dxx,dyy,dtt) — identica a la formula
// original de antes de agregar el AD de 4to orden. La usa TODA la suite
// excepto Navier-Stokes/-Unsteady (ver apply_binary_ad_ext mas abajo).
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

// ─── Version EXTENDIDA (agrega dxy/dxt/dyt/3er/4to orden) — usada UNICAMENTE
// por el camino ad_eval_ext_t de Navier-Stokes/-Unsteady, ver comentario junto
// a apply_unary_ad_ext.
AD apply_binary_ad_ext(NodeType type, const AD& L, const AD& R) {
    AD r;
    if (type == NodeType::ADD) {
        r.v = L.v+R.v; r.dx = L.dx+R.dx; r.dy = L.dy+R.dy; r.dt = L.dt+R.dt;
        r.dxx = L.dxx+R.dxx; r.dyy = L.dyy+R.dyy; r.dtt = L.dtt+R.dtt;
        r.dxy = L.dxy+R.dxy; r.dxt = L.dxt+R.dxt; r.dyt = L.dyt+R.dyt;
        r.dxxx = L.dxxx+R.dxxx; r.dxxy = L.dxxy+R.dxxy; r.dxyy = L.dxyy+R.dxyy; r.dyyy = L.dyyy+R.dyyy;
        r.dxxt = L.dxxt+R.dxxt; r.dyyt = L.dyyt+R.dyyt;
        r.dxxxx = L.dxxxx+R.dxxxx; r.dxxyy = L.dxxyy+R.dxxyy; r.dyyyy = L.dyyyy+R.dyyyy;
    } else if (type == NodeType::SUB) {
        r.v = L.v-R.v; r.dx = L.dx-R.dx; r.dy = L.dy-R.dy; r.dt = L.dt-R.dt;
        r.dxx = L.dxx-R.dxx; r.dyy = L.dyy-R.dyy; r.dtt = L.dtt-R.dtt;
        r.dxy = L.dxy-R.dxy; r.dxt = L.dxt-R.dxt; r.dyt = L.dyt-R.dyt;
        r.dxxx = L.dxxx-R.dxxx; r.dxxy = L.dxxy-R.dxxy; r.dxyy = L.dxyy-R.dxyy; r.dyyy = L.dyyy-R.dyyy;
        r.dxxt = L.dxxt-R.dxxt; r.dyyt = L.dyyt-R.dyyt;
        r.dxxxx = L.dxxxx-R.dxxxx; r.dxxyy = L.dxxyy-R.dxxyy; r.dyyyy = L.dyyyy-R.dyyyy;
    } else if (type == NodeType::MUL) {
        r = apply_mul_ad(L, R);
    } else if (type == NodeType::DIV) {
        // L/R = L * (1/R). 1/R via apply_composition con f(u)=1/u (f'=-1/u^2,
        // f''=2/u^3, f'''=-6/u^4, f''''=24/u^5), luego producto via
        // apply_mul_ad — reusa las dos reglas ya verificadas en vez de derivar
        // el cociente por separado.
        if (std::abs(R.v.real()) < 1e-12) { r.v = NAN; return r; }
        Complex inv = 1.0/R.v, inv2 = inv*inv, inv3 = inv2*inv, inv4 = inv3*inv, inv5 = inv4*inv;
        AD Rinv = apply_composition(inv, -inv2, 2.0*inv3, -6.0*inv4, 24.0*inv5, R);
        r = apply_mul_ad(L, Rinv);
    } else if (type == NodeType::POW) {
        // pow(L,R) = exp(R*log(L)): compone log (via apply_composition),
        // multiplica por R (via apply_mul_ad), y exponencia el resultado (via
        // apply_composition) — evita derivar directamente la regla de la
        // potencia de dos variables (base Y exponente diferenciables) a 4to
        // orden, que es mucho mas propensa a error. Preserva generalidad
        // completa (R puede depender de x/y/t, igual que el codigo de 2do
        // orden anterior) al no asumir R constante en ningun paso.
        if (L.v.real() < 0.0) { r.v = NAN; return r; }
        Complex lv = L.v, linv = 1.0/lv, linv2 = linv*linv, linv3 = linv2*linv, linv4 = linv3*linv;
        AD logL = apply_composition(std::log(lv), linv, -linv2, 2.0*linv3, -6.0*linv4, L);
        AD M = apply_mul_ad(R, logL);
        Complex ev = std::exp(M.v);
        r = apply_composition(ev, ev, ev, ev, ev, M);
    } else { // Polinomios Ortogonales: P_n(L(x,y,t)), composicion de una funcion
             // de una variable (el polinomio) con L — misma maquinaria que
             // SIN/EXP/etc, usando value+1ro..4to derivada de eval_poly_all.
        if (!std::isfinite(R.v.real())) { r.v = NAN; return r; }
        double pv, pdv, pdvv, pdvvv, pdvvvv; int n = std::clamp((int)std::round(R.v.real()), 0, 10);
        eval_poly_all(type, n, L.v.real(), pv, pdv, pdvv, pdvvv, pdvvvv);
        r = apply_composition(pv, pdv, pdvv, pdvvv, pdvvvv, L);
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
// Todas las derivadas de orden >=2 de x, y, t o una constante son 0 — ad_eval_t
// ya deja los 12 campos nuevos en su default (0), asi que delega directo.
AD TerminalNode::ad_eval_ext(double x, double y, int dim) const { return ad_eval_t(x, y, 0.0, dim); }
AD TerminalNode::ad_eval_ext_t(double x, double y, double t, int dim) const { return ad_eval_t(x, y, t, dim); }
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
AD UnaryNode::ad_eval_ext(double x, double y, int dim) const { return ad_eval_ext_t(x, y, 0.0, dim); }
AD UnaryNode::ad_eval_ext_t(double x, double y, double t, int dim) const {
    if (!child) return AD(0.0);
    AD C = child->ad_eval_ext_t(x, y, t, dim);
    return apply_unary_ad_ext(type, C);
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
AD BinaryNode::ad_eval_ext(double x, double y, int dim) const { return ad_eval_ext_t(x, y, 0.0, dim); }
AD BinaryNode::ad_eval_ext_t(double x, double y, double t, int dim) const {
    if (!left || !right) return AD(0.0);
    AD L = left->ad_eval_ext_t(x, y, t, dim); AD R = right->ad_eval_ext_t(x, y, t, dim);
    return apply_binary_ad_ext(type, L, R);
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

    // Distribucion (ley distributiva): A*(B+C) -> A*B + A*C, y su version SUB.
    // Sin esto, un factor reconocible como el ansatz de frontera B(x)=x(x-1)
    // multiplicando una suma nunca se "disuelve" en la expresion final — queda
    // visible tal cual, aunque el resto se haya simplificado, delatando el
    // mecanismo interno de busqueda en vez de una formula genuinamente
    // reducida. Se re-simplifica el resultado para que la distribucion se
    // encadene con las demas reglas (cancelacion, combinacion de terminos).
    if (type == NodeType::MUL) {
        if (rt == NodeType::ADD || rt == NodeType::SUB) {
            auto* rb = dynamic_cast<const BinaryNode*>(sr.get());
            return make_binary(rt,
                make_binary(NodeType::MUL, sl->clone(), rb->left->clone()),
                make_binary(NodeType::MUL, sl->clone(), rb->right->clone()))->simplify();
        }
        if (lt == NodeType::ADD || lt == NodeType::SUB) {
            auto* lb = dynamic_cast<const BinaryNode*>(sl.get());
            return make_binary(lt,
                make_binary(NodeType::MUL, lb->left->clone(), sr->clone()),
                make_binary(NodeType::MUL, lb->right->clone(), sr->clone()))->simplify();
        }
    }

    auto is_zero = [](double v) { return std::isfinite(v) && std::abs(v) < 1e-6; };
    auto is_one = [](double v) { return std::isfinite(v) && std::abs(v - 1.0) < 1e-6; };
    double lv_v = is_constant(lt) ? sl->eval_t(0,0,0).real() : NAN;
    double rv_v = is_constant(rt) ? sr->eval_t(0,0,0).real() : NAN;
    // Deteccion numerica de cancelacion exacta (A + (-A) = 0, A - A = 0) para
    // casos donde los dos subarboles son equivalentes pero no sintacticamente
    // identicos (ej. "exp(-0.5x^2) + -exp(-0.5x^2)", frecuente tras mutacion/
    // cruzamiento) — la comparacion por print_str() de abajo solo detecta
    // duplicados textuales exactos, se le escapan estos casos. Se evalua en
    // varios puntos no triviales; si la suma/resta es ~0 en todos, se asume
    // cancelacion (no es una prueba formal, pero es la misma heuristica de
    // sondeo numerico que ya usa probe_priors() para simetrias).
    if ((type == NodeType::ADD || type == NodeType::SUB) && sl->contains_variables() && sr->contains_variables()) {
        bool cancels = true;
        for (double xp : {0.13, 0.37, 0.61, 0.89}) {
            Complex a = sl->eval_t(xp, xp * 0.7, 0.0);
            Complex b = sr->eval_t(xp, xp * 0.7, 0.0);
            Complex combined = (type == NodeType::ADD) ? (a + b) : (a - b);
            double scale = std::max({std::abs(a), std::abs(b), 1e-9});
            if (!std::isfinite(combined.real()) || std::abs(combined) > 1e-6 * scale) { cancels = false; break; }
        }
        if (cancels) return make_erc(0.0);
    }
    // Combinacion de multiplos escalados del mismo sub-arbol: c1*X + c2*X ->
    // (c1+c2)*X (generaliza el caso "X+X=2X" de abajo, que es el caso
    // particular c1=c2=1). Sin esto, terminos como "0.7*sin(x) + 1.3*sin(x)"
    // (tipicos tras mutacion/cruzamiento aditivo) quedaban sin fusionar.
    if (type == NodeType::ADD || type == NodeType::SUB) {
        auto extract_coef = [](const NodePtr& n) -> std::pair<double, const Node*> {
            if (auto* bn = dynamic_cast<const BinaryNode*>(n.get())) {
                if (bn->type == NodeType::MUL) {
                    if (is_constant(bn->left->get_type())) return {bn->left->eval_t(0,0,0).real(), bn->right.get()};
                    if (is_constant(bn->right->get_type())) return {bn->right->eval_t(0,0,0).real(), bn->left.get()};
                }
            }
            return {1.0, n.get()};
        };
        auto [cl, bl] = extract_coef(sl);
        auto [cr, br] = extract_coef(sr);
        if (bl->contains_variables() && bl->print_str() == br->print_str()) {
            double combined = (type == NodeType::ADD) ? (cl + cr) : (cl - cr);
            if (is_zero(combined)) return make_erc(0.0);
            NodePtr base = bl->clone();
            if (is_one(combined)) return base;
            if (std::abs(combined + 1.0) < 1e-6) return make_binary(NodeType::SUB, make_erc(0.0), std::move(base))->simplify();
            return make_binary(NodeType::MUL, make_erc(combined), std::move(base));
        }
    }
    if (type == NodeType::ADD) {
        if (is_zero(lv_v)) return sr; if (is_zero(rv_v)) return sl;
    }
    if (type == NodeType::SUB) { if (is_zero(rv_v)) return sl; }
    if (type == NodeType::MUL) {
        if (is_zero(lv_v) || is_zero(rv_v)) return make_erc(0.0);
        if (is_one(lv_v)) return sr; if (is_one(rv_v)) return sl;
        // X*X -> X^2: sin esto la distribucion de arriba (A*(B+C)) puede dejar
        // productos como "x*x" sueltos en vez de la forma habitual x^2.
        if (sl->contains_variables() && sl->print_str() == sr->print_str())
            return make_binary(NodeType::POW, std::move(sl), make_erc(2.0));
    }
    if (type == NodeType::DIV) { if (is_zero(lv_v)) return make_erc(0.0); if (is_one(rv_v)) return sl; if (sl->print_str() == sr->print_str()) return make_erc(1.0); }
    if (type == NodeType::POW) {
        if (is_zero(rv_v)) return make_erc(1.0);
        if (is_one(rv_v)) return sl;
        if (is_zero(lv_v)) return make_erc(0.0);
        if (is_one(lv_v)) return make_erc(1.0);
    }
    // Polinomio ortogonal de grado 0: por definicion, P_0(x)=H_0(x)=T_0(x)=
    // L_0(x)=1 para las 4 familias (ver eval_poly_all) — es una constante que
    // no depende en absoluto del argumento, asi que colapsa a 1 en vez de
    // dejar el nodo (con el argumento adentro sin usar) para imprimir despues.
    if (is_polynomial(type) && is_zero(rv_v)) return make_erc(1.0);
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
AD SeriesNode::ad_eval_ext(double x, double y, int dim) const { return ad_eval_ext_t(x, y, 0.0, dim); }
AD SeriesNode::ad_eval_ext_t(double x, double y, double t, int dim) const {
    AD r; r.v = 0.0; int save_n = current_n;
    for (int n = 1; n <= n_terms; ++n) { current_n = n; AD C = child->ad_eval_ext_t(x, y, t, dim);
        Complex coef = coeffs[n - 1]; r.v += coef * C.v; r.dx += coef * C.dx; r.dy += coef * C.dy; r.dt += coef * C.dt;
        r.dxx += coef * C.dxx; r.dyy += coef * C.dyy; r.dtt += coef * C.dtt;
        r.dxy += coef * C.dxy; r.dxt += coef * C.dxt; r.dyt += coef * C.dyt;
        r.dxxx += coef * C.dxxx; r.dxxy += coef * C.dxxy; r.dxyy += coef * C.dxyy; r.dyyy += coef * C.dyyy;
        r.dxxt += coef * C.dxxt; r.dyyt += coef * C.dyyt;
        r.dxxxx += coef * C.dxxxx; r.dxxyy += coef * C.dxxyy; r.dyyyy += coef * C.dyyyy;
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
// Estiramiento de coordenadas (log-transform) cerca de una singularidad tipo
// polo: la tecnica clasica de perturbaciones singulares sustituye s=log(r)
// para volver regular una EDO que es singular en r=0 (ej. el termino 1/sqrt(r)
// de Thomas-Fermi, o el 2/x de Lane-Emden). En vez de armar la transformacion
// completa (requeriria un nodo de cambio de variable dedicado), se le da a la
// busqueda log(x+eps) como bloque de construccion DIRECTO — mucho mas
// probable que descubra la forma correcta de un termino singular via
// mutacion/cruzamiento a partir de ahi que esperando que log() y division se
// combinen por casualidad desde un arbol generico.
static NodePtr skeleton_log_stretch(int depth, std::mt19937& gen, const PDEProblem& prob) {
    auto var = (prob.dim == 1 || std::uniform_real_distribution<double>(0,1)(gen) < 0.5) ? make_var('x') : make_var('y');
    double eps = 0.05 + std::uniform_real_distribution<double>(0.0, 0.1)(gen);
    NodePtr log_var = make_unary(NodeType::LOG, make_binary(NodeType::ADD, std::move(var), make_erc(eps)));
    int mode = std::uniform_int_distribution<int>(0, 2)(gen);
    if (mode == 2 || depth <= 1) return log_var;
    NodePtr rest = random_tree(std::max(depth - 2, 1), gen, prob);
    return make_binary(mode == 0 ? NodeType::MUL : NodeType::ADD, std::move(log_var), std::move(rest));
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

// Combinacion lineal rotada a*x+b*y (a=cos(theta), b=sin(theta), angulo
// aleatorio) envuelta en una funcion unaria: building block DIRECTO para
// soluciones tipo "cresta"/onda plana f(ax+by), en vez de esperar que
// ERC*x+ERC*y emerja por mutacion al azar desde un arbol generico — mismo
// espiritu que skeleton_log_stretch para el termino singular. Ver
// PDEPriors::rotation_invariant.
static NodePtr skeleton_rotated(int depth, std::mt19937& gen, const PDEProblem& prob) {
    constexpr double PI_LOCAL = 3.14159265358979323846;
    double theta = std::uniform_real_distribution<double>(0.0, 2.0 * PI_LOCAL)(gen);
    NodePtr lin = make_binary(NodeType::ADD,
        make_binary(NodeType::MUL, make_erc(std::cos(theta)), make_var('x')),
        make_binary(NodeType::MUL, make_erc(std::sin(theta)), special_var2(prob)));

    static const NodeType unary_opts[] = {NodeType::SIN, NodeType::COS, NodeType::EXP,
                                           NodeType::TANH, NodeType::SINH, NodeType::COSH};
    NodeType op = unary_opts[std::uniform_int_distribution<int>(0, 5)(gen)];
    NodePtr wrapped = make_unary(op, std::move(lin));

    int mode = std::uniform_int_distribution<int>(0, 2)(gen);
    if (mode == 2 || depth <= 1) return wrapped;
    NodePtr rest = random_tree(std::max(depth - 2, 1), gen, prob);
    return make_binary(mode == 0 ? NodeType::MUL : NodeType::ADD, std::move(wrapped), std::move(rest));
}

// Oscilador amortiguado exp(-a*v)*sin(b*v) con v en {x,y}: building block
// DIRECTO para la firma u''+2*zeta*omega*u'+omega^2*u=0 (ver
// PDEPriors::damped_oscillator) — en vez de esperar que EXP y SIN se
// combinen por casualidad via mutacion/cruzamiento, mismo espiritu que
// skeleton_log_stretch/skeleton_rotated.
static NodePtr skeleton_damped_oscillator(int depth, std::mt19937& gen, const PDEProblem& prob) {
    // En 2D, el sobre (envolvente) y la oscilacion pueden ir en variables
    // DISTINTAS (ej. exp(-a*x)*sin(b*y)) o la misma (ej. exp(-a*x)*sin(b*x),
    // el oscilador amortiguado clasico 1D) — 50/50 al azar. La forma cruzada
    // es la que realmente necesita Navier-Stokes: su solucion real es
    // exp(lambda*x)*sin(2*pi*y)/(2*pi*Re), sobre en x pero oscilacion en y —
    // antes esta funcion solo generaba la forma de una sola variable, que
    // nunca calza con esa estructura por mas mutacion/cruzamiento que se le
    // aplique. En 1D solo existe x, asi que ambas coinciden por construccion.
    char env_c = 'x', osc_c = 'x';
    if (prob.dim >= 2) {
        env_c = (std::uniform_real_distribution<double>(0.0, 1.0)(gen) < 0.5) ? 'x' : 'y';
        bool cross = std::uniform_real_distribution<double>(0.0, 1.0)(gen) < 0.5;
        osc_c = cross ? (env_c == 'x' ? 'y' : 'x') : env_c;
    }
    NodePtr var_env = make_var(env_c);
    NodePtr var_osc = make_var(osc_c);
    double a = std::uniform_real_distribution<double>(0.1, 3.0)(gen);
    double b = std::uniform_real_distribution<double>(0.5, 10.0)(gen);

    NodePtr envelope = make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-a), std::move(var_env)));
    NodePtr osc = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_erc(b), std::move(var_osc)));
    NodePtr damped = make_binary(NodeType::MUL, std::move(envelope), std::move(osc));

    int mode = std::uniform_int_distribution<int>(0, 2)(gen);
    if (mode == 2 || depth <= 1) return damped;
    NodePtr rest = random_tree(std::max(depth - 2, 1), gen, prob);
    return make_binary(mode == 0 ? NodeType::MUL : NodeType::ADD, std::move(damped), std::move(rest));
}

NodePtr random_tree_special(int depth, std::mt19937& gen, const PDEProblem& prob, const PDEPriors& priors) {
    std::vector<int> valid_choices;
    for (int i = 0; i <= 9; ++i) valid_choices.push_back(i);
    if (priors.pole_at_origin) {
        for (int i = 0; i < 5; ++i) valid_choices.push_back(10);
        for (int i = 0; i < 5; ++i) valid_choices.push_back(13); // skeleton_log_stretch
    }
    if (priors.autonomous_x || priors.autonomous_y) { for (int i = 0; i < 5; ++i) valid_choices.push_back(11); }
    if (priors.scale_invariant) { for (int i = 0; i < 5; ++i) valid_choices.push_back(12); }
    if (priors.rotation_invariant && prob.dim >= 2) { for (int i = 0; i < 5; ++i) valid_choices.push_back(14); }
    if (priors.damped_oscillator) { for (int i = 0; i < 5; ++i) valid_choices.push_back(15); }
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
        case 13: return skeleton_log_stretch(depth, gen, prob);
        case 14: return skeleton_rotated(depth, gen, prob);
        case 15: return skeleton_damped_oscillator(depth, gen, prob);
        // Inalcanzable en la práctica (valid_choices sólo puebla 0-15), pero si algún
        // día deja de serlo, degradamos a un árbol genérico en vez de filtrar la
        // solución exacta del PDE (ver auditoría: get_exact_solution_tree eliminada).
        default: return random_tree(depth, gen, prob);
    }
}

// Relabela in-place cada VAR_X del arbol al tipo de variable dado (usado para
// convertir un sub-arbol generado "solo en x" en uno "solo en y" o "solo en
// t", sin generar dos veces con logica separada).
static void relabel_var_x(Node* n, NodeType target) {
    if (!n) return;
    if (auto* tn = dynamic_cast<TerminalNode*>(n)) { if (tn->type == NodeType::VAR_X) tn->type = target; return; }
    if (auto* un = dynamic_cast<UnaryNode*>(n)) { relabel_var_x(un->child.get(), target); return; }
    if (auto* bn = dynamic_cast<BinaryNode*>(n)) { relabel_var_x(bn->left.get(), target); relabel_var_x(bn->right.get(), target); return; }
    if (auto* sn = dynamic_cast<SeriesNode*>(n)) { relabel_var_x(sn->child.get(), target); return; }
}

NodePtr random_separable_tree(int depth, std::mt19937& gen, const PDEProblem& prob, bool multiplicative) {
    // prob1d fuerza a random_tree a nunca elegir VAR_Y ni VAR_T (dim=1,
    // is_unsteady=false) — asi fx queda garantizado "solo en x". fy se genera
    // igual (tambien solo en x) y despues se relabela x->y.
    PDEProblem prob1d = prob;
    prob1d.dim = 1;
    prob1d.is_unsteady = false;
    NodePtr fx = random_tree(depth, gen, prob1d);
    NodePtr fy = random_tree(depth, gen, prob1d);
    relabel_var_x(fy.get(), NodeType::VAR_Y);
    return make_binary(multiplicative ? NodeType::MUL : NodeType::ADD, std::move(fx), std::move(fy));
}

NodePtr random_triple_separable_tree(int depth, std::mt19937& gen, const PDEProblem& prob) {
    PDEProblem prob1d = prob;
    prob1d.dim = 1;
    prob1d.is_unsteady = false;
    NodePtr fx = random_tree(depth, gen, prob1d);
    NodePtr fy = random_tree(depth, gen, prob1d);
    NodePtr ft = random_tree(depth, gen, prob1d);
    relabel_var_x(fy.get(), NodeType::VAR_Y);
    relabel_var_x(ft.get(), NodeType::VAR_T);
    return make_binary(NodeType::MUL, make_binary(NodeType::MUL, std::move(fx), std::move(fy)), std::move(ft));
}

NodePtr random_modal_sum_tree(int depth, std::mt19937& gen, const PDEProblem& prob) {
    PDEProblem prob1d = prob;
    prob1d.dim = 1;
    prob1d.is_unsteady = false;
    bool pure_is_y = std::uniform_real_distribution<double>(0.0, 1.0)(gen) < 0.5;
    NodePtr pure = random_tree(depth, gen, prob1d);
    relabel_var_x(pure.get(), pure_is_y ? NodeType::VAR_Y : NodeType::VAR_X);

    int sub_depth = std::max(depth - 1, 1);
    NodePtr px = random_tree(sub_depth, gen, prob1d);
    NodePtr py = random_tree(sub_depth, gen, prob1d);
    relabel_var_x(py.get(), NodeType::VAR_Y);
    NodePtr prod = make_binary(NodeType::MUL, std::move(px), std::move(py));

    return make_binary(NodeType::ADD, std::move(pure), std::move(prod));
}

static bool is_v(const Node* n, double target) {
    if (!n || n->contains_variables()) return false;
    return std::abs(n->eval_t(0,0,0).real() - target) < 1e-6;
}

void TerminalNode::print_formal(std::ostream& os, int parent_prec) const { print_latex(os); }
void UnaryNode::print_formal(std::ostream& os, int parent_prec) const {
    // "^{2}" con llaves (no "^2" pelado): si este SQR queda anidado como base
    // de un POW exterior (ej. POW(SQR(x),2)), el POW imprime "^{" + exp + "}"
    // a continuacion — sin llaves aca el resultado era "x^2^2", doble
    // superindice invalido en LaTeX ("! Double superscript.").
    if (type == NodeType::SQR) { bool need_paren = (parent_prec > 2); if (need_paren) os << "("; child->print_formal(os, 3); os << "^{2}"; if (need_paren) os << ")"; }
    else {
        // Antes: cualquier tipo unario que no fuera SQR caia directo a
        // print_latex() — una funcion COMPLETAMENTE DISTINTA que usa su
        // propia recursion (print_latex de los hijos, no print_formal), asi
        // que TODO lo que quedara anidado dentro de un exp/sin/cos/log/tanh/
        // sinh/cosh/gaussian se renderizaba con el motor viejo sin ninguno de
        // los fixes de esta sesion (parentesis, "^2" sin llaves para SQR
        // dentro de print_latex especificamente, fusion de signos, etc.) — la
        // causa real del "Double superscript" en formulas con exp(...) por
        // dentro. Ahora se llama print_formal recursivamente igual que los
        // demas tipos, para que las mismas reglas apliquen sin importar que
        // funcion envuelva al subarbol.
        if (type == NodeType::SIN) os << "\\sin(";
        else if (type == NodeType::COS) os << "\\cos(";
        else if (type == NodeType::EXP) os << "\\exp(";
        else if (type == NodeType::GAUSSIAN) os << "\\exp(-0.5 ";
        else if (type == NodeType::TANH) os << "\\tanh(";
        else if (type == NodeType::LOG) os << "\\log(";
        else if (type == NodeType::SINH) os << "\\sinh(";
        else if (type == NodeType::COSH) os << "\\cosh(";
        else os << "u(";
        if (type == NodeType::GAUSSIAN) {
            // GAUSSIAN eleva su hijo al cuadrado por definicion (exp(-0.5*hijo^2)).
            // Si el hijo ya termina en su propio superindice (ej. es un SQR,
            // "x^{2}"), pegarle "^{2}" directo da "x^{2}^{2}" (doble
            // superindice invalido). Envolver el hijo entre llaves es siempre
            // valido en LaTeX sin importar que forma tenga adentro.
            os << "{"; if (child) child->print_formal(os, 0); os << "}^{2}";
        } else {
            if (child) child->print_formal(os, 0);
        }
        os << ")";
    }
}
void BinaryNode::print_formal(std::ostream& os, int parent_prec) const {
    if (type == NodeType::ADD) { if (is_v(left.get(), 0.0)) { right->print_formal(os, parent_prec); return; } if (is_v(right.get(), 0.0)) { left->print_formal(os, parent_prec); return; } }
    if (type == NodeType::SUB) {
        if (is_v(right.get(), 0.0)) { left->print_formal(os, parent_prec); return; }
        if (is_v(left.get(), 0.0)) { os << "-"; right->print_formal(os, 2); return; }
    }
    if (type == NodeType::MUL) {
        if (is_v(left.get(), 0.0) || is_v(right.get(), 0.0)) { os << "0"; return; }
        if (is_v(left.get(), 1.0)) { right->print_formal(os, parent_prec); return; } if (is_v(right.get(), 1.0)) { left->print_formal(os, parent_prec); return; }
        if (is_v(left.get(), -1.0)) { os << "-"; right->print_formal(os, 2); return; } if (is_v(right.get(), -1.0)) { os << "-"; left->print_formal(os, 2); return; }
    }
    int own_prec = (type == NodeType::MUL || type == NodeType::DIV) ? 1 : ((type == NodeType::POW) ? 2 : ((type >= NodeType::LEGENDRE && type <= NodeType::LAGUERRE) ? 3 : 0));
    bool need_paren = (own_prec < parent_prec); if (need_paren) os << "(";
    if (type == NodeType::DIV) { os << "\\frac{"; left->print_formal(os, 0); os << "}{"; right->print_formal(os, 0); os << "}"; }
    else if (type == NodeType::POW) {
        // Si la base ya termina en su propio superindice (POW o SQR anidado,
        // ej. POW(SQR(x),2)), pasar parent_prec=2 (igual a la propia
        // precedencia de POW) NO agrega parentesis porque el chequeo usa ">"
        // estricto — el resultado queda "x^{2}^{2}", LaTeX no admite dos
        // superindices consecutivos sin agrupar ("Double superscript"). Se
        // envuelve la base explicitamente en esos casos, sin tocar el umbral
        // general de parent_prec (que podria agregar parentesis de mas en
        // otros contextos).
        bool base_is_pow = (left->get_type() == NodeType::POW || left->get_type() == NodeType::SQR);
        if (base_is_pow) os << "{";
        left->print_formal(os, 2);
        if (base_is_pow) os << "}";
        os << "^{"; right->print_formal(os, 0); os << "}";
    }
    else if (type >= NodeType::LEGENDRE && type <= NodeType::LAGUERRE) {
        os << (type == NodeType::LEGENDRE ? "P_" : (type == NodeType::HERMITE ? "H_" : (type == NodeType::CHEBYSHEV ? "T_" : "L_")));
        // El grado se imprime como entero limpio (nunca "-1.000" o "3.000")
        // acotado al mismo rango [0,10] que ya usa la evaluacion (eval_poly_all
        // via apply_binary/apply_binary_ad) — asi lo mostrado siempre coincide
        // con lo que realmente se calcula, incluso si el nodo derecho no paso
        // por round_constants (ej. constante recien mutada, no pulida todavia).
        int deg = std::clamp((int)std::round(right->eval_t(0,0,0).real()), 0, 10);
        os << deg; os << "("; left->print_formal(os, 0); os << ")";
    } else if (type == NodeType::ADD) {
        // Si el lado derecho imprime con signo negativo al frente (ej. viene de
        // MUL(-1,X) que no calzo ninguno de los atajos de arriba, u otra forma
        // negativa), se fusiona en "A - X" en vez de "A + -X".
        left->print_formal(os, own_prec);
        std::ostringstream rhs; right->print_formal(rhs, own_prec);
        std::string rs = rhs.str();
        if (!rs.empty() && rs[0] == '-') { os << " - " << rs.substr(1); }
        else { os << " + " << rs; }
    } else if (type == NodeType::MUL) {
        left->print_formal(os, own_prec);
        // Espacio ambiguo si el lado derecho empieza en '-' (ej. "H(x) -3.871"
        // se lee como resta aunque sea multiplicacion por -3.871) — se usa
        // \cdot explicito en ese caso para desambiguar; simple yuxtaposicion
        // (mas legible) en el resto.
        std::ostringstream rhs; right->print_formal(rhs, own_prec);
        std::string rs = rhs.str();
        os << (!rs.empty() && rs[0] == '-' ? " \\cdot " : " ") << rs;
    } else {
        left->print_formal(os, own_prec);
        if (type == NodeType::SUB) os << " - ";
        right->print_formal(os, own_prec);
    }
    if (need_paren) os << ")";
}
void SeriesNode::print_formal(std::ostream& os, int parent_prec) const {
    // Mismo motivo que el fix de UnaryNode::print_formal: llamar print_latex()
    // aca bypasea print_formal (y sus fixes) para todo lo anidado adentro.
    os << "\\sum_{n=1}^{" << n_terms << "} C_n ";
    if (child) child->print_formal(os, 3);
}

void TerminalNode::round_constants(double epsilon) {
    if (type == NodeType::ERC) { double r = erc_val.real(); double nearest = std::round(r); if (std::abs(r - nearest) < epsilon) erc_val = Complex(nearest, erc_val.imag()); }
}
void UnaryNode::round_constants(double epsilon) { if (child) child->round_constants(epsilon); }
void BinaryNode::round_constants(double epsilon) { 
    if (left) left->round_constants(epsilon); 
    if (right) {
        if (is_polynomial(type)) {
            // Snapping agresivo para el grado del polinomio (siempre entero, y
            // acotado a [0,10] — el mismo rango que ya usa apply_binary/
            // apply_binary_ad al evaluar (eval_poly_all clampea ahi, pero antes
            // esto no clampeaba aca, asi que el grado ALMACENADO podia quedar
            // negativo o fuera de rango — ej. -1.3 -> -1 — y se imprimia tal
            // cual ("H_-1.000(x)"), sin corresponder a lo que realmente se
            // evaluaba).
            double r = right->eval_t(0,0,0).real();
            double nearest = std::clamp(std::round(r), 0.0, 10.0);
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
