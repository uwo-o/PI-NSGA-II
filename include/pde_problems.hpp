#pragma once
// =============================================================================
// pde_problems.hpp  —  Definición de Laplace, Poisson y Helmholtz
//   Dominio Ω = [0,1]², condiciones de Dirichlet.
// =============================================================================

#include "common.hpp"
#include "dimensions.hpp"
#include <vector>
#include <functional>

class Node;

// ─── Análisis a priori de la física (Physics-Guided Search) ───────────────
// Declarado antes de PDEProblem para que este pueda cachear su propio resultado
// (ver campo `priors` más abajo) y así llegar hasta PIIndividual::evaluate sin
// tener que agregar un parámetro nuevo a esa función en todos sus call sites.
struct PDEPriors {
    bool autonomous_x = false;
    bool autonomous_y = false;
    bool autonomous_t = false;    // el residuo no depende explícitamente de t
    bool pole_at_origin = false;
    bool even_parity_x = false;

    // Ansatz de frontera de UN SOLO LADO: U = u0 + x*N(x) (1D) o
    // U = u0 + r*N(x,y) (2D, r=sqrt(x^2+y^2)), anclando SOLO el origen (u0,
    // el valor real de la condicion de frontera ahi) en vez de un segundo
    // punto/arista inventado. Derivado de pole_at_origin (no un sondeo
    // aparte): un operador con un termino que diverge en el origen (ej.
    // 2/x en Lane-Emden, 1/sqrt(r) en Thomas-Fermi) tipicamente viene de
    // una EDO planteada en un dominio semi-infinito [0,inf) truncado a
    // [0,1] solo por conveniencia computacional — el origen tiene una
    // condicion genuina (regularidad/valor inicial de la EDO real), pero el
    // otro extremo del dominio truncado NO es un dato del problema, es un
    // corte artificial. Usar un ansatz de dos lados ahi obliga a inventar un
    // valor de frontera en el corte — si ese valor sale de tablas/solucion
    // numerica de la EDO real (como pasaba antes con Thomas-Fermi, ver
    // historial), es fuga de informacion: se le regala al modelo la
    // respuesta en un punto interior de la fisica real. Ver
    // apply_boundary_ansatz en pi_solver.cpp. Verificado contra la suite
    // actual: coincide exactamente con Lane-Emden y Thomas-Fermi, ambas EDOs
    // genuinamente semi-infinitas truncadas — no se sondea por nombre de EDP
    // en ningun lado.
    bool one_sided_boundary = false;

    // Simetría especular x <-> 1-x (resp. y <-> 1-y), la relevante para el
    // dominio real [0,1] — a diferencia de even_parity_x (que compara contra
    // x=-1, fuera del dominio). Sólo se marca true si TANTO el operador COMO
    // la condición de frontera son invariantes bajo la reflexión: si sólo el
    // operador lo es pero la frontera no, la solución real no tiene por qué
    // ser simétrica, y forzarla sería una restricción falsa (peor que no
    // tener el prior). Con ambos invariantes, la solución sí está obligada a
    // serlo (unicidad: aplicar la reflexión a una solución de un problema con
    // operador+frontera simétricos da otra solución del mismo problema).
    bool mirror_symmetric_x = false;
    bool mirror_symmetric_y = false;

    // Priors de Nivel Superior
    bool scale_invariant = false; // Invarianza de escala (Auto-similaridad)
    int  max_deriv_order = 0;     // ADN Diferencial (Orden máximo de derivación)
    bool is_conservative = false; // Leyes de conservación (Divergencia nula)

    // Separabilidad de variables (solo 2D): si el OPERADOR de la EDP (no la
    // solucion — eso seria fuga de informacion, ver comentario en
    // probe_priors()) es consistente con u(x,y)=f(x)+g(y) o u(x,y)=f(x)g(y),
    // toda la poblacion inicial se construye directamente en esa forma en vez
    // de arboles genericos — inspirado en el metodo clasico de separacion de
    // variables. Mutuamente excluyentes en la practica (se prioriza aditiva
    // si ambas dieran positivo, poco probable).
    bool additive_separable = false;
    bool multiplicative_separable = false;

    // Separabilidad triple (2D dependiente del tiempo): u(x,y,t)=f(x)g(y)h(t).
    // Mismo sondeo generico sobre el OPERADOR (no la solucion) que la
    // separabilidad de 2 variables, extendido a probar las 3 derivadas
    // cruzadas (xy, xt, yt). Se activa para CUALQUIER EDP dependiente del
    // tiempo que pase el sondeo, sin excluir ningun tipo por nombre — asi no
    // es una solucion "hecha a medida" de un solo problema del benchmark.
    bool triple_separable = false;

    // Suma de un termino puro + un termino producto: u(x,y)=f(v)+g(x)h(y),
    // con v in {x,y}. Generaliza additive/multiplicative_separable (expansion
    // en modos/autofunciones, ej. Fourier) — cubre estructuras como
    // y - exp(lambda*x)*sin(2*pi*y)/(2*pi*Re) (Navier-Stokes) que no son ni
    // puramente aditivas ni puramente multiplicativas. Sondeado sobre el
    // OPERADOR con una funcion generica (ver probe_priors), nunca sobre la
    // solucion real. No excluye ningun tipo de EDP por nombre.
    bool modal_sum_separable = false;

    // El operador trata x e y de forma intercambiable (isotropo/simetrico),
    // sondeado evaluando el residuo con una funcion de sondeo generica de una
    // sola variable en un punto y su version con roles x<->y intercambiados
    // en el punto reflejado — si coinciden, el operador es consistente con
    // soluciones tipo "cresta"/onda plana f(a*x+b*y) para cualquier angulo,
    // no solo un eje. Ver skeleton_rotated en tree_node.cpp.
    bool rotation_invariant = false;

    // Termino de amortiguamiento lineal en la primera derivada (du/dx o
    // du/dy) mas alla de lo que ya aporta el Laplaciano — la firma
    // caracteristica de un oscilador amortiguado (u''+2*zeta*omega*u'+
    // omega^2*u=0). Sondeado perturbando SOLO esa derivada (dejando el resto
    // fijo) y viendo si el residuo reacciona — ver skeleton_damped_oscillator
    // en tree_node.cpp.
    bool damped_oscillator = false;
};

// ─── Problema PDE ─────────────────────────────────────────────────────────────
struct PDEProblem {
    PDE    type;
    int    dim = 2;    // 1 o 2 dimensiones
    double k2 = 0.0;   // coeficiente k² en Helmholtz: ∇²u + k²u = f
    bool   is_numerical = false; // Indica si requiere validación vía NumericalSolver
    bool   is_unsteady = false;  // Indica si depende del tiempo
    std::vector<Complex> numerical_truth; // Malla de referencia pre-calculada
    PDEPriors priors; // Cacheado por PISolver al construirse (probe_priors), usado
                       // por PIIndividual::evaluate para exigir simetrías detectadas.

    // Dimensiones para coherencia física
    Dimension dim_u = Units::None;
    Dimension dim_x = Units::Length;
    Dimension dim_y = Units::Length;
    Dimension dim_t = Units::Time;

    // Puntos de colocación interior (rejilla uniforme)
    std::vector<Point> domain_points(int n) const;

    // Puntos de condición de frontera (Dirichlet sobre ∂Ω)
    std::vector<Point> boundary_points(int n) const;

    // Solución exacta conocida u*(x,y,t)
    virtual Complex exact(double x, double y, double t = 0.0) const;
    Complex numerical_exact(double x, double y, double t = 0.0) const;
    Complex pde_second_derivative(double x, Complex u) const;

    // Término fuente f(x,y,t) del lado derecho
    virtual Complex source(double x, double y, double t = 0.0) const;

    // Condición de frontera (valor de u en ∂Ω a tiempo t)
    virtual Complex bc(double x, double y, double t = 0.0) const;

    // Cálculo unificado del residuo (incluye AD o Híbrido AD-FDM)
    virtual Complex compute_residual(const Node* tree, const Point& p) const;

    // Cálculo unificado del error de frontera (Dirichlet, Neumann, etc.)
    virtual double compute_boundary_error(const Node* tree, const std::vector<Point>& bnd) const;

    // Residuo del PDE usando AD (para método propuesto)
    Complex pde_residual_ad(const AD& ad, double x, double y, double t = 0.0) const;

    // Nombre legible
    std::string name() const;
};

PDEPriors probe_priors(const PDEProblem& prob);

// ─── Fabricación de problemas ─────────────────────────────────────────────────
PDEProblem make_laplace(int dim = 2);
PDEProblem make_poisson(int dim = 2);
PDEProblem make_helmholtz(int dim = 2, double k = 1.0);
PDEProblem make_schrodinger(int dim = 2);
PDEProblem make_nonlinear_poisson();
PDEProblem make_liouville();
PDEProblem make_sine_gordon();
PDEProblem make_airy(int dim);
PDEProblem make_harmonic_oscillator(int dim);
PDEProblem make_navier_stokes();
PDEProblem make_navier_stokes_unsteady();
PDEProblem make_fisher(int dim);
PDEProblem make_duffing(int dim);
PDEProblem make_thomas_fermi(int dim);
PDEProblem make_bratu();
PDEProblem make_allen_cahn();
PDEProblem make_lane_emden();
PDEProblem make_troesch();
PDEProblem make_ginzburg_landau();
PDEProblem make_painleve1();
