#ifndef NUMERICAL_SOLVER_HPP
#define NUMERICAL_SOLVER_HPP

#include "common.hpp"
#include "pde_problems.hpp"
#include <vector>

namespace NumericalSolver {
    // Genera una malla de puntos u(x,y) usando métodos numéricos estándar
    // para problemas que no tienen solución analítica.
    std::vector<Complex> solve(const PDEProblem& prob, int resolution);

    // Implementaciones internas (pueden ser útiles fuera)
    std::vector<Complex> solve_rk4_1d(const PDEProblem& prob, int resolution);
    std::vector<Complex> solve_fd_2d(const PDEProblem& prob, int resolution);
}

#endif
