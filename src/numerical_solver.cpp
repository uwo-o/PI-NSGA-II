#include "numerical_solver.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace NumericalSolver {

std::vector<Complex> solve_rk4_1d(const PDEProblem& prob, int resolution) {
    std::vector<Complex> results(resolution);
    double h = 1.0 / (resolution - 1);
    
    // Suponemos BCs en x=0 y x=1. Para RK4 necesitamos u(0) y u'(0).
    // Como solo tenemos u(0) y u(1), esto es un BVP (Boundary Value Problem).
    // Usaremos un método de "Shooting" simple.
    
    auto shoot = [&](Complex guessed_v0) {
        Complex u = prob.bc(0.0, 0.0);
        Complex v = guessed_v0;
        std::vector<Complex> path(resolution);
        path[0] = u;

        for (int i = 0; i < resolution - 1; ++i) {
            double x = i * h;
            auto safe_f = [&](double cx, Complex cu) {
                Complex res = prob.pde_second_derivative(cx, cu);
                return std::isfinite(res.real()) ? res : Complex(1e10);
            };

            Complex k1u = v;
            Complex k1v = safe_f(x, u);

            Complex k2u = v + 0.5 * h * k1v;
            Complex k2v = safe_f(x + 0.5 * h, u + 0.5 * h * k1u);

            Complex k3u = v + 0.5 * h * k2v;
            Complex k3v = safe_f(x + 0.5 * h, u + 0.5 * h * k2u);

            Complex k4u = v + h * k3v;
            Complex k4v = safe_f(x + h, u + h * k3u);

            u += (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
            v += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
            
            if (!std::isfinite(u.real()) || std::abs(u) > 1e10) {
                u = 1e10; 
                for (int j = i; j < resolution; ++j) path[j] = u;
                return std::make_pair(u, path);
            }
            path[i+1] = u;
        }
        return std::make_pair(u, path);
    };

    // Búsqueda del v0 correcto (u(1) debe coincidir con prob.bc(1,0))
    Complex target_u1 = prob.bc(1.0, 0.0);
    Complex v0_low(-10.0, -10.0), v0_high(10.0, 10.0);
    
    // Semillas iniciales informadas por la física del problema
    Complex v_a, v_b;
    if (prob.type == PDE::HARMONIC_OSCILLATOR) {
        // Estado base Gaussiano: u'(0) = 0 (función par)
        v_a = Complex(0.0, 0.0);
        v_b = Complex(-0.1, 0.0);  // perturbación pequeña
    } else if (prob.type == PDE::AIRY) {
        // Ai'(0) ≈ -0.2588
        v_a = Complex(-0.2588, 0.0);
        v_b = Complex(-0.3, 0.0);
    } else {
        v_a = Complex(0.0, 0.0);
        v_b = Complex(-0.5, 0.0);
    }
    auto res_a = shoot(v_a);
    auto res_b = shoot(v_b);
    
    for (int iter = 0; iter < 20; ++iter) {
        Complex f_a = res_a.first - target_u1;
        Complex f_b = res_b.first - target_u1;
        
        if (std::abs(f_b) < 1e-9) break;
        if (std::abs(f_b - f_a) < 1e-15) break;
        
        Complex v_next = v_b - f_b * (v_b - v_a) / (f_b - f_a);
        
        // Limitar el salto para evitar explosiones
        if (std::abs(v_next) > 1e6) v_next = v_next / std::abs(v_next) * 1e6;

        v_a = v_b; res_a = res_b;
        v_b = v_next; res_b = shoot(v_b);

        if (!std::isfinite(v_b.real())) break;
    }
    
    return res_b.second;
}

std::vector<Complex> solve_fd_2d(const PDEProblem& prob, int resolution) {
    int N = resolution;
    std::vector<Complex> grid(N * N, 0.0);
    double h = 1.0 / (N - 1);

    // Inicializar fronteras
    for (int i = 0; i < N; ++i) {
        grid[0 * N + i] = prob.bc(0.0, i * h);         // x=0
        grid[(N - 1) * N + i] = prob.bc(1.0, i * h);   // x=1
        grid[i * N + 0] = prob.bc(i * h, 0.0);         // y=0
        grid[i * N + (N - 1)] = prob.bc(i * h, 1.0);   // y=1
    }

    // Relajación Jacobi
    for (int iter = 0; iter < 1000; ++iter) {
        std::vector<Complex> next = grid;
        double max_diff = 0;
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                // ∇²u + k²u = f  => (u_i+1 + u_i-1 + u_j+1 + u_j-1 - 4u)/h² + k²u = f
                Complex neighbors = grid[(i+1)*N + j] + grid[(i-1)*N + j] + 
                                   grid[i*N + (j+1)] + grid[i*N + (j-1)];
                
                // u = (neighbors - f*h²) / (4 - k²*h²)
                Complex val = (neighbors - prob.source(i*h, j*h) * h * h) / (4.0 - prob.k2 * h * h);
                next[i*N + j] = val;
                max_diff = std::max(max_diff, std::abs(val - grid[i*N+j]));
            }
        }
        grid = next;
        if (max_diff < 1e-6) break;
    }

    return grid;
}

std::vector<Complex> solve(const PDEProblem& prob, int resolution) {
    // printf("[NumericalSolver] Resolviendo %s (%dD, res=%d)...\n", prob.name().c_str(), prob.dim, resolution);
    if (prob.dim == 1) return solve_rk4_1d(prob, resolution);
    return solve_fd_2d(prob, resolution);
}

}
