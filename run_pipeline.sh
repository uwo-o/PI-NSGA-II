#!/bin/bash
set -e

# Asegurar que el script se ejecuta en su propio directorio
cd "$(dirname "$0")"

# 1. Build
mkdir -p build && cd build
cmake .. && make
cd ..

echo ">>> Step 1: Symbolic Benchmark"
./build/PISR-EMOAD "$@"

echo ">>> Step 2: SOTA Methods"
cd sota 
sh sota_benchmark.sh "$@"
cd ..

echo ">>> Step 3: DeepXDE Baseline"
./.venv/bin/python3.12 scripts/pinn_baseline.py --cores 8

echo ">>> Step 4: Analysis and Plotting"
./.venv/bin/python3 scripts/plot_pareto.py
./.venv/bin/python3 scripts/plot_solutions.py
./.venv/bin/python3 scripts/stats_analysis.py
./.venv/bin/python3 scripts/plot_extra_report_figures.py

echo ">>> Step 5: Report Generation"
./.venv/bin/python3 report/generate_report.py
./.venv/bin/python3 report/generate_formulas_table.py


echo ">>> Step 6: LaTeX Compilation"
cd report
pdflatex -interaction=nonstopmode results.tex > /dev/null
pdflatex -interaction=nonstopmode results.tex > /dev/null
cd ..
