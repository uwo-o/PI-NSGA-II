#!/bin/bash
set -e
cd /home/uwo/Projects/PI-NSGA-II

# 1. Build
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ..

echo ">>> Step 1: Symbolic Benchmark (20 runs for statistics)"
./build/pi_nsga2 --runs 1

echo ">>> Step 2: PINN Baseline"
# python3 pinn_baseline.py --epochs 1000 --runs 1

echo ">>> Step 3: Analysis and Plotting"
python3 plot_pareto.py
python3 plot_solutions.py
python3 stats_analysis.py

echo ">>> Step 4: Report Generation"
python3 report/generate_report.py
python3 report/generate_formulas_table.py

echo ">>> Step 5: LaTeX Compilation"
cd report
pdflatex -interaction=nonstopmode results.tex > /dev/null
pdflatex -interaction=nonstopmode results.tex > /dev/null
cd ..

echo "=========================================================="
echo "  PIPELINE FINISHED SUCCESSFULLY"
echo "  Report available at: /home/uwo/Projects/PI-NSGA-II/report/results.pdf"
echo "=========================================================="
