#!/bin/bash
set -e
cd /home/uwo/Projects/PI-NSGA-II

echo "Step 1: Generating stats runs (10 runs)"
./build/pi_nsga2 --runs 20

echo "Step 2: Stats Analysis"
python3 stats_analysis.py

echo "Step 3: Single run for 3D grids and best formulas"
./build/pi_nsga2 > single_run.log

echo "Step 4: Plot Pareto"
python3 plot_pareto.py

echo "Step 5: Plot Solutions 3D"
python3 plot_solutions.py

echo "Step 6: Generate Report tables and copy figures"
python3 report/generate_report.py
python3 report/generate_formulas_table.py

echo "Step 7: Compile LaTeX PDF"
cd report
pdflatex -interaction=nonstopmode results.tex > /dev/null
pdflatex -interaction=nonstopmode results.tex > /dev/null
echo "PIPELINE COMPLETED SUCCESSFULLY"
