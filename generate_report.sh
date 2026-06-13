#!/bin/bash

./.venv/bin/python3 plot_pareto.py
./.venv/bin/python3 plot_solutions.py
./.venv/bin/python3 stats_analysis.py
./.venv/bin/python3 plot_extra_report_figures.py

echo ">>> Step 4: Report Generation"
./.venv/bin/python3 report/generate_report.py
./.venv/bin/python3 report/generate_formulas_table.py


echo ">>> Step 5: LaTeX Compilation"
cd report
pdflatex -interaction=nonstopmode results.tex > /dev/null
pdflatex -interaction=nonstopmode results.tex > /dev/null
cd ..