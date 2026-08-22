#!/bin/bash

./.venv/bin/python3 scripts/plot_pareto.py
# plot_pareto.py nombra "{EDP}_pareto_pairs.pdf"; generate_report.py espera
# "pareto_pairs_{EDP}.pdf" — se renombra al copiar.
for f in results/*_pareto_pairs.pdf; do
    [ -e "$f" ] || continue
    base=$(basename "$f" "_pareto_pairs.pdf")
    cp "$f" "report/figures/pareto_pairs_${base}.pdf"
done
./.venv/bin/python3 scripts/plot_solutions.py
./.venv/bin/python3 scripts/stats_analysis.py
./.venv/bin/python3 report/data_efficiency_comparison.py
./.venv/bin/python3 scripts/plot_extra_report_figures.py

echo ">>> Step 4: Report Generation"
./.venv/bin/python3 report/generate_report.py
./.venv/bin/python3 report/generate_formulas_table.py


echo ">>> Step 5: LaTeX Compilation"
cd report
pdflatex -interaction=nonstopmode results.tex > /dev/null
pdflatex -interaction=nonstopmode results.tex > /dev/null
cd ..
