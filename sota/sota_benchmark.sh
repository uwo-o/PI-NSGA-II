#!/bin/bash
# Automates the SOTA benchmarking suite

# Navigate to the script's directory
cd "$(dirname "$0")"

# Setup virtual environment if needed (assuming user has packages installed or will install them)
# echo "Installing dependencies..."
# pip install -r requirements.txt -q

PROBLEMS=(
    "Airy_1D"
    "Airy_2D"
    "Duffing_1D"
    "Duffing_2D"
    "Thomas-Fermi_1D"
    "Thomas-Fermi_2D"
    "Lane-Emden_1D"
    "Troesch_1D"
    "Painleve-I_1D"
    "Fisher_1D"
    "Fisher_2D"
    "Ginzburg-Landau_1D"
)

echo "=========================================================="
echo "    STARTING SOTA BENCHMARK (PySR & PySINDy)              "
echo "=========================================================="

for prob in "${PROBLEMS[@]}"; do
    echo " "
    echo "----------------------------------------------------------"
    echo ">>> Benchmarking Problem: $prob"
    echo "----------------------------------------------------------"
    
    DATA_FILE="../results/grid_${prob}_PI-NSGA-II.csv"
    
    # 1. Generate High-Fidelity Data (Skipped because we use PI-NSGA-II data)
    echo "[1/3] Data already available from PI-NSGA-II."

    # 2. Run PySINDy
    echo "[2/3] Running PySINDy discovery..."
    # The script expects base problem name without _1D suffix for its internal naming,
    # but we will just pass the clean problem name and let it save properly.
    # Wait, the script appends suffix = f"_{dim}D" so if we pass Airy_1D, it becomes Airy_1D_1D.
    # Let's clean the name for SOTA scripts.
    clean_prob=$(echo "$prob" | sed -E 's/_(1D|2D)//')
    
    echo "      Saved SINDy output to results/sindy_${prob}.log"
    
    # 3. Run PySR
    echo "[3/3] Running PySR discovery (This might take a while)..."
    uv run python run_pysr.py --problem "$clean_prob" --dataset "$DATA_FILE" > "results/pysr_${prob}.log" 2>&1
    echo "      Saved PySR output to results/pysr_${prob}.log"
done

echo " "
echo "=========================================================="
echo "    SOTA BENCHMARK COMPLETE                               "
echo "    Check the 'sota/results/' directory for the models.   "
echo "=========================================================="
