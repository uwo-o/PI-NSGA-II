#!/bin/bash
# Automates the SOTA benchmarking suite

# Navigate to the script's directory
cd "$(dirname "$0")"

# Setup virtual environment if needed (assuming user has packages installed or will install them)
# echo "Installing dependencies..."
# pip install -r requirements.txt -q

RUNS=1
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --runs) RUNS="$2"; shift ;;
        *) shift ;;
    esac
    shift
done

PROBLEMS=(
    "Laplace_1D"
    "Poisson_1D"
    "HarmonicOscillator_1D"
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
    "Navier-Stokes_2D"
    "Navier-Stokes-Unsteady_2D"
)

echo "=========================================================="
echo "    STARTING SOTA BENCHMARK (PySR & PySINDy)              "
echo "=========================================================="

for prob in "${PROBLEMS[@]}"; do
    for run_id in $(seq 1 $RUNS); do
        echo " "
        echo "----------------------------------------------------------"
        echo ">>> Benchmarking Problem: $prob (Run $run_id/$RUNS)"
        echo "----------------------------------------------------------"
        
        # Con --runs 1 (el modo normal), PISR-NSGA-II escribe directo en
        # results/ sin subcarpeta run_N — esa solo se crea con --runs >1.
        # Antes esto asumia siempre run_N y nunca encontraba el archivo.
        if [ "$RUNS" -gt 1 ]; then
            DATA_FILE="../results/run_${run_id}/grid_${prob}_PISR-EMOAD.csv"
        else
            DATA_FILE="../results/grid_${prob}_PISR-EMOAD.csv"
        fi
        
        # 1. Generate High-Fidelity Data (Skipped because we use PISR-EMOAD data)
        echo "[1/3] Data already available from PISR-EMOAD."

        # 2. Run PySINDy
        echo "[2/3] Running PySINDy discovery..."
        clean_prob=$(echo "$prob" | sed -E 's/_(1D|2D)//')
        uv run python run_pysindy.py --problem "$clean_prob" --dataset "$DATA_FILE" --run_id "$run_id" > "results/sindy_${prob}_run${run_id}.log" 2>&1
        echo "      Saved PySINDy output to results/sindy_${prob}_run${run_id}.log"

        # 3. Run PySR
        echo "[3/3] Running PySR discovery (This might take a while)..."
        uv run python run_pysr.py --problem "$clean_prob" --dataset "$DATA_FILE" --run_id "$run_id" > "results/pysr_${prob}_run${run_id}.log" 2>&1
        echo "      Saved PySR output to results/pysr_${prob}_run${run_id}.log"
    done
done

echo " "
echo "=========================================================="
echo "    SOTA BENCHMARK COMPLETE                               "
echo "    Check the 'sota/results/' directory for the models.   "
echo "=========================================================="
