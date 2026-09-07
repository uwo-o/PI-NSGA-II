#!/bin/bash
# Barrido de robustez a ruido para el benchmark SOTA (PySR & PySINDy).
#
# Motivacion: PySR/PySINDy reciben hoy datos etiquetados PERFECTOS (el grid
# exacto/numerico exportado por PISR-NSGA-II) — en un entorno real esos datos
# vienen de mediciones con error. Este script corre cada EDP a varios niveles
# de ruido gaussiano relativo (0% = baseline limpio, para comparar de forma
# justa) y deja cada corrida en un archivo separado (metrics/log con sufijo
# _noiseNpct) para poder armar despues una curva de degradacion MSE vs ruido.
#
# Requisito: los archivos results/grid_<PDE>_PISR-EMOAD.csv deben existir ya
# (correr el binario ./build/PISR-NSGA-II para cada EDP primero) — este
# script NO corre PISR-NSGA-II, solo los baselines de comparacion sobre los
# datos que PISR-NSGA-II ya exporto.
#
# Uso:
#   ./sota_benchmark_noise.sh                                # todo, niveles default
#   ./sota_benchmark_noise.sh --noise_levels 0.0,0.05,0.10,0.20
#   ./sota_benchmark_noise.sh --pdes Airy_1D,Thomas-Fermi_1D  # subset rapido
#   ./sota_benchmark_noise.sh --runs 3                        # usa results/run_N/

cd "$(dirname "$0")"

RUNS=1
NOISE_LEVELS_CSV="0.0,0.05,0.10,0.20"
PDES_CSV=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --runs) RUNS="$2"; shift ;;
        --noise_levels) NOISE_LEVELS_CSV="$2"; shift ;;
        --pdes) PDES_CSV="$2"; shift ;;
        *) shift ;;
    esac
    shift
done
IFS=',' read -ra NOISE_LEVELS <<< "$NOISE_LEVELS_CSV"

ALL_PROBLEMS=(
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

if [ -n "$PDES_CSV" ]; then
    IFS=',' read -ra PROBLEMS <<< "$PDES_CSV"
else
    PROBLEMS=("${ALL_PROBLEMS[@]}")
fi

TOTAL=$(( ${#PROBLEMS[@]} * ${#NOISE_LEVELS[@]} * RUNS ))
echo "=========================================================="
echo "  SOTA NOISE-ROBUSTNESS BENCHMARK (PySR & PySINDy)"
echo "  EDPs: ${#PROBLEMS[@]}   Niveles de ruido: ${NOISE_LEVELS[*]}   Runs: $RUNS"
echo "  Total de combinaciones (cada una corre PySINDy + PySR): $TOTAL"
echo "  ATENCION: PySR tarda varios minutos por combinacion — esto puede"
echo "  tomar horas para la suite completa. Considera --pdes para probar"
echo "  un subconjunto primero."
echo "=========================================================="

mkdir -p results

for prob in "${PROBLEMS[@]}"; do
    for run_id in $(seq 1 "$RUNS"); do
        if [ "$RUNS" -gt 1 ]; then
            DATA_FILE="../results/run_${run_id}/grid_${prob}_PISR-EMOAD.csv"
        else
            DATA_FILE="../results/grid_${prob}_PISR-EMOAD.csv"
        fi

        if [ ! -f "$DATA_FILE" ]; then
            echo " "
            echo "[SKIP] $prob (run $run_id): no se encontro $DATA_FILE"
            echo "       -> corre primero: ./build/PISR-NSGA-II --only $prob --pop 150 --gen 150 --replicable"
            continue
        fi

        clean_prob=$(echo "$prob" | sed -E 's/_(1D|2D)//')

        for noise in "${NOISE_LEVELS[@]}"; do
            noise_pct_int=$(awk "BEGIN{printf \"%d\", $noise*100 + 0.5}")
            noise_tag="noise${noise_pct_int}pct"

            echo " "
            echo "----------------------------------------------------------"
            echo ">>> $prob  (run $run_id/$RUNS, ruido=${noise} -> ${noise_tag})"
            echo "----------------------------------------------------------"

            echo "[1/2] PySINDy..."
            uv run python run_pysindy.py --problem "$clean_prob" --dataset "$DATA_FILE" \
                --run_id "$run_id" --noise_pct "$noise" \
                > "results/sindy_${prob}_run${run_id}_${noise_tag}.log" 2>&1
            echo "      -> results/sindy_${prob}_run${run_id}_${noise_tag}.log"

            echo "[2/2] PySR (puede tardar varios minutos)..."
            uv run python run_pysr.py --problem "$clean_prob" --dataset "$DATA_FILE" \
                --run_id "$run_id" --noise_pct "$noise" \
                > "results/pysr_${prob}_run${run_id}_${noise_tag}.log" 2>&1
            echo "      -> results/pysr_${prob}_run${run_id}_${noise_tag}.log"
        done
    done
done

echo " "
echo "=========================================================="
echo "  SOTA NOISE BENCHMARK COMPLETE"
echo "  Metricas en: sota/results/[run_N/]<PDE>_{pysr,sindy}_metrics[_noiseNpct].csv"
echo "  (noise_pct=0.0 usa el nombre SIN sufijo, igual que antes — asi la"
echo "  version limpia queda intacta para comparar de forma justa)"
echo "=========================================================="
