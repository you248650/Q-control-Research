#!/bin/bash

dim=16
date=$(date +"%Y.%m.%d")

handle_error() {
    echo "[Error] Encountered an error. Removing Result/RL4Quantum directory and retrying..."
    rm -rf Result/RL4Quantum
    rm -rf Notes/{$date}
    sh run.sh
}

handle_success() {
    echo "[Success] Execution completed without errors."
}


python3 Double_harmonic_oscillator_model/main.py --dim $dim --date $date && \
python3 Plot_Result/density_matrix_plot.py --dim $dim --date $date && \
python3 Plot_Result/reward_plot.py --date $date


if [ $? -ne 0 ]; then
    handle_error
else
    handle_success
fi


if [ -d "/home/RL4Quantum/Notes/{$date}/" ]; then
    PID_nums=$(ls -1 /home/RL4Quantum/Notes/{$date}/Result_rho_PID*.png 2>/dev/null | awk -F 'PID' '{print $2}' | awk -F '.' '{print $1}' | sort -n | tail -1)

    if [ -n "$PID_nums" ]; then
        for i in $(seq 0 $PID_nums); do
            eog "/home/RL4Quantum/Notes/{$date}/Result_rho_PID${i}.png"
        done

        for i in $(seq 0 $PID_nums); do
            eog "/home/RL4Quantum/Notes/{$date}/Result_reward_PID${i}.png"
        done
    else
        echo "No image files found."
    fi
fi