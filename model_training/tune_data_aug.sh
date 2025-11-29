#!/bin/bash
GPUS=(1 3)

trap 'kill $(jobs -p)' SIGINT

# Loop and launch
for gpu in "${GPUS[@]}"; do
    echo "Launching worker on GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python aug_param_tuning.py > "aug_param_worker_gpu${gpu}.log" 2>&1 &
done

echo "All workers started. Tail the logs or check Optuna dashboard. nvidia-smi and kill -9 processes as needed."