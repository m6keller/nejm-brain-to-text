#!/bin/bash

cleanup() {
    echo "Caught Interrupt. Killing all background processes..."
    kill $(jobs -p)
    exit 1
}

trap cleanup SIGINT

for i in 1 2  # Adjust as needed
do
   CUDA_VISIBLE_DEVICES=$i python hyperparam_tuning.py &
done

wait