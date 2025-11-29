#!/bin/bash

# This script runs ensembling experiments on top k models for k=1 to N

# ================= CONFIGURATION =================

GPUS=(0 1)

PYTHON_SCRIPT="llm_ensemble.py" 
BATCH_SIZE=32
LOG_OUTPUT_DIRECTORY="ensemble-model-logs"
mkdir -p $LOG_OUTPUT_DIRECTORY

CANDIDATE_LIST=(
    "../../nav/model_training/trained_models/baseline_rnn4/baseline_rnn_val_predicted_sentences_20251125_050218.csv" # BiGRU original data aug
    "trained_models/data_aug_models/conv_w_data_aug_w_512_d_8/baseline_rnn_val_predicted_sentences_20251129_152458.csv" # ConFormer tuned data aug
    "trained_models/relative_w_512_d_8/baseline_rnn_val_predicted_sentences_20251127_181955.csv" # RelConformer original data aug
    "trained_models/data_aug_models/conv_w_data_aug_w_512_d_8/baseline_rnn_val_predicted_sentences_20251129_152458.csv" # Relative Conformer tuned data aug
)

# =================================================

cleanup() {
    echo ""
    echo "Killing all background Python jobs..."
    kill $(jobs -p) 2>/dev/null
    echo "Done."
    exit 1
}

# Trap SIGINT (Ctrl+C) and SIGTERM (kill command)
trap cleanup SIGINT SIGTERM

NUM_GPUS=${#GPUS[@]}
GPU_IDX=0

TOTAL_MODELS=${#CANDIDATE_LIST[@]}

echo "Found $TOTAL_MODELS candidate models."
echo "Distributing jobs across ${NUM_GPUS} GPUs: ${GPUS[*]}"

# Run on first one to make sure our setup is correct
for k in $(seq 1 $TOTAL_MODELS); do
    
    CURRENT_INPUTS=("${CANDIDATE_LIST[@]:0:k}")
    
    OUTPUT_NAME="ensemble_top_${k}_models.csv"
    
    CURRENT_GPU=${GPUS[$GPU_IDX]}
    
    echo "------------------------------------------------"
    echo "Launching job on GPU $CURRENT_GPU: Ensembling top $k models..."

    LOG_FILE="$LOG_OUTPUT_DIRECTORY/ensemble_top_${k}_models.log"
    OUTPUT_NAME="ensemble_top_${k}_models_validation.csv"
    
    CUDA_VISIBLE_DEVICES=$CURRENT_GPU python -u $PYTHON_SCRIPT \
        --candidates ${CURRENT_INPUTS[@]} \
        --output $OUTPUT_NAME \
        > "$LOG_FILE" 2>&1 &

    # Get next gpu index
    GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
    
    # This block waits if the number of running background jobs equals the number of GPUs.
    while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
        sleep 5
    done

done

echo "------------------------------------------------"
echo "All jobs launched. Waiting for completion..."
wait
echo "All experiments finished."
echo "------------------------------------------------"