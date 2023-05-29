#!/bin/bash

# shellcheck disable=SC2116
export WORK_DIR=$(echo "$PWD")
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src/main"
export TF_CPP_MIN_LOG_LEVEL=3
#export TF_FORCE_GPU_ALLOW_GROWTH=true

SCRIPT_PATH='src/main/learning/train_model.py'
PARAMS_FILE=${PARAMS_FILE}

MODEL=${MODEL}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH}
MODEL_OUTPUT_PATH=${MODEL_OUTPUT_PATH}
CHECKPOINT_OUTPUT_PATH=${CHECKPOINT_OUTPUT_PATH}
FAILURE_REASON_PATH=${FAILURE_REASON_PATH}

python $SCRIPT_PATH --model "$MODEL" \
        --train_dataset_path "$TRAIN_DATASET_PATH" \
        --model_storage_path "$MODEL_OUTPUT_PATH" \
        --checkpoint_storage_path "$CHECKPOINT_OUTPUT_PATH" \
        --failure_reason_path "$FAILURE_REASON_PATH" \
        $(jq -r 'to_entries[] | "--\(.key)=\(.value)"' "${PARAMS_FILE}")
