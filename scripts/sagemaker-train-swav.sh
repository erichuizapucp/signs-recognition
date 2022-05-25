#!/bin/bash

# shellcheck disable=SC2116
export WORK_DIR=$(echo "$PWD")
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src/main"
export TF_GPU_ALLOCATOR=cuda_malloc_async
#export TF_FORCE_GPU_ALLOW_GROWTH=true

PARAMS_FILE='/opt/ml/input/config/hyperparameters.json'
SCRIPT_PATH='src/main/learning/train_model.py'

MODEL='swav'
TRAIN_DATASET_PATH='/opt/ml/input/data/training'
BATCH_SIZE=$(jq -r '.batch_size' "${PARAMS_FILE:-64}")
PERSON_DETECTION_MODEL_NAME='centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
PERSON_DETECTION_CHECKOUT_PREFIX='ckpt-0'
NO_REPLICAS=$(jq -r '.no_replicas' "${PARAMS_FILE:-4}")
DETECT_PERSON=$(jq -r '.detect_person' "${PARAMS_FILE:-'True'}")
MIRRORED_TRAINING=$(jq -r '.mirrored_training' "${PARAMS_FILE:-'True'}")

python $SCRIPT_PATH \
        --model $MODEL \
        --train_dataset_path $TRAIN_DATASET_PATH \
        --batch_size "$BATCH_SIZE" \
        "$(if [ "$DETECT_PERSON" = 'True' ]; then echo '--detect_person'; fi)" \
        --person_detection_model_name $PERSON_DETECTION_MODEL_NAME \
        --person_detection_checkout_prefix $PERSON_DETECTION_CHECKOUT_PREFIX \
        "$(if [ "$MIRRORED_TRAINING" = 'True' ]; then echo '--mirrored_training'; fi)" \
        --no_replicas "$NO_REPLICAS"
