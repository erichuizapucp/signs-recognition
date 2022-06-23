#!/bin/bash

# shellcheck disable=SC2116
export WORK_DIR=$(echo "$PWD")
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src/main"
export TF_GPU_ALLOCATOR=cuda_malloc_async

SCRIPT_PATH='src/main/learning/train_model.py'

PARAMS_FILE='/opt/ml/input/config/hyperparameters.json'
BATCH_SIZE=$(jq -r '.batch_size' "${PARAMS_FILE:-64}")
NO_REPLICAS=$(jq -r '.no_replicas' "${PARAMS_FILE:-4}")
DETECT_PERSON=$(jq -r '.detect_person' "${PARAMS_FILE:-'True'}")
MIRRORED_TRAINING=$(jq -r '.mirrored_training' "${PARAMS_FILE:-'True'}")
NO_EPOCHS=$(jq -r '.no_epochs' "${PARAMS_FILE:-5}")

MODEL='fake_swav'
TRAIN_DATASET_PATH='/opt/ml/input/data/training'
MODEL_OUTPUT_PATH='/opt/ml/model'
CHECKPOINT_OUTPUT_PATH='/opt/ml/checkpoint'
FAILURE_REASON_PATH='/opt/ml/output/failure'
PERSON_DETECTION_MODEL_NAME='centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
PERSON_DETECTION_CHECKOUT_PREFIX='ckpt-0'

python $SCRIPT_PATH \
        --model $MODEL \
        --train_dataset_path $TRAIN_DATASET_PATH \
        --model_storage_path $MODEL_OUTPUT_PATH \
        --checkpoint_storage_path $CHECKPOINT_OUTPUT_PATH \
        --failure_reason_path $FAILURE_REASON_PATH \
        --no_epochs "$NO_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        "$(if [ "$DETECT_PERSON" = 'True' ]; then echo '--detect_person'; fi)" \
        --person_detection_model_name $PERSON_DETECTION_MODEL_NAME \
        --person_detection_checkout_prefix $PERSON_DETECTION_CHECKOUT_PREFIX \
        "$(if [ "$MIRRORED_TRAINING" = 'True' ]; then echo '--mirrored_training'; fi)" \
        --no_replicas "$NO_REPLICAS"
