#!/bin/bash

# shellcheck disable=SC2116
export WORK_DIR=$(echo "$PWD")
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src/main"
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

SCRIPT_PATH='src/main/learning/train_model.py'
MODEL='swav'
TRAIN_DATASET_PATH='dataset'
BATCH_SIZE=1
PERSON_DETECTION_MODEL_NAME='centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
PERSON_DETECTION_CHECKOUT_PREFIX='ckpt-0'
NO_REPLICAS=2

python $SCRIPT_PATH \
        --model $MODEL \
        --train_dataset_path $TRAIN_DATASET_PATH \
        --batch_size $BATCH_SIZE \
        --detect_person \
        --person_detection_model_name $PERSON_DETECTION_MODEL_NAME \
        --person_detection_checkout_prefix $PERSON_DETECTION_CHECKOUT_PREFIX \
        --mirrored_training \
        --no_replicas $NO_REPLICAS
