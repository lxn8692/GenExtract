#!/bin/bash
SEED=42
NUM_ITER=5
BLOCK_SIZE=512
NUM_GEN_PER_LABEL=100
PSEUDO_EMPTY_RATIO=0.1
PSEUDO_EMPTY_THRESHOLD=2

TASK_NAME=wiki2wiki
DATA_DIR=dataset/${TASK_NAME}
GEN_LABEL_PATH=data/${TASK_NAME}/generator_unseen_label_${TASK_NAME}_code.json
SAVE_DIR=experiments/${TASK_NAME}/output_llama

GEN_PLM_PATH=Meta-Llama-3-8B
EXT_PLM_PATH=bart-large


CUDA_VISIBLE_DEVICES=0,1 python warpper_rl.py \
  --path-train ${DATA_DIR}/train.jsonl \
  --path-dev ${DATA_DIR}/dev.jsonl \
  --path-test ${DATA_DIR}/test.jsonl \
  --seen-meta ${DATA_DIR}/meta_seen.json \
  --unseen-meta ${DATA_DIR}/meta_unseen.json \
  --ontology-dict-path ${DATA_DIR}/ontology.csv \
  --ontology-test-dict-path ${DATA_DIR}/ontology.csv \
  --generator-unseen-label-path ${GEN_LABEL_PATH} \
  --save-dir ${SAVE_DIR} \
  --seed ${SEED} \
  --num-iter ${NUM_ITER} \
  --block-size ${BLOCK_SIZE} \
  --task-name ${TASK_NAME} \
  --num-gen-per-label ${NUM_GEN_PER_LABEL} \
  --pseudo-empty-ratio ${PSEUDO_EMPTY_RATIO} \
  --pseudo-empty-threshold ${PSEUDO_EMPTY_THRESHOLD} \
  --gen-plm-path ${GEN_PLM_PATH} \
  --ext-plm-path ${EXT_PLM_PATH} \
  --with-train \
  --by-rel
