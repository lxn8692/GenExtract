#!/bin/bash
# ============================================================================
# Three-Agent Event Extraction Pipeline
# Detector + Extractor + Revisor
#
# Usage:
#   bash run_three_agent.sh
#
# This script runs the three-agent pipeline on the RAMS dataset.
# Modify the paths below to match your environment.
# ============================================================================

set -e

# --- Model paths (modify these to match your environment) ---
GEN_PLM_PATH="Meta-Llama-3-8B"          # Generator & default Revisor LLM
EXT_PLM_PATH="facebook/bart-large"       # Detector & Extractor base model
REV_PLM_PATH="Meta-Llama-3-8B"          # Revisor LLM (can be different from Generator)

# --- Dataset: RAMS → RAMS ---
TASK_NAME="rams2rams"
PATH_TRAIN="dataset/rams2rams/train.jsonl"
PATH_DEV="dataset/rams2rams/dev.jsonl"
PATH_TEST="dataset/rams2rams/test.jsonl"
SEEN_META="dataset/rams2rams/meta_seen.json"
UNSEEN_META="dataset/rams2rams/meta_unseen.json"
ONTOLOGY="dataset/rams2rams/ontology.csv"
UNSEEN_LABELS="data/rams2rams/generator_unseen_label_rams2rams_code.json"
SAVE_DIR="experiments/three_agent/${TASK_NAME}"

# --- Training config ---
SEED=42
NUM_ITER=5
NUM_GEN=100
BLOCK_SIZE=512
PSEUDO_EMPTY_RATIO=0.1
PSEUDO_EMPTY_THRESHOLD=2

echo "=============================================="
echo "  Three-Agent Event Extraction Pipeline"
echo "  Task: ${TASK_NAME}"
echo "  Iterations: ${NUM_ITER}"
echo "  Save dir: ${SAVE_DIR}"
echo "=============================================="

python pipeline_three_agent.py \
    --path-train "${PATH_TRAIN}" \
    --path-dev "${PATH_DEV}" \
    --path-test "${PATH_TEST}" \
    --seen-meta "${SEEN_META}" \
    --unseen-meta "${UNSEEN_META}" \
    --ontology-dict-path "${ONTOLOGY}" \
    --ontology-test-dict-path "${ONTOLOGY}" \
    --generator-unseen-label-path "${UNSEEN_LABELS}" \
    --save-dir "${SAVE_DIR}" \
    --seed ${SEED} \
    --num-iter ${NUM_ITER} \
    --task-name "${TASK_NAME}" \
    --gen-plm-path "${GEN_PLM_PATH}" \
    --ext-plm-path "${EXT_PLM_PATH}" \
    --rev-plm-path "${REV_PLM_PATH}" \
    --num-gen-per-label ${NUM_GEN} \
    --block-size ${BLOCK_SIZE} \
    --pseudo-empty-ratio ${PSEUDO_EMPTY_RATIO} \
    --pseudo-empty-threshold ${PSEUDO_EMPTY_THRESHOLD}

echo "=============================================="
echo "  Pipeline Complete!"
echo "=============================================="
