#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

RESULT_DIR="./results/biovid_ebcap_methodology_bs16"
mkdir -p "${RESULT_DIR}"

python online_tta_ebm.py \
  --config ./configs \
  --dataset biovid \
  --target-root /projets/AT46120/BioVid_Video \
  --source-checkpoint ./checkpoints/biovid_source/source_best.pt \
  --target-subjects 1,2,3,4,5,6,7,8,9,10 \
  --device cuda \
  --shuffle-within-subject \
  --stream-seed 42 \
  --batch-size 16 \
  --clip-len 16 \
  --frame-stride 1 \
  --sgld-steps 20 \
  --sgld-step-size 0.01 \
  --sgld-noise-scale 0.10 \
  --ebm-samples-per-class 3 \
  --positive-capacity 5 \
  --negative-capacity 4 \
  --ebm-logit-weight 1.0 \
  --positive-logit-weight 1.0 \
  --negative-logit-weight 1.0 \
  --entropy-alpha 1.0 \
  --entropy-beta 1.0 \
  --entropy-warmup-windows 5 \
  --entropy-fallback-tau-positive 0.5 \
  --entropy-fallback-tau-negative 0.8 \
  --save-metrics "${RESULT_DIR}/metrics.json" \
  --save-metrics-txt "${RESULT_DIR}/metrics.txt"
