#!/usr/bin/env bash
# Wraps lerobot-train for the core-task pi0.5 checkpoint (plan §5), submitted
# as an HF Jobs run on a 1x Nvidia A100-large (80GB, $2.50/hr).
#
# Usage: train_pi05.sh <dataset_repo_id> [output_dir] [timeout]

set -euo pipefail

DATASET_REPO_ID="${1:?usage: train_pi05.sh <dataset_repo_id> [output_dir] [timeout]}"
OUTPUT_DIR="${2:-outputs/pi05_clip_mug}"
TIMEOUT="${3:-6h}"

HF_USER="${HF_USER:?set HF_USER in your environment}"

echo "Submitting HF Jobs pi0.5 fine-tune run for ${DATASET_REPO_ID}"
echo "Remember: hf jobs cancel any run left over-time, or the $30/person credit bleeds fast."

hf jobs run \
  --flavor a100-large \
  --timeout "${TIMEOUT}" \
  huggingface/lerobot-gpu \
  lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="himalayan-clip-mug-pi05"

# NOTE: verify the exact `hf jobs run` image/flavor names and lerobot-train
# flags against current docs (huggingface.co/docs/hub/en/jobs,
# huggingface.co/docs/hub/en/jobs-pricing) at kickoff -- both move fast.
