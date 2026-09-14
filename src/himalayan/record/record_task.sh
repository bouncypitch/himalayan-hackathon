#!/usr/bin/env bash
# Wraps lerobot-record for the core task (plan §2/§3), against the
# makermods-robotics/lerobot fork, branch robot/makermods-maker-arm.
#
# Usage: record_task.sh <num_episodes> [episode_time_s] [reset_time_s]

set -euo pipefail

# NOTE: verify exact flag names against `lerobot-record --help` on the
# makermods-robotics fork at kickoff (plan §2) — CLI flags on a community
# fork can drift from upstream LeRobot's documented flags.

NUM_EPISODES="${1:?usage: record_task.sh <num_episodes> [episode_time_s] [reset_time_s]}"
EPISODE_TIME_S="${2:-30}"
RESET_TIME_S="${3:-10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_YAML="${REPO_ROOT}/configs/tasks/pick_clip_place_mug.yaml"
RIG_YAML="${REPO_ROOT}/configs/rig/rig1.yaml"

INSTRUCTION="Pick up the clip and place it in the mug"
HF_USER="${HF_USER:?set HF_USER in your environment}"
REPO_ID="${HF_USER}/himalayan-clip-mug-raw"

echo "Recording task: ${INSTRUCTION}"
echo "Rig config:     ${RIG_YAML}"
echo "Dataset repo:   ${REPO_ID}"

lerobot-record \
  --robot.type=maker_follower \
  --robot.config_path="${RIG_YAML}" \
  --teleop.type=rebot_102_leader \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.single_task="${INSTRUCTION}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false
