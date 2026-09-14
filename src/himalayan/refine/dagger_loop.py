"""One-round DAgger refinement pass on the core task's checkpoint (plan §6b).

Sequence: roll out the current fine-tuned policy on the real rig, have a
human take over via the leader arm the moment it's about to fail, log the
corrective action at that state, aggregate into a new dataset, retrain.

This is a *recording* helper, not a fully automated loop -- the human decides
when to intervene by physically grabbing the leader arm, same as during
lerobot-record teleop. What this script adds over plain lerobot-record is:
running the policy client-side between interventions (so episodes start from
policy-visited states, not from scratch), and tagging the resulting dataset
distinctly from the base recordings so it can be merged in via
merge_datasets.py without confusion.

Usage:
    python dagger_loop.py --checkpoint outputs/pi05_clip_mug --server-url ws://<server>:8000 \
        --rig-config ../../../configs/rig/rig1.yaml --num-episodes 10 \
        --output-repo {HF_USER}/himalayan-clip-mug-dagger-round1

NOTE: the exact API surface below (teleop.is_being_actuated(), robot.fps,
make_policy_client, etc.) is a best guess at the fork's interface, not
verified against actual source -- confirm against
github.com/makermods-robotics/lerobot (branch robot/makermods-maker-arm) and
LeRobot's own record/eval scripts before running this for real, and adjust
names as needed. The control flow (policy runs, human interventions logged
as corrections, single episode buffer) is the part that matters.
"""

import argparse

from lerobot.robots import make_robot_from_config  # verify exact import path against the fork
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy_client  # thin client, plan §9

from himalayan.eval.safety import SafetyGate


TASK_INSTRUCTION = "Pick up the clip and place it in the mug"


def run_dagger_round(
    rig_config_path: str,
    server_url: str,
    output_repo_id: str,
    num_episodes: int,
    episode_time_s: float,
) -> None:
    robot = make_robot_from_config(rig_config_path)
    teleop = make_teleoperator_from_config(rig_config_path)  # leader arm, for human takeover
    policy_client = make_policy_client(server_url)
    safety = SafetyGate.from_rig_config(rig_config_path)

    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=robot.fps,
        features=robot.observation_features | robot.action_features,
    )

    robot.connect()
    try:
        for episode_idx in range(num_episodes):
            print(f"Episode {episode_idx + 1}/{num_episodes} -- policy running, "
                  "grab the leader arm to intervene on an impending failure.")
            dataset.start_episode(task=TASK_INSTRUCTION)
            elapsed = 0.0
            while elapsed < episode_time_s:
                observation = robot.get_observation()
                if teleop.is_being_actuated():  # human has taken over
                    action = teleop.get_action()
                else:
                    action = policy_client.select_action(observation, task=TASK_INSTRUCTION)
                action = safety.clamp(action, observation)
                robot.send_action(action)
                dataset.add_frame(observation, action)
                elapsed += 1.0 / robot.fps
            dataset.save_episode()
    finally:
        robot.disconnect()

    print(f"Recorded {num_episodes} DAgger episodes to {output_repo_id}.")
    print("Next: merge_datasets.py this in with the base dataset, then re-run train_pi05.sh.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-time-s", type=float, default=30.0)
    args = parser.parse_args()
    run_dagger_round(
        args.rig_config, args.server_url, args.output_repo,
        args.num_episodes, args.episode_time_s,
    )


if __name__ == "__main__":
    main()
