"""Main closed-loop demo runner (plan §9). Runs on the laptop as a thin
client: reads rig + camera observations, sends them to a remote policy
server over websocket, executes the returned action chunk locally with RTC
blending, and runs the rope-pull detector locally each tick as a parallel
interrupt.

NOTE: the exact client API (make_robot_from_config, PolicyClient,
select_action signature, RTC blending helper) is a best guess at the fork's
and LeRobot's remote-inference interface -- verify against the fork's own
serve_policy.py / lerobot-rollout --inference.type=rtc equivalent
(plan §2, §9) before running for real.

Usage:
    python run_policy_with_pull_interrupt.py \
        --rig-config ../../../configs/rig/rig1.yaml \
        --rope-pull-config ../../../configs/tasks/rope_pull_hold.yaml \
        --server-url ws://<hf-jobs-l4-host>:8000 \
        --max-episode-s 90
"""

import argparse
import time

import yaml

from lerobot.robots import make_robot_from_config  # verify path against the fork
from lerobot.policies.client import PolicyClient  # thin websocket client, plan §9

from himalayan.vision.rope_pull_detector import RopePullDetector, DetectorState
from himalayan.eval.safety import SafetyGate, KillSwitch
from himalayan.voice.asr_bridge import AsrBridge, CANONICAL_INSTRUCTION
from himalayan.voice.phrases import ROPE_PULL_DETECTED, TASK_COMPLETE


def rope_pull_hold_controller(robot, observation) -> dict:
    """Scripted 'close gripper, hold pose' controller (plan §4) -- holds the
    current joint positions and closes the gripper. Replace `gripper` key
    name with whatever the fork's action dict actually calls it.
    """
    action = dict(observation)  # hold current pose
    action["gripper"] = 1.0  # fully closed; confirm convention against the driver
    return action


def run(
    rig_config_path: str,
    rope_pull_config_path: str,
    server_url: str,
    max_episode_s: float,
    voice_announcer=None,  # optional callable(text) -> None, wired to the LiveKit agent (plan §7)
) -> None:
    robot = make_robot_from_config(rig_config_path)
    policy_client = PolicyClient(server_url)
    safety = SafetyGate.from_rig_config(rig_config_path)
    detector = RopePullDetector.from_yaml(rope_pull_config_path)
    with open(rope_pull_config_path) as f:
        detector_camera_key = yaml.safe_load(f)["camera_key"]

    asr_bridge = AsrBridge(on_task_start=voice_announcer)
    asr_bridge.current_prompt = CANONICAL_INSTRUCTION  # single core task, plan §7

    kill_switch = KillSwitch(max_episode_s=max_episode_s)

    robot.connect()
    kill_switch.arm()
    try:
        while not kill_switch.triggered and not kill_switch.check_watchdog():
            observation = robot.get_observation()

            if detector_camera_key in observation:
                prev_state = detector.state
                new_state = detector.update(observation[detector_camera_key])
                if new_state == DetectorState.PULL_DETECTED and prev_state != DetectorState.PULL_DETECTED:
                    if voice_announcer is not None:
                        voice_announcer(ROPE_PULL_DETECTED)

            if detector.state in (DetectorState.PULL_DETECTED, DetectorState.HOLDING):
                action = rope_pull_hold_controller(robot, observation)
                if detector.state == DetectorState.PULL_DETECTED:
                    detector.confirm_holding()
            else:
                action = policy_client.select_action(observation, task=asr_bridge.current_prompt)

            action = safety.clamp(action, observation)
            robot.send_action(action)

            time.sleep(1.0 / robot.fps)
    finally:
        kill_switch.disarm()
        robot.disconnect()

    if voice_announcer is not None:
        voice_announcer(TASK_COMPLETE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--rope-pull-config", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--max-episode-s", type=float, default=90.0,
                         help="Hard watchdog limit; the ~90s live demo script (plan §10) fits inside this.")
    args = parser.parse_args()
    run(args.rig_config, args.rope_pull_config, args.server_url, args.max_episode_s)


if __name__ == "__main__":
    main()
