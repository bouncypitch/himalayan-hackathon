# HIL-SERL specialist — rope-pull-and-hold (plan §6)

Scoped narrowly to the rope-pull-and-hold behavior only. Not a from-scratch
implementation — use LeRobot's own HIL-SERL actor/learner pipeline
(`lerobot/scripts/rl/` on the fork, or wherever it lives on
`robot/makermods-maker-arm` — check at kickoff) rather than reimplementing
SAC here.

## Plan for this directory

1. **Reward classifier**: train a binary success/fail classifier off labels
   on already-recorded rope-pull episodes (from the detector calibration
   clips, `configs/tasks/rope_pull_hold.yaml`) to avoid live-labeling
   overhead. A short script here (`train_reward_classifier.py`, not yet
   written) should wrap whatever classifier-training entrypoint the fork
   exposes.
2. **Actor/learner config**: a config here (`hilserl_config.yaml`, not yet
   written) pointing the fork's HIL-SERL actor/learner scripts at
   `configs/rig/rig1.yaml` and a gamepad device, scoped to short-horizon,
   single-arm, binary-success episodes only.
3. **Real hardware only** — no `gym-hil` simulation step (plan §6's
   reasoning: unproven Maker-arm sim integration + rope-in-sim tuning cost
   for no net benefit).

Do not start this until the core-task checkpoint (`src/himalayan/train`) is
already demo-able end-to-end, and after the DAgger pass
(`src/himalayan/refine`) if both are being done — see the priority ordering
in plan §6/§6b and the Build Order.

**Fallback**: if this doesn't get built in time, the frozen pi0.5
reactive-grasp response (the scripted controller in
`src/himalayan/eval/run_policy_with_pull_interrupt.py`,
`rope_pull_hold_controller`) is the shipped behavior — this directory is
purely additive.
