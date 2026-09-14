# Himalayan

Track 2 ("Action") submission for the Himalaya Robotics Hackathon.

A single-rig Maker-arm robot system that interacts with mountaineering gear —
a carabiner clip, a mug, and a rope — combining two behaviors into one demo:

1. **Voice-triggered manipulation**: say "Pick up the clip and place it in
   the mug" and a π0.5 (Physical Intelligence) VLA policy, fine-tuned on this
   hardware, executes it — then speaks back what it's doing.
2. **Reactive rope-pull-and-hold**: a vision-based detector notices when the
   rope is pulled and the arm grabs and holds it — a tabletop analog of a
   belay/fall-arrest response, framed for extreme-condition relevance.

## Why this scope

We have exactly **one physical rig** (a Maker follower arm + ReBot 102
leader, community LeRobot fork) for the whole ~31.5-hour hackathon window.
Rather than spread that rig-time across many tasks, we deliberately narrowed
to these two behaviors so that voice I/O, a live camera viewer, DAgger
refinement, and an HIL-SERL RL specialist could all be built to a genuinely
demo-ready standard instead of half-finished across a longer task list.

## Stack

- **Hardware**: Maker follower + ReBot 102 leader, on
  [`makermods-robotics/lerobot`](https://github.com/makermods-robotics/lerobot),
  branch `robot/makermods-maker-arm`.
- **Policy**: [π0.5](https://www.physicalintelligence.company/) fine-tuned
  via LeRobot's native integration (not `openpi` directly), supervised
  fine-tuning on real teleop demonstrations, `train_expert_only` +
  `freeze_vision_encoder`.
- **Refinement**: one round of DAgger on the core task
  (`src/himalayan/refine`); a scoped HIL-SERL specialist for the rope-pull
  behavior (`src/himalayan/rl/rope_pull_hilserl`).
- **Voice**: [LiveKit](https://livekit.io/) agent, full duplex — STT for
  task triggering, TTS for status narration (`src/himalayan/voice`).
- **Compute**: [Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs)
  — A100-large for fine-tuning, L4 for the live-demo inference server.
- **Inference architecture**: laptop as a thin client (no local GPU),
  remote policy server over websocket, Real-Time Chunking for latency
  absorption (`src/himalayan/eval`).

## Repo layout

```
configs/            rig, camera, and task configuration
src/himalayan/
  record/            lerobot-record wrapper for the core task
  datasets/          dataset merge/aggregation
  vision/
    rope_pull_detector/   optical-flow reactive trigger
    camera_viewer/         live wrist+side camera viewer
  train/             lerobot-train wrapper (HF Jobs)
  refine/            DAgger refinement pass
  rl/rope_pull_hilserl/  scoped HIL-SERL specialist
  voice/             LiveKit agent, ASR bridge, TTS phrases
  sim/               optional MuJoCo stretch goal (not core scope)
  eval/              main closed-loop demo runner + safety
data/, outputs/      gitignored — local datasets and checkpoints
```

## Running things

See the `--help` / module docstrings in each script under `src/himalayan/`
for usage. Several scripts have `NOTE:` comments flagging API surface that
was written against best-available knowledge of the fork/LiveKit/LeRobot
APIs and needs verification against the actual fork at kickoff, since this
stack moves fast and a community fork can drift from documented interfaces.

Full project plan, rationale, and build order: see the team's planning
notes (not checked into this repo).
