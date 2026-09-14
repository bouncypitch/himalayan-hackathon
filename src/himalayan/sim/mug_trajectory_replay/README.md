# MuJoCo real-to-sim stretch goal (plan §8)

**Not core scope. Skip unless there's clear rig-time surplus** after the
core task, the DAgger pass, and rope-pull-hold work are all solid — see plan
§8 for the full reasoning (this competes with §3/§6b for the only physical
rig, and no MJCF model has been verified for the Maker-arm hardware yet).

If pursued: trajectory-level augmentation replayed on the real robot only —
never rendered synthetic images (domain-gap risk to the pretrained VLA's
visual grounding). Hard budget: 2–3 hours, go/no-go checkpoint at 90 minutes.

Nothing implemented here yet, deliberately — this directory is a placeholder
so the decision to skip or pursue is explicit rather than accidental.
