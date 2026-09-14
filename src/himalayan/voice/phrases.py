"""Templated TTS phrases (plan §7) -- deliberately not LLM-generated, since
the task set is fixed and known in advance. Map current task/state directly
to a canned phrase and pass it straight to TTS.
"""

TASK_START = "Picking up the clip and placing it in the mug."
ROPE_PULL_DETECTED = "Rope pull detected. Grabbing."  # keep in sync with configs/tasks/rope_pull_hold.yaml's announcement_on_trigger
ROPE_PULL_HOLDING_RELEASED = "Rope released. Resuming."
TASK_COMPLETE = "Done."
TASK_FAILED = "Task failed. Resetting."
