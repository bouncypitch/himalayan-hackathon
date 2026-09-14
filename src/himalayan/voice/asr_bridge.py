"""Forwards a LiveKit STT transcript to `current_prompt` for the eval loop
(plan §7, §9 step 3). Kept as a thin, swappable piece so the eval loop
doesn't need to know anything about LiveKit directly.
"""

from __future__ import annotations

import difflib

from .phrases import TASK_START

CANONICAL_INSTRUCTION = "Pick up the clip and place it in the mug"


def normalize_transcript(transcript: str, canonical: str = CANONICAL_INSTRUCTION,
                          similarity_threshold: float = 0.5) -> str | None:
    """Light transcript-to-canonical-phrase normalization (plan §7, "nice to
    have, not required"). Returns the canonical instruction if the transcript
    is plausibly a request for the core task, else None (caller should ignore
    unrecognized speech rather than passing it straight to the policy).
    """
    ratio = difflib.SequenceMatcher(None, transcript.lower(), canonical.lower()).ratio()
    if ratio >= similarity_threshold:
        return canonical
    return None


class AsrBridge:
    """Owns `current_prompt`, updated from LiveKit transcripts, read by the
    eval loop's closed-loop tick (plan §9). `on_task_start` fires the
    templated TTS announcement via the LiveKit agent's speaker output.
    """

    def __init__(self, on_task_start=None) -> None:
        self.current_prompt: str | None = None
        self._on_task_start = on_task_start

    def handle_transcript(self, transcript: str) -> None:
        instruction = normalize_transcript(transcript)
        if instruction is None:
            return
        is_new_task = instruction != self.current_prompt
        self.current_prompt = instruction
        if is_new_task and self._on_task_start is not None:
            self._on_task_start(TASK_START)
