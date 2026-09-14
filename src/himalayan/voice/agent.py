"""LiveKit voice agent, bidirectional: STT (mic) -> AsrBridge.current_prompt,
and speaker output (state changes -> templated phrase -> TTS). One agent for
both directions per plan §7, to qualify for the Voice-Use and LiveKit
challenges with a genuinely full-duplex entry rather than input-only.

NOTE: this is a skeleton against LiveKit's agents framework API surface as of
research time -- verify current class/method names against
https://docs.livekit.io/agents/ before running; the framework has moved fast.
The structure (STT callback -> AsrBridge, state changes -> TTS say()) is what
matters and should carry over even if exact API names have shifted.

Usage:
    python agent.py start
"""

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai as livekit_openai  # or whichever STT/TTS provider is configured

from .asr_bridge import AsrBridge
from .phrases import ROPE_PULL_DETECTED


class HimalayanVoiceAgent(Agent):
    """Wires STT transcripts into AsrBridge.current_prompt, and exposes
    `announce_rope_pull()` / `announce(text)` for the eval loop (plan §9) to
    call when the rope-pull detector fires or a task starts.
    """

    def __init__(self, asr_bridge: AsrBridge) -> None:
        super().__init__(instructions="Relay task instructions and status for the Himalayan robot demo.")
        self.asr_bridge = asr_bridge

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        if is_final:
            self.asr_bridge.handle_transcript(transcript)

    async def announce(self, text: str) -> None:
        await self.session.say(text)

    async def announce_rope_pull(self) -> None:
        await self.announce(ROPE_PULL_DETECTED)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    def on_task_start(text: str) -> None:
        # Fire-and-forget announce; the eval loop (plan §9) calls
        # asr_bridge.handle_transcript() -> this callback synchronously, so
        # keep it non-blocking here rather than awaiting directly.
        import asyncio
        asyncio.create_task(agent.announce(text))

    asr_bridge = AsrBridge(on_task_start=on_task_start)
    agent = HimalayanVoiceAgent(asr_bridge)

    session = AgentSession(
        stt=livekit_openai.STT(),
        tts=livekit_openai.TTS(),
    )
    await session.start(agent=agent, room=ctx.room)

    # Expose asr_bridge to the rest of the process (e.g. the eval loop
    # running in the same process, or via a small local IPC/queue if the
    # eval loop runs separately) -- wire this up during Day 1 integration.
    ctx.asr_bridge = asr_bridge  # type: ignore[attr-defined]


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
