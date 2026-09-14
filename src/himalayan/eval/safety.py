"""Layered safety for any closed-loop run on the real rig (plan §9, step 5).

Three independent layers, deliberately redundant:
1. max_relative_target clamp (same one active during teleop) -- lives in the
   robot driver itself, not here; confirm it's wired via rig1.yaml.
2. A joint-delta sanity clamp here in SafetyGate, independent of the driver's
   own clamping -- catches a bad policy output before it reaches the motors.
3. A hard per-episode time limit + explicit kill switch -- see KillSwitch,
   reuses LeRobot's record-loop keyboard-listener pattern and works entirely
   locally even if the network link to the policy server drops.

Test all three deliberately (with a deliberately bad policy output) before
any closed-loop run happens in front of people -- see plan's Verification
section, "Safety".
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np
import yaml


@dataclass
class SafetyGate:
    max_joint_delta: dict[str, float]  # per-joint max |delta| per control tick, degrees

    @classmethod
    def from_rig_config(cls, rig_config_path: str) -> "SafetyGate":
        with open(rig_config_path) as f:
            cfg = yaml.safe_load(f)
        clamp = cfg["follower"]["max_relative_target"]
        # Single scalar in rig1.yaml applies uniformly; override per-joint here if needed.
        return cls(max_joint_delta={"_default": clamp})

    def clamp(self, action: dict[str, float], observation: dict[str, float]) -> dict[str, float]:
        """Clamp `action` so no joint moves more than max_joint_delta from its
        current position in `observation`, independent of any clamp already
        applied inside the robot driver.
        """
        default = self.max_joint_delta.get("_default", 15.0)
        clamped = {}
        for joint, target in action.items():
            if joint not in observation:
                clamped[joint] = target
                continue
            current = observation[joint]
            limit = self.max_joint_delta.get(joint, default)
            delta = np.clip(target - current, -limit, limit)
            clamped[joint] = current + delta
        return clamped


class KillSwitch:
    """Local, network-independent kill switch + per-episode watchdog.

    Usage:
        kill_switch = KillSwitch(max_episode_s=60)
        kill_switch.arm()
        while not kill_switch.triggered:
            ...control loop tick...
        kill_switch.disarm()
    """

    def __init__(self, max_episode_s: float, key: str = "k") -> None:
        self.max_episode_s = max_episode_s
        self.key = key
        self.triggered = False
        self._start_time: float | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop_listener = threading.Event()

    def arm(self) -> None:
        self.triggered = False
        self._start_time = time.time()
        self._stop_listener.clear()
        self._listener_thread = threading.Thread(target=self._listen_for_key, daemon=True)
        self._listener_thread.start()

    def disarm(self) -> None:
        self._stop_listener.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=1.0)

    def check_watchdog(self) -> bool:
        """Call once per control-loop tick. Returns True (and sets triggered)
        if the per-episode time limit has been exceeded.
        """
        if self._start_time is not None and time.time() - self._start_time > self.max_episode_s:
            self.triggered = True
        return self.triggered

    def _listen_for_key(self) -> None:
        # Reuses LeRobot's record-loop keyboard-listener pattern -- swap in
        # the fork's actual listener implementation (e.g. `pynput` or
        # LeRobot's own `KeyboardListener`) rather than reimplementing here.
        try:
            from pynput import keyboard

            def on_press(k):
                try:
                    if k.char == self.key:
                        self.triggered = True
                except AttributeError:
                    pass

            with keyboard.Listener(on_press=on_press) as listener:
                while not self._stop_listener.is_set() and not self.triggered:
                    time.sleep(0.05)
                listener.stop()
        except ImportError:
            print("pynput not installed -- kill switch keyboard listener disabled. "
                  "`pip install pynput` or wire in LeRobot's own listener before any real run.")
