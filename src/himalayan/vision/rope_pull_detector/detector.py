"""Rope-pull-and-hold reactive detector (plan §4).

Lightweight vision trigger, decoupled from the manipulation policy: wrist-camera
optical flow magnitude within a configured ROI, debounced over N consecutive
frames. Not learned end-to-end and not dependent on which policy (pi0.5 or the
HIL-SERL specialist, plan §6) is active downstream.

State machine: IDLE/TASK -> PULL_DETECTED -> HOLDING -> (reset) -> IDLE/TASK
"""

from dataclasses import dataclass
from enum import Enum, auto
import time

import cv2
import numpy as np
import yaml


class DetectorState(Enum):
    IDLE = auto()
    PULL_DETECTED = auto()
    HOLDING = auto()


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


class RopePullDetector:
    """Feed consecutive grayscale wrist-camera frames via `update()`.

    Call `update(frame)` once per control-loop tick. Returns the current
    DetectorState; check for a state transition to know when to preempt the
    active policy (plan §9, step 4).
    """

    def __init__(
        self,
        roi: ROI,
        flow_magnitude_threshold: float,
        debounce_frames: int,
        hold_timeout_s: float,
    ) -> None:
        self.roi = roi
        self.flow_magnitude_threshold = flow_magnitude_threshold
        self.debounce_frames = debounce_frames
        self.hold_timeout_s = hold_timeout_s

        self.state = DetectorState.IDLE
        self._prev_gray: np.ndarray | None = None
        self._consecutive_over_threshold = 0
        self._holding_since: float | None = None
        self.last_magnitude: float = 0.0

    @classmethod
    def from_yaml(cls, config_path: str) -> "RopePullDetector":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            roi=ROI(**cfg["roi"]),
            flow_magnitude_threshold=cfg["flow_magnitude_threshold"],
            debounce_frames=cfg["debounce_frames"],
            hold_timeout_s=cfg["hold_timeout_s"],
        )

    def _mean_flow_magnitude(self, gray: np.ndarray) -> float:
        if self._prev_gray is None:
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))

    def update(self, frame_bgr: np.ndarray) -> DetectorState:
        gray = cv2.cvtColor(self.roi.crop(frame_bgr), cv2.COLOR_BGR2GRAY)
        magnitude = self._mean_flow_magnitude(gray)
        self._prev_gray = gray
        self.last_magnitude = magnitude

        if self.state == DetectorState.HOLDING:
            if self._holding_since is not None and (
                time.time() - self._holding_since > self.hold_timeout_s
            ):
                self.reset()
            return self.state

        if magnitude >= self.flow_magnitude_threshold:
            self._consecutive_over_threshold += 1
        else:
            self._consecutive_over_threshold = 0

        if (
            self.state == DetectorState.IDLE
            and self._consecutive_over_threshold >= self.debounce_frames
        ):
            self.state = DetectorState.PULL_DETECTED

        return self.state

    def confirm_holding(self) -> None:
        """Call once the scripted 'close gripper, hold pose' controller has engaged."""
        self.state = DetectorState.HOLDING
        self._holding_since = time.time()

    def reset(self) -> None:
        self.state = DetectorState.IDLE
        self._consecutive_over_threshold = 0
        self._holding_since = None
