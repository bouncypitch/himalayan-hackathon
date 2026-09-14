"""Live wrist + side camera visualization tool (plan §4b).

Minimal viable version: a local OpenCV window showing both feeds side by
side, reading from the same camera indices the rig's get_observation() call
uses. Optional overlay mode shows the rope-pull detector's ROI box and live
flow-magnitude value once that detector exists (plan §4).

Usage:
    python viewer.py --rig-config ../../../../configs/rig/rig1.yaml [--overlay-detector]

Press 'q' to quit.
"""

import argparse

import cv2
import yaml

from himalayan.vision.rope_pull_detector import ROI, RopePullDetector


def open_capture(index_or_path, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index_or_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index_or_path}")
    return cap


def draw_roi_overlay(frame, roi: ROI, magnitude: float, triggered: bool) -> None:
    color = (0, 0, 255) if triggered else (0, 255, 0)
    cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), color, 2)
    cv2.putText(frame, f"flow={magnitude:.2f}", (roi.x, max(0, roi.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--rope-pull-config", default=None,
                         help="Path to configs/tasks/rope_pull_hold.yaml, enables --overlay-detector")
    parser.add_argument("--overlay-detector", action="store_true")
    args = parser.parse_args()

    with open(args.rig_config) as f:
        rig_cfg = yaml.safe_load(f)
    cams_cfg = rig_cfg["cameras"]

    caps = {
        name: open_capture(cam["index_or_path"], cam["width"], cam["height"])
        for name, cam in cams_cfg.items()
    }

    detector = None
    if args.overlay_detector:
        if args.rope_pull_config is None:
            raise SystemExit("--overlay-detector requires --rope-pull-config")
        detector = RopePullDetector.from_yaml(args.rope_pull_config)
        detector_cam = yaml.safe_load(open(args.rope_pull_config))["camera_key"]

    print("Camera viewer running. Press 'q' to quit.")
    try:
        while True:
            frames = {}
            for name, cap in caps.items():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Camera {name} dropped a frame")
                frames[name] = frame

            if detector is not None and detector_cam in frames:
                state_before = detector.state
                new_state = detector.update(frames[detector_cam])
                triggered = new_state != state_before
                draw_roi_overlay(
                    frames[detector_cam], detector.roi, detector.last_magnitude, triggered,
                )

            stacked = cv2.hconcat(list(frames.values()))
            cv2.imshow("Himalayan camera viewer", stacked)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
