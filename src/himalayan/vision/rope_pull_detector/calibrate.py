"""Threshold calibration for the rope-pull detector (plan §4).

Feed a directory of short labeled clips (pull vs. static/hard-negative) and
print the mean flow magnitude per clip, so a threshold can be picked by
inspection -- no training loop required. Use the camera viewer's flow overlay
(plan §4b) alongside this for live tuning against the real rig.

Usage:
    python calibrate.py --clips-dir data/rope_pull_clips --roi-config ../../../../configs/tasks/rope_pull_hold.yaml
"""

import argparse
from pathlib import Path

import cv2
import yaml

from detector import ROI, RopePullDetector  # noqa: E402  (script run directly, not as a package)


def score_clip(video_path: Path, roi: ROI) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    prev_gray = None
    magnitudes = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(roi.crop(frame), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(mag.mean()))
        prev_gray = gray
    cap.release()
    return magnitudes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", required=True, type=Path,
                         help="Directory of .mp4 clips, named pull_*.mp4 / negative_*.mp4")
    parser.add_argument("--roi-config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.roi_config) as f:
        cfg = yaml.safe_load(f)
    roi = ROI(**cfg["roi"])

    for clip_path in sorted(args.clips_dir.glob("*.mp4")):
        magnitudes = score_clip(clip_path, roi)
        if not magnitudes:
            print(f"{clip_path.name}: no frames read")
            continue
        print(f"{clip_path.name}: mean={sum(magnitudes)/len(magnitudes):.2f} "
              f"max={max(magnitudes):.2f}")

    print("\nPick flow_magnitude_threshold roughly halfway between the max of the "
          "negative clips and the typical pull-clip mean, then update "
          "configs/tasks/rope_pull_hold.yaml.")


if __name__ == "__main__":
    main()
