"""Combine recording sessions (re-recorded/resumed batches, DAgger corrections, plan §6b)
into one canonical LeRobotDataset for training (plan §1, §5).

Usage:
    python merge_datasets.py --output {HF_USER}/himalayan-clip-mug-v1 \
        --sources {HF_USER}/himalayan-clip-mug-raw {HF_USER}/himalayan-clip-mug-dagger-round1
"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets  # verify exact import path against the fork


def merge(output_repo_id: str, source_repo_ids: list[str], root: str | None = None) -> None:
    datasets = [LeRobotDataset(repo_id, root=root) for repo_id in source_repo_ids]
    aggregate_datasets(datasets, output_repo_id=output_repo_id, root=root)
    print(f"Merged {len(source_repo_ids)} datasets into {output_repo_id}")
    print("Run `lerobot-edit-dataset --operation.type recompute_stats` on the merged "
          "dataset before training pi0.5 — it requires quantile stats in meta/stats.json.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output repo_id for the merged dataset")
    parser.add_argument("--sources", required=True, nargs="+", help="Source repo_ids to merge, in order")
    parser.add_argument("--root", default=None, help="Local LeRobotDataset root (default: HF cache)")
    args = parser.parse_args()
    merge(args.output, args.sources, args.root)


if __name__ == "__main__":
    main()
