#!/usr/bin/env python3
"""Run VGGT matching on a small ScanNet pair file and summarize results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_pairs(path: Path, max_pairs: int | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pairs.append((parts[0], parts[1]))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def read_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/scannet_sample/pairs_first_5.txt"))
    parser.add_argument("--images_dir", type=Path, default=Path("data/scannet_sample/images"))
    parser.add_argument("--out_root", type=Path, default=Path("outputs/matching"))
    parser.add_argument("--tag", type=str, default="scannet_sample")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=0.0)
    parser.add_argument("--ransac_threshold", type=float, default=1.5)
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument("--keypoints", choices=["aliked", "good_features", "grid"], default="aliked")
    parser.add_argument("--aliked_threshold", type=float, default=0.005)
    parser.add_argument("--compute_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--force", action="store_true", help="Re-run pairs even if summary.txt exists.")
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs, args.max_pairs)
    if not pairs:
        raise SystemExit(f"No pairs found in {args.pairs}")

    rows: list[tuple[int, str, str, dict[str, str]]] = []
    for idx, (image0, image1) in enumerate(pairs):
        out_dir = args.out_root / f"{args.tag}_{idx:03d}"
        summary_path = out_dir / "summary.txt"
        if summary_path.exists() and not args.force:
            print(f"[skip] {out_dir} already has summary.txt")
            rows.append((idx, image0, image1, read_summary(summary_path)))
            continue

        cmd = [
            sys.executable,
            "demo_match_lowmem.py",
            "--image0",
            str(args.images_dir / image0),
            "--image1",
            str(args.images_dir / image1),
            "--out_dir",
            str(out_dir),
            "--max_points",
            str(args.max_points),
            "--iters",
            str(args.iters),
            "--vis_threshold",
            str(args.vis_threshold),
            "--conf_threshold",
            str(args.conf_threshold),
            "--ransac_threshold",
            str(args.ransac_threshold),
            "--preprocess",
            args.preprocess,
            "--keypoints",
            args.keypoints,
            "--aliked_threshold",
            str(args.aliked_threshold),
            "--compute_dtype",
            args.compute_dtype,
        ]
        print(f"\n[{idx + 1}/{len(pairs)}] {image0} <-> {image1}")
        subprocess.run(cmd, check=True)
        rows.append((idx, image0, image1, read_summary(summary_path)))

    print("\nSummary")
    print("idx valid inliers ratio image0 image1")
    for idx, image0, image1, summary in rows:
        valid = summary.get("valid_matches", "NA")
        inliers = summary.get("ransac_inliers", "NA")
        ratio = summary.get("ransac_inlier_ratio", "NA")
        print(f"{idx:03d} {valid:>5} {inliers:>7} {ratio:>8} {image0} {image1}")


if __name__ == "__main__":
    main()
