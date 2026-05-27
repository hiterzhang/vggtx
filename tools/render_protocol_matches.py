#!/usr/bin/env python3
"""Render match visualizations from saved protocol .npz files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vggt.utils.load_fn import load_and_preprocess_images


def image_tensor_to_uint8(image) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)


def draw_matches(
    image0,
    image1,
    src: np.ndarray,
    dst: np.ndarray,
    inliers: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    max_draw: int,
) -> None:
    left = image_tensor_to_uint8(image0)
    right = image_tensor_to_uint8(image1)
    height = max(left.shape[0], right.shape[0])
    width0 = left.shape[1]
    canvas = np.zeros((height, width0 + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], :width0] = left
    canvas[: right.shape[0], width0:] = right

    order = np.argsort(-scores)
    if max_draw > 0:
        order = order[:max_draw]

    for idx in order:
        color = (0, 220, 0) if bool(inliers[idx]) else (230, 80, 60)
        p0 = tuple(np.round(src[idx]).astype(int))
        p1 = tuple(np.round(dst[idx] + np.array([width0, 0], dtype=np.float32)).astype(int))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def render_one(npz_path: Path, out_name: str, max_draw: int, preprocess: str) -> Path:
    data = np.load(npz_path)
    image_names = [str(x) for x in data["image_names"]]
    images = load_and_preprocess_images(image_names, mode=preprocess)

    out_path = npz_path.with_name(out_name)
    draw_matches(
        images[0],
        images[1],
        data["src_preprocessed"],
        data["dst_preprocessed"],
        data["visual_inliers"],
        data["scores"],
        out_path,
        max_draw=max_draw,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/matching_protocol_scannet1500"))
    parser.add_argument("--pattern", type=str, default="vggt_scannet1500_*/matches_protocol.npz")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_draw", type=int, default=300)
    parser.add_argument("--out_name", type=str, default="matches_protocol.png")
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    args = parser.parse_args()

    npz_files = sorted(args.root.glob(args.pattern))
    if args.max_files is not None:
        npz_files = npz_files[: args.max_files]
    if not npz_files:
        raise SystemExit(f"No npz files found under {args.root} with pattern {args.pattern}")

    for idx, npz_path in enumerate(npz_files, start=1):
        out_path = render_one(npz_path, args.out_name, args.max_draw, args.preprocess)
        print(f"[{idx}/{len(npz_files)}] {out_path}")


if __name__ == "__main__":
    main()
