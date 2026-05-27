#!/usr/bin/env python3
"""Run VGGT image matching on adjacent frames from selected hydrant images.

Input filenames are expected to look like:
    167_18184_34441_frame000001.jpg

For each sequence prefix, this script sorts frames by frame number, matches
adjacent pairs, writes per-pair visualizations, and creates one overview strip
where adjacent images are connected by match lines.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import demo_match_lowmem as demo_match  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


@dataclass(frozen=True)
class ImageItem:
    sequence: str
    frame_number: int
    path: Path


@dataclass
class PairResult:
    sequence: str
    pair_index: int
    image0: Path
    image1: Path
    frame0: int
    frame1: int
    preprocessed_hw: tuple[int, int]
    query_points: int
    valid_matches: int
    ransac_inliers: int
    ransac_inlier_ratio: float
    mean_vis: float
    mean_conf: float
    src: np.ndarray
    dst: np.ndarray
    inliers: np.ndarray
    scores: np.ndarray


def parse_selected_images(selected_dir: Path) -> dict[str, list[ImageItem]]:
    groups: dict[str, list[ImageItem]] = {}
    for path in sorted(selected_dir.glob("*.jpg")):
        stem = path.stem
        if "_frame" not in stem:
            continue
        sequence, frame_part = stem.split("_frame", 1)
        frame_number = int(frame_part)
        groups.setdefault(sequence, []).append(ImageItem(sequence, frame_number, path))

    for sequence in groups:
        groups[sequence].sort(key=lambda item: item.frame_number)
    return dict(sorted(groups.items()))


def load_model(device: str, dtype: torch.dtype) -> VGGT:
    model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=True)
    state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    load_result = model.load_state_dict(state, strict=False)
    del state
    model.eval()
    model.track_head.to(device=device, dtype=torch.float32)
    print(f"Loaded model. Ignored unexpected keys: {len(load_result.unexpected_keys)}")
    return model


def match_pair(
    model: VGGT,
    image0: Path,
    image1: Path,
    args: argparse.Namespace,
    device: str,
    compute_dtype: torch.dtype,
) -> PairResult:
    images = load_and_preprocess_images([str(image0), str(image1)], mode=args.preprocess).to(device)
    images_batched = images[None]
    _, _, height, width = images.shape

    query_np = demo_match.detect_query_points(
        images[0],
        max_points=args.max_points,
        margin=args.margin,
        method=args.keypoints,
        aliked_threshold=args.aliked_threshold,
    )
    query_points = torch.from_numpy(query_np).to(device=device, dtype=torch.float32)[None]

    model.aggregator.to(device=device, dtype=compute_dtype)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=compute_dtype):
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_batched)

    needed_layers = set(model.track_head.feature_extractor.intermediate_layer_idx)
    selected_tokens = [None] * len(aggregated_tokens_list)
    for layer_idx in sorted(needed_layers):
        selected_tokens[layer_idx] = aggregated_tokens_list[layer_idx].float()

    del aggregated_tokens_list
    model.aggregator.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        coord_preds, vis_scores, conf_scores = model.track_head(
            selected_tokens,
            images=images_batched.float(),
            patch_start_idx=patch_start_idx,
            query_points=query_points,
            iters=args.iters,
        )

    tracks = coord_preds[-1][0].detach().float().cpu().numpy()
    vis = vis_scores[0, 1].detach().float().cpu().numpy()
    conf = conf_scores[0, 1].detach().float().cpu().numpy() if conf_scores is not None else np.ones_like(vis)

    src = tracks[0]
    dst = tracks[1]
    scores = vis if args.conf_threshold <= 0 else vis * conf
    finite_tracks = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    dst_in_bounds = (dst[:, 0] >= 0) & (dst[:, 0] < width) & (dst[:, 1] >= 0) & (dst[:, 1] < height)
    valid = finite_tracks & dst_in_bounds & (vis >= args.vis_threshold) & (conf >= args.conf_threshold)

    src_valid = src[valid]
    dst_valid = dst[valid]
    scores_valid = scores[valid]

    if len(src_valid) >= 8:
        _, inlier_mask = cv2.findFundamentalMat(
            src_valid,
            dst_valid,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=args.ransac_threshold,
            confidence=0.999,
            maxIters=10000,
        )
        inliers = inlier_mask.reshape(-1).astype(bool) if inlier_mask is not None else np.zeros(len(src_valid), dtype=bool)
    else:
        inliers = np.zeros(len(src_valid), dtype=bool)

    result = PairResult(
        sequence="",
        pair_index=-1,
        image0=image0,
        image1=image1,
        frame0=-1,
        frame1=-1,
        preprocessed_hw=(height, width),
        query_points=int(query_points.shape[1]),
        valid_matches=int(len(src_valid)),
        ransac_inliers=int(inliers.sum()),
        ransac_inlier_ratio=float(inliers.sum() / max(len(src_valid), 1)),
        mean_vis=float(np.mean(vis)),
        mean_conf=float(np.mean(conf)),
        src=src_valid,
        dst=dst_valid,
        inliers=inliers,
        scores=scores_valid,
    )

    result._images_cpu = images.detach().cpu()  # type: ignore[attr-defined]

    del images, images_batched, selected_tokens, query_points, coord_preds, vis_scores, conf_scores
    torch.cuda.empty_cache()
    return result


def save_pair_outputs(result: PairResult, out_dir: Path, max_draw: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_cpu = result._images_cpu  # type: ignore[attr-defined]
    demo_match.draw_matches(
        images_cpu[0],
        images_cpu[1],
        result.src,
        result.dst,
        result.inliers,
        result.scores,
        str(out_dir / "matches.png"),
        max_draw=max_draw,
    )
    np.savez_compressed(
        out_dir / "matches.npz",
        image_names=np.array([str(result.image0), str(result.image1)]),
        src=result.src,
        dst=result.dst,
        scores=result.scores,
        inliers=result.inliers,
        preprocessed_hw=np.array(result.preprocessed_hw),
    )
    with (out_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"sequence: {result.sequence}\n")
        handle.write(f"pair_index: {result.pair_index}\n")
        handle.write(f"image0: {result.image0}\n")
        handle.write(f"image1: {result.image1}\n")
        handle.write(f"frame0: {result.frame0}\n")
        handle.write(f"frame1: {result.frame1}\n")
        handle.write(f"preprocessed_hw: {result.preprocessed_hw}\n")
        handle.write(f"query_points: {result.query_points}\n")
        handle.write(f"valid_matches: {result.valid_matches}\n")
        handle.write(f"ransac_inliers: {result.ransac_inliers}\n")
        handle.write(f"ransac_inlier_ratio: {result.ransac_inlier_ratio:.6f}\n")
        handle.write(f"mean_vis: {result.mean_vis:.6f}\n")
        handle.write(f"mean_conf: {result.mean_conf:.6f}\n")


def draw_sequence_overview(
    sequence: str,
    items: list[ImageItem],
    pair_results: list[PairResult],
    out_path: Path,
    max_draw_per_pair: int,
    preprocess: str,
) -> None:
    images = load_and_preprocess_images([str(item.path) for item in items], mode=preprocess)
    image_u8 = [demo_match.image_tensor_to_uint8(image) for image in images]
    height, width = image_u8[0].shape[:2]
    label_h = 34
    canvas = np.full((height + label_h, width * len(image_u8), 3), 255, dtype=np.uint8)

    for idx, image in enumerate(image_u8):
        x0 = idx * width
        canvas[label_h : label_h + height, x0 : x0 + width] = image
        cv2.putText(
            canvas,
            f"{items[idx].frame_number}",
            (x0 + 8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    for result in pair_results:
        pair_idx = result.pair_index
        if result.valid_matches == 0:
            continue
        order = np.argsort(-result.scores)
        if max_draw_per_pair > 0:
            order = order[:max_draw_per_pair]
        for match_idx in order:
            color = (0, 220, 0) if result.inliers[match_idx] else (230, 80, 60)
            p0 = np.round(result.src[match_idx] + np.array([pair_idx * width, label_h])).astype(int)
            p1 = np.round(result.dst[match_idx] + np.array([(pair_idx + 1) * width, label_h])).astype(int)
            cv2.line(canvas, tuple(p0), tuple(p1), color, 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p0), 2, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p1), 2, color, -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        sequence,
        (8, height + label_h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected_dir", type=Path, default=Path("data/hydrant_stride_1_2_10f_selected"))
    parser.add_argument("--out_root", type=Path, default=Path("outputs/matching/hydrant_stride_adjacent"))
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--max_draw", type=int, default=200)
    parser.add_argument("--overview_max_draw_per_pair", type=int, default=25)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=0.0)
    parser.add_argument("--ransac_threshold", type=float, default=1.5)
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument("--keypoints", choices=["aliked", "good_features", "grid"], default="aliked")
    parser.add_argument("--aliked_threshold", type=float, default=0.005)
    parser.add_argument("--compute_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    args = parser.parse_args()

    demo_match.normalize_proxy_env()
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT matching requires CUDA.")

    groups = parse_selected_images(args.selected_dir)
    if not groups:
        raise SystemExit(f"No selected images found in {args.selected_dir}")

    device = "cuda"
    compute_dtype = demo_match.choose_dtype(args.compute_dtype)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Aggregator dtype: {compute_dtype}")
    print(f"Sequences: {[(seq, len(items)) for seq, items in groups.items()]}")

    model = load_model(device, compute_dtype)
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []

    for sequence, items in groups.items():
        if len(items) < 2:
            continue
        print(f"\nSequence {sequence}: {len(items)} frames")
        seq_out = args.out_root / sequence
        pair_results: list[PairResult] = []

        for pair_index in range(len(items) - 1):
            left = items[pair_index]
            right = items[pair_index + 1]
            print(f"  [{pair_index:02d}] frame {left.frame_number} -> {right.frame_number}")
            result = match_pair(model, left.path, right.path, args, device, compute_dtype)
            result.sequence = sequence
            result.pair_index = pair_index
            result.frame0 = left.frame_number
            result.frame1 = right.frame_number
            pair_results.append(result)

            pair_dir = seq_out / f"pair_{pair_index:02d}_f{left.frame_number:06d}_f{right.frame_number:06d}"
            save_pair_outputs(result, pair_dir, args.max_draw)
            summary_rows.append(
                {
                    "sequence": sequence,
                    "pair_index": f"{pair_index:02d}",
                    "frame0": str(left.frame_number),
                    "frame1": str(right.frame_number),
                    "query_points": str(result.query_points),
                    "valid_matches": str(result.valid_matches),
                    "ransac_inliers": str(result.ransac_inliers),
                    "ransac_inlier_ratio": f"{result.ransac_inlier_ratio:.6f}",
                    "mean_vis": f"{result.mean_vis:.6f}",
                    "mean_conf": f"{result.mean_conf:.6f}",
                    "matches_png": str(pair_dir / "matches.png"),
                }
            )

        draw_sequence_overview(
            sequence,
            items,
            pair_results,
            seq_out / "overview_adjacent_matches.png",
            args.overview_max_draw_per_pair,
            args.preprocess,
        )
        print(f"  overview: {seq_out / 'overview_adjacent_matches.png'}")

    csv_path = args.out_root / "summary.csv"
    fieldnames = [
        "sequence",
        "pair_index",
        "frame0",
        "frame1",
        "query_points",
        "valid_matches",
        "ransac_inliers",
        "ransac_inlier_ratio",
        "mean_vis",
        "mean_conf",
        "matches_png",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {csv_path}")


if __name__ == "__main__":
    main()
