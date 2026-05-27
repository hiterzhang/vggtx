#!/usr/bin/env python3
"""Evaluate saved hydrant matches with CO3D ground-truth camera poses.

The earlier matching visualizations color lines using a Fundamental Matrix
estimated from the matches themselves. This script instead reads CO3D
frame_annotations.jgz, builds the ground-truth epipolar geometry for each image
pair, and colors matches by their true Sampson epipolar error.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import eval_co3d_hydrant_camera_pose as camera_eval  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


@dataclass(frozen=True)
class FrameId:
    sequence: str
    frame_number: int


@dataclass
class PairEval:
    sequence: str
    pair_dir: Path
    image0: Path
    image1: Path
    frame0: int
    frame1: int
    src: np.ndarray
    dst: np.ndarray
    scores: np.ndarray
    ransac_inliers: np.ndarray
    gt_errors_px: np.ndarray
    gt_inliers: np.ndarray


def load_annotations(category_dir: Path) -> dict[tuple[str, int], dict]:
    with gzip.open(category_dir / "frame_annotations.jgz", "rt", encoding="utf-8") as handle:
        frames = json.load(handle)
    return {(frame["sequence_name"], int(frame["frame_number"])): frame for frame in frames}


def parse_frame_id(path: Path) -> FrameId:
    stem = path.stem
    match = re.match(r"(?P<seq>.+)_frame(?P<num>\d+)$", stem)
    if match:
        return FrameId(match.group("seq"), int(match.group("num")))

    if path.parent.name == "images" and path.stem.startswith("frame"):
        return FrameId(path.parent.parent.name, int(path.stem.replace("frame", "")))

    raise ValueError(f"Cannot parse CO3D sequence/frame from {path}")


def co3d_ndc_intrinsics_to_pixels(frame: dict, image_path: Path) -> np.ndarray:
    """Convert CO3D/PyTorch3D ndc_isotropic intrinsics to pixel intrinsics.

    PyTorch3D's documented NDC-to-screen relation is:
      fx_screen = fx_ndc * min(W, H) / 2
      px_screen = W / 2 - px_ndc * min(W, H) / 2
    and similarly for y.
    """
    with Image.open(image_path) as image:
        width, height = image.size

    viewpoint = frame["viewpoint"]
    if viewpoint.get("intrinsics_format") != "ndc_isotropic":
        raise ValueError(f"Unsupported intrinsics_format: {viewpoint.get('intrinsics_format')}")

    focal = np.array(viewpoint["focal_length"], dtype=np.float64)
    principal = np.array(viewpoint["principal_point"], dtype=np.float64)
    scale = min(width, height) / 2.0
    fx = focal[0] * scale
    fy = focal[1] * scale
    cx = width / 2.0 - principal[0] * scale
    cy = height / 2.0 - principal[1] * scale
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def preprocess_affine(
    image_path: Path,
    mode: str,
    final_hw: tuple[int, int],
    target_size: int = 518,
) -> np.ndarray:
    with Image.open(image_path) as image:
        width, height = image.size

    if mode == "crop":
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        scale_x = new_width / width
        scale_y = new_height / height
        offset_x = 0.0
        offset_y = 0.0
        out_width = new_width
        out_height = new_height
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            offset_y -= start_y
            out_height = target_size
    elif mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (target_size / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (target_size / height) / 14) * 14
        scale_x = new_width / width
        scale_y = new_height / height
        offset_x = float((target_size - new_width) // 2)
        offset_y = float((target_size - new_height) // 2)
        out_width = target_size
        out_height = target_size
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    final_h, final_w = final_hw
    offset_x += (final_w - out_width) // 2
    offset_y += (final_h - out_height) // 2

    return np.array(
        [[scale_x, 0.0, offset_x], [0.0, scale_y, offset_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def make_gt_fundamental(
    frame0: dict,
    frame1: dict,
    image0: Path,
    image1: Path,
    final_hw: tuple[int, int],
    preprocess: str,
) -> np.ndarray:
    extri0 = camera_eval.convert_pt3d_rt_to_opencv(
        np.array(frame0["viewpoint"]["R"], dtype=np.float64),
        np.array(frame0["viewpoint"]["T"], dtype=np.float64),
    )
    extri1 = camera_eval.convert_pt3d_rt_to_opencv(
        np.array(frame1["viewpoint"]["R"], dtype=np.float64),
        np.array(frame1["viewpoint"]["T"], dtype=np.float64),
    )
    se3_0 = np.eye(4, dtype=np.float64)
    se3_1 = np.eye(4, dtype=np.float64)
    se3_0[:3, :4] = extri0
    se3_1[:3, :4] = extri1
    rel_1_from_0 = se3_1 @ np.linalg.inv(se3_0)
    r_rel = rel_1_from_0[:3, :3]
    t_rel = rel_1_from_0[:3, 3]
    essential = skew(t_rel) @ r_rel

    k0_orig = co3d_ndc_intrinsics_to_pixels(frame0, image0)
    k1_orig = co3d_ndc_intrinsics_to_pixels(frame1, image1)
    a0 = preprocess_affine(image0, preprocess, final_hw)
    a1 = preprocess_affine(image1, preprocess, final_hw)
    k0_pre = a0 @ k0_orig
    k1_pre = a1 @ k1_orig
    fundamental = np.linalg.inv(k1_pre).T @ essential @ np.linalg.inv(k0_pre)
    norm = np.linalg.norm(fundamental)
    return fundamental / norm if norm > 0 else fundamental


def sampson_error_px(src: np.ndarray, dst: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    if len(src) == 0:
        return np.zeros((0,), dtype=np.float64)
    ones = np.ones((len(src), 1), dtype=np.float64)
    x0 = np.concatenate([src.astype(np.float64), ones], axis=1)
    x1 = np.concatenate([dst.astype(np.float64), ones], axis=1)
    fx0 = (fundamental @ x0.T).T
    ftx1 = (fundamental.T @ x1.T).T
    residual = np.sum(x1 * fx0, axis=1)
    denom = fx0[:, 0] ** 2 + fx0[:, 1] ** 2 + ftx1[:, 0] ** 2 + ftx1[:, 1] ** 2
    return np.abs(residual) / np.sqrt(np.maximum(denom, 1e-12))


def image_tensor_to_uint8(image) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)


def draw_pair(
    image0: Path,
    image1: Path,
    src: np.ndarray,
    dst: np.ndarray,
    inliers: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    preprocess: str,
    max_draw: int,
) -> None:
    images = load_and_preprocess_images([str(image0), str(image1)], mode=preprocess)
    left = image_tensor_to_uint8(images[0])
    right = image_tensor_to_uint8(images[1])
    height = max(left.shape[0], right.shape[0])
    width0 = left.shape[1]
    canvas = np.zeros((height, width0 + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], :width0] = left
    canvas[: right.shape[0], width0:] = right

    order = np.argsort(-scores)
    if max_draw > 0:
        order = order[:max_draw]

    for idx in order:
        color = (0, 220, 0) if inliers[idx] else (230, 80, 60)
        p0 = tuple(np.round(src[idx]).astype(int))
        p1 = tuple(np.round(dst[idx] + np.array([width0, 0], dtype=np.float32)).astype(int))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def draw_sequence_overview(
    sequence: str,
    pair_evals: list[PairEval],
    out_path: Path,
    preprocess: str,
    max_draw_per_pair: int,
    inlier_first: bool,
) -> None:
    if not pair_evals:
        return
    image_paths = [pair_evals[0].image0] + [pair.image1 for pair in pair_evals]
    frame_numbers = [pair_evals[0].frame0] + [pair.frame1 for pair in pair_evals]
    images = load_and_preprocess_images([str(path) for path in image_paths], mode=preprocess)
    image_u8 = [image_tensor_to_uint8(image) for image in images]
    height, width = image_u8[0].shape[:2]
    label_h = 34
    canvas = np.full((height + label_h, width * len(image_u8), 3), 255, dtype=np.uint8)

    for idx, image in enumerate(image_u8):
        x0 = idx * width
        canvas[label_h : label_h + height, x0 : x0 + width] = image
        cv2.putText(canvas, str(frame_numbers[idx]), (x0 + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

    for pair in pair_evals:
        if len(pair.src) == 0:
            continue
        if inlier_first:
            inlier_order = np.where(pair.gt_inliers)[0]
            outlier_order = np.where(~pair.gt_inliers)[0]
            inlier_order = inlier_order[np.argsort(-pair.scores[inlier_order])]
            outlier_order = outlier_order[np.argsort(-pair.scores[outlier_order])]
            order = np.concatenate([inlier_order, outlier_order])
        else:
            order = np.argsort(-pair.scores)
        if max_draw_per_pair > 0:
            order = order[:max_draw_per_pair]
        for match_idx in order:
            color = (0, 220, 0) if pair.gt_inliers[match_idx] else (230, 80, 60)
            p0 = np.round(pair.src[match_idx] + np.array([pair.pair_dir_index * width, label_h])).astype(int)
            p1 = np.round(pair.dst[match_idx] + np.array([(pair.pair_dir_index + 1) * width, label_h])).astype(int)
            cv2.line(canvas, tuple(p0), tuple(p1), color, 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p0), 2, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p1), 2, color, -1, cv2.LINE_AA)

    cv2.putText(canvas, f"{sequence}  GT-pose epipolar", (8, height + label_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def evaluate_match_file(
    match_path: Path,
    annotations: dict[tuple[str, int], dict],
    args: argparse.Namespace,
) -> PairEval:
    data = np.load(match_path)
    image_names = [Path(str(name)) for name in data["image_names"]]
    image0, image1 = image_names
    fid0 = parse_frame_id(image0)
    fid1 = parse_frame_id(image1)
    if fid0.sequence != fid1.sequence:
        raise ValueError(f"Pair crosses sequences: {image0}, {image1}")
    frame0 = annotations[(fid0.sequence, fid0.frame_number)]
    frame1 = annotations[(fid1.sequence, fid1.frame_number)]
    final_hw = tuple(int(x) for x in data["preprocessed_hw"])
    fundamental = make_gt_fundamental(frame0, frame1, image0, image1, final_hw, args.preprocess)
    src = data["src"].astype(np.float64)
    dst = data["dst"].astype(np.float64)
    errors = sampson_error_px(src, dst, fundamental)
    gt_inliers = errors <= args.threshold_px
    ransac_inliers = data["inliers"].astype(bool) if "inliers" in data.files else np.zeros(len(src), dtype=bool)

    pair = PairEval(
        sequence=fid0.sequence,
        pair_dir=match_path.parent,
        image0=image0,
        image1=image1,
        frame0=fid0.frame_number,
        frame1=fid1.frame_number,
        src=src,
        dst=dst,
        scores=data["scores"].astype(np.float64),
        ransac_inliers=ransac_inliers,
        gt_errors_px=errors,
        gt_inliers=gt_inliers,
    )
    pair.pair_dir_index = int(re.search(r"pair_(\d+)", match_path.parent.name).group(1))  # type: ignore[attr-defined]
    return pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches_root", type=Path, default=Path("outputs/matching/hydrant_stride_adjacent"))
    parser.add_argument("--co3d_root", type=Path, default=Path("data"))
    parser.add_argument("--category", type=str, default="hydrant")
    parser.add_argument("--out_root", type=Path, default=Path("outputs/matching/hydrant_stride_adjacent_gt_pose"))
    parser.add_argument("--threshold_px", type=float, default=2.0)
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument("--max_draw", type=int, default=200)
    parser.add_argument("--overview_max_draw_per_pair", type=int, default=25)
    parser.add_argument(
        "--overview_inlier_first",
        action="store_true",
        help="Draw highest-scoring GT inliers first in overview instead of highest score overall.",
    )
    args = parser.parse_args()

    annotations = load_annotations(args.co3d_root / args.category)
    match_files = sorted(args.matches_root.glob("*/pair_*/matches.npz"))
    if not match_files:
        raise SystemExit(f"No matches.npz files found under {args.matches_root}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    by_sequence: dict[str, list[PairEval]] = {}

    for match_file in match_files:
        pair = evaluate_match_file(match_file, annotations, args)
        by_sequence.setdefault(pair.sequence, []).append(pair)

        out_dir = args.out_root / pair.sequence / pair.pair_dir.name
        draw_pair(
            pair.image0,
            pair.image1,
            pair.src,
            pair.dst,
            pair.gt_inliers,
            pair.scores,
            out_dir / "matches_gt_pose.png",
            args.preprocess,
            args.max_draw,
        )
        np.savez_compressed(
            out_dir / "matches_gt_pose.npz",
            image_names=np.array([str(pair.image0), str(pair.image1)]),
            src=pair.src,
            dst=pair.dst,
            scores=pair.scores,
            ransac_inliers=pair.ransac_inliers,
            gt_inliers=pair.gt_inliers,
            gt_epipolar_error_px=pair.gt_errors_px,
        )

        tp = int(np.sum(pair.ransac_inliers & pair.gt_inliers))
        ransac_count = int(np.sum(pair.ransac_inliers))
        gt_count = int(np.sum(pair.gt_inliers))
        precision = tp / max(ransac_count, 1)
        recall = tp / max(gt_count, 1)
        median_error = float(np.median(pair.gt_errors_px)) if len(pair.gt_errors_px) else float("inf")
        mean_error = float(np.mean(pair.gt_errors_px)) if len(pair.gt_errors_px) else float("inf")

        with (out_dir / "summary_gt_pose.txt").open("w", encoding="utf-8") as handle:
            handle.write(f"sequence: {pair.sequence}\n")
            handle.write(f"image0: {pair.image0}\n")
            handle.write(f"image1: {pair.image1}\n")
            handle.write(f"frame0: {pair.frame0}\n")
            handle.write(f"frame1: {pair.frame1}\n")
            handle.write(f"threshold_px: {args.threshold_px}\n")
            handle.write(f"matches: {len(pair.src)}\n")
            handle.write(f"gt_inliers: {gt_count}\n")
            handle.write(f"gt_inlier_ratio: {gt_count / max(len(pair.src), 1):.6f}\n")
            handle.write(f"median_gt_epipolar_error_px: {median_error:.6f}\n")
            handle.write(f"mean_gt_epipolar_error_px: {mean_error:.6f}\n")
            handle.write(f"ransac_inliers: {ransac_count}\n")
            handle.write(f"ransac_vs_gt_precision: {precision:.6f}\n")
            handle.write(f"ransac_vs_gt_recall: {recall:.6f}\n")

        rows.append(
            {
                "sequence": pair.sequence,
                "pair": pair.pair_dir.name,
                "frame0": str(pair.frame0),
                "frame1": str(pair.frame1),
                "matches": str(len(pair.src)),
                "gt_inliers": str(gt_count),
                "gt_inlier_ratio": f"{gt_count / max(len(pair.src), 1):.6f}",
                "median_gt_epipolar_error_px": f"{median_error:.6f}",
                "mean_gt_epipolar_error_px": f"{mean_error:.6f}",
                "ransac_inliers": str(ransac_count),
                "ransac_inlier_ratio": f"{ransac_count / max(len(pair.src), 1):.6f}",
                "ransac_vs_gt_precision": f"{precision:.6f}",
                "ransac_vs_gt_recall": f"{recall:.6f}",
                "matches_gt_pose_png": str(out_dir / "matches_gt_pose.png"),
            }
        )

    for sequence, pair_evals in by_sequence.items():
        pair_evals.sort(key=lambda pair: pair.pair_dir_index)  # type: ignore[attr-defined]
        draw_sequence_overview(
            sequence,
            pair_evals,
            args.out_root / sequence / "overview_gt_pose_matches.png",
            args.preprocess,
            args.overview_max_draw_per_pair,
            args.overview_inlier_first,
        )

    csv_path = args.out_root / "summary_gt_pose.csv"
    fieldnames = [
        "sequence",
        "pair",
        "frame0",
        "frame1",
        "matches",
        "gt_inliers",
        "gt_inlier_ratio",
        "median_gt_epipolar_error_px",
        "mean_gt_epipolar_error_px",
        "ransac_inliers",
        "ransac_inlier_ratio",
        "ransac_vs_gt_precision",
        "ransac_vs_gt_recall",
        "matches_gt_pose_png",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluated {len(rows)} pairs with CO3D GT pose.")
    print(f"Threshold: {args.threshold_px:.3f} px")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
