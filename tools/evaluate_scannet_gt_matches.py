#!/usr/bin/env python3
"""Evaluate saved VGGT matches with ScanNet-style pair-file ground truth."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PairGt:
    image0: str
    image1: str
    k0: np.ndarray
    k1: np.ndarray
    t_0to1: np.ndarray


def parse_pairs(path: Path) -> list[PairGt]:
    pairs: list[PairGt] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 38:
            raise ValueError(f"Unexpected pair line with {len(parts)} fields: {line[:120]}...")

        image0, image1 = parts[0], parts[1]
        values = np.array([float(x) for x in parts[4:]], dtype=np.float64)
        k0 = values[0:9].reshape(3, 3)
        k1 = values[9:18].reshape(3, 3)
        t_0to1 = values[18:34].reshape(4, 4)
        pairs.append(PairGt(image0=image0, image1=image1, k0=k0, k1=k1, t_0to1=t_0to1))
    return pairs


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def preprocess_affine(
    width: int,
    height: int,
    mode: str,
    final_hw: tuple[int, int] | None,
    target_size: int = 518,
) -> np.ndarray:
    """Return A where preprocessed_pixel ~= A @ original_pixel_homogeneous."""
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
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        scale_x = new_width / width
        scale_y = new_height / height
        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2
        offset_x = float(pad_left)
        offset_y = float(pad_top)
        out_width = target_size
        out_height = target_size
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    if final_hw is not None:
        final_h, final_w = final_hw
        offset_x += (final_w - out_width) // 2
        offset_y += (final_h - out_height) // 2

    return np.array(
        [
            [scale_x, 0.0, offset_x],
            [0.0, scale_y, offset_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental_from_gt(k0: np.ndarray, k1: np.ndarray, t_0to1: np.ndarray) -> np.ndarray:
    r = t_0to1[:3, :3]
    t = t_0to1[:3, 3]
    e = skew(t) @ r
    f = np.linalg.inv(k1).T @ e @ np.linalg.inv(k0)
    norm = np.linalg.norm(f)
    return f / norm if norm > 0 else f


def sampson_error_px(f: np.ndarray, points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    if len(points0) == 0:
        return np.empty((0,), dtype=np.float64)

    ones = np.ones((len(points0), 1), dtype=np.float64)
    x0 = np.concatenate([points0.astype(np.float64), ones], axis=1)
    x1 = np.concatenate([points1.astype(np.float64), ones], axis=1)

    fx0 = (f @ x0.T).T
    ftx1 = (f.T @ x1.T).T
    numerator = np.sum(x1 * fx0, axis=1) ** 2
    denominator = fx0[:, 0] ** 2 + fx0[:, 1] ** 2 + ftx1[:, 0] ** 2 + ftx1[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def summarize_errors(errors: np.ndarray) -> dict[str, str]:
    if len(errors) == 0:
        return {
            "gt_precision_1px": "NA",
            "gt_precision_3px": "NA",
            "gt_precision_5px": "NA",
            "gt_median_error_px": "NA",
            "gt_mean_error_px": "NA",
        }

    return {
        "gt_precision_1px": f"{np.mean(errors <= 1.0):.6f}",
        "gt_precision_3px": f"{np.mean(errors <= 3.0):.6f}",
        "gt_precision_5px": f"{np.mean(errors <= 5.0):.6f}",
        "gt_median_error_px": f"{np.median(errors):.6f}",
        "gt_mean_error_px": f"{np.mean(errors):.6f}",
    }


def normalize_points(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float64)

    ones = np.ones((len(points), 1), dtype=np.float64)
    points_h = np.concatenate([points.astype(np.float64), ones], axis=1)
    normalized = (np.linalg.inv(k) @ points_h.T).T
    normalized = normalized[:, :2] / normalized[:, 2:3]
    return normalized


def angle_error_mat(r_est: np.ndarray, r_gt: np.ndarray) -> float:
    cos = (np.trace(r_est.T @ r_gt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def angle_error_vec(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    t_est = t_est.reshape(-1)
    t_gt = t_gt.reshape(-1)
    denom = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
    if denom < 1e-12:
        return float("inf")
    cos = float(np.dot(t_est, t_gt) / denom)
    return float(np.degrees(np.arccos(np.clip(abs(cos), -1.0, 1.0))))


def estimate_pose_error(
    points0: np.ndarray,
    points1: np.ndarray,
    k0: np.ndarray,
    k1: np.ndarray,
    t_0to1: np.ndarray,
    ransac_threshold_px: float,
) -> dict[str, float]:
    if len(points0) < 5:
        return {
            "pose_error_deg": float("inf"),
            "rot_error_deg": float("inf"),
            "trans_error_deg": float("inf"),
            "pose_inliers": 0.0,
        }

    points0_norm = normalize_points(points0, k0)
    points1_norm = normalize_points(points1, k1)
    focal = float(np.mean([k0[0, 0], k0[1, 1], k1[0, 0], k1[1, 1]]))
    threshold = ransac_threshold_px / max(focal, 1e-12)

    essential, mask = cv2.findEssentialMat(
        points0_norm,
        points1_norm,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=threshold,
    )
    if essential is None:
        return {
            "pose_error_deg": float("inf"),
            "rot_error_deg": float("inf"),
            "trans_error_deg": float("inf"),
            "pose_inliers": 0.0,
        }

    r_gt = t_0to1[:3, :3]
    t_gt = t_0to1[:3, 3]
    best: dict[str, float] | None = None

    essential = essential.reshape(-1, 3, 3)
    for essential_i in essential:
        try:
            inliers, r_est, t_est, _ = cv2.recoverPose(
                essential_i,
                points0_norm,
                points1_norm,
                cameraMatrix=np.eye(3),
                mask=mask,
            )
        except cv2.error:
            continue

        rot_error = angle_error_mat(r_est, r_gt)
        trans_error = angle_error_vec(t_est, t_gt)
        pose_error = max(rot_error, trans_error)
        candidate = {
            "pose_error_deg": pose_error,
            "rot_error_deg": rot_error,
            "trans_error_deg": trans_error,
            "pose_inliers": float(inliers),
        }
        if best is None or candidate["pose_inliers"] > best["pose_inliers"]:
            best = candidate

    if best is None:
        return {
            "pose_error_deg": float("inf"),
            "rot_error_deg": float("inf"),
            "trans_error_deg": float("inf"),
            "pose_inliers": 0.0,
        }
    return best


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    return f"{value:.6f}"


def error_auc(errors: list[float], thresholds: list[float]) -> dict[float, float]:
    if not errors:
        return {threshold: 0.0 for threshold in thresholds}

    sorted_errors = np.array(sorted(errors), dtype=np.float64)
    recall = np.arange(1, len(sorted_errors) + 1, dtype=np.float64) / len(sorted_errors)
    aucs: dict[float, float] = {}
    for threshold in thresholds:
        last_index = int(np.searchsorted(sorted_errors, threshold, side="right"))
        if last_index == 0:
            aucs[threshold] = 0.0
            continue
        x = np.concatenate([[0.0], sorted_errors[:last_index], [threshold]])
        y = np.concatenate([[0.0], recall[:last_index], [recall[last_index - 1]]])
        aucs[threshold] = float(np.trapz(y, x) / threshold)
    return aucs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/scannet_sample/pairs_first_5.txt"))
    parser.add_argument("--images_dir", type=Path, default=Path("data/scannet_sample/images"))
    parser.add_argument("--results_root", type=Path, default=Path("outputs/matching"))
    parser.add_argument("--tag", type=str, default="scannet_sample")
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument("--out_csv", type=Path, default=Path("outputs/matching/scannet_sample_gt_eval.csv"))
    parser.add_argument("--pose_ransac_threshold_px", type=float, default=1.0)
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    rows: list[dict[str, str]] = []
    pose_errors: list[float] = []

    for idx, pair in enumerate(pairs):
        result_dir = args.results_root / f"{args.tag}_{idx:03d}"
        matches_path = result_dir / "matches.npz"
        row = {
            "idx": f"{idx:03d}",
            "image0": pair.image0,
            "image1": pair.image1,
            "matches": "0",
        }

        if not matches_path.exists():
            row["status"] = f"missing {matches_path}"
            row.update(summarize_errors(np.empty((0,), dtype=np.float64)))
            row.update(
                {
                    "pose_error_deg": "inf",
                    "rot_error_deg": "inf",
                    "trans_error_deg": "inf",
                    "pose_inliers": "0",
                }
            )
            pose_errors.append(float("inf"))
            rows.append(row)
            continue

        matches = np.load(matches_path)
        points0 = matches["src"]
        points1 = matches["dst"]
        final_hw = tuple(int(x) for x in matches["preprocessed_hw"])

        width0, height0 = image_size(args.images_dir / pair.image0)
        width1, height1 = image_size(args.images_dir / pair.image1)
        a0 = preprocess_affine(width0, height0, args.preprocess, final_hw)
        a1 = preprocess_affine(width1, height1, args.preprocess, final_hw)
        k0 = a0 @ pair.k0
        k1 = a1 @ pair.k1

        f_gt = fundamental_from_gt(k0, k1, pair.t_0to1)
        errors = sampson_error_px(f_gt, points0, points1)
        pose_result = estimate_pose_error(
            points0,
            points1,
            k0,
            k1,
            pair.t_0to1,
            ransac_threshold_px=args.pose_ransac_threshold_px,
        )

        row["status"] = "ok"
        row["matches"] = str(len(points0))
        row.update(summarize_errors(errors))
        row.update(
            {
                "pose_error_deg": format_float(pose_result["pose_error_deg"]),
                "rot_error_deg": format_float(pose_result["rot_error_deg"]),
                "trans_error_deg": format_float(pose_result["trans_error_deg"]),
                "pose_inliers": str(int(pose_result["pose_inliers"])),
            }
        )
        pose_errors.append(pose_result["pose_error_deg"])
        rows.append(row)

    aucs = error_auc(pose_errors, [5.0, 10.0, 20.0])

    fieldnames = [
        "idx",
        "status",
        "matches",
        "gt_precision_1px",
        "gt_precision_3px",
        "gt_precision_5px",
        "gt_median_error_px",
        "gt_mean_error_px",
        "pose_error_deg",
        "rot_error_deg",
        "trans_error_deg",
        "pose_inliers",
        "image0",
        "image1",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("idx status matches p@1px p@3px p@5px median_px pose_err rot_err trans_err pose_inliers image0 image1")
    for row in rows:
        print(
            f"{row['idx']} {row['status']} {row['matches']:>5} "
            f"{row['gt_precision_1px']:>7} {row['gt_precision_3px']:>7} "
            f"{row['gt_precision_5px']:>7} {row['gt_median_error_px']:>9} "
            f"{row['pose_error_deg']:>8} {row['rot_error_deg']:>8} "
            f"{row['trans_error_deg']:>9} {row['pose_inliers']:>12} "
            f"{row['image0']} {row['image1']}"
        )
    print()
    print(
        "Pose AUC: "
        f"AUC@5={aucs[5.0] * 100.0:.2f}, "
        f"AUC@10={aucs[10.0] * 100.0:.2f}, "
        f"AUC@20={aucs[20.0] * 100.0:.2f}"
    )
    print(f"\nSaved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
