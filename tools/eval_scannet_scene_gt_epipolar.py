#!/usr/bin/env python3
"""Evaluate VGGT matches on one ScanNet-1500 scene with pose AUC and GT epipolar error.

This script is a ScanNet counterpart of the CO3D GT-pose epipolar check:

1. filter pairs_calibrated.txt by scene,
2. run ALIKED + VGGT tracking to obtain matches,
3. estimate relative pose from matches and report pose AUC@5/10/20,
4. use ScanNet ground-truth relative pose to compute each match's Sampson
   epipolar error and report GT epipolar inlier ratios.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from tools.eval_vggt_scannet1500_protocol import (  # noqa: E402
    MODEL_URL,
    PairGt,
    apply_homogeneous,
    choose_dtype,
    estimate_pose_from_matches,
    image_size,
    pose_auc,
    preprocess_affine,
    resize_min_affine,
)


def normalize_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.startswith("socks://"):
            os.environ[key] = "socks5://" + value[len("socks://") :]


def parse_scene_pairs(path: Path, scene: str, max_pairs: int | None = None) -> list[PairGt]:
    pairs: list[PairGt] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts[0].startswith(scene + "/"):
            continue

        value_start = 4 if len(parts) >= 38 else 2
        values = np.array([float(x) for x in parts[value_start:]], dtype=np.float64)
        pairs.append(
            PairGt(
                image0=parts[0],
                image1=parts[1],
                k0=values[0:9].reshape(3, 3),
                k1=values[9:18].reshape(3, 3),
                t_0to1=values[18:34].reshape(4, 4),
                raw_line=line,
            )
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def image_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)


def detect_aliked_points(extractor, image: torch.Tensor, max_points: int, margin: int) -> np.ndarray:
    with torch.no_grad():
        keypoint_data = extractor.extract(image, invalid_mask=None)
    points = keypoint_data["keypoints"].detach().float().cpu().numpy()
    if points.ndim == 3:
        points = points[0]

    height, width = image.shape[-2:]
    valid = (
        (points[:, 0] >= margin)
        & (points[:, 0] < width - margin)
        & (points[:, 1] >= margin)
        & (points[:, 1] < height - margin)
    )
    return points[valid][:max_points].astype(np.float32)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def gt_fundamental_from_pair(pair: PairGt, k0: np.ndarray, k1: np.ndarray) -> np.ndarray:
    r = pair.t_0to1[:3, :3]
    t = pair.t_0to1[:3, 3]
    essential = skew(t) @ r
    fundamental = np.linalg.inv(k1).T @ essential @ np.linalg.inv(k0)
    norm = np.linalg.norm(fundamental)
    return fundamental / norm if norm > 0 else fundamental


def sampson_error_px(points0: np.ndarray, points1: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    if len(points0) == 0:
        return np.zeros((0,), dtype=np.float64)
    ones = np.ones((len(points0), 1), dtype=np.float64)
    x0 = np.concatenate([points0.astype(np.float64), ones], axis=1)
    x1 = np.concatenate([points1.astype(np.float64), ones], axis=1)
    fx0 = (fundamental @ x0.T).T
    ftx1 = (fundamental.T @ x1.T).T
    residual = np.sum(x1 * fx0, axis=1)
    denom = fx0[:, 0] ** 2 + fx0[:, 1] ** 2 + ftx1[:, 0] ** 2 + ftx1[:, 1] ** 2
    return np.abs(residual) / np.sqrt(np.maximum(denom, 1e-12))


def draw_matches(
    image0: torch.Tensor,
    image1: torch.Tensor,
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
        color = (0, 220, 0) if inliers[idx] else (230, 80, 60)
        p0 = tuple(np.round(src[idx]).astype(int))
        p1 = tuple(np.round(dst[idx] + np.array([width0, 0], dtype=np.float32)).astype(int))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


class ScanNetGtEpipolarEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError("VGGT ScanNet matching evaluation requires CUDA.")

        self.dtype = choose_dtype(args.compute_dtype)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Aggregator dtype: {self.dtype}")

        from lightglue import ALIKED

        self.aliked = ALIKED(max_num_keypoints=args.max_keypoints, detection_threshold=args.aliked_threshold).to(self.device).eval()

        self.model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=True)
        state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
        load_result = self.model.load_state_dict(state, strict=False)
        del state
        self.model.eval()
        self.model.track_head.to(device=self.device, dtype=torch.float32)
        print(f"Loaded model. Ignored unexpected keys: {len(load_result.unexpected_keys)}")

    def run_pair(self, pair: PairGt, idx: int) -> dict[str, str]:
        image0_path = self.args.images_dir / pair.image0
        image1_path = self.args.images_dir / pair.image1
        out_dir = self.args.out_root / f"{self.args.tag}_{idx:04d}"
        if self.args.save_pair_outputs:
            out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx:04d}] {pair.image0} <-> {pair.image1}")
        images = load_and_preprocess_images([str(image0_path), str(image1_path)], mode=self.args.preprocess).to(self.device)
        images_batched = images[None]
        _, _, height, width = images.shape

        query_np = detect_aliked_points(self.aliked, images[0], self.args.max_keypoints, self.args.margin)
        query_points_all = torch.from_numpy(query_np).to(device=self.device, dtype=torch.float32)[None]
        print(f"Preprocessed shape: {tuple(images.shape)}")
        print(f"ALIKED query points: {query_points_all.shape[1]}")

        self.model.aggregator.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(images_batched)

        needed_layers = set(self.model.track_head.feature_extractor.intermediate_layer_idx)
        selected_tokens = [None] * len(aggregated_tokens_list)
        for layer_idx in sorted(needed_layers):
            selected_tokens[layer_idx] = aggregated_tokens_list[layer_idx].float()
        del aggregated_tokens_list
        self.model.aggregator.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            feature_maps = self.model.track_head.feature_extractor(
                selected_tokens,
                images_batched.float(),
                patch_start_idx,
            )

        coord_chunks: list[np.ndarray] = []
        vis_chunks: list[np.ndarray] = []
        conf_chunks: list[np.ndarray] = []
        for start in range(0, query_points_all.shape[1], self.args.track_chunk):
            end = min(start + self.args.track_chunk, query_points_all.shape[1])
            query_chunk = query_points_all[:, start:end]
            with torch.no_grad():
                coord_preds, vis_scores, conf_scores = self.model.track_head.tracker(
                    query_points=query_chunk,
                    fmaps=feature_maps,
                    iters=self.args.iters,
                )
            coord_chunks.append(coord_preds[-1][0].detach().float().cpu().numpy())
            vis_chunks.append(vis_scores[0, 1].detach().float().cpu().numpy())
            if conf_scores is None:
                conf_chunks.append(np.ones((end - start,), dtype=np.float32))
            else:
                conf_chunks.append(conf_scores[0, 1].detach().float().cpu().numpy())

        tracks = np.concatenate(coord_chunks, axis=1)
        vis = np.concatenate(vis_chunks, axis=0)
        conf = np.concatenate(conf_chunks, axis=0)
        src = tracks[0]
        dst = tracks[1]
        scores = vis if self.args.conf_threshold <= 0 else vis * conf

        finite_tracks = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
        dst_in_bounds = (dst[:, 0] >= 0) & (dst[:, 0] < width) & (dst[:, 1] >= 0) & (dst[:, 1] < height)
        valid = finite_tracks & dst_in_bounds & (vis >= self.args.vis_threshold) & (conf >= self.args.conf_threshold)
        src_pre = src[valid]
        dst_pre = dst[valid]
        scores_valid = scores[valid]

        if self.args.max_matches_for_pose > 0 and len(src_pre) > self.args.max_matches_for_pose:
            order = np.argsort(-scores_valid)[: self.args.max_matches_for_pose]
            src_pre = src_pre[order]
            dst_pre = dst_pre[order]
            scores_valid = scores_valid[order]

        width0, height0 = image_size(image0_path)
        width1, height1 = image_size(image1_path)
        final_hw = (height, width)
        a0 = preprocess_affine(width0, height0, self.args.preprocess, final_hw)
        a1 = preprocess_affine(width1, height1, self.args.preprocess, final_hw)
        src_orig = apply_homogeneous(src_pre, np.linalg.inv(a0))
        dst_orig = apply_homogeneous(dst_pre, np.linalg.inv(a1))

        eval0 = resize_min_affine(width0, height0, self.args.eval_resize_min)
        eval1 = resize_min_affine(width1, height1, self.args.eval_resize_min)
        src_eval = apply_homogeneous(src_orig, eval0)
        dst_eval = apply_homogeneous(dst_orig, eval1)
        k0_eval = eval0 @ pair.k0
        k1_eval = eval1 @ pair.k1

        pose = estimate_pose_from_matches(
            src_eval,
            dst_eval,
            k0_eval,
            k1_eval,
            pair.t_0to1,
            ransac_px=self.args.pose_ransac_px,
            confidence=self.args.ransac_confidence,
            max_iters=self.args.ransac_iters,
        )

        gt_f = gt_fundamental_from_pair(pair, k0_eval, k1_eval)
        gt_errors = sampson_error_px(src_eval, dst_eval, gt_f)
        gt_inliers = gt_errors <= self.args.gt_epipolar_threshold_px

        if self.args.save_pair_outputs:
            np.savez_compressed(
                out_dir / "matches_scannet_gt_epipolar.npz",
                image_names=np.array([str(image0_path), str(image1_path)]),
                src_preprocessed=src_pre,
                dst_preprocessed=dst_pre,
                src_eval=src_eval,
                dst_eval=dst_eval,
                scores=scores_valid,
                gt_epipolar_error_px=gt_errors,
                gt_inliers=gt_inliers,
                k0_eval=k0_eval,
                k1_eval=k1_eval,
                t_0to1=pair.t_0to1,
            )
            draw_matches(
                images[0],
                images[1],
                src_pre,
                dst_pre,
                gt_inliers,
                scores_valid,
                out_dir / "matches_scannet_gt_epipolar.png",
                self.args.max_draw,
            )

        gt_count = int(gt_inliers.sum())
        matches = int(len(src_eval))
        gt_ratio = gt_count / max(matches, 1)
        median_gt_error = float(np.median(gt_errors)) if len(gt_errors) else float("inf")
        mean_gt_error = float(np.mean(gt_errors)) if len(gt_errors) else float("inf")
        row = {
            "idx": f"{idx:04d}",
            "image0": pair.image0,
            "image1": pair.image1,
            "query_points": str(int(query_points_all.shape[1])),
            "matches": str(matches),
            "pose_error_deg": format_float(pose["pose_error_deg"]),
            "rot_error_deg": format_float(pose["rot_error_deg"]),
            "trans_error_deg": format_float(pose["trans_error_deg"]),
            "pose_inliers": str(int(pose["pose_inliers"])),
            "gt_epipolar_inliers": str(gt_count),
            "gt_epipolar_inlier_ratio": f"{gt_ratio:.6f}",
            "median_gt_epipolar_error_px": format_float(median_gt_error),
            "mean_gt_epipolar_error_px": format_float(mean_gt_error),
            "gt_epipolar_threshold_px": f"{self.args.gt_epipolar_threshold_px:.6f}",
        }

        if self.args.save_pair_outputs:
            with (out_dir / "summary_scannet_gt_epipolar.txt").open("w", encoding="utf-8") as handle:
                for key, value in row.items():
                    handle.write(f"{key}: {value}\n")
        print(
            f"matches={matches} pose_error={row['pose_error_deg']} "
            f"gt_epi={gt_count}/{matches} ({gt_ratio:.3f}) "
            f"median_epi={row['median_gt_epipolar_error_px']}px"
        )

        del images, images_batched, feature_maps, selected_tokens, query_points_all
        gc.collect()
        torch.cuda.empty_cache()
        return row


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/scannet1500/scannet1500/pairs_calibrated.txt"))
    parser.add_argument("--images_dir", type=Path, default=Path("data/scannet1500/scannet1500"))
    parser.add_argument("--scene", type=str, default="scene0707_00")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--out_root", type=Path, default=Path("outputs/matching_scannet_gt_epipolar"))
    parser.add_argument("--tag", type=str, default="scene0707_00")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument("--max_keypoints", type=int, default=5000)
    parser.add_argument("--max_matches_for_pose", type=int, default=5000)
    parser.add_argument("--track_chunk", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--aliked_threshold", type=float, default=0.005)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=0.0)
    parser.add_argument("--eval_resize_min", type=int, default=480)
    parser.add_argument("--pose_ransac_px", type=float, default=0.5)
    parser.add_argument("--gt_epipolar_threshold_px", type=float, default=2.0)
    parser.add_argument("--ransac_confidence", type=float, default=0.99999)
    parser.add_argument("--ransac_iters", type=int, default=10000)
    parser.add_argument("--max_draw", type=int, default=300)
    parser.add_argument("--compute_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--save_pair_outputs", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_proxy_env()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.out_root.mkdir(parents=True, exist_ok=True)
    pairs = parse_scene_pairs(args.pairs, args.scene, args.max_pairs)
    if not pairs:
        raise SystemExit(f"No pairs found for scene {args.scene} in {args.pairs}")
    print(f"Scene: {args.scene}")
    print(f"Pairs: {len(pairs)}")

    evaluator = ScanNetGtEpipolarEvaluator(args)
    rows = [evaluator.run_pair(pair, idx) for idx, pair in enumerate(pairs)]

    pose_errors = [float(row["pose_error_deg"]) if row["pose_error_deg"] != "inf" else float("inf") for row in rows]
    aucs = pose_auc(pose_errors, [5.0, 10.0, 20.0])
    total_matches = sum(int(row["matches"]) for row in rows)
    total_gt_inliers = sum(int(row["gt_epipolar_inliers"]) for row in rows)
    gt_ratios = [float(row["gt_epipolar_inlier_ratio"]) for row in rows]
    med_errors = [float(row["median_gt_epipolar_error_px"]) for row in rows]
    mean_errors = [float(row["mean_gt_epipolar_error_px"]) for row in rows]

    metrics = {
        "scene": args.scene,
        "num_pairs": len(rows),
        "pose_auc@5": aucs[5.0] * 100.0,
        "pose_auc@10": aucs[10.0] * 100.0,
        "pose_auc@20": aucs[20.0] * 100.0,
        "median_pose_error_deg": float(np.median([err if np.isfinite(err) else 180.0 for err in pose_errors])),
        "total_matches": total_matches,
        "total_gt_epipolar_inliers": total_gt_inliers,
        "global_gt_epipolar_inlier_ratio": total_gt_inliers / max(total_matches, 1),
        "mean_pair_gt_epipolar_inlier_ratio": float(np.mean(gt_ratios)) if gt_ratios else 0.0,
        "median_gt_epipolar_error_px_avg": float(np.mean(med_errors)) if med_errors else float("inf"),
        "mean_gt_epipolar_error_px_avg": float(np.mean(mean_errors)) if mean_errors else float("inf"),
        "settings": {
            "max_keypoints": args.max_keypoints,
            "max_matches_for_pose": args.max_matches_for_pose,
            "aliked_threshold": args.aliked_threshold,
            "vis_threshold": args.vis_threshold,
            "conf_threshold": args.conf_threshold,
            "eval_resize_min": args.eval_resize_min,
            "pose_ransac_px": args.pose_ransac_px,
            "gt_epipolar_threshold_px": args.gt_epipolar_threshold_px,
            "preprocess": args.preprocess,
        },
    }

    csv_path = args.out_root / f"{args.tag}_results.csv"
    json_path = args.out_root / f"{args.tag}_metrics.json"
    fieldnames = [
        "idx",
        "image0",
        "image1",
        "query_points",
        "matches",
        "pose_error_deg",
        "rot_error_deg",
        "trans_error_deg",
        "pose_inliers",
        "gt_epipolar_inliers",
        "gt_epipolar_inlier_ratio",
        "median_gt_epipolar_error_px",
        "mean_gt_epipolar_error_px",
        "gt_epipolar_threshold_px",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nScanNet Scene GT Epipolar Summary")
    print(f"Scene: {args.scene}")
    print(f"Pairs: {metrics['num_pairs']}")
    print(f"Pose AUC@5 : {metrics['pose_auc@5']:.2f}")
    print(f"Pose AUC@10: {metrics['pose_auc@10']:.2f}")
    print(f"Pose AUC@20: {metrics['pose_auc@20']:.2f}")
    print(f"GT epipolar inlier ratio: {metrics['global_gt_epipolar_inlier_ratio'] * 100:.2f}%")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
