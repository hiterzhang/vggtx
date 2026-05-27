#!/usr/bin/env python3
"""VGGT two-view matching evaluation in a ScanNet-1500-style protocol.

This script is intended to be a fuller local reproduction scaffold for the
VGGT paper's Image Matching section:

1. read ScanNet/SuperGlue-style pair files with intrinsics and relative pose,
2. detect ALIKED keypoints in image0,
3. track those query points into image1 with VGGT's tracking head,
4. estimate an essential matrix with RANSAC,
5. recover relative pose and report AUC@5/10/20.

It also works on the tiny SuperGlue ScanNet sample pair file used in this repo.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


@dataclass(frozen=True)
class PairGt:
    image0: str
    image1: str
    k0: np.ndarray
    k1: np.ndarray
    t_0to1: np.ndarray
    raw_line: str


def normalize_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.startswith("socks://"):
            os.environ[key] = "socks5://" + value[len("socks://") :]


def choose_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def parse_pairs(path: Path, max_pairs: int | None = None) -> list[PairGt]:
    pairs: list[PairGt] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 36:
            raise ValueError(f"Unexpected pair line with {len(parts)} fields: {line[:120]}...")

        # SuperGlue sample format:
        #   image0 image1 rot0 rot1 K0(9) K1(9) T_0to1(16)
        # ETH/CVG scannet1500.zip format:
        #   image0 image1 K0(9) K1(9) T_0to1(16)
        value_start = 4 if len(parts) >= 38 else 2
        values = np.array([float(x) for x in parts[value_start:]], dtype=np.float64)
        if len(values) < 34:
            raise ValueError(f"Unexpected numeric payload with {len(values)} values: {line[:120]}...")
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
    """Return A where preprocessed_pixel ~= A @ original_pixel_h."""
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
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    if final_hw is not None:
        final_h, final_w = final_hw
        offset_x += (final_w - out_width) // 2
        offset_y += (final_h - out_height) // 2

    return np.array(
        [[scale_x, 0.0, offset_x], [0.0, scale_y, offset_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def resize_min_affine(width: int, height: int, min_side: int | None) -> np.ndarray:
    if min_side is None or min_side <= 0:
        scale = 1.0
    else:
        scale = float(min_side) / float(min(width, height))
    return np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float64)
    ones = np.ones((len(points), 1), dtype=np.float64)
    points_h = np.concatenate([points.astype(np.float64), ones], axis=1)
    out = (matrix @ points_h.T).T
    return out[:, :2] / out[:, 2:3]


def detect_aliked_points(
    image: torch.Tensor,
    max_points: int,
    margin: int,
    detection_threshold: float,
) -> np.ndarray:
    from lightglue import ALIKED

    extractor = ALIKED(max_num_keypoints=max_points, detection_threshold=detection_threshold).to(image.device).eval()
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


def normalize_points(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return apply_homogeneous(points, np.linalg.inv(k))


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


def compute_pose_error(t_0to1: np.ndarray, r_est: np.ndarray, t_est: np.ndarray) -> tuple[float, float, float]:
    r_gt = t_0to1[:3, :3]
    t_gt = t_0to1[:3, 3]
    rot_error = angle_error_mat(r_est, r_gt)
    trans_error = angle_error_vec(t_est, t_gt)
    return max(rot_error, trans_error), rot_error, trans_error


def estimate_pose_from_matches(
    points0: np.ndarray,
    points1: np.ndarray,
    k0: np.ndarray,
    k1: np.ndarray,
    t_0to1: np.ndarray,
    ransac_px: float,
    confidence: float,
    max_iters: int,
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
    norm_thresh = ransac_px / max(focal, 1e-12)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC

    essential, mask = cv2.findEssentialMat(
        points0_norm,
        points1_norm,
        cameraMatrix=np.eye(3),
        method=method,
        prob=confidence,
        threshold=norm_thresh,
        maxIters=max_iters,
    )
    if essential is None:
        return {
            "pose_error_deg": float("inf"),
            "rot_error_deg": float("inf"),
            "trans_error_deg": float("inf"),
            "pose_inliers": 0.0,
        }

    best: dict[str, float] | None = None
    for essential_i in essential.reshape(-1, 3, 3):
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
        pose_error, rot_error, trans_error = compute_pose_error(t_0to1, r_est, t_est)
        candidate = {
            "pose_error_deg": pose_error,
            "rot_error_deg": rot_error,
            "trans_error_deg": trans_error,
            "pose_inliers": float(inliers),
        }
        if best is None:
            best = candidate
            continue
        if candidate["pose_inliers"] > best["pose_inliers"]:
            best = candidate
        elif candidate["pose_inliers"] == best["pose_inliers"] and candidate["pose_error_deg"] < best["pose_error_deg"]:
            best = candidate

    if best is None:
        return {
            "pose_error_deg": float("inf"),
            "rot_error_deg": float("inf"),
            "trans_error_deg": float("inf"),
            "pose_inliers": 0.0,
        }
    return best


def pose_auc(errors: list[float], thresholds: list[float]) -> dict[float, float]:
    if not errors:
        return {threshold: 0.0 for threshold in thresholds}

    finite_cap = max(thresholds) + 1.0
    sorted_errors = np.array(sorted([error if np.isfinite(error) else finite_cap for error in errors]), dtype=np.float64)
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


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    return f"{value:.6f}"


def read_summary(path: Path) -> dict[str, str]:
    row: dict[str, str] = {}
    if not path.exists():
        return row
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        row[key.strip()] = value.strip()
    return row


def image_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)


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

    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


class VGGTEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError("VGGT-1B evaluation requires CUDA.")

        self.dtype = choose_dtype(args.compute_dtype)
        self.model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=True)
        state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
        load_result = self.model.load_state_dict(state, strict=False)
        del state
        self.model.eval()
        self.model.track_head.to(device=self.device, dtype=torch.float32)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Aggregator dtype: {self.dtype}")
        print(f"Loaded model. Ignored unexpected keys: {len(load_result.unexpected_keys)}")

    def run_pair(self, pair: PairGt, idx: int) -> dict[str, str]:
        image0_path = self.args.images_dir / pair.image0
        image1_path = self.args.images_dir / pair.image1
        out_dir = self.args.out_root / f"{self.args.tag}_{idx:04d}"
        summary_path = out_dir / "summary_protocol.txt"
        save_pair_outputs = self.args.save_matches or self.args.save_visualizations or self.args.save_per_pair_summary
        if save_pair_outputs or self.args.resume:
            out_dir.mkdir(parents=True, exist_ok=True)

        if self.args.resume and summary_path.exists():
            row = read_summary(summary_path)
            required = {"idx", "status", "matches", "pose_error_deg", "rot_error_deg", "trans_error_deg"}
            if required.issubset(row):
                print(f"\n[{idx:04d}] reuse {summary_path}")
                return row

        print(f"\n[{idx:04d}] {pair.image0} <-> {pair.image1}")
        images = load_and_preprocess_images([str(image0_path), str(image1_path)], mode=self.args.preprocess).to(self.device)
        images_batched = images[None]
        _, _, height, width = images.shape

        query_np = detect_aliked_points(
            images[0],
            max_points=self.args.max_keypoints,
            margin=self.args.margin,
            detection_threshold=self.args.aliked_threshold,
        )
        if len(query_np) == 0:
            raise RuntimeError(f"ALIKED returned no keypoints for {image0_path}")

        if self.args.shuffle_keypoints:
            order = np.random.permutation(len(query_np))
            query_np = query_np[order]

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
        vis_ok = vis >= self.args.vis_threshold
        conf_ok = conf >= self.args.conf_threshold
        valid = finite_tracks & dst_in_bounds & vis_ok & conf_ok
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

        # Visualization RANSAC in preprocessed coordinates for quick inspection.
        if len(src_pre) >= 8:
            _, f_mask = cv2.findFundamentalMat(
                src_pre,
                dst_pre,
                method=cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC,
                ransacReprojThreshold=self.args.visual_ransac_px,
                confidence=self.args.ransac_confidence,
                maxIters=self.args.ransac_iters,
            )
            vis_inliers = f_mask.reshape(-1).astype(bool) if f_mask is not None else np.zeros(len(src_pre), dtype=bool)
        else:
            vis_inliers = np.zeros(len(src_pre), dtype=bool)

        if self.args.save_matches:
            np.savez_compressed(
                out_dir / "matches_protocol.npz",
                image_names=np.array([str(image0_path), str(image1_path)]),
                src_preprocessed=src_pre,
                dst_preprocessed=dst_pre,
                src_original=src_orig,
                dst_original=dst_orig,
                src_eval=src_eval,
                dst_eval=dst_eval,
                scores=scores_valid,
                visual_inliers=vis_inliers,
                preprocessed_hw=np.array(final_hw),
                k0_eval=k0_eval,
                k1_eval=k1_eval,
                t_0to1=pair.t_0to1,
            )

        if self.args.save_visualizations:
            draw_matches(
                images[0],
                images[1],
                src_pre,
                dst_pre,
                vis_inliers,
                scores_valid,
                out_dir / "matches_protocol.png",
                max_draw=self.args.max_draw,
            )

        row = {
            "idx": f"{idx:04d}",
            "status": "ok",
            "image0": pair.image0,
            "image1": pair.image1,
            "query_points": str(query_points_all.shape[1]),
            "finite_tracks": str(int(finite_tracks.sum())),
            "vis_ok": str(int(vis_ok.sum())),
            "conf_ok": str(int(conf_ok.sum())),
            "dst_in_bounds": str(int(dst_in_bounds.sum())),
            "matches": str(len(src_pre)),
            "visual_inliers": str(int(vis_inliers.sum())),
            "visual_inlier_ratio": format_float(float(vis_inliers.mean()) if len(vis_inliers) else 0.0),
            "pose_inliers": str(int(pose["pose_inliers"])),
            "pose_error_deg": format_float(pose["pose_error_deg"]),
            "rot_error_deg": format_float(pose["rot_error_deg"]),
            "trans_error_deg": format_float(pose["trans_error_deg"]),
            "mean_vis": format_float(float(np.mean(vis)) if len(vis) else 0.0),
            "mean_conf": format_float(float(np.mean(conf)) if len(conf) else 0.0),
        }

        if self.args.save_per_pair_summary:
            with summary_path.open("w", encoding="utf-8") as handle:
                for key, value in row.items():
                    handle.write(f"{key}: {value}\n")
                handle.write(f"preprocess: {self.args.preprocess}\n")
                handle.write(f"keypoint_method: aliked\n")
                handle.write(f"aliked_threshold: {self.args.aliked_threshold}\n")
                handle.write(f"eval_resize_min: {self.args.eval_resize_min}\n")
                handle.write(f"pose_ransac_px: {self.args.pose_ransac_px}\n")

        del images, images_batched, feature_maps, selected_tokens, query_points_all
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"matches={row['matches']} pose_error={row['pose_error_deg']} "
            f"R={row['rot_error_deg']} t={row['trans_error_deg']}"
        )
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/scannet_sample/pairs_first_5.txt"))
    parser.add_argument("--images_dir", type=Path, default=Path("data/scannet_sample/images"))
    parser.add_argument("--out_root", type=Path, default=Path("outputs/matching_protocol"))
    parser.add_argument("--tag", type=str, default="vggt_scannet")
    parser.add_argument("--max_pairs", type=int, default=None)
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
    parser.add_argument("--visual_ransac_px", type=float, default=0.5)
    parser.add_argument("--ransac_confidence", type=float, default=0.99999)
    parser.add_argument("--ransac_iters", type=int, default=10000)
    parser.add_argument("--compute_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--shuffle_keypoints", action="store_true")
    parser.add_argument(
        "--save-visualizations",
        "--save_visualizations",
        dest="save_visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-matches",
        "--save_matches",
        dest="save_matches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-pair matches_protocol.npz files.",
    )
    parser.add_argument(
        "--save-per-pair-summary",
        "--save_per_pair_summary",
        dest="save_per_pair_summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-pair summary_protocol.txt files.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse per-pair summary_protocol.txt files when present.")
    parser.add_argument("--max_draw", type=int, default=300)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    normalize_proxy_env()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.out_csv is None:
        args.out_csv = args.out_root / f"{args.tag}_results.csv"
    if args.out_json is None:
        args.out_json = args.out_root / f"{args.tag}_metrics.json"

    pairs = parse_pairs(args.pairs, args.max_pairs)
    if not pairs:
        raise SystemExit(f"No pairs found in {args.pairs}")

    evaluator = VGGTEvaluator(args)
    rows: list[dict[str, str]] = []
    pose_errors: list[float] = []

    for idx, pair in enumerate(pairs):
        try:
            row = evaluator.run_pair(pair, idx)
        except Exception as exc:
            print(f"[error] pair {idx:04d}: {exc}")
            row = {
                "idx": f"{idx:04d}",
                "status": f"error: {exc}",
                "image0": pair.image0,
                "image1": pair.image1,
                "query_points": "0",
                "finite_tracks": "0",
                "vis_ok": "0",
                "conf_ok": "0",
                "dst_in_bounds": "0",
                "matches": "0",
                "visual_inliers": "0",
                "visual_inlier_ratio": "0.000000",
                "pose_inliers": "0",
                "pose_error_deg": "inf",
                "rot_error_deg": "inf",
                "trans_error_deg": "inf",
                "mean_vis": "0.000000",
                "mean_conf": "0.000000",
            }
        rows.append(row)
        pose_errors.append(float(row["pose_error_deg"]) if row["pose_error_deg"] != "inf" else float("inf"))

    aucs = pose_auc(pose_errors, [5.0, 10.0, 20.0])
    metrics = {
        "num_pairs": len(rows),
        "auc@5": aucs[5.0] * 100.0,
        "auc@10": aucs[10.0] * 100.0,
        "auc@20": aucs[20.0] * 100.0,
        "mean_matches": float(np.mean([int(row["matches"]) for row in rows])) if rows else 0.0,
        "median_pose_error_deg": float(np.median([err if np.isfinite(err) else 180.0 for err in pose_errors])),
        "settings": {
            "max_keypoints": args.max_keypoints,
            "max_matches_for_pose": args.max_matches_for_pose,
            "aliked_threshold": args.aliked_threshold,
            "vis_threshold": args.vis_threshold,
            "conf_threshold": args.conf_threshold,
            "eval_resize_min": args.eval_resize_min,
            "pose_ransac_px": args.pose_ransac_px,
            "ransac_confidence": args.ransac_confidence,
            "ransac_iters": args.ransac_iters,
        },
    }

    fieldnames = [
        "idx",
        "status",
        "matches",
        "pose_error_deg",
        "rot_error_deg",
        "trans_error_deg",
        "pose_inliers",
        "visual_inliers",
        "visual_inlier_ratio",
        "query_points",
        "finite_tracks",
        "vis_ok",
        "conf_ok",
        "dst_in_bounds",
        "mean_vis",
        "mean_conf",
        "image0",
        "image1",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nProtocol Summary")
    print(f"Pairs: {metrics['num_pairs']}")
    print(f"AUC@5 : {metrics['auc@5']:.2f}")
    print(f"AUC@10: {metrics['auc@10']:.2f}")
    print(f"AUC@20: {metrics['auc@20']:.2f}")
    print(f"CSV: {args.out_csv}")
    print(f"JSON: {args.out_json}")


if __name__ == "__main__":
    main()
