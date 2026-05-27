#!/usr/bin/env python3
"""Evaluate VGGT on EuRoC V103 with GT camera pose and GT epipolar matching.

This script is a local EuRoC counterpart of the earlier CO3D camera-pose and
ScanNet GT-epipolar scripts:

1. read EuRoC MAV cam0 images, camera calibration, and state ground truth,
2. select a middle segment with visible ground-truth motion,
3. undistort the selected images into a pinhole camera model,
4. run VGGT camera-pose evaluation on the selected frames,
5. run ALIKED + VGGT tracking on adjacent selected-frame pairs and evaluate
   both pose AUC and GT epipolar inlier ratio.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gc
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import eval_co3d_hydrant_camera_pose as camera_base  # noqa: E402
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
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


@dataclass(frozen=True)
class EurocFrame:
    index: int
    timestamp_ns: int
    filename: str
    raw_image_path: Path
    image_path: Path
    t_cw: np.ndarray
    gt_dt_ns: int


@dataclass(frozen=True)
class MotionSelection:
    start_index: int
    frame_stride: int
    path_length_m: float
    baseline_m: float
    rotation_deg: float
    score: float


def normalize_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.startswith("socks://"):
            os.environ[key] = "socks5://" + value[len("socks://") :]


def homogeneous_inverse(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def quat_wxyz_to_rot(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.astype(np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise ValueError(f"Invalid quaternion: {quat}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    cos = (np.trace(rot_a.T @ rot_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def read_euroc_camera_yaml(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fu, fv, cu, cv = [float(value) for value in data["intrinsics"]]
    intrinsics = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
    t_bs = np.array(data["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    width, height = int(data["resolution"][0]), int(data["resolution"][1])
    return intrinsics, distortion, t_bs, (width, height)


def read_camera_rows(camera_dir: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with (camera_dir / "data.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            rows.append((int(row[0]), row[1]))
    return rows


def read_groundtruth_rows(path: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            timestamp = int(row[0])
            position = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=np.float64)
            quat = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])], dtype=np.float64)
            rows.append((timestamp, position, quat))
    return rows


def nearest_groundtruth(
    timestamp_ns: int,
    gt_rows: list[tuple[int, np.ndarray, np.ndarray]],
    gt_timestamps: list[int],
) -> tuple[int, np.ndarray, np.ndarray] | None:
    insert_at = bisect.bisect_left(gt_timestamps, timestamp_ns)
    candidates: list[tuple[int, tuple[int, np.ndarray, np.ndarray]]] = []
    for idx in (insert_at - 1, insert_at):
        if 0 <= idx < len(gt_rows):
            gt = gt_rows[idx]
            candidates.append((abs(gt[0] - timestamp_ns), gt))
    if not candidates:
        return None
    _, best = min(candidates, key=lambda item: item[0])
    return best


def camera_center_from_t_cw(t_cw: np.ndarray) -> np.ndarray:
    return homogeneous_inverse(t_cw)[:3, 3]


def build_aligned_frames(
    euroc_root: Path,
    camera: str,
    max_gt_dt_ns: int,
) -> tuple[list[EurocFrame], np.ndarray, np.ndarray, tuple[int, int]]:
    camera_dir = euroc_root / "mav0" / camera
    gt_path = euroc_root / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    intrinsics, distortion, t_bc, image_size_hw = read_euroc_camera_yaml(camera_dir / "sensor.yaml")
    camera_rows = read_camera_rows(camera_dir)
    gt_rows = read_groundtruth_rows(gt_path)
    gt_timestamps = [row[0] for row in gt_rows]

    frames: list[EurocFrame] = []
    for index, (timestamp_ns, filename) in enumerate(camera_rows):
        gt = nearest_groundtruth(timestamp_ns, gt_rows, gt_timestamps)
        if gt is None:
            continue
        gt_timestamp, position_wb, quat_wb = gt
        dt = abs(gt_timestamp - timestamp_ns)
        if dt > max_gt_dt_ns:
            continue
        raw_image_path = camera_dir / "data" / filename
        if not raw_image_path.exists():
            continue

        # EuRoC q_RS is body/sensor-to-world orientation and p_RS_R is body
        # position in world/reference coordinates. Camera yaml T_BS maps camera
        # sensor coordinates into the body frame, so T_WC = T_WB @ T_BC.
        t_wb = np.eye(4, dtype=np.float64)
        t_wb[:3, :3] = quat_wxyz_to_rot(quat_wb)
        t_wb[:3, 3] = position_wb
        t_wc = t_wb @ t_bc
        t_cw = homogeneous_inverse(t_wc)

        frames.append(
            EurocFrame(
                index=index,
                timestamp_ns=timestamp_ns,
                filename=filename,
                raw_image_path=raw_image_path,
                image_path=raw_image_path,
                t_cw=t_cw,
                gt_dt_ns=dt,
            )
        )
    return frames, intrinsics, distortion, image_size_hw


def score_window(frames: list[EurocFrame], indices: list[int], baseline_weight: float, rotation_weight: float) -> MotionSelection:
    centers = [camera_center_from_t_cw(frames[idx].t_cw) for idx in indices]
    path_length = sum(float(np.linalg.norm(centers[i + 1] - centers[i])) for i in range(len(centers) - 1))
    baseline = float(np.linalg.norm(centers[-1] - centers[0]))

    rotations = [homogeneous_inverse(frames[idx].t_cw)[:3, :3] for idx in indices]
    rotation = sum(rotation_angle_deg(rotations[i], rotations[i + 1]) for i in range(len(rotations) - 1))
    score = path_length + baseline_weight * baseline + rotation_weight * (rotation / 180.0)
    return MotionSelection(
        start_index=indices[0],
        frame_stride=indices[1] - indices[0] if len(indices) > 1 else 0,
        path_length_m=path_length,
        baseline_m=baseline,
        rotation_deg=rotation,
        score=score,
    )


def select_motion_frames(
    frames: list[EurocFrame],
    num_frames: int,
    frame_stride: int,
    search_start_fraction: float,
    search_end_fraction: float,
    baseline_weight: float,
    rotation_weight: float,
    selection_strategy: str,
    manual_start_index: int | None,
) -> tuple[list[EurocFrame], MotionSelection]:
    if len(frames) < num_frames:
        raise ValueError(f"Need at least {num_frames} aligned frames, got {len(frames)}")
    span = (num_frames - 1) * frame_stride
    last_start = len(frames) - span - 1
    if last_start < 0:
        raise ValueError(f"Not enough frames for num_frames={num_frames}, frame_stride={frame_stride}")

    if manual_start_index is not None:
        start = min(max(0, manual_start_index), last_start)
        indices = [start + i * frame_stride for i in range(num_frames)]
        selection = score_window(frames, indices, baseline_weight, rotation_weight)
        return [frames[idx] for idx in indices], selection

    lo = min(max(0, int(round(search_start_fraction * len(frames)))), last_start)
    hi = min(max(lo, int(round(search_end_fraction * len(frames)))), last_start)

    best_selection: MotionSelection | None = None
    best_indices: list[int] | None = None
    for start in range(lo, hi + 1):
        indices = [start + i * frame_stride for i in range(num_frames)]
        selection = score_window(frames, indices, baseline_weight, rotation_weight)
        if best_selection is None:
            best_selection = selection
            best_indices = indices
            continue
        if selection_strategy == "max_motion" and selection.score > best_selection.score:
            best_selection = selection
            best_indices = indices
        elif selection_strategy == "min_motion" and selection.score < best_selection.score:
            best_selection = selection
            best_indices = indices

    if best_selection is None or best_indices is None:
        raise ValueError("No valid EuRoC frame window found")
    return [frames[idx] for idx in best_indices], best_selection


def undistort_selected_frames(
    selected: list[EurocFrame],
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    resolution: tuple[int, int],
    out_dir: Path,
    alpha: float,
    enabled: bool,
) -> tuple[list[EurocFrame], np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = resolution
    if enabled:
        new_k, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (width, height), alpha, (width, height))
    else:
        new_k = intrinsics.copy()

    updated: list[EurocFrame] = []
    for out_idx, frame in enumerate(selected):
        out_path = out_dir / f"{out_idx:03d}_{frame.timestamp_ns}.png"
        if not out_path.exists():
            if enabled:
                image = cv2.imread(str(frame.raw_image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(frame.raw_image_path)
                undistorted = cv2.undistort(image, intrinsics, distortion, None, new_k)
                cv2.imwrite(str(out_path), undistorted)
            else:
                shutil.copy2(frame.raw_image_path, out_path)
        updated.append(
            EurocFrame(
                index=frame.index,
                timestamp_ns=frame.timestamp_ns,
                filename=frame.filename,
                raw_image_path=frame.raw_image_path,
                image_path=out_path,
                t_cw=frame.t_cw,
                gt_dt_ns=frame.gt_dt_ns,
            )
        )
    return updated, new_k.astype(np.float64)


def write_selected_frames_csv(path: Path, selected: list[EurocFrame], selection: MotionSelection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selected_idx",
        "dataset_index",
        "timestamp_ns",
        "filename",
        "raw_image_path",
        "eval_image_path",
        "gt_dt_ns",
        "camera_center_x",
        "camera_center_y",
        "camera_center_z",
        "selection_path_length_m",
        "selection_baseline_m",
        "selection_rotation_deg",
        "selection_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for selected_idx, frame in enumerate(selected):
            center = camera_center_from_t_cw(frame.t_cw)
            writer.writerow(
                {
                    "selected_idx": f"{selected_idx:03d}",
                    "dataset_index": frame.index,
                    "timestamp_ns": frame.timestamp_ns,
                    "filename": frame.filename,
                    "raw_image_path": str(frame.raw_image_path),
                    "eval_image_path": str(frame.image_path),
                    "gt_dt_ns": frame.gt_dt_ns,
                    "camera_center_x": f"{center[0]:.9f}",
                    "camera_center_y": f"{center[1]:.9f}",
                    "camera_center_z": f"{center[2]:.9f}",
                    "selection_path_length_m": f"{selection.path_length_m:.9f}",
                    "selection_baseline_m": f"{selection.baseline_m:.9f}",
                    "selection_rotation_deg": f"{selection.rotation_deg:.9f}",
                    "selection_score": f"{selection.score:.9f}",
                }
            )


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    return f"{value:.6f}"


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


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


def relative_t_0to1(frame0: EurocFrame, frame1: EurocFrame) -> np.ndarray:
    return frame1.t_cw @ homogeneous_inverse(frame0.t_cw)


def build_adjacent_pairs(selected: list[EurocFrame], intrinsics: np.ndarray) -> list[PairGt]:
    pairs: list[PairGt] = []
    for idx in range(len(selected) - 1):
        frame0 = selected[idx]
        frame1 = selected[idx + 1]
        pairs.append(
            PairGt(
                image0=str(frame0.image_path),
                image1=str(frame1.image_path),
                k0=intrinsics.copy(),
                k1=intrinsics.copy(),
                t_0to1=relative_t_0to1(frame0, frame1),
                raw_line=f"{idx:04d} {frame0.timestamp_ns} {frame1.timestamp_ns}",
            )
        )
    return pairs


def maybe_resize_images(images: torch.Tensor, long_side: int) -> torch.Tensor:
    return camera_base.maybe_resize_images(images, long_side)


class EurocEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if not torch.cuda.is_available():
            raise RuntimeError("VGGT EuRoC evaluation requires CUDA.")
        self.device = "cuda"
        self.dtype = choose_dtype(args.compute_dtype)

        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Aggregator dtype: {self.dtype}")
        self.model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=True)
        if args.model_path:
            print(f"Loading checkpoint: {args.model_path}")
            state = torch.load(args.model_path, map_location="cpu")
        else:
            print(f"Loading checkpoint URL: {MODEL_URL}")
            state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
        load_result = self.model.load_state_dict(state, strict=False)
        del state
        self.model.eval()
        print(f"Missing keys: {len(load_result.missing_keys)}")
        print(f"Ignored unexpected keys: {len(load_result.unexpected_keys)}")

        from lightglue import ALIKED

        self.aliked = ALIKED(max_num_keypoints=args.max_keypoints, detection_threshold=args.aliked_threshold).to(
            self.device
        ).eval()

    def run_camera_pose(self, selected: list[EurocFrame]) -> dict:
        print("\nCamera Pose Estimation")
        self.model.track_head.to("cpu")
        self.model.aggregator.to(device=self.device, dtype=self.dtype)
        self.model.camera_head.to(device=self.device, dtype=torch.float32)

        image_names = [str(frame.image_path) for frame in selected]
        gt_extrinsic_np = np.stack([frame.t_cw[:3, :] for frame in selected], axis=0)
        print(f"Frames: {[frame.timestamp_ns for frame in selected]}")
        images = load_and_preprocess_images(image_names, mode=self.args.preprocess).to(self.device)
        images = maybe_resize_images(images, self.args.input_size)
        print(f"Preprocessed shape: {tuple(images.shape)}")

        images_batched = images[None]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                aggregated_tokens_list, _ = self.model.aggregator(images_batched)
            aggregated_tokens_list = [tokens.float() for tokens in aggregated_tokens_list]
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            pred_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        gt_extrinsic = torch.from_numpy(gt_extrinsic_np).to(self.device)
        r_error, t_error = camera_base.relative_pose_errors(pred_extrinsic[0], gt_extrinsic)
        pose_error = np.maximum(r_error, t_error)

        pair_rows: list[dict[str, str]] = []
        pair_i, pair_j = torch.combinations(torch.arange(len(selected)), 2, with_replacement=False).unbind(-1)
        for idx, (i, j) in enumerate(zip(pair_i.cpu().numpy(), pair_j.cpu().numpy())):
            pair_rows.append(
                {
                    "pair_idx": f"{idx:04d}",
                    "frame_i": str(int(i)),
                    "frame_j": str(int(j)),
                    "timestamp_i": str(selected[int(i)].timestamp_ns),
                    "timestamp_j": str(selected[int(j)].timestamp_ns),
                    "rot_error_deg": format_float(float(r_error[idx])),
                    "trans_error_deg": format_float(float(t_error[idx])),
                    "pose_error_deg": format_float(float(pose_error[idx])),
                }
            )

        metrics = {
            "dataset": "EuRoC V103",
            "camera": self.args.camera,
            "num_frames": len(selected),
            "num_pairs": int(len(pose_error)),
            "auc@30": camera_base.calculate_auc_np(r_error, t_error, 30) * 100.0,
            "auc@15": camera_base.calculate_auc_np(r_error, t_error, 15) * 100.0,
            "auc@5": camera_base.calculate_auc_np(r_error, t_error, 5) * 100.0,
            "auc@3": camera_base.calculate_auc_np(r_error, t_error, 3) * 100.0,
            "r_acc@5": float(np.mean(r_error < 5.0) * 100.0),
            "t_acc@5": float(np.mean(t_error < 5.0) * 100.0),
            "median_pose_error_deg": float(np.median(pose_error)),
            "median_r_error_deg": float(np.median(r_error)),
            "median_t_error_deg": float(np.median(t_error)),
            "mean_pose_error_deg": float(np.mean(pose_error)),
            "sampled_timestamps": [frame.timestamp_ns for frame in selected],
            "settings": camera_settings(self.args),
        }

        csv_path = self.args.out_root / f"{self.args.tag}_camera_pose_pairs.csv"
        json_path = self.args.out_root / f"{self.args.tag}_camera_pose_metrics.json"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)
        json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"AUC@30: {metrics['auc@30']:.2f}")
        print(f"AUC@15: {metrics['auc@15']:.2f}")
        print(f"AUC@5 : {metrics['auc@5']:.2f}")
        print(f"AUC@3 : {metrics['auc@3']:.2f}")
        print(f"Median pose error: {metrics['median_pose_error_deg']:.2f} deg")
        print(f"Saved camera pose CSV: {csv_path}")
        print(f"Saved camera pose JSON: {json_path}")

        del images, images_batched, aggregated_tokens_list, pose_enc, pred_extrinsic, gt_extrinsic
        self.model.camera_head.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return metrics

    def detect_aliked_points(self, image: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            keypoint_data = self.aliked.extract(image, invalid_mask=None)
        points = keypoint_data["keypoints"].detach().float().cpu().numpy()
        if points.ndim == 3:
            points = points[0]

        height, width = image.shape[-2:]
        valid = (
            (points[:, 0] >= self.args.margin)
            & (points[:, 0] < width - self.args.margin)
            & (points[:, 1] >= self.args.margin)
            & (points[:, 1] < height - self.args.margin)
        )
        return points[valid][: self.args.max_keypoints].astype(np.float32)

    def run_matching_pair(self, pair: PairGt, idx: int) -> dict[str, str]:
        image0_path = Path(pair.image0)
        image1_path = Path(pair.image1)
        out_dir = self.args.out_root / f"{self.args.tag}_matching_{idx:04d}"
        if self.args.save_pair_outputs:
            out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx:04d}] {image0_path.name} <-> {image1_path.name}")
        self.model.camera_head.to("cpu")
        self.model.track_head.to(device=self.device, dtype=torch.float32)

        images = load_and_preprocess_images([str(image0_path), str(image1_path)], mode=self.args.preprocess).to(
            self.device
        )
        images_batched = images[None]
        _, _, height, width = images.shape

        query_np = self.detect_aliked_points(images[0])
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
        matches = int(len(src_eval))
        gt_count = int(gt_inliers.sum())
        gt_ratio = gt_count / max(matches, 1)
        median_gt_error = float(np.median(gt_errors)) if len(gt_errors) else float("inf")
        mean_gt_error = float(np.mean(gt_errors)) if len(gt_errors) else float("inf")

        row = {
            "idx": f"{idx:04d}",
            "image0": str(image0_path),
            "image1": str(image1_path),
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
            np.savez_compressed(
                out_dir / "matches_euroc_gt_epipolar.npz",
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
                out_dir / "matches_euroc_gt_epipolar.png",
                self.args.max_draw,
            )
            with (out_dir / "summary_euroc_gt_epipolar.txt").open("w", encoding="utf-8") as handle:
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

    def run_matching(self, selected: list[EurocFrame], intrinsics: np.ndarray) -> dict:
        print("\nImage Matching")
        pairs = build_adjacent_pairs(selected, intrinsics)
        rows = [self.run_matching_pair(pair, idx) for idx, pair in enumerate(pairs)]

        pose_errors = [float(row["pose_error_deg"]) if row["pose_error_deg"] != "inf" else float("inf") for row in rows]
        aucs = pose_auc(pose_errors, [5.0, 10.0, 20.0])
        total_matches = sum(int(row["matches"]) for row in rows)
        total_gt_inliers = sum(int(row["gt_epipolar_inliers"]) for row in rows)
        gt_ratios = [float(row["gt_epipolar_inlier_ratio"]) for row in rows]
        med_errors = [float(row["median_gt_epipolar_error_px"]) for row in rows]
        mean_errors = [float(row["mean_gt_epipolar_error_px"]) for row in rows]

        metrics = {
            "dataset": "EuRoC V103",
            "camera": self.args.camera,
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
            "sampled_timestamps": [frame.timestamp_ns for frame in selected],
            "settings": matching_settings(self.args),
        }

        csv_path = self.args.out_root / f"{self.args.tag}_matching_results.csv"
        json_path = self.args.out_root / f"{self.args.tag}_matching_metrics.json"
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

        print("\nEuRoC Image Matching Summary")
        print(f"Pairs: {metrics['num_pairs']}")
        print(f"Pose AUC@5 : {metrics['pose_auc@5']:.2f}")
        print(f"Pose AUC@10: {metrics['pose_auc@10']:.2f}")
        print(f"Pose AUC@20: {metrics['pose_auc@20']:.2f}")
        print(f"GT epipolar inlier ratio: {metrics['global_gt_epipolar_inlier_ratio'] * 100.0:.2f}%")
        print(f"CSV: {csv_path}")
        print(f"JSON: {json_path}")
        return metrics


def camera_settings(args: argparse.Namespace) -> dict:
    return {
        "num_frames": args.num_frames,
        "frame_stride": args.frame_stride,
        "search_start_fraction": args.search_start_fraction,
        "search_end_fraction": args.search_end_fraction,
        "preprocess": args.preprocess,
        "input_size": args.input_size,
        "undistort": args.undistort,
        "undistort_alpha": args.undistort_alpha,
    }


def matching_settings(args: argparse.Namespace) -> dict:
    return {
        "num_frames": args.num_frames,
        "frame_stride": args.frame_stride,
        "pair_mode": "adjacent_selected_frames",
        "max_keypoints": args.max_keypoints,
        "max_matches_for_pose": args.max_matches_for_pose,
        "aliked_threshold": args.aliked_threshold,
        "vis_threshold": args.vis_threshold,
        "conf_threshold": args.conf_threshold,
        "eval_resize_min": args.eval_resize_min,
        "pose_ransac_px": args.pose_ransac_px,
        "gt_epipolar_threshold_px": args.gt_epipolar_threshold_px,
        "preprocess": args.preprocess,
        "undistort": args.undistort,
        "undistort_alpha": args.undistort_alpha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--euroc_root", type=Path, default=Path("data/V103"))
    parser.add_argument("--camera", choices=["cam0", "cam1"], default="cam0")
    parser.add_argument("--task", choices=["camera_pose", "matching", "both"], default="both")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--frame_stride", type=int, default=10, help="Frame stride in the aligned 20 Hz camera stream.")
    parser.add_argument("--search_start_fraction", type=float, default=0.35)
    parser.add_argument("--search_end_fraction", type=float, default=0.65)
    parser.add_argument(
        "--selection_strategy",
        choices=["max_motion", "min_motion"],
        default="max_motion",
        help="Pick the highest- or lowest-motion window in the search interval.",
    )
    parser.add_argument(
        "--manual_start_index",
        type=int,
        default=None,
        help="Aligned-frame start index. If set, bypasses automatic window search.",
    )
    parser.add_argument("--baseline_weight", type=float, default=0.5)
    parser.add_argument("--rotation_weight", type=float, default=0.1)
    parser.add_argument("--max_gt_dt_ns", type=int, default=3_000_000)
    parser.add_argument("--undistort", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--undistort_alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument("--input_size", type=int, default=0)
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
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=Path("outputs/euroc_v103_eval"))
    parser.add_argument("--tag", type=str, default="euroc_v103_mid_motion_10f")
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
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    args.out_root.mkdir(parents=True, exist_ok=True)
    frames, intrinsics, distortion, resolution = build_aligned_frames(args.euroc_root, args.camera, args.max_gt_dt_ns)
    if not frames:
        raise SystemExit(f"No aligned frames found under {args.euroc_root}")

    selected_raw, selection = select_motion_frames(
        frames,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        search_start_fraction=args.search_start_fraction,
        search_end_fraction=args.search_end_fraction,
        baseline_weight=args.baseline_weight,
        rotation_weight=args.rotation_weight,
        selection_strategy=args.selection_strategy,
        manual_start_index=args.manual_start_index,
    )
    selected, eval_k = undistort_selected_frames(
        selected_raw,
        intrinsics,
        distortion,
        resolution,
        args.out_root / "selected_images" / args.camera,
        alpha=args.undistort_alpha,
        enabled=args.undistort,
    )
    write_selected_frames_csv(args.out_root / f"{args.tag}_selected_frames.csv", selected, selection)

    selection_summary = {
        "dataset": "EuRoC V103",
        "camera": args.camera,
        "aligned_frames": len(frames),
        "selected_frames": len(selected),
        "selected_indices": [frame.index for frame in selected],
        "selected_timestamps": [frame.timestamp_ns for frame in selected],
        "selected_filenames": [frame.filename for frame in selected],
        "selection": {
            "start_index": selection.start_index,
            "frame_stride": selection.frame_stride,
            "path_length_m": selection.path_length_m,
            "baseline_m": selection.baseline_m,
            "rotation_deg": selection.rotation_deg,
            "score": selection.score,
            "selection_strategy": args.selection_strategy,
            "manual_start_index": args.manual_start_index,
            "search_start_fraction": args.search_start_fraction,
            "search_end_fraction": args.search_end_fraction,
        },
        "raw_intrinsics": intrinsics.tolist(),
        "eval_intrinsics": eval_k.tolist(),
        "distortion": distortion.tolist(),
        "undistort": args.undistort,
        "undistort_alpha": args.undistort_alpha,
    }
    selection_path = args.out_root / f"{args.tag}_selection.json"
    selection_path.write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")

    print(f"Aligned frames: {len(frames)}")
    print(f"Selected timestamps: {[frame.timestamp_ns for frame in selected]}")
    print(
        "Selected motion: "
        f"path={selection.path_length_m:.3f}m baseline={selection.baseline_m:.3f}m "
        f"rotation={selection.rotation_deg:.3f}deg"
    )
    print(f"Selected frames CSV: {args.out_root / f'{args.tag}_selected_frames.csv'}")
    print(f"Selection JSON: {selection_path}")

    evaluator = EurocEvaluator(args)
    if args.task in ("camera_pose", "both"):
        evaluator.run_camera_pose(selected)
    if args.task in ("matching", "both"):
        evaluator.run_matching(selected, eval_k)


if __name__ == "__main__":
    main()
