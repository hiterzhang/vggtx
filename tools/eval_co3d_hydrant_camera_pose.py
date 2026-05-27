#!/usr/bin/env python3
"""Lightweight CO3Dv2 hydrant camera-pose evaluation for VGGT.

This follows the feed-forward part of the official VGGT CO3D camera-pose
protocol: sample N frames from each sequence, predict camera extrinsics, compute
all pairwise relative rotation/translation angular errors, then report
AUC@30/15/5/3.

It is intentionally scoped for local single-sequence CO3Dv2 subsets, such as the
hydrant sequences downloaded into data/hydrant.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.rotation import mat_to_quat


MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


@dataclass(frozen=True)
class FrameRecord:
    sequence_name: str
    frame_number: int
    filepath: str
    r_pt3d: np.ndarray
    t_pt3d: np.ndarray


def choose_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def convert_pt3d_rt_to_opencv(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Convert CO3D/PyTorch3D camera parameters to OpenCV camera-from-world."""
    rot_pt3d = np.array(rot, dtype=np.float64)
    trans_pt3d = np.array(trans, dtype=np.float64)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    return np.hstack((rot_pt3d, trans_pt3d[:, None]))


def build_pair_index(num_frames: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    i1, i2 = torch.combinations(torch.arange(num_frames, device=device), 2, with_replacement=False).unbind(-1)
    return i1, i2


def rotation_angle(rot_gt: torch.Tensor, rot_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    return torch.arccos(1 - 2 * loss_q) * 180.0 / np.pi


def compare_translation_by_angle(
    t_gt: torch.Tensor,
    t_pred: torch.Tensor,
    eps: float = 1e-15,
    default_err: float = 1e6,
) -> torch.Tensor:
    t_pred = t_pred / (torch.norm(t_pred, dim=1, keepdim=True) + eps)
    t_gt = t_gt / (torch.norm(t_gt, dim=1, keepdim=True) + eps)
    loss_t = torch.clamp_min(1.0 - torch.sum(t_pred * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(t_gt: torch.Tensor, t_pred: torch.Tensor, ambiguity: bool = True) -> torch.Tensor:
    rel_tangle_deg = compare_translation_by_angle(t_gt, t_pred) * 180.0 / np.pi
    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    return rel_tangle_deg


def relative_pose_errors(
    pred_extrinsic: torch.Tensor,
    gt_extrinsic: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all-pairs relative pose errors.

    Both inputs are OpenCV camera-from-world extrinsics with shape Nx3x4.
    """
    device = pred_extrinsic.device
    num_frames = pred_extrinsic.shape[0]
    add_row = torch.tensor([0, 0, 0, 1], dtype=torch.float64, device=device).expand(num_frames, 1, 4)
    pred_se3 = torch.cat((pred_extrinsic.to(torch.float64), add_row), dim=1)
    gt_se3 = torch.cat((gt_extrinsic.to(torch.float64), add_row), dim=1)

    pair_i1, pair_i2 = build_pair_index(num_frames, device)
    relative_gt = gt_se3[pair_i1].bmm(closed_form_inverse_se3(gt_se3[pair_i2]))
    relative_pred = pred_se3[pair_i1].bmm(closed_form_inverse_se3(pred_se3[pair_i2]))

    r_error = rotation_angle(relative_gt[:, :3, :3], relative_pred[:, :3, :3])
    t_error = translation_angle(relative_gt[:, :3, 3], relative_pred[:, :3, 3])
    return r_error.detach().cpu().numpy(), t_error.detach().cpu().numpy()


def calculate_auc_np(r_error: np.ndarray, t_error: np.ndarray, max_threshold: int) -> float:
    """Official-style binned pose AUC over max(RRA, RTA)."""
    if len(r_error) == 0:
        return 0.0
    max_errors = np.max(np.stack([r_error, t_error], axis=1), axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized_histogram = histogram.astype(np.float64) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized_histogram)))


def load_frame_annotations(category_dir: Path) -> dict[tuple[str, int], dict]:
    with gzip.open(category_dir / "frame_annotations.jgz", "rt", encoding="utf-8") as handle:
        frames = json.load(handle)
    return {(frame["sequence_name"], int(frame["frame_number"])): frame for frame in frames}


def load_sequence_quality(category_dir: Path) -> dict[str, float]:
    with gzip.open(category_dir / "sequence_annotations.jgz", "rt", encoding="utf-8") as handle:
        sequences = json.load(handle)
    return {
        sequence["sequence_name"]: float(sequence.get("viewpoint_quality_score", float("nan")))
        for sequence in sequences
    }


def records_from_set_list(
    co3d_root: Path,
    category: str,
    set_list_name: str,
    subset: str,
    annotations: dict[tuple[str, int], dict],
) -> dict[str, list[FrameRecord]]:
    set_list_path = co3d_root / category / "set_lists" / f"set_lists_{set_list_name}.json"
    data = json.loads(set_list_path.read_text(encoding="utf-8"))
    if subset not in data:
        raise ValueError(f"Subset {subset!r} not in {set_list_path}")

    by_sequence: dict[str, list[FrameRecord]] = {}
    for sequence_name, frame_number, filepath in data[subset]:
        image_path = co3d_root / filepath
        if not image_path.exists():
            continue
        frame = annotations.get((sequence_name, int(frame_number)))
        if frame is None:
            continue
        viewpoint = frame["viewpoint"]
        by_sequence.setdefault(sequence_name, []).append(
            FrameRecord(
                sequence_name=sequence_name,
                frame_number=int(frame_number),
                filepath=filepath,
                r_pt3d=np.array(viewpoint["R"], dtype=np.float64),
                t_pt3d=np.array(viewpoint["T"], dtype=np.float64),
            )
        )
    return by_sequence


def sample_records(records: list[FrameRecord], num_frames: int, mode: str) -> list[FrameRecord]:
    if len(records) < num_frames:
        raise ValueError(f"Need at least {num_frames} frames, got {len(records)}")
    records = sorted(records, key=lambda record: record.frame_number)
    if mode == "random":
        ids = np.random.choice(len(records), num_frames, replace=False)
        return [records[int(i)] for i in ids]
    if mode == "linspace":
        ids = np.linspace(0, len(records) - 1, num_frames).round().astype(int)
        return [records[int(i)] for i in ids]
    if mode == "first":
        return records[:num_frames]
    raise ValueError(f"Unknown sample mode: {mode}")


def maybe_resize_images(images: torch.Tensor, long_side: int) -> torch.Tensor:
    if long_side <= 0:
        return images
    height, width = images.shape[-2:]
    scale = float(long_side) / float(max(height, width))
    new_height = max(14, int(round((height * scale) / 14.0) * 14))
    new_width = max(14, int(round((width * scale) / 14.0) * 14))
    if (new_height, new_width) == (height, width):
        return images
    return F.interpolate(images, size=(new_height, new_width), mode="bicubic", align_corners=False)


def load_model(args: argparse.Namespace, device: str, dtype: torch.dtype) -> VGGT:
    print("Initializing VGGT camera-only model...")
    model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        state = torch.load(args.model_path, map_location="cpu")
    else:
        print(f"Loading checkpoint URL: {MODEL_URL}")
        state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    load_result = model.load_state_dict(state, strict=False)
    del state
    model.eval()
    model.aggregator.to(device=device, dtype=dtype)
    model.camera_head.to(device=device, dtype=torch.float32)
    print(f"Missing keys: {len(load_result.missing_keys)}")
    print(f"Ignored unexpected keys: {len(load_result.unexpected_keys)}")
    return model


def evaluate_sequence(
    model: VGGT,
    records: list[FrameRecord],
    co3d_root: Path,
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
) -> dict:
    chosen = sample_records(records, args.num_frames, args.sample_mode)
    image_names = [str(co3d_root / record.filepath) for record in chosen]
    gt_extrinsic_np = np.stack(
        [convert_pt3d_rt_to_opencv(record.r_pt3d, record.t_pt3d) for record in chosen],
        axis=0,
    )

    print(f"  frames: {[record.frame_number for record in chosen]}")
    images = load_and_preprocess_images(image_names, mode=args.preprocess).to(device)
    images = maybe_resize_images(images, args.input_size)
    print(f"  preprocessed: {tuple(images.shape)}")

    images_batched = images[None]
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, _ = model.aggregator(images_batched)
        aggregated_tokens_list = [tokens.float() for tokens in aggregated_tokens_list]
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        pred_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    gt_extrinsic = torch.from_numpy(gt_extrinsic_np).to(device)
    r_error, t_error = relative_pose_errors(pred_extrinsic[0], gt_extrinsic)
    pose_error = np.maximum(r_error, t_error)

    del images, images_batched, aggregated_tokens_list, pose_enc, pred_extrinsic, gt_extrinsic
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "sampled_frames": [record.frame_number for record in chosen],
        "num_pairs": int(len(pose_error)),
        "r_error": r_error,
        "t_error": t_error,
        "pose_error": pose_error,
        "auc@30": calculate_auc_np(r_error, t_error, 30) * 100.0,
        "auc@15": calculate_auc_np(r_error, t_error, 15) * 100.0,
        "auc@5": calculate_auc_np(r_error, t_error, 5) * 100.0,
        "auc@3": calculate_auc_np(r_error, t_error, 3) * 100.0,
        "r_acc@5": float(np.mean(r_error < 5.0) * 100.0),
        "t_acc@5": float(np.mean(t_error < 5.0) * 100.0),
        "median_pose_error_deg": float(np.median(pose_error)),
        "median_r_error_deg": float(np.median(r_error)),
        "median_t_error_deg": float(np.median(t_error)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--co3d_root", type=Path, default=Path("data"))
    parser.add_argument("--category", type=str, default="hydrant")
    parser.add_argument(
        "--set_lists",
        type=str,
        default="manyview_dev_0,manyview_dev_1,manyview_test_0",
        help="Comma-separated set list names without the set_lists_ prefix.",
    )
    parser.add_argument("--subset", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--min_num_images", type=int, default=10)
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.5,
        help="Official CO3D preprocessing keeps sequences with viewpoint_quality_score > min_quality.",
    )
    parser.add_argument("--sample_mode", choices=["random", "linspace", "first"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument(
        "--input_size",
        type=int,
        default=0,
        help="Optional low-VRAM long side after VGGT preprocessing. Use 0 for the default 518.",
    )
    parser.add_argument("--compute_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=Path("outputs/camera_pose_co3d_hydrant"))
    parser.add_argument("--tag", type=str, default="vggt_hydrant_camera_pose")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT camera pose evaluation requires CUDA.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    args.out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    dtype = choose_dtype(args.compute_dtype)

    category_dir = args.co3d_root / args.category
    annotations = load_frame_annotations(category_dir)
    sequence_quality = load_sequence_quality(category_dir)

    sequences: list[tuple[str, str, float, list[FrameRecord]]] = []
    for set_list in [item.strip() for item in args.set_lists.split(",") if item.strip()]:
        by_sequence = records_from_set_list(args.co3d_root, args.category, set_list, args.subset, annotations)
        for sequence_name, records in sorted(by_sequence.items()):
            quality = sequence_quality.get(sequence_name, float("nan"))
            if not np.isfinite(quality) or quality <= args.min_quality:
                print(f"Skipping {set_list}/{sequence_name}: viewpoint_quality_score={quality}")
                continue
            if len(records) >= args.min_num_images:
                sequences.append((set_list, sequence_name, quality, records))

    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit("No local CO3D sequences with enough images were found.")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {dtype}")
    print(f"Sequences: {[(set_list, seq, quality, len(records)) for set_list, seq, quality, records in sequences]}")

    model = load_model(args, device, dtype)

    rows: list[dict[str, str]] = []
    all_r_errors: list[np.ndarray] = []
    all_t_errors: list[np.ndarray] = []

    for set_list, sequence_name, quality, records in sequences:
        print(f"\n[{set_list}] {args.category}/{sequence_name} ({len(records)} available frames, quality={quality:.4f})")
        try:
            result = evaluate_sequence(model, records, args.co3d_root, args, device, dtype)
            all_r_errors.append(result["r_error"])
            all_t_errors.append(result["t_error"])
            row = {
                "set_list": set_list,
                "sequence": sequence_name,
                "status": "ok",
                "sequence_quality": f"{quality:.6f}",
                "available_frames": str(len(records)),
                "sampled_frames": " ".join(map(str, result["sampled_frames"])),
                "num_pairs": str(result["num_pairs"]),
                "auc@30": f"{result['auc@30']:.6f}",
                "auc@15": f"{result['auc@15']:.6f}",
                "auc@5": f"{result['auc@5']:.6f}",
                "auc@3": f"{result['auc@3']:.6f}",
                "r_acc@5": f"{result['r_acc@5']:.6f}",
                "t_acc@5": f"{result['t_acc@5']:.6f}",
                "median_pose_error_deg": f"{result['median_pose_error_deg']:.6f}",
                "median_r_error_deg": f"{result['median_r_error_deg']:.6f}",
                "median_t_error_deg": f"{result['median_t_error_deg']:.6f}",
            }
            print(
                "  "
                f"AUC@30={row['auc@30']} AUC@15={row['auc@15']} "
                f"AUC@5={row['auc@5']} AUC@3={row['auc@3']} "
                f"median_pose={row['median_pose_error_deg']} deg"
            )
        except Exception as exc:
            row = {
                "set_list": set_list,
                "sequence": sequence_name,
                "status": f"error: {exc}",
                "sequence_quality": f"{quality:.6f}",
                "available_frames": str(len(records)),
                "sampled_frames": "",
                "num_pairs": "0",
                "auc@30": "0.000000",
                "auc@15": "0.000000",
                "auc@5": "0.000000",
                "auc@3": "0.000000",
                "r_acc@5": "0.000000",
                "t_acc@5": "0.000000",
                "median_pose_error_deg": "inf",
                "median_r_error_deg": "inf",
                "median_t_error_deg": "inf",
            }
            print(f"  error: {exc}")
        rows.append(row)

    if all_r_errors:
        r_errors = np.concatenate(all_r_errors)
        t_errors = np.concatenate(all_t_errors)
    else:
        r_errors = np.array([], dtype=np.float64)
        t_errors = np.array([], dtype=np.float64)

    pose_errors = np.maximum(r_errors, t_errors) if len(r_errors) else np.array([], dtype=np.float64)
    metrics = {
        "category": args.category,
        "num_sequences": len(all_r_errors),
        "num_pairs": int(len(pose_errors)),
        "auc@30": calculate_auc_np(r_errors, t_errors, 30) * 100.0,
        "auc@15": calculate_auc_np(r_errors, t_errors, 15) * 100.0,
        "auc@5": calculate_auc_np(r_errors, t_errors, 5) * 100.0,
        "auc@3": calculate_auc_np(r_errors, t_errors, 3) * 100.0,
        "r_acc@5": float(np.mean(r_errors < 5.0) * 100.0) if len(r_errors) else 0.0,
        "t_acc@5": float(np.mean(t_errors < 5.0) * 100.0) if len(t_errors) else 0.0,
        "median_pose_error_deg": float(np.median(pose_errors)) if len(pose_errors) else float("inf"),
        "median_r_error_deg": float(np.median(r_errors)) if len(r_errors) else float("inf"),
        "median_t_error_deg": float(np.median(t_errors)) if len(t_errors) else float("inf"),
        "settings": {
            "co3d_root": str(args.co3d_root),
            "set_lists": args.set_lists,
            "subset": args.subset,
            "num_frames": args.num_frames,
            "min_num_images": args.min_num_images,
            "min_quality": args.min_quality,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "preprocess": args.preprocess,
            "input_size": args.input_size,
            "compute_dtype": str(dtype),
        },
    }

    csv_path = args.out_root / f"{args.tag}_results.csv"
    json_path = args.out_root / f"{args.tag}_metrics.json"
    fieldnames = [
        "set_list",
        "sequence",
        "status",
        "sequence_quality",
        "available_frames",
        "sampled_frames",
        "num_pairs",
        "auc@30",
        "auc@15",
        "auc@5",
        "auc@3",
        "r_acc@5",
        "t_acc@5",
        "median_pose_error_deg",
        "median_r_error_deg",
        "median_t_error_deg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nCO3D Camera Pose Summary")
    print(f"Sequences: {metrics['num_sequences']}")
    print(f"Pairs: {metrics['num_pairs']}")
    print(f"AUC@30: {metrics['auc@30']:.2f}")
    print(f"AUC@15: {metrics['auc@15']:.2f}")
    print(f"AUC@5 : {metrics['auc@5']:.2f}")
    print(f"AUC@3 : {metrics['auc@3']:.2f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
