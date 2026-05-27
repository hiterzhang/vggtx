#!/usr/bin/env python3
"""Deterministic stride-frame CO3Dv2 hydrant camera-pose test.

Default frame selection is:
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19

The metric computation matches the lightweight CO3D camera-pose script:
VGGT predicts camera extrinsics for the selected frames, then all frame pairs
are evaluated with relative rotation/translation angular errors and AUC.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import eval_co3d_hydrant_camera_pose as base  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--co3d_root", type=Path, default=Path("data"))
    parser.add_argument("--category", type=str, default="hydrant")
    parser.add_argument(
        "--set_lists",
        type=str,
        default="manyview_dev_0,manyview_dev_1,manyview_test_0",
        help="Used only to discover local sequence names when --sequence_names is not set.",
    )
    parser.add_argument(
        "--sequence_names",
        type=str,
        default=None,
        help="Comma-separated sequence names. Default: sequences discovered from --set_lists.",
    )
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--start_frame", type=int, default=1)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.5,
        help="Skip sequences with viewpoint_quality_score <= this value.",
    )
    parser.add_argument(
        "--include_unqualified",
        action="store_true",
        help="Do not skip nan/low-quality CO3D sequences.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument(
        "--input_size",
        type=int,
        default=0,
        help="Optional low-VRAM long side after VGGT preprocessing. Use 0 for the default 518.",
    )
    parser.add_argument("--compute_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=Path("outputs/camera_pose_co3d_hydrant_stride"))
    parser.add_argument("--tag", type=str, default="vggt_hydrant_camera_pose_stride_1_2_10f")
    return parser.parse_args()


def discover_sequences(args: argparse.Namespace) -> list[str]:
    if args.sequence_names:
        return [item.strip() for item in args.sequence_names.split(",") if item.strip()]

    names: list[str] = []
    seen: set[str] = set()
    category_dir = args.co3d_root / args.category
    for set_list in [item.strip() for item in args.set_lists.split(",") if item.strip()]:
        set_list_path = category_dir / "set_lists" / f"set_lists_{set_list}.json"
        data = json.loads(set_list_path.read_text(encoding="utf-8"))
        for subset in ("train", "val", "test"):
            for sequence_name, _, _ in data.get(subset, []):
                if sequence_name not in seen:
                    seen.add(sequence_name)
                    names.append(sequence_name)
    return names


def records_for_sequence(
    co3d_root: Path,
    category: str,
    sequence_name: str,
    annotations: dict[tuple[str, int], dict],
) -> dict[int, base.FrameRecord]:
    records: dict[int, base.FrameRecord] = {}
    image_dir = co3d_root / category / sequence_name / "images"
    for image_path in sorted(image_dir.glob("frame*.jpg")):
        frame_number = int(image_path.stem.replace("frame", ""))
        frame = annotations.get((sequence_name, frame_number))
        if frame is None:
            continue
        viewpoint = frame["viewpoint"]
        records[frame_number] = base.FrameRecord(
            sequence_name=sequence_name,
            frame_number=frame_number,
            filepath=f"{category}/{sequence_name}/images/{image_path.name}",
            r_pt3d=np.array(viewpoint["R"], dtype=np.float64),
            t_pt3d=np.array(viewpoint["T"], dtype=np.float64),
        )
    return records


def select_stride_records(records_by_frame: dict[int, base.FrameRecord], args: argparse.Namespace) -> list[base.FrameRecord]:
    frame_numbers = [args.start_frame + args.frame_stride * i for i in range(args.num_frames)]
    missing = [frame_number for frame_number in frame_numbers if frame_number not in records_by_frame]
    if missing:
        raise ValueError(f"Missing requested frames: {missing}")
    return [records_by_frame[frame_number] for frame_number in frame_numbers]


def evaluate_chosen_records(
    model,
    chosen: list[base.FrameRecord],
    co3d_root: Path,
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
) -> dict:
    image_names = [str(co3d_root / record.filepath) for record in chosen]
    gt_extrinsic_np = np.stack(
        [base.convert_pt3d_rt_to_opencv(record.r_pt3d, record.t_pt3d) for record in chosen],
        axis=0,
    )

    print(f"  frames: {[record.frame_number for record in chosen]}")
    images = load_and_preprocess_images(image_names, mode=args.preprocess).to(device)
    images = base.maybe_resize_images(images, args.input_size)
    print(f"  preprocessed: {tuple(images.shape)}")

    images_batched = images[None]
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, _ = model.aggregator(images_batched)
        aggregated_tokens_list = [tokens.float() for tokens in aggregated_tokens_list]
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        pred_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    gt_extrinsic = torch.from_numpy(gt_extrinsic_np).to(device)
    r_error, t_error = base.relative_pose_errors(pred_extrinsic[0], gt_extrinsic)
    pose_error = np.maximum(r_error, t_error)

    del images, images_batched, aggregated_tokens_list, pose_enc, pred_extrinsic, gt_extrinsic
    torch.cuda.empty_cache()

    return {
        "sampled_frames": [record.frame_number for record in chosen],
        "num_pairs": int(len(pose_error)),
        "r_error": r_error,
        "t_error": t_error,
        "pose_error": pose_error,
        "auc@30": base.calculate_auc_np(r_error, t_error, 30) * 100.0,
        "auc@15": base.calculate_auc_np(r_error, t_error, 15) * 100.0,
        "auc@5": base.calculate_auc_np(r_error, t_error, 5) * 100.0,
        "auc@3": base.calculate_auc_np(r_error, t_error, 3) * 100.0,
        "r_acc@5": float(np.mean(r_error < 5.0) * 100.0),
        "t_acc@5": float(np.mean(t_error < 5.0) * 100.0),
        "median_pose_error_deg": float(np.median(pose_error)),
        "median_r_error_deg": float(np.median(r_error)),
        "median_t_error_deg": float(np.median(t_error)),
    }


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
    dtype = base.choose_dtype(args.compute_dtype)

    category_dir = args.co3d_root / args.category
    annotations = base.load_frame_annotations(category_dir)
    sequence_quality = base.load_sequence_quality(category_dir)
    sequence_names = discover_sequences(args)

    selected_sequences: list[tuple[str, float, dict[int, base.FrameRecord]]] = []
    for sequence_name in sequence_names:
        quality = sequence_quality.get(sequence_name, float("nan"))
        if not args.include_unqualified and (not np.isfinite(quality) or quality <= args.min_quality):
            print(f"Skipping {sequence_name}: viewpoint_quality_score={quality}")
            continue
        records_by_frame = records_for_sequence(args.co3d_root, args.category, sequence_name, annotations)
        if len(records_by_frame) < args.num_frames:
            print(f"Skipping {sequence_name}: only {len(records_by_frame)} local annotated frames")
            continue
        selected_sequences.append((sequence_name, quality, records_by_frame))

    if not selected_sequences:
        raise SystemExit("No local sequences can satisfy the stride-frame request.")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {dtype}")
    print(
        "Stride frames: "
        f"{[args.start_frame + args.frame_stride * i for i in range(args.num_frames)]}"
    )
    print(f"Sequences: {[(name, quality, len(records)) for name, quality, records in selected_sequences]}")

    model = base.load_model(args, device, dtype)

    rows: list[dict[str, str]] = []
    all_r_errors: list[np.ndarray] = []
    all_t_errors: list[np.ndarray] = []

    for sequence_name, quality, records_by_frame in selected_sequences:
        print(f"\n{args.category}/{sequence_name} ({len(records_by_frame)} local frames, quality={quality})")
        try:
            chosen = select_stride_records(records_by_frame, args)
            result = evaluate_chosen_records(model, chosen, args.co3d_root, args, device, dtype)
            all_r_errors.append(result["r_error"])
            all_t_errors.append(result["t_error"])
            row = {
                "sequence": sequence_name,
                "status": "ok",
                "sequence_quality": f"{quality:.6f}" if np.isfinite(quality) else "nan",
                "available_frames": str(len(records_by_frame)),
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
                "sequence": sequence_name,
                "status": f"error: {exc}",
                "sequence_quality": f"{quality:.6f}" if np.isfinite(quality) else "nan",
                "available_frames": str(len(records_by_frame)),
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
        "auc@30": base.calculate_auc_np(r_errors, t_errors, 30) * 100.0,
        "auc@15": base.calculate_auc_np(r_errors, t_errors, 15) * 100.0,
        "auc@5": base.calculate_auc_np(r_errors, t_errors, 5) * 100.0,
        "auc@3": base.calculate_auc_np(r_errors, t_errors, 3) * 100.0,
        "r_acc@5": float(np.mean(r_errors < 5.0) * 100.0) if len(r_errors) else 0.0,
        "t_acc@5": float(np.mean(t_errors < 5.0) * 100.0) if len(t_errors) else 0.0,
        "median_pose_error_deg": float(np.median(pose_errors)) if len(pose_errors) else float("inf"),
        "median_r_error_deg": float(np.median(r_errors)) if len(r_errors) else float("inf"),
        "median_t_error_deg": float(np.median(t_errors)) if len(t_errors) else float("inf"),
        "settings": {
            "co3d_root": str(args.co3d_root),
            "set_lists": args.set_lists,
            "sequence_names": args.sequence_names,
            "num_frames": args.num_frames,
            "start_frame": args.start_frame,
            "frame_stride": args.frame_stride,
            "min_quality": args.min_quality,
            "include_unqualified": args.include_unqualified,
            "preprocess": args.preprocess,
            "input_size": args.input_size,
            "compute_dtype": str(dtype),
        },
    }

    csv_path = args.out_root / f"{args.tag}_results.csv"
    json_path = args.out_root / f"{args.tag}_metrics.json"
    fieldnames = [
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

    print("\nStride Camera Pose Summary")
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
