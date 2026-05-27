# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Lightweight local image matching demo for VGGT's tracking branch.

import argparse
import gc
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def normalize_proxy_env() -> None:
    """httpx expects socks5:// rather than socks:// if proxy envs are present."""
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


def resolve_image_pair(args: argparse.Namespace) -> list[str]:
    if args.image0 and args.image1:
        return [args.image0, args.image1]

    if not args.image_folder:
        raise ValueError("Provide either --image0/--image1 or --image_folder.")

    image_names = sorted(
        glob.glob(os.path.join(args.image_folder, "*.jpg"))
        + glob.glob(os.path.join(args.image_folder, "*.jpeg"))
        + glob.glob(os.path.join(args.image_folder, "*.png"))
    )
    if len(image_names) < 2:
        raise ValueError(f"Need at least two images in {args.image_folder}")

    left_idx, right_idx = [int(x.strip()) for x in args.indices.split(",")]
    return [image_names[left_idx], image_names[right_idx]]


def image_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)


def detect_aliked_points(
    image: torch.Tensor,
    max_points: int,
    margin: int,
    detection_threshold: float,
) -> np.ndarray:
    try:
        from lightglue import ALIKED
    except ImportError as exc:
        raise RuntimeError(
            "ALIKED requires LightGlue. Install it with: "
            "python -m pip install 'git+https://github.com/jytime/LightGlue.git#egg=lightglue'"
        ) from exc

    device = image.device
    extractor = ALIKED(max_num_keypoints=max_points, detection_threshold=detection_threshold).to(device).eval()
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


def detect_query_points(
    image: torch.Tensor,
    max_points: int,
    margin: int,
    method: str,
    aliked_threshold: float,
) -> np.ndarray:
    image_u8 = image_tensor_to_uint8(image)
    height, width = image_u8.shape[:2]

    if method == "aliked":
        points = detect_aliked_points(
            image,
            max_points=max_points,
            margin=margin,
            detection_threshold=aliked_threshold,
        )
        if len(points) >= max(8, max_points // 20):
            return points
        return detect_query_points(
            image,
            max_points=max_points,
            margin=margin,
            method="good_features",
            aliked_threshold=aliked_threshold,
        )

    if method == "grid":
        step = int(np.sqrt((width - 2 * margin) * (height - 2 * margin) / max(max_points, 1)))
        step = max(step, 8)
        xs = np.arange(margin, width - margin, step)
        ys = np.arange(margin, height - margin, step)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1).astype(np.float32)
        return points[:max_points]

    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points * 3,
        qualityLevel=0.005,
        minDistance=8,
        blockSize=7,
    )

    if corners is None or len(corners) < max(32, max_points // 10):
        return detect_query_points(
            image,
            max_points=max_points,
            margin=margin,
            method="grid",
            aliked_threshold=aliked_threshold,
        )

    points = corners.reshape(-1, 2).astype(np.float32)
    valid = (
        (points[:, 0] >= margin)
        & (points[:, 0] < width - margin)
        & (points[:, 1] >= margin)
        & (points[:, 1] < height - margin)
    )
    points = points[valid]
    return points[:max_points]


def draw_matches(
    image0: torch.Tensor,
    image1: torch.Tensor,
    src: np.ndarray,
    dst: np.ndarray,
    inliers: np.ndarray,
    scores: np.ndarray,
    out_path: str,
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

    for rank, idx in enumerate(order):
        color = (0, 220, 0) if inliers[idx] else (230, 80, 60)
        p0 = tuple(np.round(src[idx]).astype(int))
        p1 = tuple(np.round(dst[idx] + np.array([width0, 0], dtype=np.float32)).astype(int))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)

    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def run(args: argparse.Namespace) -> None:
    normalize_proxy_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This demo expects CUDA; VGGT-1B is too heavy for a practical CPU matching test.")

    image_names = resolve_image_pair(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Image pair:")
    print(f"  0: {image_names[0]}")
    print(f"  1: {image_names[1]}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    images = load_and_preprocess_images(image_names, mode=args.preprocess).to(device)
    images_batched = images[None]
    _, _, height, width = images.shape
    print(f"Preprocessed shape: {tuple(images.shape)}")

    query_np = detect_query_points(
        images[0],
        max_points=args.max_points,
        margin=args.margin,
        method=args.keypoints,
        aliked_threshold=args.aliked_threshold,
    )
    query_points = torch.from_numpy(query_np).to(device=device, dtype=torch.float32)[None]
    print(f"Query points: {query_points.shape[1]}")

    compute_dtype = choose_dtype(args.compute_dtype)
    print(f"Aggregator dtype: {compute_dtype}")

    model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=True)
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    load_result = model.load_state_dict(state, strict=False)
    del state
    print(f"Loaded model. Ignored unexpected keys: {len(load_result.unexpected_keys)}")
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    model.aggregator.to(device=device, dtype=compute_dtype)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=compute_dtype):
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_batched)

    needed_layers = set(model.track_head.feature_extractor.intermediate_layer_idx)
    selected_tokens = [None] * len(aggregated_tokens_list)
    for idx in sorted(needed_layers):
        selected_tokens[idx] = aggregated_tokens_list[idx].float()

    del aggregated_tokens_list
    model.aggregator.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    model.track_head.to(device=device, dtype=torch.float32)
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
    dst_in_bounds = (
        (dst[:, 0] >= 0)
        & (dst[:, 0] < width)
        & (dst[:, 1] >= 0)
        & (dst[:, 1] < height)
    )
    vis_ok = vis >= args.vis_threshold
    conf_ok = conf >= args.conf_threshold
    valid = (
        finite_tracks
        & vis_ok
        & conf_ok
        & dst_in_bounds
    )

    src_valid = src[valid]
    dst_valid = dst[valid]
    scores_valid = scores[valid]
    print(
        "Filter diagnostics: "
        f"finite={int(finite_tracks.sum())}, "
        f"vis_ok={int(vis_ok.sum())}, "
        f"conf_ok={int(conf_ok.sum())}, "
        f"dst_in_bounds={int(dst_in_bounds.sum())}"
    )
    print(f"Valid matches after score/bounds filtering: {len(src_valid)}")

    if len(src_valid) >= 8:
        _, inlier_mask = cv2.findFundamentalMat(
            src_valid,
            dst_valid,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=args.ransac_threshold,
            confidence=0.999,
            maxIters=10000,
        )
        if inlier_mask is None:
            inliers = np.zeros(len(src_valid), dtype=bool)
        else:
            inliers = inlier_mask.reshape(-1).astype(bool)
    else:
        inliers = np.zeros(len(src_valid), dtype=bool)

    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(len(src_valid), 1)
    print(f"RANSAC F inliers: {inlier_count}/{len(src_valid)} ({inlier_ratio:.3f})")
    print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    np.savez_compressed(
        out_dir / "matches.npz",
        image_names=np.array(image_names),
        src=src_valid,
        dst=dst_valid,
        scores=scores_valid,
        inliers=inliers,
        preprocessed_hw=np.array([height, width]),
    )

    draw_matches(
        images[0],
        images[1],
        src_valid,
        dst_valid,
        inliers,
        scores_valid,
        str(out_dir / "matches.png"),
        max_draw=args.max_draw,
    )

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"image0: {image_names[0]}\n")
        f.write(f"image1: {image_names[1]}\n")
        f.write(f"preprocessed_shape: {tuple(images.shape)}\n")
        f.write(f"keypoint_method: {args.keypoints}\n")
        if args.keypoints == "aliked":
            f.write(f"aliked_threshold: {args.aliked_threshold}\n")
        f.write(f"query_points: {query_points.shape[1]}\n")
        f.write(f"finite_tracks: {int(finite_tracks.sum())}\n")
        f.write(f"vis_ok: {int(vis_ok.sum())}\n")
        f.write(f"conf_ok: {int(conf_ok.sum())}\n")
        f.write(f"dst_in_bounds: {int(dst_in_bounds.sum())}\n")
        f.write(f"mean_vis: {float(np.mean(vis)):.6f}\n")
        f.write(f"mean_conf: {float(np.mean(conf)):.6f}\n")
        f.write(f"valid_matches: {len(src_valid)}\n")
        f.write(f"ransac_inliers: {inlier_count}\n")
        f.write(f"ransac_inlier_ratio: {inlier_ratio:.6f}\n")

    print(f"Saved visualization: {out_dir / 'matches.png'}")
    print(f"Saved raw matches:   {out_dir / 'matches.npz'}")
    print(f"Saved summary:       {out_dir / 'summary.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VGGT lightweight two-view image matching demo")
    parser.add_argument("--image0", type=str, default=None)
    parser.add_argument("--image1", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="examples/kitchen_3/images")
    parser.add_argument("--indices", type=str, default="0,1", help="Pair indices when using --image_folder")
    parser.add_argument("--out_dir", type=str, default="outputs/matching/kitchen_00_01")
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--max_draw", type=int, default=200)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=0.0)
    parser.add_argument("--ransac_threshold", type=float, default=1.5)
    parser.add_argument("--preprocess", choices=["crop", "pad"], default="crop")
    parser.add_argument("--keypoints", choices=["aliked", "good_features", "grid"], default="aliked")
    parser.add_argument("--aliked_threshold", type=float, default=0.005)
    parser.add_argument("--compute_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
