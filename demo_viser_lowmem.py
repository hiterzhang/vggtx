# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Low-VRAM viser demo wrapper for local reproduction on 8GB GPUs.

import argparse
import gc
import glob
import os
import time

import numpy as np
import torch


def _normalize_proxy_env() -> None:
    """httpx/gradio expects socks5:// rather than socks://."""
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.startswith("socks://"):
            os.environ[key] = "socks5://" + value[len("socks://") :]


_normalize_proxy_env()

from demo_viser import viser_wrapper  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


DEFAULT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def _select_compute_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _load_vggt_state_dict() -> dict:
    model_path = os.environ.get("VGGT_MODEL_PATH")
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"VGGT_MODEL_PATH does not exist: {model_path}. "
                "Mount model.pt into the container or unset VGGT_MODEL_PATH to allow downloading."
            )
        print(f"Loading VGGT weights from {model_path}")
        return torch.load(model_path, map_location="cpu")

    url = os.environ.get("VGGT_MODEL_URL", DEFAULT_MODEL_URL)
    print(f"Loading VGGT weights from {url}")
    return torch.hub.load_state_dict_from_url(url, map_location="cpu")


def run_lowmem_inference(
    image_folder: str,
    compute_dtype: torch.dtype,
    frame_chunk_size: int,
    output_npz: str | None,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This low-VRAM runner is intended for CUDA inference.")

    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    if not image_names:
        raise ValueError(f"No images found under {image_folder}")

    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"Using aggregator dtype: {compute_dtype}")
    print(f"Found {len(image_names)} images")

    print("Initializing and loading VGGT model...")
    model = VGGT()
    state = _load_vggt_state_dict()
    model.load_state_dict(state)
    del state
    model.eval()

    print(f"Loading images from {image_folder}...")
    images = load_and_preprocess_images(image_names).to(device)
    images_batched = images[None]
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    start = time.time()
    print("Running aggregator in low-VRAM mode...")
    model.aggregator.to(device=device, dtype=compute_dtype)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=compute_dtype):
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_batched)
    torch.cuda.synchronize()
    print(f"Aggregator finished in {time.time() - start:.2f}s")

    # DPT heads only use these four layers; camera uses the final layer.
    needed_layers = set(model.depth_head.intermediate_layer_idx + [len(aggregated_tokens_list) - 1])
    selected_tokens = [None] * len(aggregated_tokens_list)
    for idx in sorted(needed_layers):
        selected_tokens[idx] = aggregated_tokens_list[idx].float()
    del aggregated_tokens_list
    model.aggregator.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Kept token layers: {sorted(needed_layers)}")

    print("Running camera head in FP32...")
    model.camera_head.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pose_enc = model.camera_head(selected_tokens)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batched.shape[-2:])
    model.camera_head.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    print("Running depth head in FP32...")
    model.depth_head.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        depth, depth_conf = model.depth_head(
            selected_tokens,
            images=images_batched.float(),
            patch_start_idx=patch_start_idx,
            frames_chunk_size=frame_chunk_size,
        )
    model.depth_head.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    predictions = {
        "images": images_batched.cpu().numpy().squeeze(0),
        "depth": depth.cpu().numpy().squeeze(0),
        "depth_conf": depth_conf.cpu().numpy().squeeze(0),
        "extrinsic": extrinsic.cpu().numpy().squeeze(0),
        "intrinsic": intrinsic.cpu().numpy().squeeze(0),
    }

    # The default viser path uses depth + camera. Keep these placeholders so the
    # original wrapper can still unpack the standard VGGT prediction dictionary.
    num_frames, _, height, width = predictions["images"].shape
    predictions["world_points"] = np.zeros((num_frames, height, width, 3), dtype=np.float32)
    predictions["world_points_conf"] = np.zeros((num_frames, height, width), dtype=np.float32)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak CUDA memory: {peak_gb:.2f} GB")

    if output_npz:
        os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
        np.savez_compressed(output_npz, **predictions)
        print(f"Saved predictions to {output_npz}")

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-VRAM VGGT viser demo")
    parser.add_argument("--image_folder", type=str, default="examples/kitchen_3/images")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=25.0)
    parser.add_argument("--compute_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--frame_chunk_size", type=int, default=1)
    parser.add_argument("--background_mode", action="store_true")
    parser.add_argument("--mask_sky", action="store_true")
    parser.add_argument(
        "--output_npz",
        type=str,
        default="outputs/kitchen_3_lowmem_predictions.npz",
        help="Where to save the low-VRAM predictions before launching viser.",
    )
    args = parser.parse_args()

    predictions = run_lowmem_inference(
        image_folder=args.image_folder,
        compute_dtype=_select_compute_dtype(args.compute_dtype),
        frame_chunk_size=args.frame_chunk_size,
        output_npz=args.output_npz,
    )

    print("Starting viser visualization...")
    viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=False,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )


if __name__ == "__main__":
    main()
