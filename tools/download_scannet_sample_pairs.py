#!/usr/bin/env python3
"""Download a tiny ScanNet-style image-pair sample for local matching tests.

This script downloads the small ScanNet sample distributed with
magicleap/SuperGluePretrainedNetwork, not the full ScanNet-1500 benchmark.
It is useful when disk space is tight and you only want a few indoor pairs
with intrinsics / relative pose metadata.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path


PAIR_FILE_URL = (
    "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/"
    "master/assets/scannet_sample_pairs_with_gt.txt"
)
IMAGE_BASE_URL = (
    "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/"
    "master/assets/scannet_sample_images"
)


def normalize_proxy_env() -> None:
    """urllib supports socks only with extra handlers; prefer HTTP proxy envs."""
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.startswith("socks://"):
            os.environ.pop(key, None)


def download_url(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst}")
        return

    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    dst.write_bytes(data)
    print(f"[saved] {dst} ({dst.stat().st_size / 1024:.1f} KB)")


def parse_pair_file(path: Path, max_pairs: int) -> list[tuple[str, str, str]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    pairs = []
    for line in lines[:max_pairs]:
        parts = line.split()
        if len(parts) < 2:
            continue
        pairs.append((parts[0], parts[1], line))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path("data/scannet_sample"))
    parser.add_argument("--max_pairs", type=int, default=5)
    args = parser.parse_args()

    normalize_proxy_env()

    pair_file = args.out_dir / "pairs_with_gt.txt"
    images_dir = args.out_dir / "images"

    download_url(PAIR_FILE_URL, pair_file)
    pairs = parse_pair_file(pair_file, args.max_pairs)

    selected_pair_file = args.out_dir / f"pairs_first_{len(pairs)}.txt"
    selected_pair_file.write_text("\n".join(pair[2] for pair in pairs) + "\n")

    needed_images = sorted({name for pair in pairs for name in pair[:2]})
    for image_name in needed_images:
        download_url(f"{IMAGE_BASE_URL}/{image_name}", images_dir / image_name)

    print()
    print(f"Downloaded {len(pairs)} pairs / {len(needed_images)} images")
    print(f"Images: {images_dir}")
    print(f"Pair file: {selected_pair_file}")
    print()
    print("Example VGGT command:")
    first0, first1, _ = pairs[0]
    print(
        "python demo_match_lowmem.py "
        f"--image0 {images_dir / first0} "
        f"--image1 {images_dir / first1} "
        "--out_dir outputs/matching/scannet_sample_000 "
        "--max_points 512"
    )


if __name__ == "__main__":
    main()
