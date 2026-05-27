#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_ROOT="${VGGT_EXAMPLES_ROOT:-${ROOT}/examples}"
PYTHON_BIN="${VGGT_PYTHON:-/home/zzh/anaconda3/envs/vggt/bin/python}"

scene="${1:-room}"
count="${2:-all}"
port="${3:-8080}"

if [[ "${scene}" == "--list" || "${scene}" == "list" ]]; then
  if [[ ! -d "${EXAMPLES_ROOT}" ]]; then
    echo "Examples root not found: ${EXAMPLES_ROOT}" >&2
    exit 1
  fi
  find "${EXAMPLES_ROOT}" -mindepth 2 -maxdepth 2 -type d -name images \
    | sed "s#${EXAMPLES_ROOT%/}/##; s#/images##" \
    | sort
  exit 0
fi

src_dir="${EXAMPLES_ROOT%/}/${scene}/images"
if [[ ! -d "${src_dir}" ]]; then
  echo "Example scene not found: ${scene}" >&2
  echo "Examples root: ${EXAMPLES_ROOT}" >&2
  echo "Available scenes:" >&2
  "${BASH_SOURCE[0]}" --list >&2
  exit 1
fi

if [[ "${count}" == "all" ]]; then
  image_folder="${src_dir}"
  count_label="all"
else
  if ! [[ "${count}" =~ ^[0-9]+$ ]] || [[ "${count}" -lt 1 ]]; then
    echo "Count must be a positive integer or 'all'." >&2
    exit 1
  fi

  mapfile -t images < <(find "${src_dir}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort)
  if [[ "${count}" -gt "${#images[@]}" ]]; then
    echo "${scene} only has ${#images[@]} images; requested ${count}." >&2
    exit 1
  fi

  subset_dir="${ROOT}/outputs/selected_examples/${scene}_${count}/images"
  rm -rf "${subset_dir}"
  mkdir -p "${subset_dir}"
  for ((i = 0; i < count; i++)); do
    image="${images[$i]}"
    ext="${image##*.}"
    ln -s "$(realpath "${image}")" "${subset_dir}/$(printf '%03d' "${i}").${ext}"
  done
  image_folder="${subset_dir}"
  count_label="${count}"
fi

output_npz="${ROOT}/outputs/${scene}_${count_label}_lowmem_predictions.npz"

if [[ "${VGGT_REQUIRE_LOCAL_WEIGHTS:-0}" == "1" ]]; then
  weight_path="${VGGT_MODEL_PATH:-${TORCH_HOME:-/opt/vggt-cache/torch}/hub/checkpoints/model.pt}"
  if [[ ! -s "${weight_path}" ]]; then
    cat >&2 <<EOF
VGGT model weight was not found:
  ${weight_path}

For production Docker deploys, mount model.pt to:
  /opt/vggt-cache/torch/hub/checkpoints/model.pt

Or set VGGT_MODEL_PATH to the mounted checkpoint path.
EOF
    exit 1
  fi
  export VGGT_MODEL_PATH="${weight_path}"
fi

echo "Scene: ${scene}"
echo "Examples root: ${EXAMPLES_ROOT}"
echo "Images: ${image_folder}"
echo "Port: ${port}"
echo "Output: ${output_npz}"

exec "${PYTHON_BIN}" -u "${ROOT}/demo_viser_lowmem.py" \
  --image_folder "${image_folder}" \
  --port "${port}" \
  --conf_threshold 25 \
  --output_npz "${output_npz}"
