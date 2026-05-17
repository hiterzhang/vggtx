FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV TORCH_HOME=/opt/vggt-cache/torch
ENV VGGT_PYTHON=/opt/vggt-env/bin/python
ENV PATH=/opt/vggt-env/bin:$PATH

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/vggt-env \
    && /opt/vggt-env/bin/python -m pip install --upgrade pip setuptools wheel

WORKDIR /app/vggt

COPY pyproject.toml requirements.txt requirements_demo.txt LICENSE.txt README.md CODE_OF_CONDUCT.md CONTRIBUTING.md ./
COPY *.py ./
COPY vggt ./vggt
COPY docs ./docs
COPY training ./training
COPY run_example_lowmem.sh ./run_example_lowmem.sh

RUN chmod +x ./run_example_lowmem.sh \
    && /opt/vggt-env/bin/python -m pip install \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && /opt/vggt-env/bin/python -m pip install -e ".[demo]" \
    && /opt/vggt-env/bin/python -m pip install "httpx[socks]" socksio

RUN mkdir -p "${TORCH_HOME}/hub/checkpoints" \
    && /opt/vggt-env/bin/python - <<'PY'
import torch

print("torch", torch.__version__, "cuda_runtime", torch.version.cuda)
PY

EXPOSE 8080

CMD ["./run_example_lowmem.sh", "room", "all", "8080"]
