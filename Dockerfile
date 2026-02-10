# Prebuilt PyTorch with Volta (sm_70) support
# No PyTorch source builds

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# ---------- env ----------
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.0" \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/root/.cache/huggingface \
    NVIDIA_DISABLE_REQUIRE=1

# ---------- system deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    curl \
    ca-certificates \
    patchelf \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- sanity check: sm_70 present ----------
# (safe at runtime; this image already has CUDA)
RUN python3 -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"

# ---------- Python deps ----------
RUN pip install --no-cache-dir \
    numpy \
    pyyaml \
    packaging \
    typing_extensions \
    aiohttp

# ---------- quant / inference deps ----------
RUN pip install --no-cache-dir \
    bitsandbytes \
    auto-round

# ---------- install vLLM (prebuilt wheel) ----------
# vLLM wheels are CUDA-version aware; this works with cu128
RUN pip install --no-cache-dir vllm==0.14.1

# ---------- runtime tuning ----------
ENV VLLM_ATTENTION_BACKEND=xformers \
    VLLM_TARGET_DEVICE=cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
