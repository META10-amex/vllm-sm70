# vLLM with sm_70 (Volta) support
# Uses PREBUILT PyTorch — no source builds

ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# -------- build args --------
ARG VLLM_VERSION=0.14.1

# -------- env --------
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.0" \
    HF_HOME=/root/.cache/huggingface \
    NVIDIA_DISABLE_REQUIRE=1 \
    VLLM_TARGET_DEVICE=cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1

# -------- system deps --------
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

# -------- python deps --------
RUN pip install --no-cache-dir \
    numpy \
    pyyaml \
    packaging \
    typing_extensions \
    aiohttp \
    bitsandbytes \
    auto-round \
    "transformers>=5.1.0"

# -------- install vLLM --------
RUN pip install --no-cache-dir vllm==${VLLM_VERSION}

# -------- sanity check (safe at build time) --------
RUN python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
