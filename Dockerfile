# vLLM v0.12.0 + PyTorch (source) + sm_70 (Volta / CMP 100-210)
# Explicit support: xFormers, SDPA, bitsandbytes, AutoRound

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.0" \
    CUDAARCHS="70" \
    MAX_JOBS=4 \
    USE_CUDA=1 \
    USE_CUDNN=1 \
    BUILD_TEST=0 \
    PYTORCH_BUILD_VERSION=2.9.0+sm70 \
    PYTORCH_BUILD_NUMBER=1 \
    HF_HOME=/root/.cache/huggingface \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_DISABLE_REQUIRE=1

# ---- system deps ----
RUN apt update && apt install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    cmake \
    ninja-build \
    curl \
    ca-certificates \
    libopenblas-dev \
    libomp-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    numpy \
    wheel \
    "setuptools>=70" \
    pyyaml \
    typing_extensions \
    packaging

# ---- build PyTorch from source (sm_70 enabled, SDPA included) ----
WORKDIR /opt
RUN git clone --branch v2.9.0 --depth 1 --recursive --shallow-submodules https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git submodule sync && \
    git submodule update --init --recursive --depth 1

ENV USE_MKLDNN=0 \
    USE_FBGEMM=0 \
    USE_NNPACK=0 \
    USE_XNNPACK=0 \
    BUILD_CAFFE2=0 \
    USE_QNNPACK=0 \
    USE_ROCM=0 \
    USE_FLASH_ATTENTION=0 \
    USE_MEM_EFF_ATTENTION=1

WORKDIR /opt/pytorch
RUN pip install --break-system-packages --no-cache-dir -e . -v --no-build-isolation && \
    rm -rf /opt/pytorch/.git /opt/pytorch/build /root/.cache/pip && \
    find /opt/pytorch -type d -name __pycache__ -exec rm -rf {} + || true

# ---- xFormers (build from source for sm_70) ----
# ---- xFormers (build from source for sm_70) ----
WORKDIR /opt
RUN git clone --branch v0.0.28 --depth 1 --recursive https://github.com/facebookresearch/xformers.git && \
    cd xformers && \
    git submodule sync && \
    git submodule update --init --recursive --depth 1


WORKDIR /opt/xformers
ENV XFORMERS_BUILD_TYPE=Release \
    XFORMERS_ENABLE_FLASH_ATTENTION=0 \
    XFORMERS_ENABLE_MEM_EFF_ATTENTION=1 \
    XFORMERS_FORCE_DISABLE_TRITON=0

RUN pip install --break-system-packages --no-cache-dir -e . -v --no-build-isolation && \
    rm -rf /opt/xformers/.git /root/.cache/pip

# ---- bitsandbytes (Volta-safe source build) ----
WORKDIR /opt
RUN git clone --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes.git

WORKDIR /opt/bitsandbytes
ENV BNB_CUDA_VERSION=128 \
    BNB_FORCE_BUILD_FROM_SOURCE=1 \
    CUDA_VERSION=128

RUN python3 setup.py install && \
    rm -rf /opt/bitsandbytes/.git /root/.cache/pip

# ---- AutoRound (pure Python, no CUDA kernels) ----
RUN pip install --break-system-packages --no-cache-dir autoround

# ---- build vLLM v0.12.0 ----
WORKDIR /opt
RUN git clone --branch v0.12.0 --depth 1 --recursive --shallow-submodules https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git submodule sync && \
    git submodule update --init --recursive --depth 1

WORKDIR /opt/vllm
RUN python3 -m pip install --break-system-packages --no-cache-dir \
        setuptools_scm \
        cmake \
        ninja \
        packaging && \
    python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation -e . && \
    rm -rf /opt/vllm/.git /root/.cache/pip && \
    find /opt/vllm -type d -name __pycache__ -exec rm -rf {} + || true && \
    find /opt/vllm -type d -name "*.egg-info" -exec rm -rf {} + || true

# ---- sanity flags (runtime selection) ----
ENV VLLM_ATTENTION_BACKEND=auto \
    PYTORCH_ENABLE_MPS_FALLBACK=0

# ---- runtime ----
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
