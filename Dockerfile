# vLLM with sm_70 (Volta) support
# Builds vLLM from source so its CUDA kernels include sm_70.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG VLLM_VERSION=0.20.2
ARG MAX_JOBS=8
ARG NVCC_THREADS=2

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.0" \
    CMAKE_CUDA_ARCHITECTURES=70 \
    CUDAARCHS=70 \
    CMAKE_CUDA_FLAGS="-gencode=arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets" \
    NVCC_PREPEND_FLAGS="-gencode=arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets" \
    HF_HOME=/root/.cache/huggingface \
    NVIDIA_DISABLE_REQUIRE=1 \
    VLLM_TARGET_DEVICE=cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

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

RUN pip install --no-cache-dir --upgrade pip "setuptools<82" wheel

RUN git clone --depth 1 --branch "v${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git /opt/vllm

WORKDIR /opt/vllm

RUN python3 -c 'from pathlib import Path; cm=Path("CMakeLists.txt"); txt=cm.read_text(); old="    # vllm-flash-attn should be last as it overwrites some CMake functions\n    include(cmake/external_projects/vllm_flash_attn.cmake)\n"; new="    message(STATUS \"Skipping vllm-flash-attn for Volta-only sm_70 build\")\n"; assert old in txt, "vllm-flash-attn CMake include block not found"; cm.write_text(txt.replace(old, new)); setup=Path("setup.py"); txt=setup.read_text(); start="if _is_cuda():\n    ext_modules.append(CMakeExtension(name=\"vllm.vllm_flash_attn._vllm_fa2_C\"))"; end="    if envs.VLLM_USE_PRECOMPILED or (\n        CUDA_HOME and get_nvcc_cuda_version() >= Version(\"12.9\")\n    ):"; i=txt.index(start); j=txt.index(end, i); repl="if _is_cuda():\n    # Volta cannot use FA2/FA3/FA4 kernels; skip building those extensions.\n"; setup.write_text(txt[:i] + repl + txt[j:])'

RUN python3 -c 'from pathlib import Path; import torch, triton; Path("/opt/vllm/constraints.txt").write_text(f"torch=={torch.__version__}\ntriton=={triton.__version__}\n")'

RUN python3 use_existing_torch.py && \
    pip install --no-cache-dir -r requirements/build/cuda.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_VERSION} \
    MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} \
        pip install --no-cache-dir --no-build-isolation -c /opt/vllm/constraints.txt -v .

RUN pip install --no-cache-dir -c /opt/vllm/constraints.txt bitsandbytes auto-round

WORKDIR /root

RUN python3 - <<'PY'
import torch
import vllm
import vllm._C

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)

arches = torch.cuda.get_arch_list()
print("cuda arch list", arches)

assert torch.__version__.startswith("2.10.0"), torch.__version__
assert "sm_70" in arches, arches
PY

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
