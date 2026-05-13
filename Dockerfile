 # vLLM with sm_70 (Volta) support
  # Uses PREBUILT PyTorch; no PyTorch source build

  ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
  FROM ${PYTORCH_IMAGE}

  ARG VLLM_VERSION=0.20.2

  ENV DEBIAN_FRONTEND=noninteractive \
      TORCH_CUDA_ARCH_LIST="7.0" \
      HF_HOME=/root/.cache/huggingface \
      NVIDIA_DISABLE_REQUIRE=1 \
      VLLM_TARGET_DEVICE=cuda \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_DEVICE_MAX_CONNECTIONS=1 \
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

  RUN pip install --no-cache-dir \
      numpy \
      pyyaml \
      packaging \
      typing_extensions \
      aiohttp \
      bitsandbytes \
      auto-round \
      "transformers>=5.1.0"

  # vLLM's wheel dependencies may replace the Volta-capable Torch build.
  RUN pip install --no-cache-dir vllm==${VLLM_VERSION} \
      && pip install --no-cache-dir --force-reinstall \
         --index-url https://download.pytorch.org/whl/cu128 \
         torch==2.7.1+cu128 \
         torchvision==0.22.1+cu128 \
         torchaudio==2.7.1+cu128

  RUN python3 - <<'PY'
  import torch
  print("torch", torch.__version__, "cuda", torch.version.cuda)
  arches = torch.cuda.get_arch_list()
  print("cuda arch list", arches)
  assert torch.__version__.startswith("2.7.1"), torch.__version__
  assert "sm_70" in arches, arches
  PY

  EXPOSE 8000

  ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
