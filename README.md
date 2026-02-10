# vLLM with sm_70 (Volta) Support — **Prebuilt PyTorch, No Source Builds**

Docker image for running **vLLM v0.14.1** on older NVIDIA GPUs with **sm_70** compute capability (Volta architecture), including:

- Tesla V100  
- Titan V  
- Quadro GV100  
- **NVIDIA CMP 100-210** (mining GPUs)

This version intentionally **does NOT build PyTorch from source**.

---

## Why This Exists

Newer official vLLM images and recent PyTorch releases increasingly **drop or de-prioritize Volta (sm_70)** support.  
At the same time, building PyTorch from source on modern Linux + Python (Ubuntu 24.04 / Python 3.12) is slow, fragile, and painful.

This project takes the pragmatic route:

- Use a **known-good prebuilt PyTorch image** that still includes `sm_70`
- Layer **vLLM on top** without touching Torch internals
- Avoid PEP-517, setuptools, editable installs, and CUDA build hell
- Focus on **actually running inference on Volta**

If you just want vLLM to run on V100 / CMP cards, this is the boring solution that works.

---

## What’s Changed vs Older Versions

**Old approach (deprecated):**
- PyTorch 2.9.x built from source
- 3–4 hour builds
- Extremely fragile on Python 3.12
- Constant breakage from packaging changes

**Current approach (this repo):**
- **Prebuilt PyTorch 2.7.1**
- CUDA 12.8
- sm_70 kernels included
- vLLM installed from wheels
- Build time: minutes, not hours

---

## Base Stack

- **Base image**: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
- **CUDA**: 12.8
- **cuDNN**: 9
- **PyTorch**: 2.7.1 (prebuilt, includes `sm_70`)
- **vLLM**: 0.14.1
- **Python**: from base image

No PyTorch source compilation. No custom CUDA builds.

---

## Pre-built Image

Pull the pre-built image from GitHub Container Registry:

```bash
docker pull ghcr.io/jajmangold/vllm-sm70:latest
