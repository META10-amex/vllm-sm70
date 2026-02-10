# vLLM with sm_70 (Volta) Support

## TL;DR

```bash
docker pull ghcr.io/jajmangold/vllm-sm70:latest
````

---

## What’s in This Repo?

A Docker image for running the **latest vLLM** on older NVIDIA GPUs with **sm_70** compute capability (Volta architecture), including:

* Tesla V100
* Titan V
* Quadro GV100
* **NVIDIA CMP 100-210** (mining GPUs)

This image is built to be **feature-complete for inference on Volta**, not a crippled fallback.

---

## What You Actually Get (Important)

Despite running on Volta, this image includes the **modern inference stack** you care about:

* ✅ **xFormers attention** (CUTLASS-backed kernels where applicable)
* ✅ **PyTorch SDPA** (scaled dot-product attention fallback)
* ✅ **bitsandbytes (bnb)** for efficient quantized weights
* ✅ **AutoRound** for W4A16 / low-bit quantization workflows
* ✅ **CUDA graphs** (enabled by default in vLLM)
* ✅ **Continuous batching** and KV cache reuse (vLLM core features)

What you *don’t* get (hardware limits, not software):

* ❌ FlashAttention v2 (requires sm_80+)
* ❌ FP8 / Hopper-only kernels
* ❌ Marlin (Ampere+)

This is the **best possible attention + quantization stack on Volta** without rebuilding PyTorch.

---

## Why This Exists

Newer official vLLM images and recent PyTorch releases increasingly **drop or de-prioritize Volta (sm_70)** support.

This project takes the pragmatic route:

* Use a **known-good prebuilt PyTorch image** that still includes `sm_70`
* Preserve **xFormers + SDPA** attention paths
* Include **bnb + AutoRound** for modern quantized inference
* Avoid PyTorch source builds, PEP-517 pain, and toolchain breakage
* Focus on **running inference on Volta**, not fighting packaging

If you just want **new vLLM versions to keep working** on V100 / CMP 100-210 cards, this is the boring solution that works.

---

## Base Stack

* **Base image**: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
* **CUDA**: 12.8
* **cuDNN**: 9
* **PyTorch**: 2.7.1 (prebuilt, includes `sm_70`)
* **vLLM**: latest (auto-built from upstream releases)
* **Attention backends**:

  * xFormers
  * PyTorch SDPA
* **Quantization tooling**:

  * bitsandbytes
  * AutoRound
* **Python**: from base image

---

## Pre-built Image

The latest version of vLLM is built **nightly** .

Pull the pre-built image from GitHub Container Registry:

```bash
docker pull ghcr.io/jajmangold/vllm-sm70:latest
```

This tag always tracks:

* the newest upstream vLLM release
* a Volta-compatible PyTorch base
* a full inference feature set (xFormers, SDPA, bnb, AutoRound)

---
