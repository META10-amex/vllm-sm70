# vLLM with sm_70 (Volta) Support

## TL;DR

```bash
docker pull ghcr.io/jajmangold/vllm-sm70:latest
````

---

## 👉 Looking for a more complete V100 stack? Use 1Cat-vLLM

For serious Volta serving — especially **Qwen3.5 / Qwen3.6** — consider
**[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)** instead. It's a
full vLLM fork purpose-built for SM70 with capabilities this image does **not**
have:

* A real **`FLASH_ATTN_V100`** attention kernel (TurboMind-derived SM70 WMMA),
  not just the Triton/SDPA fallback — enable with `--attention-backend FLASH_ATTN_V100`.
* **AWQ 4-bit** paths validated for dense and MoE Qwen models on V100.
* Validated multi-GPU **Qwen3.5 27B / 35B / 122B** (TP2/TP4) deployments.
* Distributed as prebuilt **wheels** (`vllm` + `flash_attn_v100`) on torch 2.9.1+cu128.

This repo remains a simple "build latest vLLM for sm_70" Docker image; 1Cat-vLLM
is the better choice if you need the V100 FlashAttention path or large/long-context
Qwen models.

> Note on newer vLLM + Volta: vLLM 0.20.x pins torch 2.11.0, which dropped Volta
> from its cu128/cu130 wheels — only the **cu126** build keeps sm_70. This image's
> Dockerfile is based on the cuda12.6 PyTorch image for that reason.

---

## What’s in This Repo?

A Docker image for running the **latest vLLM** on older NVIDIA GPUs with **sm_70** compute capability (Volta architecture), including:

* Tesla V100
* Titan V
* Quadro GV100
* NVIDIA CMP 100-210 (mining GPUs)

This image is built to be **feature-complete for inference on Volta**, not a crippled fallback.

---

## What You Actually Get (Important)

Despite running on Volta, this image includes the **modern inference stack** you care about:

* ✅ **Triton attention** 
* ✅ **PyTorch SDPA** (scaled dot-product attention fallback)
* ✅ **bitsandbytes (bnb)** for efficient quantized weights
* ✅ **AutoRound** for W4A16 / low-bit quantization workflows
* ✅ **CUDA graphs** (enabled by default in vLLM)
* ✅ **Continuous batching** and KV cache reuse (vLLM core features)

What you *don’t* get (hardware limits, not software):

* ❌ FlashAttention v2 (requires sm_80+)
* ❌ FP8 / Hopper-only kernels
* ❌ Marlin (Ampere+)

---

## Why This Exists

Newer official vLLM images and recent PyTorch releases increasingly **drop or de-prioritize Volta (sm_70)** support.

This project takes the pragmatic route:

* Use a **known-good prebuilt PyTorch image** that still includes `sm_70`
* Include **AutoRound** for SOTA modern quantized inference
* Avoid PyTorch source builds
* Focus on **running inference on Volta**, not fighting packaging

---

## Base Stack

* **Base image**: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
* **CUDA**: 12.8
* **cuDNN**: 9
* **PyTorch**: 2.7.1 (prebuilt, includes `sm_70`)
* **vLLM**: latest (auto-built from upstream releases)
* **Attention backends**:
  
  * Triton
  * xFormers
  * PyTorch SDPA
* **Quantization tooling**:

  * bitsandbytes
  * AutoRound
* **Python**: from base image

---

## Pre-built Image

The latest version of vLLM checked **nightly** . Builds happen when upstream releases happen.

Pull the pre-built image from GitHub Container Registry:

```bash
docker pull ghcr.io/jajmangold/vllm-sm70:latest
```

This tag always tracks:

* the newest upstream vLLM release
* a Volta-compatible PyTorch base
* a full inference feature set ( bnb, AutoRound)

---
