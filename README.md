# vLLM with sm_70 (Volta) Support — **Prebuilt PyTorch, No Source Builds**

## TLDR

```bash
docker pull ghcr.io/jajmangold/vllm-sm70:latest
```
## What's in This Repo?

Docker image for running the latest **vLLM** on older NVIDIA GPUs with **sm_70** compute capability (Volta architecture), including:

- Tesla V100  
- Titan V  
- Quadro GV100  
- **NVIDIA CMP 100-210** (mining GPUs)

---

## Why This Exists

Newer official vLLM images and recent PyTorch releases increasingly **drop or de-prioritize Volta (sm_70)** support.  

This project takes the pragmatic route:

- Use a **known-good prebuilt PyTorch image** that still includes `sm_70`
- Layer **vLLM on top** without touching Torch internals
- Focus on **running inference on Volta**

If you just want new vLLM versions to run on V100 / CMP 100-210 cards, this is the boring solution that works.

---

## Base Stack

- **Base image**: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
- **CUDA**: 12.8
- **cuDNN**: 9
- **PyTorch**: 2.7.1 (prebuilt, includes `sm_70`)
- **vLLM**: latest
- **Python**: from base image

---

## Pre-built Image

Pull the pre-built image from GitHub Container Registry:

```bash
docker pull ghcr.io/jajmangold/vllm-sm70:latest
```
