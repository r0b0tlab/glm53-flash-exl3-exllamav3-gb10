# Third-party notices

This project builds on and credits the following upstream projects. No
upstream code is redistributed in this repository; the container image built
by `scripts/build_image.sh` compiles/installs them from the pinned revisions
recorded in `runtime.lock.json`.

## ExLlamaV3
- Upstream: https://github.com/turboderp-org/exllamav3 (turboderp)
- Pinned revision: 12414d0af7b3beeabdda5990f6b554b996fa1416 (v1.5.2)
- License: MIT

## TabbyAPI
- Upstream: https://github.com/theroyallab/tabbyAPI
- Pinned revision: 816c32195887aaecea1c64528f2921566766259b
- License: GNU AGPL-3.0. We use it unmodified; AGPL source is available at
  the upstream repository at the pinned revision.

## NVIDIA CUDA base image
- nvidia/cuda:13.0.2-devel-ubuntu24.04 (digest in runtime.lock.json)
- NVIDIA Container License terms apply to the base layers.

## PyTorch
- 2.13.0+cu130 — BSD-style license (upstream).

## Models
- Target pack: our EXL3 quantization of zai-org/GLM-5.3-Flash (MIT).
- Draft: our EXL3 quantization of incoai/GLM-5.3-Flash-DFlash2
  (CC BY-NC-ND 4.0 upstream; our draft quant is published for
  research/evaluation on that basis).
