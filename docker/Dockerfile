# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=4 \
    PATH="/opt/venv/bin:$PATH" HOME=/home/runner \
    XDG_CACHE_HOME=/data/cache TRITON_CACHE_DIR=/data/triton \
    TORCH_EXTENSIONS_DIR=/data/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3.12 python3.12-dev python3.12-venv \
    git build-essential ninja-build libgomp1 && \
    rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv && \
    python -m pip install --upgrade pip setuptools wheel ninja packaging
RUN python -m pip install 'torch==2.13.0+cu130' \
    --index-url https://download.pytorch.org/whl/cu130

# The release tag and exact SHA are both checked. Compile the real native extension
# for GB10/SM121; no JIT-only source installation or prebuilt x86 wheel is allowed.
RUN git clone https://github.com/turboderp-org/exllamav3.git /opt/exllamav3-src && \
    git -C /opt/exllamav3-src checkout 12414d0af7b3beeabdda5990f6b554b996fa1416 && \
    test "$(git -C /opt/exllamav3-src rev-parse HEAD)" = "12414d0af7b3beeabdda5990f6b554b996fa1416" && \
    test "$(git -C /opt/exllamav3-src describe --tags --exact-match)" = "v1.5.2" && \
    python -m pip install --no-build-isolation /opt/exllamav3-src

# NVIDIA's cuSPARSELt 0.8.1 ARM64 wheel contains an SBSA WHEEL tag despite its
# AArch64 filename/ELF. Correct only that exact tag and RECORD after the native
# compile layer, then keep pip check strict; no CUDA library bytes are altered.
COPY scripts/repair_cusparselt_wheel.py /opt/repair_cusparselt_wheel.py
# Audit separately so a failed assertion does not invalidate the expensive compile layer.
RUN python /opt/repair_cusparselt_wheel.py && \
    python -c "import torch,exllamav3_ext; from exllamav3.version import __version__; assert torch.__version__ == '2.13.0+cu130'; assert __version__ == '1.5.2'; print('NATIVE_EXTENSION_PRESENT',__version__,exllamav3_ext.__file__)" && \
    python -m pip check

RUN git clone https://github.com/theroyallab/tabbyAPI.git /opt/tabbyAPI && \
    git -C /opt/tabbyAPI checkout 816c32195887aaecea1c64528f2921566766259b && \
    test "$(git -C /opt/tabbyAPI rev-parse HEAD)" = "816c32195887aaecea1c64528f2921566766259b" && \
    python -m pip install /opt/tabbyAPI
# Pinned TabbyAPI main.py imports uvloop on every non-Windows host, while its
# pyproject.toml declares uvloop only for x86_64. Install the native ARM64 wheel.
RUN python -m pip install 'uvloop==0.22.1' && \
    python -c "import uvloop; uvloop.install(); print('ARM64_UVLOOP_READY',uvloop.__version__)" && \
    python -m pip check

WORKDIR /opt/tabbyAPI
COPY config/config.yml /opt/tabbyAPI/config.yml
COPY scripts/smoke.py /opt/smoke.py
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin runner && \
    mkdir -p /data/cache /data/triton /data/torch /opt/tabbyAPI/logs && \
    chown -R runner:runner /data /opt/tabbyAPI/logs
ARG PROJECT_REVISION
RUN test -n "$PROJECT_REVISION"
LABEL org.opencontainers.image.revision="${PROJECT_REVISION}" \
      org.opencontainers.image.title="r0b0tlab GLM-5.3-Flash EXL3 GB10 local runtime" \
      org.opencontainers.image.exllamav3-revision="12414d0af7b3beeabdda5990f6b554b996fa1416" \
      org.opencontainers.image.tabbyapi-revision="816c32195887aaecea1c64528f2921566766259b"
USER runner
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health',timeout=5)" || exit 1
CMD ["python", "main.py"]
