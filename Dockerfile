# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3.12-slim AS python-builder

ARG TARGETOS=linux
ARG TARGETARCH=amd64

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    gcc g++ gfortran \
    make cmake pkg-config git\
    libopenblas-dev libnuma1 libnuma-dev\
    zlib1g-dev libjpeg-dev \
    rustc cargo libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python build tools required for s390x
# -----------------------------
RUN if [ "$TARGETARCH" = "s390x" ]; then \
        python -m pip install --upgrade pip && \
        python -m pip install \
            "setuptools>=77.0.3,<81.0.0" \
            "setuptools-scm>=8.0" \
            "packaging>=24.2" \
            "pillow==10.4.0" \
            "cryptography==42.0.8" \
            "setuptools-rust" wheel cffi maturin numpy \
            cmake pybind11 cython ninja scikit-build-core ; \
    fi
# torch (already working)
RUN if [ "$TARGETARCH" = "s390x" ]; then \
        python -m pip install "torch==2.10.0+cpu" \
        --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# disable triton (important for s390x)
ENV VLLM_USE_TRITON=0
ARG VLLM_BENCHMARK_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_BENCHMARK_BRANCH=v0.16.0 
ARG OPENCV_VERSION=90
ARG ENABLE_HEADLESS=1

# install vllm
RUN if [ "$TARGETARCH" = "s390x" ]; then \
        git clone --recursive https://github.com/opencv/opencv-python.git -b ${OPENCV_VERSION} && \
        cd opencv-python && \
        python -m pip install build && \
        python -m build --wheel --outdir /tmp/wheels && \
        pip install /tmp/wheels/*.whl && \
        echo "Building vLLM from source for s390x..." && \
        git clone --branch ${VLLM_BENCHMARK_BRANCH} ${VLLM_BENCHMARK_REPO} /tmp/vllm && \
        cd /tmp/vllm && \
        echo "numpy==2.4.4" > /tmp/constraints.txt && \
        sed -i '/"torch == 2.10.0"/d' pyproject.toml && \
    	sed -i '/^license\s*=.*/d' pyproject.toml && \
    	sed -i '/^\[project\.license\]/,/^\[/d' pyproject.toml && \
    	sed -i '/license-files/d' pyproject.toml && \
        sed -i '/^\[project\]/a license = { text = "Apache-2.0" }' pyproject.toml && \
    	echo "==== FINAL LICENSE STATE ====" && \
    	cat pyproject.toml | grep -n license || true && \
    	echo "=============================" && \
        VLLM_TARGET_DEVICE=empty python -m pip install -e . --no-build-isolation -c /tmp/constraints.txt  && \
        rm -rf /tmp/vllm ; \
    else \
        echo "Installing vLLM via Python Dep .." ; \
    fi

COPY Makefile Makefile
COPY pkg/preprocessing/chat_completions/ pkg/preprocessing/chat_completions/

# -----------------------------
# Create virtualenv + reuse vLLM
# -----------------------------
RUN if [ "$TARGETARCH" = "s390x" ]; then \
        python -m venv /workspace/build/venv && \
        echo "/usr/local/lib/python3.12/site-packages" > /workspace/build/venv/lib/python3.12/site-packages/system.pth ;\
    fi

RUN TARGETOS=${TARGETOS} TARGETARCH=${TARGETARCH} make install-python-deps

# Build Stage: using Go 1.24.1 image
#FROM quay.io/projectquay/golang:1.24 AS builder
FROM golang:1.24
ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace

# Install system-level dependencies first. This layer is very stable.
USER root
# Install EPEL repository directly and then ZeroMQ, as epel-release is not in default repos.
# Install all necessary dependencies including Python 3.12 for chat-completions templating.
# The builder is based on UBI8, so we need epel-release-8.
RUN dnf install -y 'https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm' && \
    dnf install -y gcc-c++ libstdc++ libstdc++-devel clang zeromq-devel pkgconfig python3.12-devel python3.12-pip openblas && \
    dnf clean all

# Copy the Go Modules manifests
COPY go.mod go.mod
COPY go.sum go.sum
# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the source code.
COPY . .

# Copy this project's own Python source code into the final image
COPY --from=python-builder /workspace/pkg/preprocessing/chat_completions /workspace/pkg/preprocessing/chat_completions
RUN make setup-venv
COPY --from=python-builder /workspace/build/venv/lib/python3.12/site-packages /workspace/build/venv/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/lib/python3.12/site-packages \
    /usr/local/lib/python3.12/site-packages

# Set the PYTHONPATH. This mirrors the Makefile's export, ensuring both this project's
# Python code and the installed libraries (site-packages) are found at runtime.
ENV PYTHONPATH=/workspace/pkg/preprocessing/chat_completions:/workspace/build/venv/lib/python3.12/site-packages
RUN python3.12 -c "import tokenizer_wrapper"

RUN make build

# Use distroless as minimal base image to package the manager binary
# Refer to https://github.com/GoogleContainerTools/distroless for more details
FROM registry.access.redhat.com/ubi9/ubi:latest
WORKDIR /
# Install zeromq runtime library needed by the manager.
# The final image is UBI9, so we need epel-release-9.
USER root
RUN dnf install -y 'https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm' && \
    dnf install -y zeromq libxcrypt-compat python3.12 python3.12-pip openblas openblas-devel libgomp && \
    dnf clean all

# Ensure correct soname exists
RUN ln -sf /usr/lib64/libopenblas.so /usr/lib64/libopenblas.so.0 || true

# Copy this project's own Python source code into the final image
COPY --from=python-builder /workspace/pkg/preprocessing/chat_completions /app/pkg/preprocessing/chat_completions
COPY --from=python-builder /workspace/build/venv/lib/python3.12/site-packages /workspace/build/venv/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/lib/python3.12/site-packages \
    /usr/local/lib/python3.12/site-packages

# Set the PYTHONPATH. This mirrors the Makefile's export, ensuring both this project's
# Python code and the installed libraries (site-packages) are found at runtime.
ENV PYTHONPATH=/app/pkg/preprocessing/chat_completions:/workspace/build/venv/lib/python3.12/site-packages
RUN python3.12 -c "import tokenizer_wrapper"

# Copy the compiled Go application
COPY --from=builder /workspace/bin/llm-d-kv-cache /app/kv-cache-manager
USER 65532:65532

# Set the entrypoint to the kv-cache-manager binary
ENTRYPOINT ["/app/kv-cache-manager"]
