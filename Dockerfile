# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY apps/ apps/
COPY include/ include/
COPY src/ src/
COPY third_party/ third_party/

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

ARG MODEL_URL=https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer
# Keep immutable model bytes below the frequently changing application layers.
RUN --mount=type=bind,source=.,target=/context,readonly \
    mkdir -p /opt/ninfer/models \
    && if [ -f /context/models/qwen3_8_27b.ninfer ]; then \
         cp /context/models/qwen3_8_27b.ninfer /opt/ninfer/models/qwen3_8_27b.ninfer; \
       else \
         curl --fail --location --retry 5 \
           --output /opt/ninfer/models/qwen3_8_27b.ninfer \
           "$MODEL_URL"; \
       fi \
    && printf '%s  %s\n' \
         eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e \
         /opt/ninfer/models/qwen3_8_27b.ninfer \
       | sha256sum --check

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "/opt/ninfer/models/qwen3_8_27b.ninfer", "--model-id", "qwen3.8-27b", "--host", "0.0.0.0", "--port", "8080", "--max-context", "262144", "--kv-capacity", "262144", "--max-concurrency", "1", "--max-pending-requests", "16", "--prefill-chunk", "1024", "--kv-dtype", "rk4v4-e8", "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft", "--prefix-checkpoint-policy", "rolling-tool", "--preserve-thinking"]
