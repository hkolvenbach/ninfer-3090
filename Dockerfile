# syntax=docker/dockerfile:1

# The dashboard is a static bundle with no runtime dependency on the engine image, so it builds in
# its own stage. Debian rather than Alpine: the lockfile resolves the glibc lightningcss binary.
FROM oven/bun:1.3.5 AS web

WORKDIR /web
# Dependencies first, so editing a panel does not re-resolve the tree.
COPY apps/web/package.json apps/web/bun.lock ./
RUN bun install --frozen-lockfile
COPY apps/web/ ./
RUN bun run build

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

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
# Only the C++ applications. apps/web is not part of the CMake build, and copying it here would
# invalidate this cached CUDA layer on every dashboard edit.
COPY apps/CMakeLists.txt apps/
COPY apps/cli/ apps/cli/
COPY apps/serve/ apps/serve/
COPY include/ include/
COPY src/ src/
COPY third_party/ third_party/

RUN --mount=type=cache,id=ninfer-sm86-cuda13.1-release,target=/build,sharing=locked \
    cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
        -DNINFER_BUILD_QWEN3_8_27B=ON \
        -DNINFER_BUILD_QWEN3_6_35B_A3B=OFF \
    && cmake --build /build --parallel 16 --target ninfer ninfer-serve \
    && install -D /build/apps/ninfer /opt/ninfer/bin/ninfer \
    && install -D /build/apps/ninfer-serve /opt/ninfer/bin/ninfer-serve

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

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

# The CUDA runtime image ships forward-compatibility libraries in
# /usr/local/cuda*/compat (a newer libcuda.so than the host driver). Forward
# compatibility is supported only on datacenter GPUs; on any GeForce card the
# loader picks these up and every CUDA call fails at startup with
#   cudaErrorCompatNotSupportedOnDevice: forward compatibility was attempted
#   on non supported HW
# Removing them lets the container use the host driver through ordinary CUDA
# minor-version compatibility, which is what an RTX 3090 needs.
RUN rm -rf /usr/local/cuda-13.1/compat /usr/local/cuda-13/compat /usr/local/cuda/compat

# Model artifact is mounted at runtime from NFS (/srv/models/ on Yulie host).
# Do not embed in the image — the 16.96 GiB download belongs on the NFS share.

COPY --from=build /opt/ninfer/bin/ninfer /usr/local/bin/ninfer
COPY --from=build /opt/ninfer/bin/ninfer-serve /usr/local/bin/ninfer-serve
COPY --from=web /web/dist /opt/ninfer/web

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM
VOLUME ["/var/cache/ninfer"]

CMD ["ninfer-serve", "/opt/ninfer/models/qwen3_8_27b.ninfer", "--model-id", "qwen3.8-27b", "--host", "0.0.0.0", "--port", "8080", "--max-context", "131072", "--kv-capacity", "131072", "--max-concurrency", "4", "--max-pending-requests", "16", "--prefill-chunk", "1024", "--kv-dtype", "int8", "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft", "--prefix-checkpoint-policy", "rolling-tool", "--continuation-cache", "l1-l2-l3", "--continuation-cache-dir", "/var/cache/ninfer", "--continuation-cache-namespace", "local", "--continuation-cache-l1-mib", "768", "--continuation-cache-l2-mib", "16384", "--continuation-cache-l3-mib", "49152", "--continuation-cache-filesystem-reserve-mib", "4096", "--preserve-thinking", "--web-dir", "/opt/ninfer/web"]
