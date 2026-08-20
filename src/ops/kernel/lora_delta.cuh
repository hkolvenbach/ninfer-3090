#pragma once

// Implements: include/ninfer/ops/lora.h
//
// Two stages over one shared activation. The down stage reduces `A * x` into a
// rank-sized FP32 partial, split over K for occupancy at small T; the up stage
// folds those partials in a fixed order and accumulates `B * temp` into each
// destination.
//
// Both stages iterate over the *distinct* adapters present in their token tile
// rather than over tokens. Two tokens that select the same adapter then share
// one read of that adapter's factor, and a tile carrying k distinct adapters
// reads k factors, which is the work the routing actually implies. The scan is
// a first-occurrence test, so the adapter order, and therefore the reduction
// order, depends only on the tile contents.

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kLoraTileTokens = 8;
inline constexpr int kLoraDownWarps  = 8;
inline constexpr int kLoraDownBlock  = kLoraDownWarps * 32;
inline constexpr int kLoraUpBlock    = 128;
inline constexpr int kLoraChunk      = 256;

struct LoraSiteParams {
    const __nv_bfloat16* a = nullptr;
    const __nv_bfloat16* b = nullptr;
    __nv_bfloat16* dest    = nullptr;
    std::int32_t n         = 0;
    std::int32_t k         = 0;
    std::int64_t a_stride  = 0; // elements between adapters
    std::int64_t b_stride  = 0;
};

struct LoraLaunchParams {
    LoraSiteParams sites[4];
    const __nv_bfloat16* x       = nullptr;
    const std::int32_t* adapters = nullptr;
    float* partials              = nullptr;
    std::int32_t site_count      = 0;
    std::int32_t tokens          = 0;
    std::int32_t k_splits        = 0;
    // Columns covered by one adapter-index entry. 0 selects one uniform adapter for every token;
    // 1 routes per column; `width` routes per sequence in a [width, batch] verify tile.
    std::int32_t adapter_stride  = 0;
};

__device__ __forceinline__ std::int32_t lora_adapter_of(const LoraLaunchParams& params,
                                                        std::int32_t token) {
    return params.adapters[params.adapter_stride == 0 ? 0 : token / params.adapter_stride];
}

/// True when `token` is the first entry of its tile selecting this adapter.
__device__ __forceinline__ bool lora_first_of_adapter(const LoraLaunchParams& params,
                                                      std::int32_t tile_begin, std::int32_t token,
                                                      std::int32_t adapter) {
    for (std::int32_t earlier = tile_begin; earlier < token; ++earlier) {
        if (lora_adapter_of(params, earlier) == adapter) { return false; }
    }
    return true;
}

// partials[((split * site_count + site) * RANK + j) * tokens + token]
__device__ __forceinline__ std::int64_t lora_partial_index(const LoraLaunchParams& params,
                                                           std::int32_t split, std::int32_t site,
                                                           std::int32_t rank, std::int32_t j,
                                                           std::int32_t token) {
    const std::int64_t plane = static_cast<std::int64_t>(split) * params.site_count + site;
    return (plane * rank + j) * params.tokens + token;
}

template <int Rank>
__launch_bounds__(kLoraDownBlock) __global__ void lora_down_kernel(LoraLaunchParams params) {
    constexpr int kPerWarp = Rank / kLoraDownWarps > 0 ? Rank / kLoraDownWarps : 1;
    constexpr int kActiveWarps = Rank < kLoraDownWarps ? Rank : kLoraDownWarps;

    const std::int32_t split = blockIdx.x;
    const std::int32_t site  = blockIdx.z;
    const std::int32_t tile  = blockIdx.y * kLoraTileTokens;
    if (site >= params.site_count || tile >= params.tokens) { return; }

    const LoraSiteParams& sp = params.sites[site];
    if (sp.a == nullptr) { return; }

    const std::int32_t tile_end = min(tile + kLoraTileTokens, params.tokens);
    const std::int32_t per_split = (sp.k + params.k_splits - 1) / params.k_splits;
    const std::int32_t k_begin   = split * per_split;
    const std::int32_t k_end     = min(k_begin + per_split, sp.k);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    __shared__ float staged[kLoraChunk][kLoraTileTokens];

    // Zero this split's partials once; every distinct adapter accumulates into them.
    for (int index = threadIdx.x; index < Rank * kLoraTileTokens; index += kLoraDownBlock) {
        const std::int32_t j     = index / kLoraTileTokens;
        const std::int32_t token = tile + index % kLoraTileTokens;
        if (token < tile_end) {
            params.partials[lora_partial_index(params, split, site, Rank, j, token)] = 0.0F;
        }
    }
    __syncthreads();
    if (k_begin >= k_end) { return; }

    for (std::int32_t token = tile; token < tile_end; ++token) {
        const std::int32_t adapter = lora_adapter_of(params, token);
        if (adapter < 0) { continue; }
        if (!lora_first_of_adapter(params, tile, token, adapter)) { continue; }

        const __nv_bfloat16* a_base = sp.a + static_cast<std::int64_t>(adapter) * sp.a_stride;
        float acc[kPerWarp][kLoraTileTokens];
#pragma unroll
        for (int r = 0; r < kPerWarp; ++r) {
#pragma unroll
            for (int t = 0; t < kLoraTileTokens; ++t) { acc[r][t] = 0.0F; }
        }

        for (std::int32_t base = k_begin; base < k_end; base += kLoraChunk) {
            const std::int32_t extent = min(kLoraChunk, k_end - base);
            for (int index = threadIdx.x; index < extent * kLoraTileTokens;
                 index += kLoraDownBlock) {
                const int local        = index / kLoraTileTokens;
                const int slot         = index % kLoraTileTokens;
                const std::int32_t col = tile + slot;
                float value            = 0.0F;
                if (col < tile_end && lora_adapter_of(params, col) == adapter) {
                    value = __bfloat162float(
                        params.x[static_cast<std::int64_t>(col) * sp.k + base + local]);
                }
                staged[local][slot] = value;
            }
            __syncthreads();

            if (warp < kActiveWarps) {
#pragma unroll
                for (int r = 0; r < kPerWarp; ++r) {
                    const int j = warp + r * kLoraDownWarps;
                    if (j >= Rank) { break; }
                    const __nv_bfloat16* row = a_base + static_cast<std::int64_t>(j) * sp.k + base;
                    for (int local = lane; local < extent; local += 32) {
                        const float weight = __bfloat162float(row[local]);
#pragma unroll
                        for (int t = 0; t < kLoraTileTokens; ++t) {
                            acc[r][t] = fmaf(weight, staged[local][t], acc[r][t]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (warp < kActiveWarps) {
#pragma unroll
            for (int r = 0; r < kPerWarp; ++r) {
                const int j = warp + r * kLoraDownWarps;
                if (j >= Rank) { break; }
#pragma unroll
                for (int t = 0; t < kLoraTileTokens; ++t) {
                    float value = acc[r][t];
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
                    }
                    const std::int32_t col = tile + t;
                    if (lane == 0 && col < tile_end &&
                        lora_adapter_of(params, col) == adapter) {
                        params.partials[lora_partial_index(params, split, site, Rank, j, col)] +=
                            value;
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <int Rank>
__launch_bounds__(kLoraUpBlock) __global__ void lora_up_kernel(LoraLaunchParams params) {
    const std::int32_t site = blockIdx.z;
    const std::int32_t tile = blockIdx.y * kLoraTileTokens;
    if (site >= params.site_count || tile >= params.tokens) { return; }

    const LoraSiteParams& sp = params.sites[site];
    if (sp.b == nullptr) { return; }

    const std::int32_t row_begin = blockIdx.x * kLoraUpBlock;
    if (row_begin >= sp.n) { return; }
    const std::int32_t tile_end = min(tile + kLoraTileTokens, params.tokens);

    __shared__ float temp[Rank][kLoraTileTokens];
    __shared__ __nv_bfloat16 staged_b[kLoraUpBlock * Rank];

    for (int index = threadIdx.x; index < Rank * kLoraTileTokens; index += kLoraUpBlock) {
        const std::int32_t j     = index / kLoraTileTokens;
        const std::int32_t slot  = index % kLoraTileTokens;
        const std::int32_t token = tile + slot;
        float total              = 0.0F;
        if (token < tile_end) {
            for (std::int32_t split = 0; split < params.k_splits; ++split) {
                total += params.partials[lora_partial_index(params, split, site, Rank, j, token)];
            }
        }
        temp[j][slot] = total;
    }

    const std::int32_t rows = min(kLoraUpBlock, sp.n - row_begin);

    for (std::int32_t token = tile; token < tile_end; ++token) {
        const std::int32_t adapter = lora_adapter_of(params, token);
        if (adapter < 0) { continue; }
        if (!lora_first_of_adapter(params, tile, token, adapter)) { continue; }

        const __nv_bfloat16* b_base = sp.b + static_cast<std::int64_t>(adapter) * sp.b_stride;
        __syncthreads();
        // [rows, Rank] is contiguous in B, so this stage is fully coalesced.
        for (int index = threadIdx.x; index < rows * Rank; index += kLoraUpBlock) {
            staged_b[index] = b_base[static_cast<std::int64_t>(row_begin) * Rank + index];
        }
        __syncthreads();

        const std::int32_t row = row_begin + static_cast<std::int32_t>(threadIdx.x);
        if (threadIdx.x < static_cast<unsigned>(rows)) {
            for (std::int32_t col = token; col < tile_end; ++col) {
                if (lora_adapter_of(params, col) != adapter) { continue; }
                const int slot = col - tile;
                float sum      = 0.0F;
#pragma unroll
                for (int j = 0; j < Rank; ++j) {
                    sum = fmaf(__bfloat162float(staged_b[threadIdx.x * Rank + j]), temp[j][slot],
                               sum);
                }
                const std::int64_t offset = static_cast<std::int64_t>(col) * sp.n + row;
                sp.dest[offset] =
                    __float2bfloat16_rn(__bfloat162float(sp.dest[offset]) + sum);
            }
        }
    }
}

} // namespace ninfer::ops
