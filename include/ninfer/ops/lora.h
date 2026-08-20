#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/// Largest rank a banked adapter may occupy. Shorter adapters zero-pad.
inline constexpr std::int32_t kMaximumLoraRank = 64;

/// Most sites one call may correct from a single shared input.
inline constexpr std::int32_t kMaximumLoraSites = 4;

/**
 * One banked low-rank factor pair.
 *
 * `a` addresses BF16 `[rank, k]` and `b` addresses BF16 `[n, rank]`, both
 * row-major and contiguous, for adapter 0. Adapter `i` begins `i * stride`
 * bytes later in each plane. A pair whose `a` or `b` is null is inactive and
 * contributes nothing.
 */
struct LoraSite {
    const void* a                = nullptr;
    const void* b                = nullptr;
    std::int32_t n               = 0;
    std::int32_t k               = 0;
    std::size_t a_adapter_stride = 0;
    std::size_t b_adapter_stride = 0;
};

/**
 * A set of sites sharing one input activation, applied by one call.
 *
 * `rank` is the bank rank, uniform across every adapter and site. An adapter
 * trained at a lower rank stores zeros in the trailing rows of `a` and the
 * trailing columns of `b`, which contribute nothing.
 */
struct LoraGroup {
    std::int32_t rank                             = 0;
    std::int32_t adapter_count                    = 0;
    std::int32_t site_count                       = 0;
    std::array<LoraSite, kMaximumLoraSites> sites = {};
};

/**
 * Returns the transient capacity lora_delta_add requires for every T in the
 * inclusive [min_tokens,max_tokens] interval at this rank and site count.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t lora_delta_add_workspace_capacity_bytes(std::int32_t rank,
                                                                 std::int32_t site_count,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Banked low-rank correction with per-token adapter selection:
 *
 *   a = adapter_index[t / (T / adapter_index.ne[0])]
 *   destination_s[:, t] += B_s[a] * (A_s[a] * x[:, t])   for every site s, when a >= 0.
 *
 * `x` is a contiguous BF16 `[K,T]` activation with T>0, shared by every site in
 * the group. Each `destination_s` is a contiguous BF16 `[N_s,T]` tensor updated
 * in place; its incoming value is the uncorrected projection result and its
 * outgoing value is that result plus the selected adapter's delta. A column
 * whose adapter index is negative is left exactly unchanged, so the base model
 * and any adapter may share one batch.
 *
 * `adapter_index` is a contiguous I32 tensor whose extent is 1 or an exact
 * divisor of T. One element selects uniformly for the whole call; T elements
 * select per column; `batch` elements select per sequence in a `[width, batch]`
 * verify tile, where each entry covers `width` consecutive columns. Every
 * non-negative index must be below `group.adapter_count`.
 *
 * A group carries `site_count` sites in `sites[0..site_count)` and the caller
 * supplies exactly that many destinations, in the same order. Two sites may
 * share one `a` plane; the Op does not deduplicate that projection.
 *
 * The complete mathematical oracle starts from the represented BF16 inputs and
 * exactly decoded factors, and evaluates `destination + B * (A * x)` naively in
 * FP32/FP64. The intermediate `A * x`, its accumulation order, its workspace
 * representation, and the kernel decomposition are private execution choices
 * rather than semantic rounding boundaries. Activations are BF16 at the public
 * boundary in every route; no activation-quantized profile is admitted, because
 * this Op runs inside the phase that CUDA Graphs capture.
 *
 * `x`, every destination, every factor plane, and live workspace must be
 * pairwise non-overlapping. Execution is enqueued on `stream` without host
 * synchronization. Workspace is caller-owned, graph-stable transient storage and
 * carries no state beyond the call. The Op holds no persistent state.
 */
void lora_delta_add(const Tensor& x, const LoraGroup& group, const Tensor& adapter_index,
                    std::span<Tensor* const> destinations, WorkspaceArena& workspace,
                    cudaStream_t stream);

} // namespace ninfer::ops
