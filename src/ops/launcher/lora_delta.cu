// Implements: include/ninfer/ops/lora.h
// Finite dispatch: one instantiation per registered bank rank; the down stage
// splits K for occupancy at small token counts and the up stage folds the
// resulting partials in a fixed order.
#include "ops/launcher/lora_delta.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/lora_delta.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

LoraLaunchParams build_params(const Tensor& x, const LoraGroup& group,
                              const Tensor& adapter_index,
                              std::span<Tensor* const> destinations, float* partials,
                              std::int32_t tokens, std::int32_t k_splits) {
    LoraLaunchParams params;
    params.x        = static_cast<const __nv_bfloat16*>(x.data);
    params.adapters = static_cast<const std::int32_t*>(adapter_index.data);
    params.partials = partials;
    params.site_count     = group.site_count;
    params.tokens         = tokens;
    params.k_splits       = k_splits;
    params.adapter_stride = adapter_index.ne[0] == 1 ? 0 : tokens / adapter_index.ne[0];

    for (std::int32_t index = 0; index < group.site_count; ++index) {
        const LoraSite& site   = group.sites[index];
        LoraSiteParams& target = params.sites[index];
        target.a    = static_cast<const __nv_bfloat16*>(site.a);
        target.b    = static_cast<const __nv_bfloat16*>(site.b);
        target.dest = static_cast<__nv_bfloat16*>(destinations[index]->data);
        target.n    = site.n;
        target.k    = site.k;
        target.a_stride =
            static_cast<std::int64_t>(site.a_adapter_stride / sizeof(__nv_bfloat16));
        target.b_stride =
            static_cast<std::int64_t>(site.b_adapter_stride / sizeof(__nv_bfloat16));
    }
    return params;
}

template <int Rank>
void launch_rank(const LoraLaunchParams& params, std::int32_t maximum_rows, cudaStream_t stream) {
    const dim3 down_grid(static_cast<unsigned>(params.k_splits),
                         static_cast<unsigned>(lora_token_tiles(params.tokens)),
                         static_cast<unsigned>(params.site_count));
    lora_down_kernel<Rank><<<down_grid, kLoraDownBlock, 0, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());

    const unsigned row_tiles =
        static_cast<unsigned>((maximum_rows + kLoraUpBlock - 1) / kLoraUpBlock);
    const dim3 up_grid(row_tiles, static_cast<unsigned>(lora_token_tiles(params.tokens)),
                       static_cast<unsigned>(params.site_count));
    lora_up_kernel<Rank><<<up_grid, kLoraUpBlock, 0, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void lora_delta_add_launch(const Tensor& x, const LoraGroup& group, const Tensor& adapter_index,
                           std::span<Tensor* const> destinations, WorkspaceArena& workspace,
                           cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(x.ne[1]);
    const std::int32_t k_splits = lora_k_splits(tokens, group.site_count);

    auto scope = workspace.scope();
    const DeviceSpan storage =
        workspace.alloc_bytes(lora_partial_bytes(group.rank, group.site_count, tokens), 256);
    auto* partials = static_cast<float*>(storage.data);

    std::int32_t maximum_rows = 0;
    for (std::int32_t index = 0; index < group.site_count; ++index) {
        maximum_rows = group.sites[index].n > maximum_rows ? group.sites[index].n : maximum_rows;
    }
    if (maximum_rows == 0) { return; }

    const LoraLaunchParams params =
        build_params(x, group, adapter_index, destinations, partials, tokens, k_splits);

    switch (group.rank) {
    case 8: launch_rank<8>(params, maximum_rows, stream); return;
    case 16: launch_rank<16>(params, maximum_rows, stream); return;
    case 32: launch_rank<32>(params, maximum_rows, stream); return;
    case 64: launch_rank<64>(params, maximum_rows, stream); return;
    default: break;
    }
    throw std::invalid_argument("lora_delta_add: unregistered bank rank");
}

} // namespace ninfer::ops::detail
