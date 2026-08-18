#pragma once

// Identity-free Qwen3.8 family runtime helper.

#include "core/arena.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_8/mtp_alignment.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_8::detail {

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const qwen3_8::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8::detail
