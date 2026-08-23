#pragma once

#include <ninfer/targets/qwen3_8/startup_features.h>
#include <ninfer/targets/qwen3_8/vision.h>

#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer {

class DeviceArena;

namespace targets::qwen3_8 {

template <class ProjectionPayload, class PostMixerPayload>
struct FullAttentionWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class ProjectionPayload, class PostMixerPayload>
struct GdnWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor convolution;
    Tensor norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class AttentionPayload, class PostMixerPayload>
struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Tensor input_norm;
    AttentionPayload attention;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
    Tensor final_norm;
};

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

// One registered low-rank correction site, banked over every registered adapter.
//
// `a` holds the registered adapters' [rank, K] factors back to back and `b` their [N, rank]
// factors, so a bank index selects a plane by a fixed byte stride. The alpha/r scale is folded
// into `b` at conversion, so no runtime scale exists. A site whose `a` is null was not trained
// by any registered adapter and is never scheduled.
struct LoraSiteWeights {
    Tensor a;
    Tensor b;
    std::size_t a_adapter_stride = 0;
    std::size_t b_adapter_stride = 0;

    [[nodiscard]] bool present() const noexcept { return a.data != nullptr; }
};

// Everything a package execution leaf needs to correct one site whose input it owns privately.
// The family passes this only when the bank is resident and the site was trained.
struct LoraApplication {
    const LoraSiteWeights* site   = nullptr;
    const Tensor* adapter_index   = nullptr;
    std::int32_t rank             = 0;
    std::int32_t adapter_count    = 0;

    [[nodiscard]] bool active() const noexcept {
        return site != nullptr && site->present() && adapter_index != nullptr;
    }
};

// The registered sites of one full-attention layer. `query` and `gate` are two row selections of
// the same source module, so they share one `a` plane and differ only in `b`.
struct LoraFullLayerWeights {
    LoraSiteWeights query;
    LoraSiteWeights gate;
    LoraSiteWeights key;
    LoraSiteWeights value;
    LoraSiteWeights output;
    LoraSiteWeights down;
};

struct LoraGdnLayerWeights {
    LoraSiteWeights output;
    LoraSiteWeights down;
};

// A startup-fixed bank of resident adapters. Every registered adapter shares one rank, so the
// rank and the per-site strides are kernel constants and the bank is one persistent allocation
// with deterministic offsets.
template <std::size_t FullAttentionLayers, std::size_t GdnLayers>
struct LoraWeights {
    std::uint32_t adapters = 0;
    std::int32_t rank      = 0;
    // Device bytes the package committed for the whole bank. The bank is its own arena outside
    // the weights arena, so this is the only route by which the family's memory summary can
    // account for it instead of leaving it as unexplained missing free memory.
    std::uint64_t device_bytes = 0;
    std::array<LoraFullLayerWeights, FullAttentionLayers> full_layers;
    std::array<LoraGdnLayerWeights, GdnLayers> gdn_layers;
};

struct DFlashLayerWeights {
    Tensor input_norm;
    Weight query_key_value;
    Weight context_key;
    Weight context_value;
    Tensor query_norm;
    Tensor key_norm;
    Weight attention_output;
    Tensor post_attention_norm;
    Weight gate_up;
    Weight down;
};

template <std::size_t Layers>
struct DFlashWeights {
    Weight feature_projection;
    Tensor context_norm;
    std::array<DFlashLayerWeights, Layers> layers;
    Tensor final_norm;
};

template <class FullProjectionPayload, class GdnProjectionPayload, class MainPostMixerPayload,
          class MtpAttentionPayload, class MtpPostMixerPayload, class DFlashPayload,
          std::size_t FullAttentionLayers, std::size_t GdnLayers>
struct ModelView {
    using FullLayer = FullAttentionWeights<FullProjectionPayload, MainPostMixerPayload>;
    using GdnLayer  = GdnWeights<GdnProjectionPayload, MainPostMixerPayload>;
    using MtpLayer  = MtpWeights<MtpAttentionPayload, MtpPostMixerPayload>;
    using DFlash    = DFlashPayload;
    using Lora      = LoraWeights<FullAttentionLayers, GdnLayers>;

    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<FullLayer, FullAttentionLayers> full_layers;
    std::array<GdnLayer, GdnLayers> gdn_layers;
    Tensor final_norm;
    Weight output_head;
    StartupFeatures features;
    std::optional<OptimizedProposalWeights> optimized_proposal;
    std::optional<MtpLayer> mtp;
    std::optional<DFlashPayload> dflash;
    std::optional<VisionWeights> vision;
    std::optional<Lora> lora;
};

} // namespace targets::qwen3_8
} // namespace ninfer
