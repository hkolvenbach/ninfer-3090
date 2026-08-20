#pragma once

// Registered LoRA adapter binding and bank residency for the Qwen3.8-27B identity.
//
// An adapter artifact is a normal `.ninfer` container carrying only BF16 `contiguous-le-v1`
// rank-two tensors under the identity `{qwen3.8-27b, lora-bf16}`. It restates no frontend
// resource, so the base artifact remains the sole authority for tokenizer, template and
// preprocessor configs.
//
// Every registered adapter shares one rank and one object inventory, so the resident bank is a
// single allocation laid out adapter-major with a fixed per-adapter slab. Every site's adapter
// stride is then the same constant and every plane address is deterministic, which is what lets
// the bank participate in a captured graph.

#include <ninfer/targets/qwen3_8/model_view.h>
#include <ninfer/types.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_8_27b::detail {

inline constexpr std::string_view kLoraWeightsId = "lora-bf16";

// One registered site. Both factors are present or neither is; a site the adapter never trained
// is simply not scheduled. Offsets are byte positions inside one adapter's slab, so the same
// plan describes every adapter in the bank.
struct LoraSitePlan {
    std::uint64_t a_offset = 0;
    std::uint64_t b_offset = 0;
    std::int32_t rows      = 0; // N of the destination this site corrects
    bool present           = false;
};

struct LoraFullLayerPlan {
    LoraSitePlan query;
    LoraSitePlan gate;
    LoraSitePlan key;
    LoraSitePlan value;
    LoraSitePlan output;
    LoraSitePlan down;
};

struct LoraGdnLayerPlan {
    LoraSitePlan output;
    LoraSitePlan down;
};

// One bound object and where it lives inside an adapter slab.
struct LoraObjectPlacement {
    artifact::ObjectHandle object;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

struct LoraBindingPlan {
    std::int32_t rank        = 0;
    std::uint64_t slab_bytes = 0;
    std::array<LoraFullLayerPlan, kFullAttentionLayers> full_layers;
    std::array<LoraGdnLayerPlan, kGdnLayers> gdn_layers;
    std::vector<LoraObjectPlacement> objects;
};

struct LoraArtifactLoadPlan {
    LoraBindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

// Discovers the adapter's rank from its objects and binds its complete registered inventory.
// A site that appears at all must carry both of its factors.
[[nodiscard]] LoraArtifactLoadPlan bind_lora_artifact(artifact::Binder& binder,
                                                      const artifact::Reader& reader);

// The resident bank plus the names that select it. Owns the single device allocation the model
// view points into, so the view is only valid while this object is alive.
struct LoadedLoraBank {
    std::unique_ptr<DeviceArena> arena;
    RuntimeModelView::Lora view;
    std::vector<std::string> names;
    std::uint64_t device_bytes = 0;
    std::uint64_t file_bytes   = 0;
};

// Opens, validates, materializes and packs every registered adapter into one bank. Rejects a
// foreign identity, a rank or inventory that disagrees with the first adapter, a duplicate name,
// and more adapters than the engine admits.
[[nodiscard]] LoadedLoraBank load_lora_bank(std::span<const LoraAdapterSpec> adapters,
                                            DeviceContext& device);

} // namespace ninfer::targets::qwen3_8_27b::detail
