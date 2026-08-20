#include "targets/qwen3_8_27b/impl/load/lora_bindings.h"

#include "artifact/reader.h"
#include "artifact/typed_binding.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <span>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::targets::qwen3_8_27b::detail {
namespace {

using artifact::NumericFormat;

constexpr std::int32_t kHidden          = 5120;
constexpr std::int32_t kQuerySize       = 6144;
constexpr std::int32_t kKeyValueSize    = 1024;
constexpr std::int32_t kAttentionValues = 6144;
constexpr std::int32_t kGdnValues       = 6144;
constexpr std::int32_t kIntermediate    = 17408;
constexpr std::uint64_t kSlabAlignment  = 256;

bool registered_rank(std::int32_t rank) {
    return rank == 8 || rank == 16 || rank == 32 || rank == 64;
}

bool is_full_attention_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

std::size_t full_attention_index(std::size_t layer) { return (layer - 3) / 4; }

// Layers 3, 7, 11, ... are full attention; every other layer is Gated DeltaNet.
std::size_t gdn_index(std::size_t layer) {
    return layer - (layer >= 3 ? full_attention_index(layer) + 1 : 0);
}

std::string layer_prefix(std::size_t layer) {
    return "text/layers/" + std::to_string(layer) + "/";
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Accumulates slab offsets while binding, so a shared factor is bound and placed exactly once.
class SlabBuilder {
public:
    SlabBuilder(artifact::Binder& binder, const artifact::Reader& reader, std::int32_t rank,
                LoraBindingPlan& plan)
        : binder_(binder), reader_(reader), rank_(rank), plan_(plan) {}

    [[nodiscard]] bool has(const std::string& name) const {
        return reader_.find(name) != nullptr;
    }

    // Binds `name` if it has not been placed yet and returns its slab offset.
    std::uint64_t place(const std::string& name, std::int32_t rows, std::int32_t columns) {
        const auto existing = placed_.find(name);
        if (existing != placed_.end()) { return existing->second; }
        const artifact::ObjectHandle object = artifact::bind_device_tensor(
            binder_, name, NumericFormat::BF16,
            {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)});
        const auto bytes =
            static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) * 2U;
        const std::uint64_t offset = cursor_;
        cursor_                    = align_up(cursor_ + bytes, kSlabAlignment);
        plan_.objects.push_back(
            LoraObjectPlacement{.object = object, .offset = offset, .bytes = bytes});
        placed_.emplace(name, offset);
        return offset;
    }

    [[nodiscard]] std::uint64_t slab_bytes() const { return cursor_; }

    [[nodiscard]] std::int32_t rank() const { return rank_; }

private:
    artifact::Binder& binder_;
    const artifact::Reader& reader_;
    std::int32_t rank_;
    LoraBindingPlan& plan_;
    std::map<std::string, std::uint64_t> placed_;
    std::uint64_t cursor_ = 0;
};

// Binds a site whose two factors are private to it.
LoraSitePlan bind_private_site(SlabBuilder& slab, const std::string& a_name,
                               const std::string& b_name, std::int32_t in_features,
                               std::int32_t out_features, std::string_view label) {
    const bool has_a = slab.has(a_name);
    const bool has_b = slab.has(b_name);
    if (!has_a && !has_b) { return LoraSitePlan{}; }
    if (has_a != has_b) {
        throw artifact::ArtifactError("LoRA site '" + std::string(label) +
                                      "' is incomplete: both factors are required");
    }
    return LoraSitePlan{
        .a_offset = slab.place(a_name, slab.rank(), in_features),
        .b_offset = slab.place(b_name, out_features, slab.rank()),
        .rows     = out_features,
        .present  = true,
    };
}

void bind_full_layer(SlabBuilder& slab, std::size_t layer, LoraFullLayerPlan& plan) {
    const std::string prefix = layer_prefix(layer);

    // The attention parent stores query rows followed by output-gate rows for each head, so the
    // two sites are row selections of one source module and share one down-projection factor.
    const std::string shared_a = prefix + "attention/query_gate/lora_a";
    const std::string query_b  = prefix + "attention/query/lora_b";
    const std::string gate_b   = prefix + "attention/gate/lora_b";
    const bool any_query_gate  = slab.has(shared_a) || slab.has(query_b) || slab.has(gate_b);
    if (any_query_gate) {
        if (!slab.has(shared_a) || !slab.has(query_b) || !slab.has(gate_b)) {
            throw artifact::ArtifactError(
                "LoRA attention query/gate group in layer " + std::to_string(layer) +
                " is incomplete: the shared lora_a and both lora_b factors are required");
        }
        const std::uint64_t a_offset = slab.place(shared_a, slab.rank(), kHidden);
        plan.query                   = LoraSitePlan{.a_offset = a_offset,
                                                    .b_offset = slab.place(query_b, kQuerySize,
                                                                           slab.rank()),
                                                    .rows     = kQuerySize,
                                                    .present  = true};
        plan.gate                    = LoraSitePlan{.a_offset = a_offset,
                                                    .b_offset = slab.place(gate_b, kQuerySize,
                                                                           slab.rank()),
                                                    .rows     = kQuerySize,
                                                    .present  = true};
    }

    plan.key    = bind_private_site(slab, prefix + "attention/key/lora_a",
                                    prefix + "attention/key/lora_b", kHidden, kKeyValueSize,
                                    "attention/key");
    plan.value  = bind_private_site(slab, prefix + "attention/value/lora_a",
                                    prefix + "attention/value/lora_b", kHidden, kKeyValueSize,
                                    "attention/value");
    plan.output = bind_private_site(slab, prefix + "attention/output/lora_a",
                                    prefix + "attention/output/lora_b", kAttentionValues, kHidden,
                                    "attention/output");
    plan.down   = bind_private_site(slab, prefix + "mlp/down/lora_a", prefix + "mlp/down/lora_b",
                                    kIntermediate, kHidden, "mlp/down");
}

void bind_gdn_layer(SlabBuilder& slab, std::size_t layer, LoraGdnLayerPlan& plan) {
    const std::string prefix = layer_prefix(layer);
    plan.output = bind_private_site(slab, prefix + "gdn/output/lora_a",
                                    prefix + "gdn/output/lora_b", kGdnValues, kHidden,
                                    "gdn/output");
    plan.down   = bind_private_site(slab, prefix + "mlp/down/lora_a", prefix + "mlp/down/lora_b",
                                    kIntermediate, kHidden, "mlp/down");
}

// Rank is a property of the artifact, not of the request. It is read from the first registered
// down-projection factor present, then enforced on every other object by exact shape binding.
std::int32_t discover_rank(const artifact::Reader& reader) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const std::string prefix = layer_prefix(layer);
        // Held by value: a braced-init-list selected through a conditional would not have its
        // backing array's lifetime extended.
        static constexpr std::array<std::pair<const char*, std::int32_t>, 5> kFullCandidates{
            {{"attention/query_gate/lora_a", kHidden},
             {"attention/key/lora_a", kHidden},
             {"attention/value/lora_a", kHidden},
             {"attention/output/lora_a", kAttentionValues},
             {"mlp/down/lora_a", kIntermediate}}};
        static constexpr std::array<std::pair<const char*, std::int32_t>, 2> kGdnCandidates{
            {{"gdn/output/lora_a", kGdnValues}, {"mlp/down/lora_a", kIntermediate}}};
        const std::span<const std::pair<const char*, std::int32_t>> candidates =
            is_full_attention_layer(layer)
                ? std::span<const std::pair<const char*, std::int32_t>>(kFullCandidates)
                : std::span<const std::pair<const char*, std::int32_t>>(kGdnCandidates);
        for (const auto& [suffix, in_features] : candidates) {
            const std::string name                   = prefix + suffix;
            const artifact::ObjectDescriptor* object = reader.find(name);
            if (object == nullptr) { continue; }
            const auto* tensor = std::get_if<artifact::TensorDescriptor>(object);
            if (tensor == nullptr || tensor->shape.size() != 2) {
                throw artifact::ArtifactError(name + ": a LoRA factor must be a rank-two tensor");
            }
            if (tensor->shape[1] != static_cast<std::uint64_t>(in_features)) {
                throw artifact::ArtifactError(name + ": lora_a must have " +
                                              std::to_string(in_features) + " columns, found " +
                                              std::to_string(tensor->shape[1]));
            }
            const auto rank = static_cast<std::int32_t>(tensor->shape[0]);
            if (!registered_rank(rank)) {
                throw artifact::ArtifactError(name + ": LoRA rank " + std::to_string(rank) +
                                              " is not registered; supported ranks are 8, 16, 32,"
                                              " 64");
            }
            return rank;
        }
    }
    throw artifact::ArtifactError(
        "LoRA artifact carries no registered site object, so it corrects nothing");
}

qwen3_8::LoraSiteWeights bind_site_view(const LoraSitePlan& plan, void* slab_base,
                                        std::uint64_t slab_bytes, std::int32_t rank,
                                        std::int32_t in_features) {
    if (!plan.present) { return {}; }
    auto* bytes = static_cast<unsigned char*>(slab_base);
    qwen3_8::LoraSiteWeights view;
    view.a = Tensor(bytes + plan.a_offset, DType::BF16, {in_features, rank});
    view.b = Tensor(bytes + plan.b_offset, DType::BF16, {rank, plan.rows});
    view.a_adapter_stride = slab_bytes;
    view.b_adapter_stride = slab_bytes;
    return view;
}

} // namespace

LoraArtifactLoadPlan bind_lora_artifact(artifact::Binder& binder, const artifact::Reader& reader) {
    LoraArtifactLoadPlan load;
    load.bindings.rank = discover_rank(reader);

    SlabBuilder slab(binder, reader, load.bindings.rank, load.bindings);
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        if (is_full_attention_layer(layer)) {
            bind_full_layer(slab, layer, load.bindings.full_layers[full_attention_index(layer)]);
        } else {
            bind_gdn_layer(slab, layer, load.bindings.gdn_layers[gdn_index(layer)]);
        }
    }
    if (load.bindings.objects.empty()) {
        throw artifact::ArtifactError("LoRA artifact bound no registered site object");
    }
    load.bindings.slab_bytes = align_up(slab.slab_bytes(), kSlabAlignment);
    load.materialization     = binder.finish();
    return load;
}

LoadedLoraBank load_lora_bank(std::span<const LoraAdapterSpec> adapters, DeviceContext& device) {
    LoadedLoraBank bank;
    if (adapters.empty()) { return bank; }
    if (adapters.size() > kMaximumLoraAdapters) {
        throw std::invalid_argument("at most " + std::to_string(kMaximumLoraAdapters) +
                                    " LoRA adapters may be registered, received " +
                                    std::to_string(adapters.size()));
    }

    std::set<std::string> seen;
    std::vector<LoraArtifactLoadPlan> plans;
    std::vector<artifact::MaterializedArtifact> materialized;
    plans.reserve(adapters.size());
    materialized.reserve(adapters.size());

    for (const LoraAdapterSpec& spec : adapters) {
        if (spec.name.empty()) {
            throw std::invalid_argument("a registered LoRA adapter must have a name");
        }
        if (!seen.insert(spec.name).second) {
            throw std::invalid_argument("LoRA adapter name '" + spec.name +
                                        "' is registered more than once");
        }
        artifact::Reader reader(spec.path);
        const artifact::ArtifactIdentity& identity = reader.identity();
        if (identity.model_id != std::string(Package::model_id) ||
            identity.weights_id != std::string(kLoraWeightsId)) {
            throw std::invalid_argument(
                "LoRA adapter '" + spec.name + "' has identity '" + identity.model_id + "/" +
                identity.weights_id + "', but this engine requires '" +
                std::string(Package::model_id) + "/" + std::string(kLoraWeightsId) + "'");
        }
        artifact::Binder binder(reader);
        LoraArtifactLoadPlan plan = bind_lora_artifact(binder, reader);
        if (!plans.empty()) {
            if (plan.bindings.rank != plans.front().bindings.rank) {
                throw std::invalid_argument(
                    "LoRA adapter '" + spec.name + "' has rank " +
                    std::to_string(plan.bindings.rank) + ", but adapter '" + bank.names.front() +
                    "' has rank " + std::to_string(plans.front().bindings.rank) +
                    "; every registered adapter must share one rank");
            }
            if (plan.bindings.slab_bytes != plans.front().bindings.slab_bytes ||
                plan.bindings.objects.size() != plans.front().bindings.objects.size()) {
                throw std::invalid_argument(
                    "LoRA adapter '" + spec.name +
                    "' targets a different site set than adapter '" + bank.names.front() +
                    "'; every registered adapter must share one inventory");
            }
        }
        bank.file_bytes += reader.file_bytes();
        materialized.push_back(artifact::materialize(reader, plan.materialization, device));
        plans.push_back(std::move(plan));
        bank.names.push_back(spec.name);
    }

    const LoraBindingPlan& reference = plans.front().bindings;
    const std::uint64_t slab         = reference.slab_bytes;
    bank.device_bytes                = slab * adapters.size();
    bank.arena = std::make_unique<DeviceArena>(
        static_cast<std::size_t>(bank.device_bytes) + kSlabAlignment);
    const DeviceSpan storage =
        bank.arena->alloc_bytes(static_cast<std::size_t>(bank.device_bytes), kSlabAlignment);
    auto* base = static_cast<unsigned char*>(storage.data);

    // Pack adapter-major so every site's adapter stride is one constant.
    for (std::size_t index = 0; index < plans.size(); ++index) {
        unsigned char* destination = base + static_cast<std::uint64_t>(index) * slab;
        for (const LoraObjectPlacement& placement : plans[index].bindings.objects) {
            CUDA_CHECK(cudaMemcpyAsync(destination + placement.offset,
                                       materialized[index].device_data(placement.object),
                                       static_cast<std::size_t>(placement.bytes),
                                       cudaMemcpyDeviceToDevice, device.stream));
        }
    }
    device.synchronize();

    bank.view.adapters = static_cast<std::uint32_t>(plans.size());
    bank.view.rank     = reference.rank;
    for (std::size_t index = 0; index < kFullAttentionLayers; ++index) {
        const LoraFullLayerPlan& source     = reference.full_layers[index];
        qwen3_8::LoraFullLayerWeights& view = bank.view.full_layers[index];
        view.query  = bind_site_view(source.query, base, slab, reference.rank, kHidden);
        view.gate   = bind_site_view(source.gate, base, slab, reference.rank, kHidden);
        view.key    = bind_site_view(source.key, base, slab, reference.rank, kHidden);
        view.value  = bind_site_view(source.value, base, slab, reference.rank, kHidden);
        view.output = bind_site_view(source.output, base, slab, reference.rank, kAttentionValues);
        view.down   = bind_site_view(source.down, base, slab, reference.rank, kIntermediate);
    }
    for (std::size_t index = 0; index < kGdnLayers; ++index) {
        const LoraGdnLayerPlan& source     = reference.gdn_layers[index];
        qwen3_8::LoraGdnLayerWeights& view = bank.view.gdn_layers[index];
        view.output = bind_site_view(source.output, base, slab, reference.rank, kGdnValues);
        view.down   = bind_site_view(source.down, base, slab, reference.rank, kIntermediate);
    }
    return bank;
}

} // namespace ninfer::targets::qwen3_8_27b::detail
