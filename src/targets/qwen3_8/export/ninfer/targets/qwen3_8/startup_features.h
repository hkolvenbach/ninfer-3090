#pragma once

#include "ninfer/ops/lora.h"
#include "ninfer/types.h"

namespace ninfer::targets::qwen3_8 {

struct StartupFeatures {
    bool vision                    = false;
    std::uint32_t vision_max_tokens = 8192;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    // Startup-fixed number of resident LoRA adapters. Zero means the LoRA leaves are never
    // emitted, so the captured graph is topologically identical to an engine built without
    // adapter support.
    std::uint32_t lora_adapters = 0;
    // Rank the workspace is sized for. The layout is frozen before any adapter artifact is read,
    // so it is sized for the largest registered rank; the bank's actual rank lives on the model
    // view and is what the Op executes. The difference is well under a MiB of transient scratch.
    std::uint32_t lora_sizing_rank = 0;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] bool lora() const noexcept { return lora_adapters > 0; }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision            = options.enable_vision,
        .vision_max_tokens = options.vision_max_tokens > 0 ? options.vision_max_tokens : 8192,
        .speculative       = options.speculative.backend,
        .proposal_head     = options.speculative.proposal_head,
        .lora_adapters     = static_cast<std::uint32_t>(options.lora_adapters.size()),
        .lora_sizing_rank  = options.lora_adapters.empty() ? 0U : ops::kMaximumLoraRank,
    };
}

} // namespace ninfer::targets::qwen3_8
