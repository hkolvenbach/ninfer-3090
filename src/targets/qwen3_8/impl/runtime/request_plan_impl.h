#include "targets/qwen3_8/impl/runtime/instance.h"
#include "targets/qwen3_8/impl/runtime/program.h"

#include "targets/qwen3_8/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS {
namespace {

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk + prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

} // namespace

RequestBasePlan
ProgramImplCore::plan_request_base(const PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.patches.empty()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->sampling                       = translate_sampling(options.sampling);
    base->adapter                        = options.adapter;
    base->allow_prefix_reuse             = options.allow_prefix_reuse;
    const std::uint32_t reserved_context_tokens =
        base->summary.prompt_tokens + (base->summary.effective_output_tokens == 0
                                           ? 0U
                                           : base->summary.effective_output_tokens - 1U);
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const std::uint32_t mtp_tokens    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(reserved_context_tokens) + draft_window - 1ULL));
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_tokens);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    }
    base->summary.admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        auto control =
            std::make_shared<qwen3_8::VisionControl>(qwen3_8::build_vision_control(prompt));
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (const qwen3_8::VisionItemControl& item : control->items) {
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last  = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end = last + 1;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > base->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            if (schedule::VisionContext::workspace_bytes(item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        base->vision_transient_bytes = schedule::VisionContext::output_transient_bytes(max_merged);
        base->vision_control         = std::move(control);
    }

    if (prompt.identity.turn_rewrite_boundary) {
        const std::uint32_t candidate = *prompt.identity.turn_rewrite_boundary;
        if (candidate == 0 || candidate >= base->summary.prompt_tokens) {
            throw std::invalid_argument("turn rewrite boundary must lie inside the prompt");
        }
        base->turn_rewrite_boundary = candidate;
    }
    if (prompt.identity.stable_prefix_boundary) {
        const std::uint32_t candidate = *prompt.identity.stable_prefix_boundary;
        if (candidate == 0 || candidate > base->summary.prompt_tokens) {
            throw std::invalid_argument("stable prefix boundary must lie inside the prompt");
        }
        qwen3_8::detail::ResidentPrefixIdentity stable_identity;
        stable_identity.assign(prompt);
        stable_identity.truncate(candidate);
        base->stable_prefix_boundary = candidate;
    }
    if (prompt.identity.user_turn_boundary) {
        const std::uint32_t candidate = *prompt.identity.user_turn_boundary;
        if (candidate == 0 || candidate >= base->summary.prompt_tokens) {
            throw std::invalid_argument("user turn boundary must lie inside the prompt");
        }
        base->user_turn_boundary = candidate;
    }
    if (base->stable_prefix_boundary && base->turn_rewrite_boundary &&
        *base->stable_prefix_boundary > *base->turn_rewrite_boundary) {
        throw std::invalid_argument("stable prefix boundary must not follow the rewrite boundary");
    }
    // The user-turn anchor is only useful strictly between the stable prefix and the rewrite
    // frontier: outside that window it duplicates an anchor the lane already holds.
    if (base->user_turn_boundary &&
        ((base->stable_prefix_boundary &&
          *base->user_turn_boundary <= *base->stable_prefix_boundary) ||
         (base->turn_rewrite_boundary &&
          *base->user_turn_boundary >= *base->turn_rewrite_boundary))) {
        base->user_turn_boundary.reset();
    }
    const std::size_t cold_prefill_splits =
        (base->vision_control != nullptr ? base->vision_control->items.size() : 0ULL) +
        (base->turn_rewrite_boundary ? 1ULL : 0ULL) + (base->user_turn_boundary ? 1ULL : 0ULL) +
        (base->stable_prefix_boundary &&
                 base->stable_prefix_boundary != base->turn_rewrite_boundary
             ? 1ULL
             : 0ULL);
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits);
    return RequestBasePlan(std::move(base));
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const RequestBasePlan& base_plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;
    plan->adapter                     = base.adapter;

    // A resident prefix produced under a different adapter is not reusable: its KV and GDN
    // recurrent state encode that adapter's weights.
    if (base.allow_prefix_reuse && prompt.identity.reusable && sequence.retained &&
        sequence.adapter == base.adapter) {
        const bool dflash_append_ready =
            speculative_backend != SpeculativeBackend::DFlash ||
            sequence.dflash_context_frontier == sequence.execution_frontier;
        if (sequence.execution_frontier != 0 && dflash_append_ready &&
            qwen3_8::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                            sequence.execution_frontier)) {
            plan->reuse      = ReusePath::AppendAtFrontier;
            plan->reuse_base = sequence.execution_frontier;
        } else if (sequence.turn_checkpoint.valid && sequence.turn_checkpoint.frontier != 0 &&
                   sequence.turn_checkpoint.frontier < prompt.token_ids.size() &&
                   qwen3_8::detail::prefix_matches(prompt, sequence.ledger,
                                                   sequence.prefix_identity,
                                                   sequence.turn_checkpoint.frontier)) {
            plan->reuse      = ReusePath::RestoreTurnCheckpoint;
            plan->reuse_base = sequence.turn_checkpoint.frontier;
        } else if (speculative_backend != SpeculativeBackend::DFlash &&
                   sequence.user_turn_anchor.valid && sequence.user_turn_anchor.frontier != 0 &&
                   sequence.user_turn_anchor.frontier < prompt.token_ids.size() &&
                   qwen3_8::detail::prefix_matches(prompt, sequence.ledger,
                                                   sequence.prefix_identity,
                                                   sequence.user_turn_anchor.frontier)) {
            // Both device anchors sit after the last user message's content, so a client that
            // rewrites that message's tail invalidates them while this one still holds.
            plan->reuse      = ReusePath::RestoreUserTurnAnchor;
            plan->reuse_base = sequence.user_turn_anchor.frontier;
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const bool append_ready =
            plan->reuse == ReusePath::AppendAtFrontier && sequence.tail_hidden_valid &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || sequence.mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready = (plan->reuse == ReusePath::RestoreTurnCheckpoint ||
                                       plan->reuse == ReusePath::RestoreUserTurnAnchor) &&
                                      decoder->mtp_cache() != nullptr &&
                                      sequence.mtp_kv_valid >= plan->reuse_base - 1;
        if (plan->reuse != ReusePath::FullReset && !append_ready && !checkpoint_ready) {
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
        }
    }

    if (plan->reuse == ReusePath::RestoreTurnCheckpoint &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash || !sequence.kv || !sequence.kv->backend ||
         sequence.dflash_context_frontier < plan->reuse_base)) {
        plan->reuse      = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    if (base.stable_prefix_boundary && *base.stable_prefix_boundary > plan->reuse_base) {
        plan->stable_checkpoint_capture_frontier = base.stable_prefix_boundary;
    }

    const std::optional<std::uint32_t> desired = base.turn_rewrite_boundary;
    const bool can_keep                        = desired && plan->reuse != ReusePath::FullReset &&
                          sequence.turn_checkpoint.valid &&
                          sequence.turn_checkpoint.frontier == *desired &&
                          qwen3_8::detail::prefix_matches(prompt, sequence.ledger,
                                                          sequence.prefix_identity, *desired);
    if (!desired) {
        plan->turn_checkpoint_action = TurnCheckpointAction::Drop;
    } else if (can_keep) {
        plan->turn_checkpoint_action = TurnCheckpointAction::KeepExisting;
    } else {
        if (*desired <= plan->reuse_base) {
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
        }
        plan->turn_checkpoint_action           = TurnCheckpointAction::CaptureNew;
        plan->turn_checkpoint_capture_frontier = desired;
    }

    // The user-turn anchor is stationary for the whole turn, so within a tool loop it is kept
    // rather than recaptured; it only moves when a new user query arrives.
    const std::optional<std::uint32_t> user_turn = base.user_turn_boundary;
    plan->keep_user_turn_anchor =
        user_turn && plan->reuse != ReusePath::FullReset && sequence.user_turn_anchor.valid &&
        sequence.user_turn_anchor.frontier == *user_turn &&
        qwen3_8::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                        *user_turn);
    if (user_turn && !plan->keep_user_turn_anchor && *user_turn > plan->reuse_base &&
        speculative_backend != SpeculativeBackend::DFlash) {
        plan->user_turn_capture_frontier = user_turn;
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (plan->reuse == ReusePath::RestoreTurnCheckpoint ||
                   plan->reuse == ReusePath::RestoreUserTurnAnchor) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = MtpBridgeMode::BeforeSuffix;
        }
    }

    if (base.vision_control != nullptr) {
        VisionPrefillPlan vision;
        vision.control = base.vision_control;
        vision.uses.reserve(base.vision_control->items.size());
        for (std::size_t index = 0; index < base.vision_control->items.size(); ++index) {
            const qwen3_8::VisionItemControl& item = base.vision_control->items[index];
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
        }
        if (!vision.uses.empty()) {
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes     = base.vision_transient_bytes;
            plan->vision                      = std::move(vision);
        }
    }

    const std::size_t prefill_splits =
        (plan->vision ? plan->vision->uses.size() : 0ULL) +
        (plan->turn_checkpoint_capture_frontier ? 1ULL : 0ULL) +
        (plan->user_turn_capture_frontier ? 1ULL : 0ULL) +
        (plan->stable_checkpoint_capture_frontier &&
                 plan->stable_checkpoint_capture_frontier !=
                     plan->turn_checkpoint_capture_frontier
             ? 1ULL
             : 0ULL);
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, plan->reuse_base, prefill_chunk, prefill_splits);
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS
