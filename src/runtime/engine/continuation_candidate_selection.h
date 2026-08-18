#pragma once

#include "runtime/cache/continuation_cache.h"
#include "ninfer/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::runtime {

struct LaneContinuationProvenance {
    ContinuationAliasKind alias_kind = ContinuationAliasKind::None;
    ContinuationSource origin        = ContinuationSource::None;
};

enum class CacheLookupAccountingTier : std::uint8_t { None, L2, L3 };

inline CacheLookupAccountingTier cache_lookup_accounting_tier(cache::CacheSource source) noexcept {
    switch (source) {
    case cache::CacheSource::L2: return CacheLookupAccountingTier::L2;
    case cache::CacheSource::L3: return CacheLookupAccountingTier::L3;
    case cache::CacheSource::None: return CacheLookupAccountingTier::None;
    }
    return CacheLookupAccountingTier::None;
}

inline LaneContinuationProvenance imported_lane_provenance(ContinuationAliasKind alias_kind,
                                                            cache::CacheSource source) noexcept {
    return {.alias_kind = alias_kind,
            .origin = source == cache::CacheSource::L3 ? ContinuationSource::L3
                                                       : ContinuationSource::L2};
}

inline LaneContinuationProvenance completion_publication_provenance(
    LaneContinuationProvenance current, ContinuationAliasKind alias_kind) noexcept {
    if (alias_kind != ContinuationAliasKind::None) current.alias_kind = alias_kind;
    current.origin = ContinuationSource::L1;
    return current;
}

inline int continuation_miss_reason_priority(ContinuationMissReason reason) noexcept {
    switch (reason) {
    case ContinuationMissReason::None: return 0;
    case ContinuationMissReason::NoAlias: return 1;
    case ContinuationMissReason::NotDeeper: return 2;
    case ContinuationMissReason::EntryUnavailableOrCorrupt: return 3;
    case ContinuationMissReason::PreflightRejected: return 4;
    case ContinuationMissReason::RollbackConflict: return 5;
    case ContinuationMissReason::RestoreFailed: return 6;
    case ContinuationMissReason::NoLane: return 7;
    case ContinuationMissReason::Disabled: return 8;
    }
    return 0;
}

inline void observe_continuation_miss(ContinuationDiagnostics& diagnostics,
                                      ContinuationMissReason reason,
                                      ContinuationAliasKind alias_kind =
                                          ContinuationAliasKind::None) noexcept {
    if (diagnostics.source == ContinuationSource::None &&
        continuation_miss_reason_priority(reason) >=
            continuation_miss_reason_priority(diagnostics.final_miss_reason)) {
        diagnostics.final_miss_reason = reason;
        if (alias_kind != ContinuationAliasKind::None) diagnostics.alias_kind = alias_kind;
    }
}

inline void reconcile_continuation_aggregate_totals(RuntimeStats& stats) noexcept {
    stats.continuation_restore_successes = stats.continuation_l1_restore_successes +
                                           stats.continuation_l2_restore_successes +
                                           stats.continuation_l3_restore_successes;
    stats.continuation_restored_tokens = stats.continuation_l1_restored_tokens +
                                         stats.continuation_l2_restored_tokens +
                                         stats.continuation_l3_restored_tokens;
    stats.continuation_restored_bytes = stats.continuation_l1_restored_bytes +
                                        stats.continuation_l2_restored_bytes +
                                        stats.continuation_l3_restored_bytes;
}

inline void classify_resident_continuation(ContinuationDiagnostics& diagnostics,
                                            std::uint64_t reused_tokens,
                                            std::uint64_t resident_bytes,
                                            LaneContinuationProvenance provenance) {
    if (reused_tokens == 0 || diagnostics.source != ContinuationSource::None) return;
    diagnostics.source = ContinuationSource::L1;
    diagnostics.alias_kind = provenance.alias_kind;
    diagnostics.final_miss_reason = ContinuationMissReason::None;
    diagnostics.restored_tokens = reused_tokens;
    diagnostics.restored_bytes = resident_bytes;
}

struct SelectedContinuationCandidate {
    std::size_t index = 0;
    std::uint64_t reusable_depth = 0;
};

[[nodiscard]] inline bool prefer_routed_candidate(std::uint64_t routed_depth,
                                                   std::uint64_t stable_depth) noexcept {
    return routed_depth != 0 && routed_depth >= stable_depth;
}

template <class ReusableDepth>
[[nodiscard]] std::vector<SelectedContinuationCandidate>
rank_reusable_candidate_descriptors(
    std::span<const cache::SessionCandidateDescriptor> candidates,
    std::uint64_t minimum_depth, std::uint64_t required_depth,
    ReusableDepth&& reusable_depth) {
    std::vector<SelectedContinuationCandidate> viable;
    for (std::size_t i = 0; i != candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        // Effective reuse is either the saved frontier or a boundary below it.
        if (candidate.status != cache::CacheLookupStatus::Hit ||
            candidate.frontier_tokens <= minimum_depth) continue;
        const std::uint64_t depth = reusable_depth(candidate);
        if (depth > minimum_depth && depth >= required_depth) {
            viable.push_back({.index = i, .reusable_depth = depth});
        }
    }
    std::ranges::sort(viable, [](const auto& lhs, const auto& rhs) {
        return lhs.reusable_depth != rhs.reusable_depth
                   ? lhs.reusable_depth > rhs.reusable_depth
                   : lhs.index < rhs.index;
    });
    return viable;
}

} // namespace ninfer::runtime
