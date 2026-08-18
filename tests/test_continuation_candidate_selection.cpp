#include "runtime/engine/continuation_candidate_selection.h"
#include "runtime/engine/l1_reuse_accounting.h"

#include <iostream>
#include <memory>
#include <vector>

namespace {
using namespace ninfer;

void check(bool condition, const char* message);

cache::SessionCandidateDescriptor candidate(std::uint8_t marker, std::uint64_t frontier,
                                             std::uint64_t boundary = 0) {
    return {.id = cache::ContentId{std::string(64, static_cast<char>('a' + marker))},
            .image_format_version = 2,
            .compatibility_key = {marker},
            .image_bytes = 1024,
            .frontier_tokens = frontier,
            .boundary_tokens = boundary,
            .frontier_prefix_digest = cache::Bytes(32, marker),
            .source = marker == 1 ? cache::CacheSource::L3 : cache::CacheSource::L2,
            .status = cache::CacheLookupStatus::Hit};
}

void test_resident_diagnostic_classification() {
    ContinuationDiagnostics session;
    session.final_miss_reason = ContinuationMissReason::NotDeeper;
    runtime::classify_resident_continuation(
        session, 96, 4096,
        {.alias_kind = ContinuationAliasKind::StablePrefix, .origin = ContinuationSource::L3});
    check(session.source == ContinuationSource::L1 &&
               session.alias_kind == ContinuationAliasKind::StablePrefix &&
               session.final_miss_reason == ContinuationMissReason::None &&
               session.restored_tokens == 96 && session.restored_bytes == 4096,
          "resident planner reuse uses actual lane provenance");

    const auto imported = runtime::imported_lane_provenance(ContinuationAliasKind::Session,
                                                            cache::CacheSource::L2);
    check(imported.alias_kind == ContinuationAliasKind::Session &&
              imported.origin == ContinuationSource::L2,
          "lane import retains alias kind and source origin");
    const auto stable_import = runtime::imported_lane_provenance(
        ContinuationAliasKind::StablePrefix, cache::CacheSource::L3);
    const auto published = runtime::completion_publication_provenance(
        stable_import, ContinuationAliasKind::Session);
    check(stable_import.alias_kind == ContinuationAliasKind::StablePrefix &&
              stable_import.origin == ContinuationSource::L3 &&
              published.alias_kind == ContinuationAliasKind::Session &&
              published.origin == ContinuationSource::L1,
          "stable restore and successful completion publication update lane provenance");
}

void test_final_reason_precedence_and_success() {
    ContinuationDiagnostics diagnostics;
    runtime::observe_continuation_miss(diagnostics, ContinuationMissReason::PreflightRejected,
                                       ContinuationAliasKind::Session);
    runtime::observe_continuation_miss(diagnostics,
                                       ContinuationMissReason::EntryUnavailableOrCorrupt,
                                       ContinuationAliasKind::StablePrefix);
    check(diagnostics.final_miss_reason == ContinuationMissReason::PreflightRejected &&
              diagnostics.alias_kind == ContinuationAliasKind::Session,
          "a less useful fallback miss does not hide a preflight rejection");
    runtime::observe_continuation_miss(diagnostics, ContinuationMissReason::RestoreFailed,
                                       ContinuationAliasKind::StablePrefix);
    check(diagnostics.final_miss_reason == ContinuationMissReason::RestoreFailed &&
              diagnostics.alias_kind == ContinuationAliasKind::StablePrefix,
          "a failed fallback restore becomes the terminal useful reason");

    diagnostics.source = ContinuationSource::L2;
    diagnostics.final_miss_reason = ContinuationMissReason::None;
    runtime::observe_continuation_miss(diagnostics, ContinuationMissReason::RestoreFailed);
    check(diagnostics.final_miss_reason == ContinuationMissReason::None,
          "failed candidates do not overwrite a successful fallback outcome");
}

void test_none_lookup_accounting() {
    check(runtime::cache_lookup_accounting_tier(cache::CacheSource::None) ==
              runtime::CacheLookupAccountingTier::None,
          "absent lookup has no L2 operation or latency tier");
    check(runtime::cache_lookup_accounting_tier(cache::CacheSource::L2) ==
              runtime::CacheLookupAccountingTier::L2,
          "L2 lookup retains its accounting tier");
}

void test_aggregate_tier_invariant() {
    RuntimeStats stats;
    stats.continuation_l1_restore_successes = 2;
    stats.continuation_l2_restore_successes = 3;
    stats.continuation_l3_restore_successes = 5;
    stats.continuation_l1_restored_tokens = 7;
    stats.continuation_l2_restored_tokens = 11;
    stats.continuation_l3_restored_tokens = 13;
    stats.continuation_l1_restored_bytes = 17;
    stats.continuation_l2_restored_bytes = 19;
    stats.continuation_l3_restored_bytes = 23;
    runtime::reconcile_continuation_aggregate_totals(stats);
    check(stats.continuation_restore_successes == 10 &&
              stats.continuation_restored_tokens == 31 &&
              stats.continuation_restored_bytes == 59,
          "aggregate restore counters equal the sum of all tiers");
}

void test_l1_frontier_byte_accounting() {
    const runtime::L1ReuseByteLayout layout{.main_page_bytes = 100,
                                            .backend_page_bytes = 40,
                                            .current_state_bytes = 30,
                                            .checkpoint_state_bytes = 30};
    const auto current = runtime::l1_reused_bytes_at_frontier(layout, 65, 64, 64, false);
    const auto checkpoint = runtime::l1_reused_bytes_at_frontier(layout, 65, 64, 64, true);
    const auto same_frontier_after_suffix =
        runtime::l1_reused_bytes_at_frontier(layout, 65, 64, 64, false);
    check(current && *current == 270 && checkpoint == current &&
              same_frontier_after_suffix == current,
          "L1 bytes depend on the selected frontier, not generated suffix or checkpoint slot");
}

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_current_mismatch_older_exact_hit() {
    const std::vector candidates{candidate(0, 90), candidate(1, 70)};
    const auto selected = runtime::rank_reusable_candidate_descriptors(
        std::span<const cache::SessionCandidateDescriptor>(candidates), 0, 0,
        [](const cache::SessionCandidateDescriptor& item) {
            return item.compatibility_key.front() == 1 ? item.frontier_tokens : 0;
        });
    check(selected.size() == 1 && selected.front().index == 1 &&
              selected.front().reusable_depth == 70,
          "an exact older checkpoint beats a mismatching current head");
    check(candidates[selected.front().index].source == cache::CacheSource::L3,
          "history selection preserves the selected candidate tier");
}

void test_deepest_exact_and_stable_tie_order() {
    const std::vector candidates{candidate(0, 80), candidate(1, 140), candidate(2, 110),
                                 candidate(3, 140)};
    const auto selected = runtime::rank_reusable_candidate_descriptors(
        std::span<const cache::SessionCandidateDescriptor>(candidates), 0, 0,
        [](const cache::SessionCandidateDescriptor& item) {
            return item.compatibility_key.front() != 0 ? item.frontier_tokens : 0;
        });
    check(selected.size() == 3 && selected[0].index == 1 && selected[1].index == 3 &&
              selected[2].index == 2,
          "the deepest exact checkpoint is selected with newest-first tie breaking");
    check(runtime::rank_reusable_candidate_descriptors(
              std::span<const cache::SessionCandidateDescriptor>(candidates), 140, 0,
              [](const auto& item) { return item.frontier_tokens; }).empty(),
          "a checkpoint no deeper than resident reuse is not selected");
}

void test_effective_checkpoint_depth_controls_selection() {
    const std::vector candidates{candidate(0, 150, 60), candidate(1, 90)};
    const auto selected = runtime::rank_reusable_candidate_descriptors(
        std::span<const cache::SessionCandidateDescriptor>(candidates), 70, 0,
        [](const cache::SessionCandidateDescriptor& item) {
            return item.compatibility_key.front() == 0 ? item.boundary_tokens
                                                        : item.frontier_tokens;
        });
    check(selected.size() == 1 && selected.front().index == 1 &&
              selected.front().reusable_depth == 90,
          "history is ranked by effective checkpoint depth rather than saved frontier");
    check(runtime::rank_reusable_candidate_descriptors(
              std::span<const cache::SessionCandidateDescriptor>(candidates).first(1), 60, 0,
              [](const auto& item) { return item.boundary_tokens; }).empty(),
          "a boundary fallback no deeper than resident reuse is rejected");
}

void test_routed_and_stable_candidates_compare_effective_depth() {
    check(!runtime::prefer_routed_candidate(60, 100),
          "a shallower routed checkpoint cannot beat a deeper stable prefix");
    check(runtime::prefer_routed_candidate(100, 100),
          "an equally deep routed checkpoint wins without changing alias history");
    check(!runtime::prefer_routed_candidate(0, 0),
          "an absent routed candidate is never preferred");
}

void test_null_corrupt_candidate_is_skipped() {
    std::vector candidates{candidate(0, 100), candidate(1, 90)};
    candidates.front().status = cache::CacheLookupStatus::UnavailableOrCorrupt;
    const auto selected = runtime::rank_reusable_candidate_descriptors(
        std::span<const cache::SessionCandidateDescriptor>(candidates), 0, 0,
        [](const auto& item) { return item.frontier_tokens; });
    check(selected.size() == 1 && selected.front().index == 1,
          "an unavailable historical image is a miss, not a selection failure");
}
} // namespace

int main() {
    test_resident_diagnostic_classification();
    test_none_lookup_accounting();
    test_aggregate_tier_invariant();
    test_final_reason_precedence_and_success();
    test_l1_frontier_byte_accounting();
    test_current_mismatch_older_exact_hit();
    test_deepest_exact_and_stable_tie_order();
    test_effective_checkpoint_depth_controls_selection();
    test_routed_and_stable_candidates_compare_effective_depth();
    test_null_corrupt_candidate_is_skipped();
    if (failures == 0) std::cout << "continuation candidate selection tests passed\n";
    return failures == 0 ? 0 : 1;
}
