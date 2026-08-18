#include "ninfer/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<ninfer::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<ninfer::PreparedPrompt>);
static_assert(std::is_move_constructible_v<ninfer::Engine>);
static_assert(!std::is_copy_constructible_v<ninfer::Engine>);
static_assert(requires(ninfer::RuntimeStats stats) {
    stats.continuation_lookup_hits;
    stats.continuation_lookup_misses;
    stats.continuation_preflight_rejections;
    stats.continuation_restore_successes;
    stats.continuation_restore_failures;
    stats.continuation_publication_successes;
    stats.continuation_publication_failures;
    stats.continuation_publication_superseded;
    stats.continuation_restored_tokens;
    stats.continuation_restored_bytes;
});
static_assert(ninfer::continuation_source_name(ninfer::ContinuationSource::L3) == "l3");
static_assert(ninfer::continuation_alias_kind_name(ninfer::ContinuationAliasKind::StablePrefix) ==
              "stable_prefix");
static_assert(ninfer::continuation_alias_kind_name(ninfer::ContinuationAliasKind::Session) ==
              "routed_session");
static_assert(ninfer::continuation_miss_reason_name(
                  ninfer::ContinuationMissReason::EntryUnavailableOrCorrupt) ==
              "entry_unavailable_or_corrupt");
static_assert(ninfer::continuation_miss_reason_name(ninfer::ContinuationMissReason::None) == "none");
static_assert(ninfer::continuation_miss_reason_name(ninfer::ContinuationMissReason::Disabled) ==
              "disabled");
static_assert(ninfer::continuation_miss_reason_name(ninfer::ContinuationMissReason::NoAlias) ==
              "no_alias");
static_assert(ninfer::continuation_miss_reason_name(ninfer::ContinuationMissReason::NotDeeper) ==
              "not_deeper");
static_assert(ninfer::continuation_miss_reason_name(
                  ninfer::ContinuationMissReason::PreflightRejected) == "preflight_rejected");
static_assert(ninfer::continuation_miss_reason_name(
                  ninfer::ContinuationMissReason::RollbackConflict) == "rollback_conflict");
static_assert(ninfer::continuation_miss_reason_name(ninfer::ContinuationMissReason::NoLane) ==
              "no_lane");
static_assert(ninfer::continuation_miss_reason_name(ninfer::ContinuationMissReason::RestoreFailed) ==
              "restore_failed");
static_assert(requires(ninfer::GenerationResult result) {
    result.continuation.lookup_microseconds;
    result.continuation.completion_publication_queued;
});

int main() {
    const ninfer::EngineOptions options;
    const ninfer::ContinuationCacheOptions& cache = options.continuation_cache;
    return options.enable_vision || cache.tiers != ninfer::ContinuationCacheTiers::L1L2 ||
                   cache.policy != ninfer::ContinuationCachePolicy::Adaptive ||
                   cache.l1_capacity_mib != 768 || cache.l2_capacity_mib != 16384 ||
                   cache.l3_capacity_mib != 49152 || !cache.directory.empty() ||
                   cache.cache_namespace != "local" || cache.l1_idle_ttl_seconds != 600 ||
                   cache.l2_idle_ttl_seconds != 7200 || cache.l3_idle_ttl_seconds != 86400 ||
                   cache.persist_interval_seconds != 60 || cache.persist_min_tokens != 8192 ||
                   cache.filesystem_reserve_mib != 0 || cache.prefix_checkpoint_history != 4
               ? 1
               : 0;
}
