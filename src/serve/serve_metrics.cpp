#include "serve/serve_metrics.h"

#include <algorithm>
#include <cstdio>
#include <string_view>

namespace ninfer::serve {

namespace {

void append_counter(std::string& out, const char* name, std::uint64_t value) {
    const std::string_view metric(name);
    out += "# TYPE ";
    out += name;
    out += metric.ends_with("_total") ? " counter\n" : " gauge\n";
    char line[160];
    std::snprintf(line, sizeof(line), "%s %llu\n", name, static_cast<unsigned long long>(value));
    out += line;
}

void append_counter(std::string& out, const char* name, double value) {
    const std::string_view metric(name);
    out += "# TYPE ";
    out += name;
    out += metric.ends_with("_total") ? " counter\n" : " gauge\n";
    char line[160];
    std::snprintf(line, sizeof(line), "%s %.6f\n", name, value);
    out += line;
}

} // namespace

void ServeMetrics::begin_request(std::uint64_t id, int prompt_tokens) {
    const std::lock_guard<std::mutex> lock(mutex_);
    active_[id] = prompt_tokens > 0 ? prompt_tokens : 0;
}

void ServeMetrics::end_request(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    active_.erase(id);
}

std::vector<std::pair<std::uint64_t, int>> ServeMetrics::active_snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return {active_.begin(), active_.end()};
}

void ServeMetrics::record(const GenerationOutcome& outcome) {
    const GenerationMetrics& m = outcome.metrics;
    const std::uint64_t cached = m.prefix_cache_hit_tokens;
    const std::uint64_t prompt =
        outcome.prompt_tokens > 0 ? static_cast<std::uint64_t>(outcome.prompt_tokens) : 0;
    const std::uint64_t computed_prefill = prompt > cached ? prompt - cached : 0;

    const std::lock_guard<std::mutex> lock(mutex_);
    requests_total_ += 1;
    prompt_tokens_total_ += computed_prefill;
    prompt_seconds_total_ += std::max(0.0, m.prefill_seconds);
    tokens_predicted_total_ +=
        outcome.completion_tokens > 0 ? static_cast<std::uint64_t>(outcome.completion_tokens) : 0;
    tokens_predicted_seconds_total_ += std::max(0.0, m.decode_seconds);
    prefix_cache_hit_tokens_total_ += cached;
    speculative_draft_tokens_total_ += m.speculative_draft_tokens;
    speculative_accepted_tokens_total_ += m.speculative_accepted_tokens;
    last_completed_.prompt_tokens = static_cast<int>(prompt);
    // Clamped like computed_prefill above: a cache figure reported larger
    // than the prompt must not advertise more resident tokens than exist.
    last_completed_.cached_tokens = static_cast<int>(std::min(cached, prompt));
}

ServeMetrics::LastCompleted ServeMetrics::last_completed() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return last_completed_;
}

std::string ServeMetrics::render(std::uint32_t max_concurrency, const RuntimeStats& runtime) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t in_flight  = active_.size();
    const std::uint64_t processing = std::min<std::uint64_t>(in_flight, max_concurrency);
    std::string out;
    out.reserve(16384);
    append_counter(out, "llamacpp:prompt_tokens_total", prompt_tokens_total_);
    append_counter(out, "llamacpp:prompt_seconds_total", prompt_seconds_total_);
    append_counter(out, "llamacpp:tokens_predicted_total", tokens_predicted_total_);
    append_counter(out, "llamacpp:tokens_predicted_seconds_total", tokens_predicted_seconds_total_);
    append_counter(out, "llamacpp:requests_processing", processing);
    append_counter(out, "llamacpp:requests_deferred", in_flight - processing);
    append_counter(out, "ninfer:requests_total", requests_total_);
    append_counter(out, "ninfer:prefix_cache_hit_tokens_total", prefix_cache_hit_tokens_total_);
    append_counter(out, "ninfer:draft_tokens_total", speculative_draft_tokens_total_);
    append_counter(out, "ninfer:draft_accepted_tokens_total", speculative_accepted_tokens_total_);
    append_counter(out, "ninfer:continuation_lookup_hits_total", runtime.continuation_lookup_hits);
    append_counter(out, "ninfer:continuation_lookup_misses_total",
                   runtime.continuation_lookup_misses);
    append_counter(out, "ninfer:continuation_preflight_rejections_total",
                   runtime.continuation_preflight_rejections);
    append_counter(out, "ninfer:continuation_restore_successes_total",
                   runtime.continuation_restore_successes);
    append_counter(out, "ninfer:continuation_restore_failures_total",
                   runtime.continuation_restore_failures);
    append_counter(out, "ninfer:continuation_publication_successes_total",
                   runtime.continuation_publication_successes);
    append_counter(out, "ninfer:continuation_publication_failures_total",
                   runtime.continuation_publication_failures);
    append_counter(out, "ninfer:continuation_publication_superseded_total",
                   runtime.continuation_publication_superseded);
    append_counter(out, "ninfer:continuation_restored_tokens_total",
                   runtime.continuation_restored_tokens);
    append_counter(out, "ninfer:continuation_restored_bytes_total",
                   runtime.continuation_restored_bytes);
    append_counter(out, "ninfer:continuation_l1_restore_successes_total",
                   runtime.continuation_l1_restore_successes);
    append_counter(out, "ninfer:continuation_l2_restore_successes_total",
                   runtime.continuation_l2_restore_successes);
    append_counter(out, "ninfer:continuation_l3_restore_successes_total",
                   runtime.continuation_l3_restore_successes);
    append_counter(out, "ninfer:continuation_l1_restored_tokens_total",
                   runtime.continuation_l1_restored_tokens);
    append_counter(out, "ninfer:continuation_l2_restored_tokens_total",
                   runtime.continuation_l2_restored_tokens);
    append_counter(out, "ninfer:continuation_l3_restored_tokens_total",
                   runtime.continuation_l3_restored_tokens);
    append_counter(out, "ninfer:continuation_l1_restored_bytes_total",
                   runtime.continuation_l1_restored_bytes);
    append_counter(out, "ninfer:continuation_l2_restored_bytes_total",
                   runtime.continuation_l2_restored_bytes);
    append_counter(out, "ninfer:continuation_l3_restored_bytes_total",
                   runtime.continuation_l3_restored_bytes);
    append_counter(out, "ninfer:continuation_session_restores_total",
                   runtime.continuation_session_restores);
    append_counter(out, "ninfer:continuation_stable_prefix_restores_total",
                   runtime.continuation_stable_prefix_restores);
    append_counter(out, "ninfer:continuation_miss_disabled_total",
                   runtime.continuation_miss_disabled);
    append_counter(out, "ninfer:continuation_miss_no_alias_total",
                   runtime.continuation_miss_no_alias);
    append_counter(out, "ninfer:continuation_miss_entry_unavailable_or_corrupt_total",
                   runtime.continuation_miss_entry_unavailable_or_corrupt);
    append_counter(out, "ninfer:continuation_miss_not_deeper_total",
                   runtime.continuation_miss_not_deeper);
    append_counter(out, "ninfer:continuation_miss_preflight_rejected_total",
                   runtime.continuation_miss_preflight_rejected);
    append_counter(out, "ninfer:continuation_miss_rollback_conflict_total",
                   runtime.continuation_miss_rollback_conflict);
    append_counter(out, "ninfer:continuation_miss_no_lane_total",
                   runtime.continuation_miss_no_lane);
    append_counter(out, "ninfer:continuation_miss_restore_failed_total",
                   runtime.continuation_miss_restore_failed);
    append_counter(out, "ninfer:continuation_l2_lookup_microseconds_total",
                   runtime.continuation_l2_lookup_microseconds);
    append_counter(out, "ninfer:continuation_l2_lookup_operations_total",
                   runtime.continuation_l2_lookup_operations);
    append_counter(out, "ninfer:continuation_l3_lookup_microseconds_total",
                   runtime.continuation_l3_lookup_microseconds);
    append_counter(out, "ninfer:continuation_l3_lookup_operations_total",
                   runtime.continuation_l3_lookup_operations);
    append_counter(out, "ninfer:continuation_preflight_microseconds_total",
                   runtime.continuation_preflight_microseconds);
    append_counter(out, "ninfer:continuation_preflight_operations_total",
                   runtime.continuation_preflight_operations);
    append_counter(out, "ninfer:continuation_l2_restore_microseconds_total",
                   runtime.continuation_l2_restore_microseconds);
    append_counter(out, "ninfer:continuation_l2_restore_operations_total",
                   runtime.continuation_l2_restore_operations);
    append_counter(out, "ninfer:continuation_l3_restore_microseconds_total",
                   runtime.continuation_l3_restore_microseconds);
    append_counter(out, "ninfer:continuation_l3_restore_operations_total",
                   runtime.continuation_l3_restore_operations);
    append_counter(out, "ninfer:continuation_l2_admission_microseconds_total",
                   runtime.continuation_l2_admission_microseconds);
    append_counter(out, "ninfer:continuation_l2_admission_operations_total",
                   runtime.continuation_l2_admission_operations);
    append_counter(out, "ninfer:continuation_l3_persistence_microseconds_total",
                   runtime.continuation_l3_persistence_microseconds);
    append_counter(out, "ninfer:continuation_l3_persistence_operations_total",
                   runtime.continuation_l3_persistence_operations);
    append_counter(out, "ninfer:continuation_persistence_queued_total",
                   runtime.continuation_persistence_queued);
    append_counter(out, "ninfer:continuation_persistence_coalesced_total",
                   runtime.continuation_persistence_coalesced);
    append_counter(out, "ninfer:continuation_persistence_successes_total",
                   runtime.continuation_persistence_successes);
    append_counter(out, "ninfer:continuation_persistence_failures_total",
                   runtime.continuation_persistence_failures);
    append_counter(out, "ninfer:continuation_l2_entries",
                   static_cast<std::uint64_t>(runtime.continuation_l2_entries));
    append_counter(out, "ninfer:continuation_l2_bytes", runtime.continuation_l2_bytes);
    append_counter(out, "ninfer:continuation_l3_entries",
                   static_cast<std::uint64_t>(runtime.continuation_l3_entries));
    append_counter(out, "ninfer:continuation_l3_bytes", runtime.continuation_l3_bytes);
    append_counter(out, "ninfer:l1_evictions_total", runtime.l1_evictions);
    append_counter(out, "ninfer:l1_demotions_total", runtime.l1_demotions);
    append_counter(out, "ninfer:l1_resident_entries",
                   static_cast<std::uint64_t>(runtime.l1_resident_entries));
    append_counter(out, "ninfer:l1_resident_bytes", runtime.l1_resident_bytes);
    return out;
}

} // namespace ninfer::serve
