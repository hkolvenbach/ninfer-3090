#include "serve/serve_metrics.h"

#include <cstdio>
#include <map>
#include <sstream>
#include <string>

namespace {

using ninfer::serve::GenerationMetrics;
using ninfer::serve::GenerationOutcome;
using ninfer::serve::ServeMetrics;

int check(bool ok, const char* label) {
    if (!ok) { std::printf("FAIL %s\n", label); }
    return ok ? 0 : 1;
}

std::map<std::string, double> parse(const std::string& body) {
    std::map<std::string, double> values;
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.starts_with('#')) { continue; }
        const auto space = line.find(' ');
        if (space == std::string::npos) { continue; }
        values[line.substr(0, space)] = std::stod(line.substr(space + 1));
    }
    return values;
}

GenerationOutcome outcome(int prompt, std::uint32_t cached, int completion, double prefill_s,
                          double decode_s, std::uint64_t drafted, std::uint64_t accepted) {
    GenerationOutcome out;
    out.prompt_tokens                       = prompt;
    out.completion_tokens                   = completion;
    out.metrics.prefix_cache_hit_tokens     = cached;
    out.metrics.prefill_seconds             = prefill_s;
    out.metrics.decode_seconds              = decode_s;
    out.metrics.speculative_draft_tokens    = drafted;
    out.metrics.speculative_accepted_tokens = accepted;
    return out;
}

} // namespace

int main() {
    int failures = 0;

    ServeMetrics metrics;
    const auto empty = parse(metrics.render(1));
    failures += check(empty.at("llamacpp:prompt_tokens_total") == 0.0, "starts at zero");
    const auto never = metrics.last_completed();
    failures += check(never.prompt_tokens == 0 && never.cached_tokens == 0,
                      "last completed starts at zero");
    failures += check(empty.at("ninfer:requests_total") == 0.0, "requests start at zero");
    failures += check(empty.at("llamacpp:requests_processing") == 0.0, "idle processing");
    failures += check(empty.at("llamacpp:requests_deferred") == 0.0, "idle deferred");

    // Two in-flight requests against one execution lane: FIFO order says the
    // older one processes and the newer one is deferred.
    metrics.begin_request(7, 500);
    metrics.begin_request(8, 900);
    const auto busy = parse(metrics.render(1));
    failures += check(busy.at("llamacpp:requests_processing") == 1.0, "one processing");
    failures += check(busy.at("llamacpp:requests_deferred") == 1.0, "one deferred");
    const auto active = metrics.active_snapshot();
    failures += check(active.size() == 2 && active[0].first == 7 && active[0].second == 500,
                      "snapshot FIFO order");
    metrics.end_request(7);
    metrics.end_request(7); // idempotent
    metrics.end_request(8);
    const auto drained = parse(metrics.render(1));
    failures += check(drained.at("llamacpp:requests_processing") == 0.0, "drained processing");
    failures += check(drained.at("llamacpp:requests_deferred") == 0.0, "drained deferred");

    // Cold request: whole prompt computed.
    metrics.record(outcome(1000, 0, 200, 0.5, 4.0, 300, 150));
    const auto cold = metrics.last_completed();
    failures += check(cold.prompt_tokens == 1000 && cold.cached_tokens == 0,
                      "last completed after cold request");
    // Warm request: 900 of 1200 prompt tokens served from the prefix cache -
    // only the 300 computed tokens may count toward the prompt counter.
    metrics.record(outcome(1200, 900, 100, 0.1, 2.0, 150, 75));
    const auto warm = metrics.last_completed();
    failures += check(warm.prompt_tokens == 1200 && warm.cached_tokens == 900,
                      "last completed after warm request");

    ninfer::RuntimeStats runtime;
    runtime.continuation_lookup_hits           = 7;
    runtime.continuation_lookup_misses         = 3;
    runtime.continuation_preflight_rejections  = 2;
    runtime.continuation_restore_successes     = 5;
    runtime.continuation_restore_failures      = 1;
    runtime.continuation_publication_successes = 4;
    runtime.continuation_publication_failures  = 2;
    runtime.continuation_publication_superseded = 3;
    runtime.continuation_restored_tokens       = 123;
    runtime.continuation_restored_bytes        = 456;
    runtime.continuation_persistence_queued    = 12;
    runtime.continuation_persistence_coalesced = 5;
    runtime.continuation_persistence_successes = 6;
    runtime.continuation_persistence_failures  = 1;
    runtime.continuation_l2_entries            = 3;
    runtime.continuation_l2_bytes              = 1000;
    runtime.continuation_l3_entries            = 4;
    runtime.continuation_l3_bytes              = 2000;
    runtime.l1_evictions                        = 8;
    runtime.l1_demotions                        = 6;
    runtime.l1_resident_entries                 = 2;
    runtime.l1_resident_bytes                   = 789;
    runtime.continuation_l1_restore_successes   = 2;
    runtime.continuation_l2_restore_successes   = 3;
    runtime.continuation_l3_restore_successes   = 4;
    runtime.continuation_miss_disabled          = 5;
    runtime.continuation_miss_no_alias          = 6;
    runtime.continuation_miss_entry_unavailable_or_corrupt = 7;
    runtime.continuation_miss_not_deeper        = 8;
    runtime.continuation_miss_preflight_rejected = 9;
    runtime.continuation_miss_rollback_conflict = 10;
    runtime.continuation_miss_no_lane           = 11;
    runtime.continuation_miss_restore_failed    = 6;
    runtime.continuation_l2_lookup_microseconds = 700;
    runtime.continuation_l3_lookup_operations   = 8;
    runtime.continuation_l2_admission_microseconds = 900;
    runtime.continuation_l3_persistence_operations = 10;
    const auto values                          = parse(metrics.render(1, runtime));
    const std::string rendered                 = metrics.render(1, runtime);
    failures += check(rendered.find("# TYPE ninfer:continuation_restore_successes_total counter") !=
                              std::string::npos &&
                          rendered.find("# TYPE ninfer:continuation_l2_bytes gauge") !=
                              std::string::npos &&
                          rendered.find("# TYPE ninfer:l1_resident_entries gauge") !=
                              std::string::npos,
                      "Prometheus counter and gauge types rendered");
    failures += check(values.at("llamacpp:prompt_tokens_total") == 1300.0, "computed prefill sum");
    failures += check(values.at("llamacpp:prompt_seconds_total") == 0.6, "prefill seconds sum");
    failures += check(values.at("llamacpp:tokens_predicted_total") == 300.0, "decode tokens sum");
    failures +=
        check(values.at("llamacpp:tokens_predicted_seconds_total") == 6.0, "decode seconds sum");
    failures += check(values.at("ninfer:requests_total") == 2.0, "request count");
    failures += check(values.at("ninfer:prefix_cache_hit_tokens_total") == 900.0, "cache hits");
    failures += check(values.at("ninfer:draft_tokens_total") == 450.0, "draft tokens");
    failures += check(values.at("ninfer:draft_accepted_tokens_total") == 225.0, "accepted tokens");
    failures += check(values.at("ninfer:continuation_lookup_hits_total") == 7.0 &&
                          values.at("ninfer:continuation_lookup_misses_total") == 3.0 &&
                          values.at("ninfer:continuation_restore_successes_total") == 5.0 &&
                            values.at("ninfer:continuation_publication_failures_total") == 2.0 &&
                            values.at("ninfer:continuation_publication_superseded_total") == 3.0 &&
                           values.at("ninfer:continuation_restored_tokens_total") == 123.0 &&
                            values.at("ninfer:continuation_restored_bytes_total") == 456.0 &&
                            values.at("ninfer:continuation_persistence_queued_total") == 12.0 &&
                            values.at("ninfer:continuation_persistence_coalesced_total") == 5.0 &&
                            values.at("ninfer:continuation_persistence_successes_total") == 6.0 &&
                            values.at("ninfer:continuation_persistence_failures_total") == 1.0 &&
                            values.at("ninfer:continuation_l2_entries") == 3.0 &&
                            values.at("ninfer:continuation_l2_bytes") == 1000.0 &&
                            values.at("ninfer:continuation_l3_entries") == 4.0 &&
                            values.at("ninfer:continuation_l3_bytes") == 2000.0 &&
                           values.at("ninfer:l1_evictions_total") == 8.0 &&
                           values.at("ninfer:l1_demotions_total") == 6.0 &&
                           values.at("ninfer:l1_resident_entries") == 2.0 &&
                            values.at("ninfer:l1_resident_bytes") == 789.0,
                       "continuation runtime counters rendered");
    failures += check(values.at("ninfer:continuation_l1_restore_successes_total") == 2.0 &&
                          values.at("ninfer:continuation_l2_restore_successes_total") == 3.0 &&
                           values.at("ninfer:continuation_l3_restore_successes_total") == 4.0 &&
                           values.at("ninfer:continuation_miss_disabled_total") == 5.0 &&
                           values.at("ninfer:continuation_miss_no_alias_total") == 6.0 &&
                           values.at(
                               "ninfer:continuation_miss_entry_unavailable_or_corrupt_total") ==
                               7.0 &&
                           values.at("ninfer:continuation_miss_not_deeper_total") == 8.0 &&
                           values.at("ninfer:continuation_miss_preflight_rejected_total") == 9.0 &&
                           values.at("ninfer:continuation_miss_rollback_conflict_total") == 10.0 &&
                           values.at("ninfer:continuation_miss_no_lane_total") == 11.0 &&
                           values.at("ninfer:continuation_miss_restore_failed_total") == 6.0 &&
                          values.at("ninfer:continuation_l2_lookup_microseconds_total") == 700.0 &&
                          values.at("ninfer:continuation_l3_lookup_operations_total") == 8.0 &&
                          values.at("ninfer:continuation_l2_admission_microseconds_total") == 900.0 &&
                          values.at("ninfer:continuation_l3_persistence_operations_total") == 10.0,
                      "stable tier, reason, latency, and persistence metric names rendered");

    // A cache hit reported larger than the prompt must clamp, not underflow.
    metrics.record(outcome(10, 50, 1, 0.0, 0.1, 0, 0));
    const auto clamped = parse(metrics.render(1));
    failures += check(clamped.at("llamacpp:prompt_tokens_total") == 1300.0, "underflow clamped");
    const auto residue = metrics.last_completed();
    failures += check(residue.prompt_tokens == 10 && residue.cached_tokens == 10,
                      "last completed cache clamped to prompt");

    std::printf("%s serve metrics\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
