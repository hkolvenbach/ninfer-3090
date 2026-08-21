#pragma once

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "ninfer/types.h"
#include "runtime/cache/continuation_cache.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/continuation_candidate_selection.h"
#include "runtime/engine/l1_retention_policy.h"
#include "runtime/engine/request_memory.h"
#include "runtime/engine/stable_prefix_flights.h"
#include "runtime/generation/generation_budget.h"
#include "targets/qwen3_8/export/ninfer/targets/qwen3_8/frontend.h"
#include "targets/qwen3_8/export/ninfer/targets/qwen3_8/prepared_prompt.h"
#include "targets/qwen3_8/export/ninfer/targets/qwen3_8/runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Namespaces a continuation-cache alias by the adapter that produced it. Every alias is scoped,
// including the base weights, so no unscoped key exists and two adapters can never collide. The
// separator is ASCII unit-separator, which no session id or stable-prefix digest contains.
inline std::string adapter_scoped_alias(std::int32_t adapter, std::string_view alias) {
    std::string scope = adapter < 0 ? std::string("base") : "lora" + std::to_string(adapter);
    scope += '\x1f';
    scope += alias;
    return scope;
}

template <class Instance>
class ConcurrentExecutor {
    struct Request;

public:
    using Package  = typename Instance::Package;
    using Program  = typename Package::Program;
    using BasePlan = typename Package::RequestBasePlan;
    using Plan     = typename Package::RequestPlan;
    using Clock    = std::chrono::steady_clock;

    ConcurrentExecutor(Instance& instance, const EngineOptions& options)
        : instance_(instance), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          auto_save_evicted_(options.auto_save_evicted),
          admission_capacity_(instance.program->admission_capacity()),
          continuation_cache_(make_continuation_cache(options.continuation_cache)),
          l1_policy_active_(options.continuation_cache.tiers != ContinuationCacheTiers::Off),
          l1_byte_budget_(mib_to_bytes(options.continuation_cache.l1_capacity_mib,
                                       "continuation L1 capacity")),
          l1_idle_ttl_(std::chrono::seconds(options.continuation_cache.l1_idle_ttl_seconds)),
          publication_l2_ttl_(std::chrono::seconds(
              options.continuation_cache.l2_idle_ttl_seconds)),
          publication_l3_ttl_(std::chrono::seconds(
              options.continuation_cache.l3_idle_ttl_seconds)) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        if (continuation_cache_) {
            publication_worker_ = std::thread([this] { publication_loop(); });
        }
        try {
            worker_ = std::thread([this] { worker_loop(); });
        } catch (...) {
            stop_publication_worker();
            throw;
        }
    }

    ~ConcurrentExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
        stop_publication_worker();
    }

    ConcurrentExecutor(const ConcurrentExecutor&)            = delete;
    ConcurrentExecutor& operator=(const ConcurrentExecutor&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    Submission submit(targets::qwen3_8::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      Clock::time_point pending_deadline = {}, HostInputLease host_input = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            ++outstanding_;
            request_id = next_request_id_++;
        }

        // A continuation produced under one adapter encodes that adapter's weights in its KV and
        // GDN recurrent state, so every cache alias is namespaced by the selected adapter. This
        // is a correctness requirement, not an optimization.
        if (options.routing_hint) {
            options.routing_hint = adapter_scoped_alias(options.execution.adapter,
                                                        *options.routing_hint);
        }

        std::shared_ptr<Request> request;
        try {
            CachedContinuation routed_continuation;
            std::optional<PendingSessionPublication> pending_session_publication;
            std::optional<std::string> stable_alias;
            CachedContinuation stable_continuation;
            std::optional<std::uint32_t> stable_boundary;
            if (continuation_lookup_enabled(static_cast<bool>(continuation_cache_),
                                            options.execution.allow_prefix_reuse)) {
                if (options.routing_hint) {
                    // Bracket the non-waiting snapshot so a publication that completes or is
                    // queued during cache I/O is still reconciled by the scheduler.
                    const auto publication_before =
                        this->pending_session_publication(*options.routing_hint);
                    routed_continuation = lookup_continuation(
                        *options.routing_hint, ContinuationAliasKind::Session, false, true);
                    const auto publication_after =
                        this->pending_session_publication(*options.routing_hint);
                    if (publication_after &&
                        (!publication_before ||
                         publication_after->sequence != publication_before->sequence ||
                         !publication_before->completed)) {
                        pending_session_publication = publication_after;
                    }
                }
                stable_alias = instance_.program->stable_prefix_alias(prompt);
                if (stable_alias) {
                    stable_alias =
                        adapter_scoped_alias(options.execution.adapter, *stable_alias);
                }
                stable_boundary = targets::qwen3_8::PreparedPromptAccess::view(prompt)
                                      .identity.stable_prefix_boundary;
                if (stable_alias) {
                    stable_continuation = lookup_continuation(
                        *stable_alias, ContinuationAliasKind::StablePrefix, false);
                }
            }
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                          options.output);
            if (Clock::now() >= pending_deadline) {
                throw RequestError(RequestErrorKind::QueueTimeout,
                                   "inference request expired before submission");
            }
            request     = std::make_shared<Request>(
                request_id, std::move(prompt), std::move(output), prompt_summary, prepare_seconds,
                std::move(options), pending_deadline, submitted, std::move(host_input),
                std::move(routed_continuation), std::move(stable_continuation),
                std::move(stable_alias), stable_boundary,
                std::move(pending_session_publication));
            initialize_stable_flight(request);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                release_stable_builder(request);
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        out.request_transient                  = instance_.request_memory.summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        RuntimeStats snapshot = published_stats_;
        snapshot.continuation_lookup_hits =
            continuation_stats_.lookup_hits.load(std::memory_order_relaxed);
        snapshot.continuation_lookup_misses =
            continuation_stats_.lookup_misses.load(std::memory_order_relaxed);
        snapshot.continuation_preflight_rejections =
            continuation_stats_.preflight_rejections.load(std::memory_order_relaxed);
        reconcile_continuation_aggregate_totals(snapshot);
        snapshot.continuation_restore_failures =
            continuation_stats_.restore_failures.load(std::memory_order_relaxed);
        snapshot.continuation_publication_successes =
            continuation_stats_.publication_successes.load(std::memory_order_relaxed);
        snapshot.continuation_publication_failures =
            continuation_stats_.publication_failures.load(std::memory_order_relaxed);
        snapshot.continuation_publication_superseded =
            continuation_stats_.publication_superseded.load(std::memory_order_relaxed);
        snapshot.continuation_l2_lookup_microseconds =
            continuation_stats_.l2_lookup_microseconds.load(std::memory_order_relaxed);
        snapshot.continuation_l2_lookup_operations =
            continuation_stats_.l2_lookup_operations.load(std::memory_order_relaxed);
        snapshot.continuation_l3_lookup_microseconds =
            continuation_stats_.l3_lookup_microseconds.load(std::memory_order_relaxed);
        snapshot.continuation_l3_lookup_operations =
            continuation_stats_.l3_lookup_operations.load(std::memory_order_relaxed);
        if (continuation_cache_) {
            const cache::CacheStats cache = continuation_cache_->stats();
            snapshot.continuation_persistence_queued = cache.persistence_queued;
            snapshot.continuation_persistence_coalesced = cache.persistence_coalesced;
            snapshot.continuation_persistence_successes = cache.persistence_successes;
            snapshot.continuation_persistence_failures = cache.persistence_failures;
            snapshot.continuation_l2_entries = static_cast<std::uint32_t>(std::min<std::size_t>(
                cache.l2_entries, std::numeric_limits<std::uint32_t>::max()));
            snapshot.continuation_l2_bytes = cache.l2_bytes;
            snapshot.continuation_l3_entries = static_cast<std::uint32_t>(std::min<std::size_t>(
                cache.l3_entries, std::numeric_limits<std::uint32_t>::max()));
            snapshot.continuation_l3_bytes = cache.l3_bytes;
            snapshot.continuation_l2_admission_microseconds = cache.l2_admission_microseconds;
            snapshot.continuation_l2_admission_operations = cache.l2_admission_operations;
            snapshot.continuation_l3_persistence_microseconds = cache.l3_persistence_microseconds;
            snapshot.continuation_l3_persistence_operations = cache.l3_persistence_operations;
        }
        return snapshot;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
        } catch (...) {}
    }

    // Session persistence entry points. Each claims the execution mutex, so GPU copies land at
    // a request boundary; the worker resumes as soon as the device round trip completes. A lane
    // with an active request is refused rather than drained. A non-empty expected_digest is a
    // precondition on the lane's resident session, checked atomically with the operation.
    // session_path is the file the operation targets; a successful save or restore binds the
    // lane to it so an involuntary eviction can spill the session back (see
    // spill_retained_lane).
    [[nodiscard]] targets::qwen3_8::RetainedSessionSnapshot
    save_retained_lane(std::uint32_t lane, std::string_view model_binding,
                       std::string_view expected_digest, std::string_view session_path = {}) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        require_session_digest(lane, expected_digest);
        auto snapshot = instance_.program->save_retained_lane(lane, model_binding);
        if (!session_path.empty()) { lane_session_path_[lane] = session_path; }
        return snapshot;
    }

    [[nodiscard]] std::pair<std::uint32_t, std::string>
    restore_retained_lane(std::uint32_t lane, std::span<const std::uint8_t> snapshot,
                          std::string_view model_binding, std::string_view session_path = {}) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        if (instance_.program->has_retained_lane(lane)) {
            // Involuntary for whatever session held the lane: the client asked for a restore,
            // not for that session's destruction.
            spill_retained_lane(lane);
            instance_.program->evict_retained_lane(lane);
            invalidate_lane_plans(lane);
        }
        lane_session_path_[lane].clear();
        const std::uint32_t tokens =
            instance_.program->restore_retained_lane(lane, snapshot, model_binding);
        invalidate_lane_plans(lane);
        if (!session_path.empty()) { lane_session_path_[lane] = session_path; }
        retained_digest_cache_[lane] = instance_.program->retained_lane_digest(lane);
        retained_checkpoints_cache_[lane] = instance_.program->retained_lane_checkpoints(lane);
        publish_runtime_stats();
        return {tokens, retained_digest_cache_[lane]};
    }

    std::uint32_t erase_retained_lane(std::uint32_t lane, std::string_view expected_digest) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        require_session_digest(lane, expected_digest);
        const std::uint32_t tokens = instance_.program->retained_lane_depth(lane);
        // Explicit erase is a deletion request: never auto-save, and drop the binding.
        lane_session_path_[lane].clear();
        if (instance_.program->has_retained_lane(lane)) {
            instance_.program->evict_retained_lane(lane);
            invalidate_lane_plans(lane);
            publish_runtime_stats();
        }
        return tokens;
    }

    // Installs the auto-save sink: the model binding save_retained_lane needs, and a consumer
    // that receives (path, snapshot) for each spilled session and writes the file off-thread.
    void set_eviction_sink(
        std::string model_binding,
        std::function<void(std::string, targets::qwen3_8::RetainedSessionSnapshot&&)> sink) {
        std::scoped_lock lock(execution_mutex_);
        eviction_model_binding_ = std::move(model_binding);
        eviction_sink_          = std::move(sink);
    }

    // Truthful per-lane occupancy: an active request's prompt size, or the retained session's
    // depth and identifying digest. Served from the snapshot the worker publishes at every unit
    // boundary - the execution mutex is held nearly continuously while a request runs, so a
    // scraper that waited on it would starve for the length of a deep prefill.
    [[nodiscard]] std::vector<SlotState> slot_states() const {
        std::lock_guard lock(stats_mutex_);
        std::vector<SlotState> states = published_slots_;
        states.resize(max_concurrency_);
        return states;
    }

private:
    static std::size_t mib_to_bytes(std::size_t mib, const char* field) {
        constexpr std::size_t mib_bytes = 1024U * 1024U;
        if (mib > std::numeric_limits<std::size_t>::max() / mib_bytes) {
            throw std::invalid_argument(std::string(field) + " exceeds the addressable byte range");
        }
        return mib * mib_bytes;
    }

    static void validate_cache_namespace(const std::string& value) {
        const std::filesystem::path path(value);
        const bool safe_characters = std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                   c == '.' || c == '_' || c == '-';
        });
        if (value.empty() || value == "." || value == ".." || path.is_absolute() ||
            path.has_root_name() || path.has_root_directory() ||
            std::distance(path.begin(), path.end()) != 1 || path.filename().string() != value ||
            !safe_characters) {
            throw std::invalid_argument(
                "continuation cache namespace must be one safe relative path component");
        }
    }

    static std::unique_ptr<cache::ContinuationCache>
    make_continuation_cache(const ContinuationCacheOptions& options) {
        if (options.policy != ContinuationCachePolicy::Adaptive) {
            throw std::invalid_argument("unsupported continuation cache policy");
        }

        switch (options.tiers) {
        case ContinuationCacheTiers::Off:
        case ContinuationCacheTiers::L1:
            return nullptr;
        case ContinuationCacheTiers::L1L2: {
            if (options.prefix_checkpoint_history == 0) {
                throw std::invalid_argument("continuation cache history must be positive");
            }
            return std::make_unique<cache::ContinuationCache>(cache::CacheConfig{
                .l2_byte_budget = mib_to_bytes(options.l2_capacity_mib, "continuation L2 capacity"),
                .session_history_depth = options.prefix_checkpoint_history,
                .l2_idle_ttl           = std::chrono::seconds(options.l2_idle_ttl_seconds),
            });
        }
        case ContinuationCacheTiers::L1L2L3: {
            if (options.directory.empty()) {
                throw std::invalid_argument("continuation L3 requires a cache directory");
            }
            if (options.prefix_checkpoint_history == 0) {
                throw std::invalid_argument("continuation cache history must be positive");
            }
            validate_cache_namespace(options.cache_namespace);
            return std::make_unique<cache::ContinuationCache>(cache::CacheConfig{
                .enable_l3      = true,
                .root           = options.directory / options.cache_namespace,
                .l2_byte_budget = mib_to_bytes(options.l2_capacity_mib, "continuation L2 capacity"),
                .l3_byte_budget = mib_to_bytes(options.l3_capacity_mib, "continuation L3 capacity"),
                .filesystem_reserve_bytes =
                    mib_to_bytes(options.filesystem_reserve_mib, "continuation filesystem reserve"),
                .session_history_depth = options.prefix_checkpoint_history,
                .l2_idle_ttl           = std::chrono::seconds(options.l2_idle_ttl_seconds),
                .l3_idle_ttl           = std::chrono::seconds(options.l3_idle_ttl_seconds),
                .persist_interval = std::chrono::seconds(options.persist_interval_seconds),
                .persist_min_tokens = options.persist_min_tokens,
            });
        }
        }
        throw std::invalid_argument("unsupported continuation cache tier selection");
    }

    struct CachedContinuation {
        std::optional<cache::ContentId> id;
        std::shared_ptr<const cache::ContinuationImage> image;
        std::vector<cache::SessionCandidateDescriptor> candidates;
        std::optional<std::uint64_t> generation;
        ContinuationAliasKind alias_kind = ContinuationAliasKind::None;
        cache::CacheSource source = cache::CacheSource::None;
        cache::CacheLookupStatus status = cache::CacheLookupStatus::Absent;
        std::uint64_t lookup_microseconds = 0;
    };

    enum class PublicationStatus : std::uint8_t { Pending, Success, Failed, Superseded };
    using PublicationTicket = std::shared_ptr<std::atomic<PublicationStatus>>;

    struct Publication {
        cache::ContinuationImage image;
        std::string session;
        std::optional<cache::ContentId> expected_head;
        std::optional<std::uint64_t> expected_generation;
        PublicationTicket ticket;
        std::uint64_t sequence = 0;
        bool immutable         = false;
        std::optional<std::pair<std::string, std::uint64_t>> stable_flight;
    };

    struct PendingSessionPublication {
        std::uint64_t sequence = 0;
        bool completed         = false;
    };

    struct LaneSession {
        std::string name;
        std::optional<cache::ContentId> expected_head;
        std::optional<std::uint64_t> expected_generation;
        PublicationTicket publication;
    };

    void refresh_lane_provenance(std::uint32_t lane) noexcept {
        if (!instance_.program->has_retained_lane(lane)) {
            lane_provenance_[lane] = {};
            return;
        }
        if (lane_sessions_[lane] && lane_sessions_[lane]->publication &&
            lane_sessions_[lane]->publication->load(std::memory_order_acquire) ==
                PublicationStatus::Success) {
            lane_provenance_[lane] = completion_publication_provenance(
                lane_provenance_[lane], ContinuationAliasKind::Session);
        }
    }

    struct AtomicContinuationStats {
        std::atomic<std::uint64_t> lookup_hits{0};
        std::atomic<std::uint64_t> lookup_misses{0};
        std::atomic<std::uint64_t> preflight_rejections{0};
        std::atomic<std::uint64_t> restore_successes{0};
        std::atomic<std::uint64_t> restore_failures{0};
        std::atomic<std::uint64_t> publication_successes{0};
        std::atomic<std::uint64_t> publication_failures{0};
        std::atomic<std::uint64_t> publication_superseded{0};
        std::atomic<std::uint64_t> restored_tokens{0};
        std::atomic<std::uint64_t> restored_bytes{0};
        std::atomic<std::uint64_t> l2_lookup_microseconds{0};
        std::atomic<std::uint64_t> l2_lookup_operations{0};
        std::atomic<std::uint64_t> l3_lookup_microseconds{0};
        std::atomic<std::uint64_t> l3_lookup_operations{0};
    };

    CachedContinuation lookup_continuation(const std::string& session,
                                             ContinuationAliasKind alias_kind,
                                             bool wait_for_publication = true,
                                             bool include_history = false,
                                             Clock::time_point deadline = {}) {
        const auto lookup_started = Clock::now();
        CachedContinuation candidate;
        candidate.alias_kind = alias_kind;
        try {
            if (wait_for_publication) {
                std::unique_lock lock(publication_mutex_);
                const auto pending = session_publications_.find(session);
                if (pending != session_publications_.end()) {
                    const std::uint64_t pending_sequence = pending->second.sequence;
                    const auto completed = [&] {
                        return publication_completed_ >= pending_sequence || publication_stopping_;
                    };
                    if (deadline == Clock::time_point{}) {
                        publication_cv_.wait(lock, completed);
                    } else if (!publication_cv_.wait_until(lock, deadline, completed)) {
                        throw RequestError(RequestErrorKind::QueueTimeout,
                                           "inference request expired before submission");
                    }
                }
            }
            if (include_history) {
                auto snapshot        = continuation_cache_->session_candidates(session);
                candidate.candidates = std::move(snapshot.newest_to_oldest);
                candidate.generation = snapshot.generation;
                if (!candidate.candidates.empty()) {
                    candidate.id    = candidate.candidates.front().id;
                    candidate.source = candidate.candidates.front().source;
                    candidate.status = candidate.candidates.front().status;
                }
                for (const auto& item : candidate.candidates) {
                    if (item.status == cache::CacheLookupStatus::UnavailableOrCorrupt) {
                        candidate.status = item.status;
                    }
                }
            } else {
                candidate.id = continuation_cache_->session_current(session);
                if (candidate.id) {
                    auto lookup = continuation_cache_->lookup_shared(*candidate.id);
                    candidate.image = std::move(lookup.image);
                    candidate.source = lookup.source;
                    candidate.status = lookup.status;
                    candidate.lookup_microseconds = lookup.io_microseconds;
                }
            }
        } catch (const RequestError&) {
            throw;
        } catch (...) {
            candidate.image.reset();
            candidate.status = cache::CacheLookupStatus::UnavailableOrCorrupt;
        }
        const bool hit = candidate.image ||
                         std::ranges::any_of(candidate.candidates, [](const auto& item) {
                             return item.status == cache::CacheLookupStatus::Hit;
                         });
        (hit ? continuation_stats_.lookup_hits : continuation_stats_.lookup_misses)
            .fetch_add(1, std::memory_order_relaxed);
        const auto account_lookup = [&](cache::CacheSource source, std::uint64_t microseconds) {
            if (cache_lookup_accounting_tier(source) == CacheLookupAccountingTier::L3) {
                continuation_stats_.l3_lookup_microseconds.fetch_add(microseconds,
                                                                     std::memory_order_relaxed);
                continuation_stats_.l3_lookup_operations.fetch_add(1, std::memory_order_relaxed);
            } else if (cache_lookup_accounting_tier(source) == CacheLookupAccountingTier::L2) {
                continuation_stats_.l2_lookup_microseconds.fetch_add(microseconds,
                                                                     std::memory_order_relaxed);
                continuation_stats_.l2_lookup_operations.fetch_add(1, std::memory_order_relaxed);
            }
        };
        if (!include_history && candidate.id) {
            account_lookup(candidate.source, candidate.lookup_microseconds);
        }
        candidate.lookup_microseconds = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - lookup_started)
                .count());
        return candidate;
    }

    [[nodiscard]] std::optional<PendingSessionPublication>
    pending_session_publication(const std::string& session) {
        std::lock_guard lock(publication_mutex_);
        const auto pending = session_publications_.find(session);
        return pending == session_publications_.end()
                   ? std::nullopt
                   : std::optional<PendingSessionPublication>(PendingSessionPublication{
                         .sequence = pending->second.sequence,
                         .completed = publication_completed_ >= pending->second.sequence ||
                                      publication_stopping_});
    }

    [[nodiscard]] bool publication_completed(std::uint64_t sequence) {
        std::lock_guard lock(publication_mutex_);
        return publication_completed_ >= sequence || publication_stopping_;
    }

    PublicationTicket queue_publication(cache::ContinuationImage image, const std::string& session,
                                         std::optional<cache::ContentId> expected_head,
                                         std::optional<std::uint64_t> expected_generation =
                                             std::nullopt,
                                         bool immutable = false,
                                        std::optional<std::pair<std::string, std::uint64_t>>
                                            stable_flight = std::nullopt) noexcept {
        try {
            auto ticket =
                std::make_shared<std::atomic<PublicationStatus>>(PublicationStatus::Pending);
            std::lock_guard lock(publication_mutex_);
            if (publication_stopping_) {
                continuation_stats_.publication_failures.fetch_add(1, std::memory_order_relaxed);
                return {};
            }
            const std::uint64_t sequence = ++publication_issued_;
            publications_.push_back(Publication{.image         = std::move(image),
                                                 .session       = session,
                                                 .expected_head = std::move(expected_head),
                                                 .expected_generation = expected_generation,
                                                .ticket        = ticket,
                                                .sequence      = sequence,
                                                .immutable     = immutable,
                                                .stable_flight = std::move(stable_flight)});
            auto& pending = session_publications_[session];
            pending.sequence = sequence;
            publication_cv_.notify_one();
            return ticket;
        } catch (...) {
            continuation_stats_.publication_failures.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
    }

    void publication_loop() noexcept {
        for (;;) {
            Publication item;
            {
                std::unique_lock lock(publication_mutex_);
                publication_cv_.wait(
                    lock, [&] { return publication_stopping_ || !publications_.empty(); });
                if (publications_.empty()) {
                    if (publication_stopping_) { return; }
                    continue;
                }
                item = std::move(publications_.front());
                publications_.pop_front();
            }
            PublicationStatus status = PublicationStatus::Failed;
            try {
                const cache::StoreOptions options{
                    .recompute_cost = static_cast<double>(item.image.frontier_tokens),
                    .l2_idle_ttl    = publication_l2_ttl_,
                    .l3_idle_ttl    = publication_l3_ttl_,
                };
                if (item.immutable) {
                    const auto frontier_tokens = item.image.frontier_tokens;
                    const cache::ContentId id =
                        continuation_cache_->admit(std::move(item.image), options);
                    if (continuation_cache_->publish_immutable_alias(item.session, id)) {
                        status = PublicationStatus::Success;
                        (void)continuation_cache_->queue_persistence(
                            item.session, id, frontier_tokens, true);
                    }
                } else {
                    const auto frontier_tokens = item.image.frontier_tokens;
                    const auto result = continuation_cache_->publish_session_l2(
                        std::move(item.image), item.session, item.expected_head, options,
                        item.expected_generation);
                    if (result.alias_advanced) {
                        (void)continuation_cache_->queue_persistence(
                            item.session, result.id, frontier_tokens);
                    }
                    status = result.alias_advanced ? PublicationStatus::Success
                                                   : (result.stored ? PublicationStatus::Superseded
                                                                    : PublicationStatus::Failed);
                }
            } catch (...) {}
            if (status == PublicationStatus::Success) {
                continuation_stats_.publication_successes.fetch_add(1, std::memory_order_relaxed);
            } else if (status == PublicationStatus::Superseded) {
                continuation_stats_.publication_superseded.fetch_add(1,
                                                                      std::memory_order_relaxed);
            } else {
                continuation_stats_.publication_failures.fetch_add(1, std::memory_order_relaxed);
            }
            item.ticket->store(status, std::memory_order_release);
            {
                std::lock_guard lock(publication_mutex_);
                publication_completed_ = item.sequence;
            }
            publication_cv_.notify_all();
            queue_cv_.notify_all();
            if (item.stable_flight) {
                {
                    std::lock_guard lock(stable_flight_mutex_);
                    (void)stable_flights_.release(item.stable_flight->first,
                                                  item.stable_flight->second);
                }
                queue_cv_.notify_all();
            }
        }
    }

    void stop_publication_worker() noexcept {
        {
            std::lock_guard lock(publication_mutex_);
            publication_stopping_ = true;
        }
        publication_cv_.notify_all();
        if (publication_worker_.joinable()) { publication_worker_.join(); }
    }

    void require_idle_lane(std::uint32_t lane) const {
        if (lane >= max_concurrency_) {
            throw std::invalid_argument("slot id is outside the Engine lane count");
        }
        if (slots_[lane] != nullptr) {
            throw RequestError(RequestErrorKind::Overloaded, "slot is processing a request");
        }
    }

    void require_session_digest(std::uint32_t lane, std::string_view expected_digest) const {
        if (expected_digest.empty()) { return; }
        if (instance_.program->retained_lane_digest(lane) != expected_digest) {
            throw SlotSessionMismatch("slot session does not match if_digest");
        }
    }

    // Best-effort spill of a retained session about to be destroyed involuntarily. The device
    // snapshot runs on the calling thread (it synchronizes the stream); the file write happens
    // on the Engine's writer thread through the sink. Only sessions bound to a slot file are
    // spilled, and a spill failure never blocks the eviction itself.
    void spill_retained_lane(std::uint32_t lane) noexcept {
        if (!auto_save_evicted_ || !eviction_sink_ || lane >= kMaximumConcurrency) { return; }
        if (lane_session_path_[lane].empty() || !instance_.program->has_retained_lane(lane)) {
            return;
        }
        try {
            auto snapshot = instance_.program->save_retained_lane(lane, eviction_model_binding_);
            eviction_sink_(lane_session_path_[lane], std::move(snapshot));
        } catch (...) {
            // The session was going to be destroyed either way; losing the spill costs the
            // client one cold prefill, exactly the pre-feature behavior.
        }
    }

    void publish_runtime_stats() {
        RuntimeStats snapshot = cumulative_stats_;
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = prefill_lane_.has_value() ? 1U : 0U;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (l1_policy_active_ && instance_.program->has_retained_lane(lane)) {
                ++snapshot.l1_resident_entries;
                const std::size_t bytes = instance_.program->retained_lane_resident_bytes(lane);
                if (bytes <= std::numeric_limits<std::uint64_t>::max() -
                                 snapshot.l1_resident_bytes) {
                    snapshot.l1_resident_bytes += bytes;
                } else {
                    snapshot.l1_resident_bytes = std::numeric_limits<std::uint64_t>::max();
                }
            }
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->decode_ready) { ++snapshot.decode_ready_requests; }
        }

        // Per-lane occupancy for /slots-style readers. Digests come from the cache the
        // completion and restore paths maintain, so publishing costs no ledger hashing.
        std::vector<SlotState> slot_snapshot(max_concurrency_);
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            SlotState& state    = slot_snapshot[lane];
            const auto& request = slots_[lane];
            if (request != nullptr) {
                state.processing    = true;
                state.prompt_tokens = request->prompt_summary.prompt_tokens;
                if (request->begin) { state.cached_tokens = request->begin->reused_prompt_tokens; }
            } else if (instance_.program->has_retained_lane(lane)) {
                state.retained       = true;
                state.prompt_tokens  = instance_.program->retained_lane_depth(lane);
                state.cached_tokens  = state.prompt_tokens;
                state.session_digest = retained_digest_cache_[lane];
                state.checkpoints    = retained_checkpoints_cache_[lane];
            }
        }

        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
        published_slots_ = std::move(slot_snapshot);
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::vector<OutputDelta> events;
        for (;;) {
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done || !request->events.empty(); });
                events.swap(request->events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    struct Request {
        Request(std::uint64_t request_identity, targets::qwen3_8::PreparedPrompt input,
                targets::qwen3_8::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, ResolvedRequestOptions request_options,
                Clock::time_point limit, Clock::time_point submit_time, HostInputLease input_lease,
                 CachedContinuation routed_cached_continuation,
                 CachedContinuation stable_cached_continuation,
                 std::optional<std::string> shared_stable_alias,
                  std::optional<std::uint32_t> shared_stable_boundary,
                  std::optional<PendingSessionPublication> pending_publication)
            : id(request_identity), host_input(std::move(input_lease)), prompt(std::move(input)),
              output(std::move(output_session)), prompt_summary(summary),
              prepare_seconds(frontend_seconds), options(std::move(request_options)),
              deadline(limit), submitted(submit_time),
              routed_continuation(std::move(routed_cached_continuation)),
               stable_continuation(std::move(stable_cached_continuation)),
               stable_alias(std::move(shared_stable_alias)),
                stable_boundary(shared_stable_boundary),
                pending_session_publication(std::move(pending_publication)) {
            continuation.lookup_microseconds = routed_continuation.lookup_microseconds +
                                               stable_continuation.lookup_microseconds;
            if (!options.execution.allow_prefix_reuse) {
                continuation.final_miss_reason = ContinuationMissReason::Disabled;
            } else {
                continuation.alias_kind = options.routing_hint
                                              ? ContinuationAliasKind::Session
                                              : (stable_alias ? ContinuationAliasKind::StablePrefix
                                                              : ContinuationAliasKind::None);
                if (!options.routing_hint && !stable_alias) {
                    continuation.final_miss_reason = ContinuationMissReason::NoAlias;
                } else if (routed_continuation.status ==
                               cache::CacheLookupStatus::UnavailableOrCorrupt ||
                       stable_continuation.status == cache::CacheLookupStatus::UnavailableOrCorrupt) {
                    continuation.final_miss_reason =
                        ContinuationMissReason::EntryUnavailableOrCorrupt;
                } else {
                    continuation.final_miss_reason = ContinuationMissReason::NoAlias;
                }
            }
        }

        const std::uint64_t id;
        HostInputLease host_input;
        targets::qwen3_8::PreparedPrompt prompt;
        targets::qwen3_8::OutputSession output;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        Clock::time_point deadline;
        Clock::time_point submitted;
        std::optional<Clock::time_point> first_token;
        std::optional<GenerationBudget> budget;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        std::string content;
        std::string reasoning;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        bool decode_ready = false;
        CachedContinuation routed_continuation;
        CachedContinuation stable_continuation;
        std::optional<std::string> stable_alias;
        std::optional<std::uint32_t> stable_boundary;
        bool continuation_restore_attempted = false;
        std::optional<PendingSessionPublication> pending_session_publication;
        bool session_lookup_deferred = false;
        std::uint64_t continuation_preflight_operations = 0;
        std::uint64_t continuation_l2_restore_microseconds = 0;
        std::uint64_t continuation_l2_restore_operations = 0;
        std::uint64_t continuation_l3_restore_microseconds = 0;
        std::uint64_t continuation_l3_restore_operations = 0;
        bool stable_flight_builder           = false;
        bool stable_flight_resolved          = false;
        bool stable_publication_pending      = false;
        ContinuationDiagnostics continuation;

        std::optional<BasePlan> base_plan;
        std::array<std::optional<Plan>, kMaximumConcurrency> lane_plans{};
        std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions{};
        AdmissionResources admission_resources;
        std::uint64_t remaining_service_work = 0;
        std::uint64_t backfill_epoch         = 0;
        BackfillClass backfill_class         = BackfillClass::None;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
        RanGpuUnit,
    };

    struct LaneChoice {
        std::uint32_t lane  = 0;
        bool evict_retained = false;
    };

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_8::PublishedOutput output) {
        if (output.empty()) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                request->events.push_back(std::move(delta));
            }
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
        for (auto& plan : request->lane_plans) { plan.reset(); }
    }

    [[nodiscard]] bool stable_flight_needed(const Request& request) const noexcept {
        return request.stable_alias && !request.stable_continuation.image;
    }

    void initialize_stable_flight(const std::shared_ptr<Request>& request) {
        request->stable_flight_resolved = !stable_flight_needed(*request);
        if (request->stable_flight_resolved) { return; }
        std::lock_guard lock(stable_flight_mutex_);
        request->stable_flight_builder =
            stable_flights_.acquire(*request->stable_alias, request->id) ==
            StablePrefixFlights::AcquireResult::Builder;
    }

    void release_stable_builder(const std::shared_ptr<Request>& request) noexcept {
        if (!request || !request->stable_alias || !request->stable_flight_builder ||
            request->stable_publication_pending) {
            return;
        }
        {
            std::lock_guard lock(stable_flight_mutex_);
            (void)stable_flights_.release(*request->stable_alias, request->id);
        }
        request->stable_flight_builder = false;
        queue_cv_.notify_all();
    }

    void refresh_stable_flights() {
        const auto queued = pending_snapshot();
        for (const auto& request : queued) {
            if (request->stable_flight_resolved || request->stable_flight_builder ||
                !request->stable_alias) {
                continue;
            }

            bool became_builder = false;
            {
                std::lock_guard lock(stable_flight_mutex_);
                became_builder =
                    stable_flights_.acquire(*request->stable_alias, request->id) ==
                    StablePrefixFlights::AcquireResult::Builder;
            }
            if (!became_builder) { continue; }

            request->stable_flight_builder = true;
            CachedContinuation candidate = lookup_continuation(
                *request->stable_alias, ContinuationAliasKind::StablePrefix, false);
            request->continuation.lookup_microseconds += candidate.lookup_microseconds;
            if (!candidate.image) { continue; }

            request->stable_continuation            = std::move(candidate);
            request->continuation_restore_attempted = false;
            request->stable_flight_resolved         = true;
            release_stable_builder(request);
        }
    }

    [[nodiscard]] static bool stable_flight_blocked(const Request& request) noexcept {
        return !request.stable_flight_resolved && !request.stable_flight_builder;
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_stable_builder(request);
        release_planning_state(request);
        request->prompt = {};
        request->host_input.reset();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        release_stable_builder(request);
        release_planning_state(request);
        request->prompt = {};
        request->host_input.reset();
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
            result.slot        = static_cast<std::int32_t>(*request->lane);
            // Empty unless the lane retained the finished session (aborts and cancels clear it).
            result.session_digest = instance_.program->retained_lane_digest(*request->lane);
            // Completion and restore are the only paths that make a lane retained, so keeping
            // the cache here means publish_runtime_stats never has to hash a ledger.
            retained_digest_cache_[*request->lane] = result.session_digest;
            retained_checkpoints_cache_[*request->lane] =
                instance_.program->retained_lane_checkpoints(*request->lane);
        }
        if (request->continuation.source != ContinuationSource::None) {
            request->continuation.final_miss_reason = ContinuationMissReason::None;
            auto add_tier = [&](std::uint64_t& successes, std::uint64_t& tokens,
                                std::uint64_t& bytes) {
                ++successes;
                tokens += request->continuation.restored_tokens;
                bytes += request->continuation.restored_bytes;
            };
            switch (request->continuation.source) {
            case ContinuationSource::L1:
                add_tier(cumulative_stats_.continuation_l1_restore_successes,
                         cumulative_stats_.continuation_l1_restored_tokens,
                         cumulative_stats_.continuation_l1_restored_bytes);
                break;
            case ContinuationSource::L2:
                add_tier(cumulative_stats_.continuation_l2_restore_successes,
                         cumulative_stats_.continuation_l2_restored_tokens,
                         cumulative_stats_.continuation_l2_restored_bytes);
                break;
            case ContinuationSource::L3:
                add_tier(cumulative_stats_.continuation_l3_restore_successes,
                         cumulative_stats_.continuation_l3_restored_tokens,
                         cumulative_stats_.continuation_l3_restored_bytes);
                break;
            case ContinuationSource::None: break;
            }
            if (request->continuation.alias_kind == ContinuationAliasKind::Session) {
                ++cumulative_stats_.continuation_session_restores;
            } else if (request->continuation.alias_kind == ContinuationAliasKind::StablePrefix) {
                ++cumulative_stats_.continuation_stable_prefix_restores;
            }
        } else {
            switch (request->continuation.final_miss_reason) {
            case ContinuationMissReason::Disabled:
                ++cumulative_stats_.continuation_miss_disabled;
                break;
            case ContinuationMissReason::NoAlias:
                ++cumulative_stats_.continuation_miss_no_alias;
                break;
            case ContinuationMissReason::EntryUnavailableOrCorrupt:
                ++cumulative_stats_.continuation_miss_entry_unavailable_or_corrupt;
                break;
            case ContinuationMissReason::NotDeeper:
                ++cumulative_stats_.continuation_miss_not_deeper;
                break;
            case ContinuationMissReason::PreflightRejected:
                ++cumulative_stats_.continuation_miss_preflight_rejected;
                break;
            case ContinuationMissReason::RollbackConflict:
                ++cumulative_stats_.continuation_miss_rollback_conflict;
                break;
            case ContinuationMissReason::NoLane:
                ++cumulative_stats_.continuation_miss_no_lane;
                break;
            case ContinuationMissReason::RestoreFailed:
                ++cumulative_stats_.continuation_miss_restore_failed;
                break;
            case ContinuationMissReason::None: break;
            }
        }
        if (request->continuation_preflight_operations != 0) {
            cumulative_stats_.continuation_preflight_operations +=
                request->continuation_preflight_operations;
            cumulative_stats_.continuation_preflight_microseconds +=
                request->continuation.preflight_microseconds;
        }
        cumulative_stats_.continuation_l2_restore_operations +=
            request->continuation_l2_restore_operations;
        cumulative_stats_.continuation_l2_restore_microseconds +=
            request->continuation_l2_restore_microseconds;
        cumulative_stats_.continuation_l3_restore_operations +=
            request->continuation_l3_restore_operations;
        cumulative_stats_.continuation_l3_restore_microseconds +=
            request->continuation_l3_restore_microseconds;
        result.continuation = request->continuation;
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    void publish_retained_completion(const std::shared_ptr<Request>& request, std::uint32_t lane,
                                     FinishReason reason) noexcept {
        retained_last_used_[lane] = Clock::now();
        lane_provenance_[lane] = completion_publication_provenance(
            lane_provenance_[lane], request->options.routing_hint
                                        ? ContinuationAliasKind::Session
                                        : (request->stable_alias
                                               ? ContinuationAliasKind::StablePrefix
                                               : request->continuation.alias_kind));
        if (!continuation_cache_ || !request->options.execution.allow_prefix_reuse ||
            reason == FinishReason::Cancelled ||
            !request->options.routing_hint) {
            lane_sessions_[lane].reset();
            return;
        }
        if (request->session_lookup_deferred) {
            // The retained lane is newer than the alias that is still being published. Do not
            // block completion or enqueue a child with a stale CAS parent. A later turn can publish
            // the newest retained state after the predecessor becomes visible.
            lane_sessions_[lane].reset();
            return;
        }
        lane_sessions_[lane] = LaneSession{.name          = *request->options.routing_hint,
                                            .expected_head = request->routed_continuation.id,
                                            .expected_generation =
                                                request->routed_continuation.generation};
        try {
            auto image                        = instance_.program->export_continuation_lane(lane);
            image.parent_id                   = lane_sessions_[lane]->expected_head;
            lane_sessions_[lane]->publication = queue_publication(
                std::move(image), lane_sessions_[lane]->name, lane_sessions_[lane]->expected_head,
                lane_sessions_[lane]->expected_generation);
            request->continuation.completion_publication_queued =
                static_cast<bool>(lane_sessions_[lane]->publication);
        } catch (...) {
            // A continuation is an optimization; generation has already completed successfully.
        }
    }

    // Every involuntary loss of a retained session funnels through here: L1 retention pressure,
    // a continuation restore taking the lane, and a FullReset admission over it. The session is
    // offered to both retention tiers - the continuation cache (L2/L3) and, when the lane is
    // bound to a slot file, that file - and the lane always ends up unbound so a later eviction
    // can never write a new session over a previous session's file. Explicit erase is a deletion
    // request and does not come through here.
    void evict_retained_lane(std::uint32_t lane) noexcept {
        spill_retained_lane(lane);
        if (lane < kMaximumConcurrency) { lane_session_path_[lane].clear(); }
        if (!instance_.program->has_retained_lane(lane)) {
            lane_sessions_[lane].reset();
            lane_provenance_[lane] = {};
            retained_last_used_[lane].reset();
            return;
        }
        bool demoted = false;
        if (continuation_cache_ && lane_sessions_[lane]) {
            const PublicationStatus status = lane_sessions_[lane]->publication
                                                 ? lane_sessions_[lane]->publication->load(
                                                       std::memory_order_acquire)
                                                 : PublicationStatus::Failed;
            if (status == PublicationStatus::Failed) {
                try {
                    auto image      = instance_.program->export_continuation_lane(lane);
                    image.parent_id = lane_sessions_[lane]->expected_head;
                    lane_sessions_[lane]->publication =
                        queue_publication(std::move(image), lane_sessions_[lane]->name,
                                          lane_sessions_[lane]->expected_head,
                                          lane_sessions_[lane]->expected_generation);
                } catch (...) {}
            }
            demoted = static_cast<bool>(lane_sessions_[lane]->publication);
        }
        instance_.program->evict_retained_lane(lane);
        lane_sessions_[lane].reset();
        lane_provenance_[lane] = {};
        retained_last_used_[lane].reset();
        invalidate_lane_plans(lane);
        if (l1_policy_active_) {
            ++cumulative_stats_.l1_evictions;
            if (demoted) { ++cumulative_stats_.l1_demotions; }
        }
    }

    [[nodiscard]] bool enforce_l1_retention() noexcept {
        if (!l1_policy_active_) { return false; }
        bool changed = false;
        try {
            for (;;) {
                const Clock::time_point now = Clock::now();
                std::array<L1RetentionEntry, kMaximumConcurrency> entries{};
                for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                    const bool resident = instance_.program->has_retained_lane(lane);
                    if (resident && !retained_last_used_[lane]) { retained_last_used_[lane] = now; }
                    if (!resident && slots_[lane] == nullptr) { retained_last_used_[lane].reset(); }
                    entries[lane] = L1RetentionEntry{
                        .lane = lane,
                        .resident_bytes = resident
                                              ? instance_.program->retained_lane_resident_bytes(lane)
                                              : 0,
                        .last_used = retained_last_used_[lane].value_or(now),
                        .resident = resident,
                        .active = slots_[lane] != nullptr,
                    };
                }
                const auto victim = select_l1_retention_victim(
                    std::span<const L1RetentionEntry>(entries.data(), max_concurrency_),
                    l1_byte_budget_, l1_idle_ttl_, now);
                if (!victim) { break; }
                evict_retained_lane(*victim);
                changed = true;
            }
        } catch (...) {
            // Retention is an optimization and must never fail active execution or admission.
        }
        return changed;
    }

    // Resolves the single token a prefill licenses. Returns the finish reason when the request
    // ended here; the caller releases the lane and republishes before waking the waiter, so a
    // client never observes its own completion ahead of the slot state that produced it.
    std::optional<FinishReason> resolve_round(const std::shared_ptr<Request>& request,
                                              TokenId token, bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->abort_lane(lane);
            lane_sessions_[lane].reset();
            append_output(request, request->output.commit_preview());
            return FinishReason::Cancelled;
        }

        const std::span<const TokenId> tokens(&token, 1);
        const OutputDecision decision = request->output.preview(
            tokens, request->budget->remaining(), request->budget->limit_reason());
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("prefill output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        request->budget->commit(1);
        auto published = request->output.commit_preview();
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published));
        if (decision.finished()) {
            publish_retained_completion(request, lane, decision.finish_reason);
            return decision.finish_reason;
        }
        return std::nullopt;
    }

    void invalidate_lane_plans(std::uint32_t lane) noexcept { ++lane_plan_versions_[lane]; }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        invalidate_lane_plans(lane);
    }

    void consume_service_work(const std::shared_ptr<Request>& request, std::uint64_t work) {
        if (work == 0 || work > request->remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request->remaining_service_work) + " remaining");
        }
        request->remaining_service_work -= work;
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    void
    cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            instance_.program->abort_lane(lane);
            lane_sessions_[lane].reset();
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            complete_cancelled(request);
            remove_completed_slot(lane);
            changed = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        if (protection_) {
            const auto removed_protected = [&](const std::shared_ptr<Request>& request) {
                return request->id == protection_->head_request_id;
            };
            if (std::any_of(cancelled.begin(), cancelled.end(), removed_protected) ||
                std::any_of(expired.begin(), expired.end(), removed_protected)) {
                protection_.reset();
            }
        }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
        if (!cancelled.empty() || !expired.empty()) { publish_runtime_stats(); }
        return have_pending;
    }

    [[nodiscard]] RoundMembership build_round_membership() const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->decode_ready) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            membership.lanes[membership.size]   = lane;
            membership.budgets[membership.size] = request->budget->round_budget();
            ++membership.size;
        }
        return membership;
    }

    [[nodiscard]] ActiveAdmissionSet active_admission_set() const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr) { continue; }
            if (request->admission_resources.active_lanes == 0 ||
                request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id            = request->id,
                .resources             = request->admission_resources,
                .remaining_work_quanta = request->remaining_service_work,
                .backfill_epoch        = request->backfill_epoch,
                .backfill_class        = request->backfill_class,
            };
        }
        return active;
    }

    void resolve_prefill_step(const std::shared_ptr<Request>& request,
                              const PrefillStepResult& step, bool cancel_at_boundary) {
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        consume_service_work(request, 1);
        if (step.host_input_consumed || step.complete) { request->host_input.reset(); }
        if (continuation_cache_ && request->options.execution.allow_prefix_reuse && request->lane &&
            request->stable_alias &&
            request->stable_flight_builder && !request->stable_publication_pending) {
            try {
                auto stable = instance_.program->take_stable_continuation_lane(*request->lane);
                if (stable) {
                    auto ticket = queue_publication(
                        std::move(*stable), *request->stable_alias, std::nullopt, std::nullopt, true,
                        std::pair<std::string, std::uint64_t>{*request->stable_alias, request->id});
                    if (ticket) {
                        request->stable_publication_pending = true;
                        request->continuation.completion_publication_queued = true;
                    } else {
                        release_stable_builder(request);
                    }
                }
            } catch (...) { release_stable_builder(request); }
        }
        if (step.complete && request->stable_flight_builder &&
            !request->stable_publication_pending) {
            release_stable_builder(request);
        }
        if (cancel_at_boundary) {
            if (!request->lane) { throw std::logic_error("cancelled prefill has no request lane"); }
            const std::uint32_t lane = *request->lane;
            if (prefill_lane_ && lane == *prefill_lane_) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            instance_.program->abort_lane(lane);
            lane_sessions_[lane].reset();
            complete_cancelled(request);
            remove_completed_slot(lane);
            return;
        }
        if (!step.complete) { return; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        if (prefill_lane_ && *request->lane == *prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        request->begin = step.summary;
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }
        if (const std::optional<FinishReason> reason =
                resolve_round(request, step.round.tokens.front(), false)) {
            remove_completed_slot(*request->lane);
            (void)enforce_l1_retention();
            publish_runtime_stats();
            complete_success(request, *reason);
        } else {
            request->decode_ready = true;
        }
    }

    void run_prefill_step() {
        if (!prefill_lane_) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane_;
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const auto unit_started       = Clock::now();
        const PrefillStepResult step  = instance_.program->advance_prefill_lane(lane);
        cumulative_stats_.prefill_seconds_total +=
            std::chrono::duration<double>(Clock::now() - unit_started).count();
        const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
        resolve_prefill_step(request, step, cancel_at_boundary);
        publish_runtime_stats();
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    void clear_protection_if_head(const std::shared_ptr<Request>& request) noexcept {
        if (protection_ && protection_->head_request_id == request->id) { protection_.reset(); }
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request_base(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.admission.active_lanes != 1 || summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    void ensure_lane_plan(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (slots_[lane] != nullptr) { return; }
        if (request->lane_plan_versions[lane] == lane_plan_versions_[lane] &&
            request->lane_plans[lane]) {
            return;
        }
        request->lane_plans[lane].reset();
        request->lane_plans[lane].emplace(
            instance_.program->plan_request_for_lane(lane, request->prompt, *request->base_plan));
        request->lane_plan_versions[lane] = lane_plan_versions_[lane];
    }

    void try_restore_continuation(const std::shared_ptr<Request>& request) noexcept {
        if (request->continuation_restore_attempted) { return; }

        try {
            std::optional<std::uint32_t> target_lane;
            std::uint32_t target_reuse        = std::numeric_limits<std::uint32_t>::max();
            std::uint32_t best_resident_reuse = 0;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) { continue; }
                ensure_lane_plan(request, lane);
                const std::uint32_t reuse =
                    request->lane_plans[lane]->summary().reusable_prompt_tokens;
                best_resident_reuse = std::max(best_resident_reuse, reuse);
                // Prefer an empty lane, otherwise replace the retained lane least useful to this
                // request. This is what makes two sessions switch through a single active lane.
                const bool empty = !instance_.program->has_retained_lane(lane);
                const bool target_empty =
                    target_lane && !instance_.program->has_retained_lane(*target_lane);
                if (!target_lane || (empty && !target_empty) ||
                    (empty == target_empty && reuse < target_reuse)) {
                    target_lane  = lane;
                    target_reuse = reuse;
                }
            }
            if (!target_lane) {
                // Lane pressure is transient. Leave the candidate and preflight state untouched so
                // restoration is retried when admission next observes an idle lane.
                return;
            }

            if (request->pending_session_publication) {
                if (publication_completed(request->pending_session_publication->sequence)) {
                    if (request->options.routing_hint) {
                        CachedContinuation refreshed = lookup_continuation(
                            *request->options.routing_hint, ContinuationAliasKind::Session, false,
                            true);
                        request->continuation.lookup_microseconds +=
                            refreshed.lookup_microseconds;
                        request->routed_continuation = std::move(refreshed);
                    }
                    request->pending_session_publication.reset();
                } else {
                    // Use the deepest already-visible L1/L2/L3 checkpoint instead of waiting for
                    // publication. Completion skips its own publication so the pending predecessor
                    // remains the only writer that can advance this session generation.
                    request->session_lookup_deferred = true;
                    request->pending_session_publication.reset();
                }
            }

            if (!request->routed_continuation.image &&
                request->routed_continuation.candidates.empty() &&
                !request->stable_continuation.image) {
                request->continuation_restore_attempted = true;
                return;
            }

            const std::uint32_t lane = *target_lane;
            const auto preflight_depth = [&](const cache::ContinuationImage& image,
                                             ContinuationAliasKind alias_kind) {
                const auto started = Clock::now();
                ++request->continuation_preflight_operations;
                const std::uint32_t depth =
                    instance_.program->preflight_continuation(image, request->prompt);
                request->continuation.preflight_microseconds += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started)
                        .count());
                if (depth == 0) {
                    continuation_stats_.preflight_rejections.fetch_add(1,
                                                                        std::memory_order_relaxed);
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::PreflightRejected,
                                              alias_kind);
                }
                return depth;
            };
            const auto metadata_preflight_depth = [&](
                const cache::SessionCandidateDescriptor& item) {
                const auto started = Clock::now();
                ++request->continuation_preflight_operations;
                const std::uint32_t depth =
                    instance_.program->preflight_continuation_metadata(item, request->prompt);
                request->continuation.preflight_microseconds += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started)
                        .count());
                if (depth == 0 && item.status == cache::CacheLookupStatus::Hit) {
                    continuation_stats_.preflight_rejections.fetch_add(1,
                                                                        std::memory_order_relaxed);
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::PreflightRejected,
                                              ContinuationAliasKind::Session);
                }
                return depth;
            };
            const auto try_candidate = [&](CachedContinuation& candidate,
                                           std::uint32_t reusable_depth = 0) {
                if (!candidate.image) {
                    if (candidate.status == cache::CacheLookupStatus::UnavailableOrCorrupt) {
                        observe_continuation_miss(
                            request->continuation,
                            ContinuationMissReason::EntryUnavailableOrCorrupt,
                            candidate.alias_kind);
                    }
                    return false;
                }
                if (candidate.image->frontier_tokens <= best_resident_reuse) {
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::NotDeeper,
                                              candidate.alias_kind);
                    candidate.image.reset();
                    return false;
                }
                if (reusable_depth == 0) {
                    reusable_depth = preflight_depth(*candidate.image, candidate.alias_kind);
                    if (reusable_depth == 0) {
                        candidate.image.reset();
                        return false;
                    }
                }
                if (reusable_depth <= best_resident_reuse) {
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::NotDeeper,
                                              candidate.alias_kind);
                    candidate.image.reset();
                    return false;
                }
                if (instance_.program->has_retained_lane(lane)) { evict_retained_lane(lane); }
                const std::uint64_t restored_bytes =
                    cache::continuation_image_bytes(*candidate.image);
                const auto restore_started = Clock::now();
                const bool restored = instance_.program->import_continuation_lane(
                    lane, *candidate.image, request->prompt, request->options.execution.adapter);
                const std::uint64_t restore_microseconds = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                                          restore_started)
                        .count());
                request->continuation.restore_microseconds += restore_microseconds;
                if (candidate.source == cache::CacheSource::L3) {
                    ++request->continuation_l3_restore_operations;
                    request->continuation_l3_restore_microseconds += restore_microseconds;
                } else {
                    ++request->continuation_l2_restore_operations;
                    request->continuation_l2_restore_microseconds += restore_microseconds;
                }
                if (restored) {
                    retained_last_used_[lane] = Clock::now();
                    lane_provenance_[lane] =
                        imported_lane_provenance(candidate.alias_kind, candidate.source);
                    continuation_stats_.restore_successes.fetch_add(1, std::memory_order_relaxed);
                    continuation_stats_.restored_tokens.fetch_add(reusable_depth,
                                                                   std::memory_order_relaxed);
                    continuation_stats_.restored_bytes.fetch_add(
                        restored_bytes, std::memory_order_relaxed);
                    request->continuation.source = candidate.source == cache::CacheSource::L3
                                                       ? ContinuationSource::L3
                                                       : ContinuationSource::L2;
                    request->continuation.alias_kind = candidate.alias_kind;
                    request->continuation.restored_tokens = reusable_depth;
                    request->continuation.restored_bytes = restored_bytes;
                    request->continuation.final_miss_reason = ContinuationMissReason::None;
                } else {
                    continuation_stats_.restore_failures.fetch_add(1, std::memory_order_relaxed);
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::RestoreFailed,
                                              candidate.alias_kind);
                }
                candidate.image.reset();
                invalidate_lane_plans(lane);
                return restored;
            };
            request->continuation_restore_attempted = true;
            bool routed_ready = false;
            std::uint32_t routed_reusable_depth = 0;
            std::uint32_t stable_reusable_depth = 0;
            if (request->stable_continuation.image &&
                request->stable_continuation.image->frontier_tokens > best_resident_reuse) {
                stable_reusable_depth = preflight_depth(*request->stable_continuation.image,
                                                        ContinuationAliasKind::StablePrefix);
            }
            if (!request->routed_continuation.candidates.empty()) {
                bool unavailable = false;
                bool available = false;
                for (const auto& item : request->routed_continuation.candidates) {
                    unavailable |=
                        item.status == cache::CacheLookupStatus::UnavailableOrCorrupt;
                    available |= item.status == cache::CacheLookupStatus::Hit;
                }
                const auto viable = rank_reusable_candidate_descriptors(
                    std::span<const cache::SessionCandidateDescriptor>(
                        request->routed_continuation.candidates),
                    best_resident_reuse, stable_reusable_depth,
                    [&](const auto& item) {
                        return metadata_preflight_depth(item);
                    });

                std::optional<std::size_t> selected_index;
                std::uint32_t selected_depth = 0;
                bool resolved_image = false;
                for (const auto& viable_item : viable) {
                    if (selected_index && viable_item.reusable_depth <= selected_depth) break;
                    const auto& descriptor =
                        request->routed_continuation.candidates[viable_item.index];
                    auto lookup = continuation_cache_->resolve_candidate(descriptor);
                    request->continuation.lookup_microseconds += lookup.io_microseconds;
                    if (cache_lookup_accounting_tier(lookup.source) ==
                        CacheLookupAccountingTier::L3) {
                        continuation_stats_.l3_lookup_microseconds.fetch_add(
                            lookup.io_microseconds, std::memory_order_relaxed);
                        continuation_stats_.l3_lookup_operations.fetch_add(
                            1, std::memory_order_relaxed);
                    } else if (cache_lookup_accounting_tier(lookup.source) ==
                               CacheLookupAccountingTier::L2) {
                        continuation_stats_.l2_lookup_microseconds.fetch_add(
                            lookup.io_microseconds, std::memory_order_relaxed);
                        continuation_stats_.l2_lookup_operations.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    if (!lookup.image) {
                        unavailable |=
                            lookup.status == cache::CacheLookupStatus::UnavailableOrCorrupt;
                        continue;
                    }
                    resolved_image = true;
                    const std::uint32_t exact =
                        preflight_depth(*lookup.image, ContinuationAliasKind::Session);
                    if (exact <= best_resident_reuse || exact < stable_reusable_depth ||
                        (selected_index && exact <= selected_depth)) {
                        continue;
                    }
                    selected_index = viable_item.index;
                    selected_depth = exact;
                    request->routed_continuation.image = std::move(lookup.image);
                    request->routed_continuation.source = lookup.source;
                    request->routed_continuation.status = lookup.status;
                }
                if (selected_index && prefer_routed_candidate(selected_depth,
                                                               stable_reusable_depth)) {
                    const auto& chosen =
                        request->routed_continuation.candidates[*selected_index];
                    if (*selected_index == 0) {
                        routed_ready = true;
                    } else if (request->options.routing_hint &&
                               request->routed_continuation.generation) {
                        const auto rollback = continuation_cache_->rollback_session_to(
                            *request->options.routing_hint, chosen.id,
                            *request->routed_continuation.generation);
                        if (rollback.rolled_back) {
                            request->routed_continuation.generation = rollback.generation;
                            request->continuation.destructive_rollback = true;
                            routed_ready = true;
                        } else {
                            observe_continuation_miss(request->continuation,
                                                      ContinuationMissReason::RollbackConflict,
                                                      ContinuationAliasKind::Session);
                        }
                    }
                    if (routed_ready) {
                        routed_reusable_depth = selected_depth;
                        request->routed_continuation.id    = chosen.id;
                    }
                } else {
                    if (unavailable && (!available || (!viable.empty() && !resolved_image))) {
                        observe_continuation_miss(
                            request->continuation,
                            ContinuationMissReason::EntryUnavailableOrCorrupt,
                            ContinuationAliasKind::Session);
                    } else if (available) {
                        observe_continuation_miss(request->continuation,
                                                  ContinuationMissReason::NotDeeper,
                                                  ContinuationAliasKind::Session);
                    }
                }
                request->routed_continuation.candidates.clear();
            } else {
                routed_ready = static_cast<bool>(request->routed_continuation.image);
                if (routed_ready) {
                    routed_reusable_depth = preflight_depth(
                        *request->routed_continuation.image, ContinuationAliasKind::Session);
                    routed_ready = prefer_routed_candidate(routed_reusable_depth,
                                                           stable_reusable_depth) &&
                                   routed_reusable_depth > best_resident_reuse;
                }
            }
            if (!routed_ready ||
                !try_candidate(request->routed_continuation, routed_reusable_depth)) {
                if (stable_reusable_depth > best_resident_reuse) {
                    (void)try_candidate(request->stable_continuation, stable_reusable_depth);
                }
            }
            request->routed_continuation.image.reset();
            request->routed_continuation.candidates.clear();
            request->stable_continuation.image.reset();
            return;
        } catch (...) {
            request->continuation_restore_attempted = true;
            continuation_stats_.restore_failures.fetch_add(1, std::memory_order_relaxed);
            observe_continuation_miss(request->continuation,
                                      ContinuationMissReason::RestoreFailed);
            request->routed_continuation.image.reset();
            request->routed_continuation.candidates.clear();
            request->stable_continuation.image.reset();
            return;
        }
    }

    // Lane choice maximizes reusable prefix; ties break toward the lane whose occupation costs
    // least to replace - an empty lane before any retained session, then the shallowest
    // retained session - so a fresh request never clobbers a deep resident session while a
    // cheaper lane is available.
    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request) {
        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        std::uint32_t selected_cost  = 0;
        const auto prefer            = [&](std::uint32_t reuse, std::uint32_t cost) {
            return !selected || reuse > selected_reuse ||
                   (reuse == selected_reuse && cost < selected_cost);
        };
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            const std::uint32_t cost  = instance_.program->retained_lane_depth(lane);
            if (instance_.program->can_admit_lane(lane, plan) && prefer(reuse, cost)) {
                selected       = LaneChoice{.lane = lane};
                selected_reuse = reuse;
                selected_cost  = cost;
            }
        }
        if (selected) { return selected; }

        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            const std::uint32_t cost  = instance_.program->retained_lane_depth(lane);
            if (instance_.program->can_admit_lane_after_retained_eviction(lane, plan) &&
                prefer(reuse, cost)) {
                selected = LaneChoice{
                    .lane           = lane,
                    .evict_retained = true,
                };
                selected_reuse = reuse;
                selected_cost  = cost;
            }
        }
        return selected;
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        clear_protection_if_head(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (Clock::now() >= request->deadline) {
            return remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            clear_protection_if_head(request);
            complete_cancelled(request);
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const std::uint32_t lane = choice.lane;
        if (!request->lane_plans[lane]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        if (choice.evict_retained) {
            for (std::uint32_t retained_lane = 0;
                 retained_lane < max_concurrency_ &&
                 !instance_.program->can_admit_lane(lane, *request->lane_plans[lane]);
                 ++retained_lane) {
                if (retained_lane != lane && slots_[retained_lane] == nullptr &&
                    instance_.program->has_retained_lane(retained_lane)) {
                    evict_retained_lane(retained_lane);
                }
            }
            if (!instance_.program->can_admit_lane(lane, *request->lane_plans[lane])) {
                throw std::logic_error("retained eviction did not make admission feasible");
            }
        }

        Plan selected_plan = std::move(*request->lane_plans[lane]);
        request->lane_plans[lane].reset();
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        release_planning_state(request);

        const RequestPlanSummary summary = selected_plan.summary();
        if (backfill_class == BackfillClass::Temporal) {
            if (!protection_ || protection_->epoch_id != backfill_epoch ||
                summary.service_work_quanta > protection_->temporal_credit) {
                throw std::logic_error("temporal backfill lost its protected credit");
            }
            protection_->temporal_credit -= summary.service_work_quanta;
        }
        clear_protection_if_head(request);

        refresh_lane_provenance(lane);
        classify_resident_continuation(
            request->continuation, summary.reusable_prompt_tokens,
            instance_.program->retained_lane_reused_bytes(lane, selected_plan),
            lane_provenance_[lane]);
        // Zero reuse means the target takes the FullReset path and destroys whatever session
        // the lane retained. Spill it first, and start the new session unbound either way so a
        // later eviction can never write it over the previous session's file.
        if (summary.reusable_prompt_tokens == 0) {
            spill_retained_lane(lane);
            lane_session_path_[lane].clear();
        }

        const bool needs_prefill = summary.reusable_prompt_tokens < summary.prompt_tokens;
        bool target_started      = false;
        try {
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
            request->lane                   = lane;
            request->admission_resources    = summary.admission;
            request->remaining_service_work = summary.service_work_quanta;
            request->backfill_epoch         = backfill_epoch;
            request->backfill_class         = backfill_class;
            slots_[lane]                    = request;
            retained_last_used_[lane]       = Clock::now();
            invalidate_lane_plans(lane);

            TransientRegion transient;
            if (needs_prefill) {
                instance_.request_memory.activate(summary.transient_bytes,
                                                  summary.transient_alignment);
                prefill_lane_ = lane;
                transient     = instance_.request_memory.region();
            }
            publish_runtime_stats();
            target_started                = true;
            const auto unit_started       = Clock::now();
            const PrefillStepResult first = instance_.program->start_prefill_lane(
                lane, std::move(request->prompt), std::move(selected_plan), transient);
            cumulative_stats_.prefill_seconds_total +=
                std::chrono::duration<double>(Clock::now() - unit_started).count();
            if (!first.complete && (!prefill_lane_ || *prefill_lane_ != lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            resolve_prefill_step(request, first, cancel_at_boundary);
            publish_runtime_stats();
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (target_started) { instance_.program->abort_lane(lane); }
            lane_sessions_[lane].reset();
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            slots_[lane].reset();
            invalidate_lane_plans(lane);
            complete_error(request, error);
            throw;
        }
        return AdmissionProgress::RanGpuUnit;
    }

    AdmissionProgress try_admit_one() {
        bool control_progress = false;
        for (;;) {
            std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
            std::erase_if(queued, [](const std::shared_ptr<Request>& request) {
                return stable_flight_blocked(*request);
            });
            if (queued.empty()) {
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.front();
            if (protection_ && protection_->head_request_id != head->id) { protection_.reset(); }
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    clear_protection_if_head(head);
                    complete_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            try_restore_continuation(head);
            const RequestPlanSummary& head_base = head->base_plan->summary();
            if (!admission_resources_fit(head_base.admission, admission_capacity_)) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }

            std::optional<LaneChoice> head_lane;
            try {
                head_lane = find_admission_lane(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_lane) {
                return admit_planned_request(head, *head_lane, BackfillClass::None, 0);
            }

            const ActiveAdmissionSet active = active_admission_set();
            if (active.size == 0) {
                throw std::logic_error("exclusive-feasible request cannot enter an idle Engine");
            }
            if (!protection_) {
                protection_.emplace(make_admission_protection(next_protection_epoch_++, head->id,
                                                              head_base.admission, active.span(),
                                                              admission_capacity_));
            }
            if (protected_head_safe_without_temporal(*protection_, active.span(),
                                                     admission_capacity_)) {
                protection_->phase = ProtectionPhase::Drain;
            }
            if (protection_->phase == ProtectionPhase::Drain) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const std::uint64_t frontier_distance =
                protection_frontier_distance(*protection_, active.span());
            for (std::size_t i = 1; i < queued.size(); ++i) {
                const std::shared_ptr<Request>& candidate = queued[i];
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        complete_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }

                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                try_restore_continuation(candidate);
                const RequestPlanSummary& candidate_base = candidate->base_plan->summary();
                if (!admission_resources_fit(candidate_base.admission, admission_capacity_)) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }

                std::optional<LaneChoice> candidate_lane;
                try {
                    candidate_lane = find_admission_lane(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!candidate_lane) { continue; }
                const RequestPlanSummary& candidate_plan =
                    candidate->lane_plans[candidate_lane->lane]->summary();

                BackfillClass backfill = BackfillClass::None;
                if (persistent_backfill_is_safe(*protection_, active.span(),
                                                candidate_plan.admission, admission_capacity_)) {
                    backfill = BackfillClass::Persistent;
                } else if (candidate_plan.service_work_quanta <= frontier_distance &&
                           candidate_plan.service_work_quanta <= protection_->temporal_credit) {
                    backfill = BackfillClass::Temporal;
                }
                if (backfill != BackfillClass::None) {
                    return admit_planned_request(candidate, *candidate_lane, backfill,
                                                 protection_->epoch_id);
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    void run_decode_round(const RoundMembership& membership) {
        const std::span<const std::uint32_t> lanes = membership.lane_span();
        const auto unit_started                    = Clock::now();
        const BatchedGeneratedRound round =
            instance_.program->decode_batch(lanes, membership.budget_span());
        cumulative_stats_.decode_seconds_total +=
            std::chrono::duration<double>(Clock::now() - unit_started).count();

        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            cancelled[row] =
                slots_[lanes[row]]->cancelled.load(std::memory_order_acquire) ? 1U : 0U;
        }

        if (round.row_stride == 0 ||
            (!round.row_counts.empty() && round.row_counts.size() != lanes.size()) ||
            round.tokens.size() < static_cast<std::size_t>(round.row_stride) * lanes.size()) {
            throw std::logic_error("decode batch returned an invalid ragged layout");
        }

        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        // Terminal requests of this round, kept alive past their slot release so they can be
        // woken after the published lane table already reflects their completion.
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> finished{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            const std::uint32_t count =
                round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[row]);
            if (count == 0 || count > round.row_stride) {
                throw std::logic_error("decode batch returned an invalid licensed row extent");
            }
            const auto row_tokens =
                round.tokens.subspan(row * round.row_stride, static_cast<std::size_t>(count));
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            const OutputDecision decision = request->output.preview(
                row_tokens, request->budget->remaining(), request->budget->limit_reason());
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                (!decision.finished() && decision.accepted_tokens != count)) {
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            accepted[row]       = decision.accepted_tokens;
            terminal[row]       = decision.finished() ? 1 : 0;
            finish_reasons[row] = decision.finish_reason;
        }

        instance_.program->resolve_pending_batch(
            lanes, std::span<const std::uint32_t>(accepted.data(), lanes.size()),
            std::span<const std::uint8_t>(terminal.data(), lanes.size()),
            std::span<const std::uint8_t>(cancelled.data(), lanes.size()));

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            if (!cancelled[row]) {
                const auto row_tokens = round.tokens.subspan(
                    row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                request->generated.insert(request->generated.end(), row_tokens.begin(),
                                          row_tokens.end());
                request->budget->commit(accepted[row]);
                consume_service_work(request, accepted[row]);
            }
            auto published = request->output.commit_preview();
            if (!request->first_token && accepted[row] != 0) {
                request->first_token = Clock::now();
            }
            append_output(request, std::move(published));
            if (terminal[row]) {
                // The waiter is not woken here: a client that observes its own completion must
                // also be able to observe the slot state that completion produced, so every
                // terminal lane in this round is released and republished first.
                publish_retained_completion(request, lane, finish_reasons[row]);
                finished[row] = request;
                remove_completed_slot(lane);
            }
        }
        (void)enforce_l1_retention();
        ++cumulative_stats_.decode_rounds;
        cumulative_stats_.decode_row_rounds += lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!cancelled[row]) { cumulative_stats_.committed_decode_tokens += accepted[row]; }
        }
        publish_runtime_stats();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (finished[row]) { complete_success(finished[row], finish_reasons[row]); }
        }
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        if (prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                lane_sessions_[lane].reset();
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    void worker_loop() noexcept {
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        if (l1_policy_active_ &&
                            l1_idle_ttl_ > Clock::duration::zero()) {
                            const auto now = Clock::now();
                            std::optional<Clock::duration> until_expiry;
                            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                                if (!retained_last_used_[lane]) { continue; }
                                const auto elapsed = now >= *retained_last_used_[lane]
                                                         ? now - *retained_last_used_[lane]
                                                         : Clock::duration::zero();
                                const auto remaining = elapsed >= l1_idle_ttl_
                                                           ? Clock::duration::zero()
                                                           : l1_idle_ttl_ - elapsed;
                                if (!until_expiry || remaining < *until_expiry) {
                                    until_expiry = remaining;
                                }
                            }
                            if (until_expiry) {
                                queue_cv_.wait_for(lock, *until_expiry,
                                                   [&] { return stopping_ || !pending_.empty(); });
                            } else {
                                queue_cv_.wait(lock,
                                               [&] { return stopping_ || !pending_.empty(); });
                            }
                        } else {
                            queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                        }
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                refresh_stable_flights();
                std::unique_lock execution_lock(execution_mutex_);
                if (enforce_l1_retention()) { publish_runtime_stats(); }
                const bool have_pending          = expire_pending_requests();
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary);
                const RoundMembership membership = build_round_membership();

                if (prefill_lane_) {
                    if (!membership.empty() && !previous_unit_was_decode) {
                        run_decode_round(membership);
                        previous_unit_was_decode = true;
                    } else {
                        run_prefill_step();
                        previous_unit_was_decode = false;
                    }
                    continue;
                }

                if (have_pending && (membership.empty() || previous_unit_was_decode)) {
                    const AdmissionProgress progress = try_admit_one();
                    if (progress == AdmissionProgress::RanGpuUnit) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                    if (progress == AdmissionProgress::ControlProgress && membership.empty()) {
                        continue;
                    }
                }

                if (!membership.empty()) {
                    run_decode_round(membership);
                    previous_unit_was_decode = true;
                    continue;
                }
                execution_lock.unlock();
                std::unique_lock queue_lock(queue_mutex_);
                queue_cv_.wait_for(queue_lock, std::chrono::milliseconds(10));
            } catch (...) {
                fail_all(std::current_exception());
                return;
            }
        }
    }

    Instance& instance_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const bool auto_save_evicted_;
    const AdmissionResources admission_capacity_;
    std::unique_ptr<cache::ContinuationCache> continuation_cache_;
    // Off preserves historical unmanaged same-lane prefix reuse. Every active tier applies this
    // policy; zero capacity or zero TTL therefore retains no inactive GPU continuation.
    const bool l1_policy_active_;
    const std::size_t l1_byte_budget_;
    const Clock::duration l1_idle_ttl_;
    const std::chrono::seconds publication_l2_ttl_;
    const std::chrono::seconds publication_l3_ttl_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    mutable std::mutex stable_flight_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_       = 0;
    std::uint64_t next_request_id_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<std::uint32_t> prefill_lane_;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::array<std::optional<LaneSession>, kMaximumConcurrency> lane_sessions_{};
    std::array<LaneContinuationProvenance, kMaximumConcurrency> lane_provenance_{};
    std::array<std::optional<Clock::time_point>, kMaximumConcurrency> retained_last_used_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    AtomicContinuationStats continuation_stats_;
    StablePrefixFlights stable_flights_;
    std::vector<SlotState> published_slots_;
    // Digest of each lane's retained session, maintained by the completion and restore paths
    // (the only ones that set `retained`) so publishing needs no ledger hashing.
    std::array<std::string, kMaximumConcurrency> retained_digest_cache_{};
    std::array<std::vector<SlotCheckpoint>, kMaximumConcurrency> retained_checkpoints_cache_{};
    // Slot file each lane's resident session was last saved to or restored from; empty means
    // unbound. Guarded by execution_mutex_ like the lane state it describes.
    std::array<std::string, kMaximumConcurrency> lane_session_path_{};
    std::string eviction_model_binding_;
    std::function<void(std::string, targets::qwen3_8::RetainedSessionSnapshot&&)> eviction_sink_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;

    std::mutex publication_mutex_;
    std::condition_variable publication_cv_;
    std::deque<Publication> publications_;
    std::unordered_map<std::string, PendingSessionPublication> session_publications_;
    std::uint64_t publication_issued_    = 0;
    std::uint64_t publication_completed_ = 0;
    bool publication_stopping_           = false;
    std::thread publication_worker_;
};

} // namespace ninfer::runtime
