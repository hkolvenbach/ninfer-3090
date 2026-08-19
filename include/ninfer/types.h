#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency = 8;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    RotatedInt8KeyInt4ValueGroup64,
    RotatedInt4KeyInt4ValueGroup64,
    RK4V4E8,
    RK2V4E8,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
};

struct LoadProgress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
};

enum class PrefixCheckpointPolicy : std::uint8_t {
    StableTurn,
    RollingTool,
};

enum class ContinuationCacheTiers : std::uint8_t {
    Off,
    L1,
    L1L2,
    L1L2L3,
};

enum class ContinuationCachePolicy : std::uint8_t {
    Adaptive,
};

struct ContinuationCacheOptions {
    ContinuationCacheTiers tiers   = ContinuationCacheTiers::L1L2;
    ContinuationCachePolicy policy = ContinuationCachePolicy::Adaptive;
    std::size_t l1_capacity_mib    = 768;
    std::size_t l2_capacity_mib    = 16384;
    std::size_t l3_capacity_mib    = 49152;
    std::filesystem::path directory;
    std::string cache_namespace             = "local";
    std::uint32_t l1_idle_ttl_seconds       = 600;
    std::uint32_t l2_idle_ttl_seconds       = 7200;
    std::uint32_t l3_idle_ttl_seconds       = 86400;
    std::uint32_t persist_interval_seconds  = 60;
    std::uint32_t persist_min_tokens        = 8192;
    std::size_t filesystem_reserve_mib      = 0;
    std::uint32_t prefix_checkpoint_history = 4;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    int device                         = 0;
    std::uint32_t max_context          = 2048; // Exact logical ceiling of each request.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    PrefixCheckpointPolicy prefix_checkpoint_policy = PrefixCheckpointPolicy::RollingTool;
    ContinuationCacheOptions continuation_cache;
    std::uint32_t vision_max_tokens = 8192;
    bool enable_vision  = false;
    bool use_cuda_graph = true;
    LoadProgress load_progress;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    // Client-provided routing hint only; exact prepared-prefix identity must authorize reuse.
    std::optional<std::string> routing_hint;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

// Owns a bounded host-input reservation whose lifetime may cross from request preparation into
// Engine execution. The concrete reservation is product-owned; Engine only releases it once the
// target reports that the retained host payload is no longer needed.
class HostInputLease {
public:
    HostInputLease() noexcept = default;

    explicit HostInputLease(std::shared_ptr<void> owner) noexcept : owner_(std::move(owner)) {}

    HostInputLease(HostInputLease&&) noexcept            = default;
    HostInputLease& operator=(HostInputLease&&) noexcept = default;

    HostInputLease(const HostInputLease&)            = delete;
    HostInputLease& operator=(const HostInputLease&) = delete;

    void reset() noexcept { owner_.reset(); }

    [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }

private:
    std::shared_ptr<void> owner_;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    std::string role;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    Low,
    Medium,
    XHigh,
};

struct ReasoningEffortCapabilities {
    bool low    = false;
    bool medium = false;
    bool xhigh  = false;
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] constexpr bool supports(ReasoningEffort effort) const noexcept {
        switch (effort) {
        case ReasoningEffort::Low:
            return low;
        case ReasoningEffort::Medium:
            return medium;
        case ReasoningEffort::XHigh:
            return xhigh;
        }
        return false;
    }
};

struct PromptCapabilities {
    bool enable_thinking = false;
    ReasoningEffortCapabilities reasoning_effort;
};

struct PromptOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
    bool add_vision_id     = false;
    std::vector<std::string> tool_jsons;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    MediaBudgetExceeded,
    Overloaded,
    QueueTimeout,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

class OutputSink {
public:
    virtual ~OutputSink()                   = default;
    virtual void publish(OutputDelta delta) = 0;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double total_seconds       = 0.0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

enum class PrefixReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreTurnCheckpoint,
    // Restored from the host-resident anchor at the last user query's opener. Shallower than a
    // turn checkpoint, but it survives a client rewriting the tail of that message.
    RestoreUserTurnAnchor,
};

enum class ContinuationSource : std::uint8_t { None, L1, L2, L3 };
enum class ContinuationAliasKind : std::uint8_t { None, Session, StablePrefix };
enum class ContinuationMissReason : std::uint8_t {
    None,
    Disabled,
    NoAlias,
    EntryUnavailableOrCorrupt,
    NotDeeper,
    PreflightRejected,
    RollbackConflict,
    NoLane,
    RestoreFailed,
};

[[nodiscard]] constexpr std::string_view continuation_source_name(ContinuationSource value) {
    switch (value) {
    case ContinuationSource::None: return "none";
    case ContinuationSource::L1: return "l1";
    case ContinuationSource::L2: return "l2";
    case ContinuationSource::L3: return "l3";
    }
    return "none";
}

[[nodiscard]] constexpr std::string_view continuation_alias_kind_name(
    ContinuationAliasKind value) {
    switch (value) {
    case ContinuationAliasKind::None: return "none";
    case ContinuationAliasKind::Session: return "routed_session";
    case ContinuationAliasKind::StablePrefix: return "stable_prefix";
    }
    return "none";
}

[[nodiscard]] constexpr std::string_view continuation_miss_reason_name(
    ContinuationMissReason value) {
    switch (value) {
    case ContinuationMissReason::None: return "none";
    case ContinuationMissReason::Disabled: return "disabled";
    case ContinuationMissReason::NoAlias: return "no_alias";
    case ContinuationMissReason::EntryUnavailableOrCorrupt:
        return "entry_unavailable_or_corrupt";
    case ContinuationMissReason::NotDeeper: return "not_deeper";
    case ContinuationMissReason::PreflightRejected: return "preflight_rejected";
    case ContinuationMissReason::RollbackConflict: return "rollback_conflict";
    case ContinuationMissReason::NoLane: return "no_lane";
    case ContinuationMissReason::RestoreFailed: return "restore_failed";
    }
    return "none";
}

struct ContinuationDiagnostics {
    ContinuationSource source                 = ContinuationSource::None;
    ContinuationAliasKind alias_kind         = ContinuationAliasKind::None;
    ContinuationMissReason final_miss_reason = ContinuationMissReason::NoAlias;
    std::uint64_t lookup_microseconds         = 0;
    std::uint64_t preflight_microseconds      = 0;
    std::uint64_t restore_microseconds        = 0;
    std::uint64_t restored_tokens             = 0;
    std::uint64_t restored_bytes              = 0;
    bool destructive_rollback                 = false;
    bool completion_publication_queued        = false;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::uint32_t reasoning_tokens     = 0;
    FinishReason finish_reason         = FinishReason::None;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationDiagnostics continuation;
    // Lane that served the request and, when it retained the finished session, that session's
    // identifying digest (see SlotState) - the handle a client needs for /slots operations.
    std::int32_t slot = -1;
    std::string session_digest;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    ArenaMemorySummary request_transient;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t cuda_graph_observed_bytes         = 0;
    std::size_t kv_payload_bytes                  = 0;
};

// Monotonic execution counters plus one boundary-consistent scheduler snapshot. Consumers derive
// interval throughput by subtracting two snapshots and dividing by their own monotonic wall time.
struct RuntimeStats {
    // Actual prompt tokens evaluated by prefill; resident prefix hits are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds                      = 0;
    std::uint64_t decode_row_rounds                  = 0;
    std::uint64_t continuation_lookup_hits           = 0;
    std::uint64_t continuation_lookup_misses         = 0;
    std::uint64_t continuation_preflight_rejections  = 0;
    // Aggregate useful restores. These equal the corresponding L1 + L2 + L3 tier totals.
    std::uint64_t continuation_restore_successes     = 0;
    std::uint64_t continuation_restore_failures      = 0;
    std::uint64_t continuation_publication_successes = 0;
    std::uint64_t continuation_publication_failures  = 0;
    std::uint64_t continuation_publication_superseded = 0;
    std::uint64_t continuation_restored_tokens       = 0;
    std::uint64_t continuation_restored_bytes        = 0;
    std::uint64_t continuation_l1_restore_successes   = 0;
    std::uint64_t continuation_l2_restore_successes   = 0;
    std::uint64_t continuation_l3_restore_successes   = 0;
    std::uint64_t continuation_l1_restored_tokens     = 0;
    std::uint64_t continuation_l2_restored_tokens     = 0;
    std::uint64_t continuation_l3_restored_tokens     = 0;
    std::uint64_t continuation_l1_restored_bytes      = 0;
    std::uint64_t continuation_l2_restored_bytes      = 0;
    std::uint64_t continuation_l3_restored_bytes      = 0;
    std::uint64_t continuation_session_restores       = 0;
    std::uint64_t continuation_stable_prefix_restores = 0;
    std::uint64_t continuation_miss_disabled          = 0;
    std::uint64_t continuation_miss_no_alias          = 0;
    std::uint64_t continuation_miss_entry_unavailable_or_corrupt = 0;
    std::uint64_t continuation_miss_not_deeper        = 0;
    std::uint64_t continuation_miss_preflight_rejected = 0;
    std::uint64_t continuation_miss_rollback_conflict = 0;
    std::uint64_t continuation_miss_no_lane           = 0;
    std::uint64_t continuation_miss_restore_failed    = 0;
    std::uint64_t continuation_l2_lookup_microseconds = 0;
    std::uint64_t continuation_l2_lookup_operations   = 0;
    std::uint64_t continuation_l3_lookup_microseconds = 0;
    std::uint64_t continuation_l3_lookup_operations   = 0;
    std::uint64_t continuation_preflight_microseconds = 0;
    std::uint64_t continuation_preflight_operations   = 0;
    std::uint64_t continuation_l2_restore_microseconds = 0;
    std::uint64_t continuation_l2_restore_operations  = 0;
    std::uint64_t continuation_l3_restore_microseconds = 0;
    std::uint64_t continuation_l3_restore_operations  = 0;
    std::uint64_t continuation_l2_admission_microseconds = 0;
    std::uint64_t continuation_l2_admission_operations = 0;
    std::uint64_t continuation_l3_persistence_microseconds = 0;
    std::uint64_t continuation_l3_persistence_operations = 0;
    std::uint64_t continuation_persistence_queued    = 0;
    std::uint64_t continuation_persistence_coalesced = 0;
    std::uint64_t continuation_persistence_successes = 0;
    std::uint64_t continuation_persistence_failures  = 0;
    std::uint64_t continuation_l2_bytes              = 0;
    std::uint64_t continuation_l3_bytes              = 0;
    std::uint32_t continuation_l2_entries            = 0;
    std::uint32_t continuation_l3_entries            = 0;
    std::uint64_t l1_evictions                        = 0;
    std::uint64_t l1_demotions                        = 0;
    std::uint64_t l1_resident_bytes                   = 0;
    std::uint32_t l1_resident_entries                 = 0;
    std::uint32_t running_requests                   = 0;
    std::uint32_t prefilling_requests                = 0;
    std::uint32_t decode_ready_requests              = 0;
    std::uint32_t waiting_requests                   = 0;
};

// Session persistence outcomes. Tokens count the resident session depth moved; bytes count the
// snapshot file payload on disk; session_digest identifies the session (see SlotState).
struct SlotSaveResult {
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
    double seconds       = 0.0;
    std::string session_digest;
};

struct SlotRestoreResult {
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
    double seconds       = 0.0;
    std::string session_digest;
};

// One Engine lane's occupancy for /slots-style reporting: an active request's prompt size, or
// the retained resident session. session_digest is a stable identifier of the exact resident
// token ledger (FNV-1a 64 as 16 hex chars) - equal digests mean the identical session; clients
// treat it as opaque and may pass it back as a slot-operation precondition.
struct SlotState {
    bool processing              = false;
    bool retained                = false;
    std::uint32_t prompt_tokens  = 0;
    std::uint32_t cached_tokens  = 0;
    std::string session_digest;
};

// Raised when a slot operation's session precondition (if_digest) does not match the lane's
// resident session.
class SlotSessionMismatch final : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

struct LoadSummary {
    std::string target;
    std::string model_id;
    std::string weights_id;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t tensor_count           = 0;
    std::size_t resource_count         = 0;
};

} // namespace ninfer
