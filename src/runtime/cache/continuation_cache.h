#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::cache {

using Bytes = std::vector<std::uint8_t>;

enum class CacheSource : std::uint8_t { None, L2, L3 };
enum class CacheLookupStatus : std::uint8_t { Hit, Absent, UnavailableOrCorrupt };

struct ContentId {
    std::string hex;

    [[nodiscard]] bool valid() const;
    friend bool operator==(const ContentId&, const ContentId&) = default;
};

// This is deliberately an owning, pointer-free wire image. compatibility_key is an opaque,
// canonical runtime/model/layout key; prefix_identity contains the exact bytes used to identify
// the tokenized prefix (not a probabilistic tokenizer-side surrogate).
struct ContinuationImage {
    std::uint32_t format_version = 2;
    Bytes compatibility_key;
    Bytes prefix_identity;
    std::uint64_t frontier_tokens = 0;
    std::uint64_t boundary_tokens = 0;
    // SHA-256 of the target's canonical exact prefix identity at each reusable depth. These are
    // negative filters only; the complete prefix_identity remains authoritative after resolution.
    Bytes frontier_prefix_digest;
    Bytes boundary_prefix_digest;
    Bytes frontier_metadata;
    Bytes boundary_metadata;
    std::optional<ContentId> parent_id;
    std::map<std::string, Bytes> segments;

    friend bool operator==(const ContinuationImage&, const ContinuationImage&) = default;
};

struct CacheConfig {
    // L3 is opt-in. When false, root is ignored and the cache never touches the filesystem.
    bool enable_l3 = false;
    std::filesystem::path root;
    std::size_t l2_byte_budget                                 = 256U * 1024U * 1024U;
    std::size_t l3_byte_budget                                 = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    std::size_t filesystem_reserve_bytes                       = 0;
    // Chunks restart at every segment boundary, so a segment whose length is not a multiple of
    // this size rewrites a partial tail chunk per turn. Cutting the chunk to 512 KiB to shrink
    // that residue was measured on the 27B swarm and rejected: L3 fell only from 476 to 438 MB
    // per entry, because the residue is small next to the GDN state and the turn's genuine KV
    // delta, while persistence rose from 6.9 to 17.9 seconds per operation on the per-file fsync
    // and directory sync that each additional chunk costs.
    std::size_t chunk_bytes                                    = 4U * 1024U * 1024U;
    std::size_t session_history_depth                          = 8;
    std::chrono::seconds l2_idle_ttl                           = std::chrono::hours(2);
    std::chrono::seconds l3_idle_ttl                           = std::chrono::hours(24);
    std::chrono::seconds persist_interval                      = std::chrono::seconds(60);
    std::uint64_t persist_min_tokens                           = 8192;
    std::function<std::chrono::system_clock::time_point()> now = [] {
        return std::chrono::system_clock::now();
    };
    // Optional synchronization hook for tests; invoked after promotion releases the cache mutex.
    std::function<void()> before_persistence_io;
    // Optional test hooks invoked outside the cache mutex.
    std::function<void()> before_l3_restore_io;
    std::function<void(std::string_view)> before_alias_persistence_io;
    std::function<void(const std::filesystem::path&)> before_mutable_replace;
};

struct StoreOptions {
    // Recompute cost in arbitrary, caller-consistent units. GDSF favors expensive small entries.
    double recompute_cost = 1.0;
    std::optional<std::chrono::seconds> l2_idle_ttl;
    std::optional<std::chrono::seconds> l3_idle_ttl;
    std::optional<std::string> session;
};

struct CacheStats {
    std::size_t entries                  = 0;
    std::size_t l2_entries               = 0;
    std::size_t l2_bytes                 = 0;
    std::size_t l3_entries               = 0;
    std::size_t l3_bytes                 = 0;
    // Capacity-driven evictions only, never a TTL expiry. The two mean opposite things: an expiry
    // reclaims state that went cold, while an eviction is a live working set failing to fit its
    // budget and is therefore the tier-level churn signal.
    std::uint64_t l2_evictions           = 0;
    std::uint64_t l2_evicted_bytes       = 0;
    std::uint64_t l3_evictions           = 0;
    std::uint64_t l3_evicted_bytes       = 0;
    std::uint64_t persistence_queued     = 0;
    std::uint64_t persistence_coalesced = 0;
    std::uint64_t persistence_successes = 0;
    std::uint64_t persistence_failures = 0;
    std::uint64_t l2_admission_microseconds = 0;
    std::uint64_t l2_admission_operations = 0;
    std::uint64_t l3_persistence_microseconds = 0;
    std::uint64_t l3_persistence_operations = 0;
};

struct PersistencePolicy {
    std::chrono::seconds interval = std::chrono::seconds(60);
    std::uint64_t min_tokens      = 8192;
};

struct PersistedState {
    std::uint64_t tokens = 0;
    std::chrono::system_clock::time_point at{};
};

[[nodiscard]] bool persistence_due(const PersistencePolicy& policy,
                                   const std::optional<PersistedState>& previous,
                                   std::uint64_t candidate_tokens,
                                   std::chrono::system_clock::time_point queued_at,
                                   std::chrono::system_clock::time_point now);

// Why a session publication did not advance the alias. A publication that is not stored is a
// capacity outcome; one that is stored but did not advance lost a CAS race.
enum class SessionPublishOutcome : std::uint8_t {
    Advanced,
    RejectedTooLarge,   // the image alone exceeds the L2 budget
    EvictedOnAdmission, // admitted, then evicted by its own admission pass
    HeadMoved,          // stored, but the alias head changed since the caller read it
    GenerationMoved,    // stored, but the alias generation changed (a rollback intervened)
    // A write-once stable-prefix alias was already owned by a different image. Two lanes that
    // computed the same stable prefix do not produce the same image: the FP32 GDN recurrence
    // accumulates over each lane's own prefill chunk boundaries, so the content ids differ even
    // though the prefix text is identical. The loser's state is still admitted and usable; it
    // simply does not own the alias, so this is a lost race and not a failure.
    AliasAlreadyOwned,
    // The caller's image was parented on a head that is no longer the one it expected. Under
    // concurrent turns of one session this is an ordinary lost race, not a malformed call, so it
    // is reported rather than thrown: the publication worker re-derives the head and retries.
    LineageMismatch,
};

[[nodiscard]] constexpr const char* session_publish_outcome_name(SessionPublishOutcome outcome) {
    switch (outcome) {
    case SessionPublishOutcome::Advanced: return "advanced";
    case SessionPublishOutcome::RejectedTooLarge: return "rejected_too_large";
    case SessionPublishOutcome::EvictedOnAdmission: return "evicted_on_admission";
    case SessionPublishOutcome::HeadMoved: return "head_moved";
    case SessionPublishOutcome::GenerationMoved: return "generation_moved";
    case SessionPublishOutcome::AliasAlreadyOwned: return "alias_already_owned";
    case SessionPublishOutcome::LineageMismatch: return "lineage_mismatch";
    }
    return "advanced";
}

struct SessionPublishResult {
    ContentId id;
    bool stored         = false;
    bool alias_advanced = false;
    SessionPublishOutcome outcome = SessionPublishOutcome::Advanced;
};

struct SessionCandidateDescriptor {
    ContentId id;
    std::uint32_t image_format_version = 0;
    Bytes compatibility_key;
    std::uint64_t image_bytes = 0;
    std::uint64_t frontier_tokens = 0;
    std::uint64_t boundary_tokens = 0;
    Bytes frontier_prefix_digest;
    Bytes boundary_prefix_digest;
    CacheSource source             = CacheSource::None;
    CacheLookupStatus status       = CacheLookupStatus::Absent;
};

struct SessionCandidates {
    // Candidates are bounded by session_history_depth and ordered newest to oldest.
    std::vector<SessionCandidateDescriptor> newest_to_oldest;
    std::uint64_t generation = 0;
};

struct SessionRollbackResult {
    bool rolled_back         = false;
    std::uint64_t generation = 0;
};

struct CacheLookupResult {
    std::shared_ptr<const ContinuationImage> image;
    CacheSource source               = CacheSource::None;
    CacheLookupStatus status         = CacheLookupStatus::Absent;
    std::uint64_t io_microseconds    = 0;
};

class ContinuationCache {
public:
    explicit ContinuationCache(CacheConfig config);
    ~ContinuationCache();
    ContinuationCache(ContinuationCache&&) noexcept;
    ContinuationCache& operator=(ContinuationCache&&) noexcept;
    ContinuationCache(const ContinuationCache&)            = delete;
    ContinuationCache& operator=(const ContinuationCache&) = delete;

    [[nodiscard]] ContentId store(const ContinuationImage& image, const StoreOptions& options = {});
    // Admission publishes only to L2. promote() durably commits an admitted content ID without
    // requiring the original runtime Program or re-exporting device state.
    [[nodiscard]] ContentId admit(const ContinuationImage& image,
                                  const StoreOptions& options = {});
    [[nodiscard]] ContentId admit(ContinuationImage&& image,
                                  const StoreOptions& options = {});
    [[nodiscard]] bool promote(const ContentId& id);
    // Stores the immutable image, then advances session only when its current head exactly matches
    // expected_head. std::nullopt explicitly means that the session is expected to have no head.
    [[nodiscard]] SessionPublishResult
    publish_session(const ContinuationImage& image, std::string_view session,
                    const std::optional<ContentId>& expected_head,
                    const StoreOptions& options = {},
                    std::optional<std::uint64_t> expected_generation = std::nullopt);
    [[nodiscard]] SessionPublishResult
    publish_session_l2(const ContinuationImage& image, std::string_view session,
                       const std::optional<ContentId>& expected_head,
                       const StoreOptions& options = {},
                       std::optional<std::uint64_t> expected_generation = std::nullopt);
    [[nodiscard]] SessionPublishResult
    publish_session_l2(ContinuationImage&& image, std::string_view session,
                       const std::optional<ContentId>& expected_head,
                       const StoreOptions& options = {},
                       std::optional<std::uint64_t> expected_generation = std::nullopt);
    // The returned immutable image is the L2 allocation itself. Its lifetime pins the cache entry;
    // callers should prefer this over get() when the image may be large.
    [[nodiscard]] std::shared_ptr<const ContinuationImage> get_shared(const ContentId& id);
    [[nodiscard]] CacheLookupResult lookup_shared(const ContentId& id);
    [[nodiscard]] std::optional<ContinuationImage> get(const ContentId& id);

    [[nodiscard]] std::optional<ContentId> session_current(std::string_view session) const;
    [[nodiscard]] std::vector<ContentId> session_history(std::string_view session) const;
    // Captures bounded metadata-only descriptors and the alias generation without payload I/O.
    [[nodiscard]] SessionCandidates session_candidates(std::string_view session);
    // Resolves and verifies the complete immutable payload for a descriptor selected by preflight.
    [[nodiscard]] CacheLookupResult resolve_candidate(const SessionCandidateDescriptor& candidate);
    // Reserved aliases are immutable selectors for content-addressed shared prefixes. Publishing
    // an existing alias is a no-op, and unlike sessions it never creates history.
    [[nodiscard]] SessionPublishResult publish_immutable_alias(std::string_view alias,
                                                               const ContentId& id);
    // Coalesces the latest admitted state per session/stable alias and promotes it asynchronously.
    [[nodiscard]] bool queue_persistence(std::string_view alias, const ContentId& id,
                                         std::uint64_t frontier_tokens, bool immutable = false);
    // depth=0 is a no-op; depth=1 selects the previous image and forgets the newer alias history.
    [[nodiscard]] bool rollback_session(std::string_view session, std::size_t depth);
    // Destructively selects id only when the alias has not changed since session_candidates().
    [[nodiscard]] SessionRollbackResult rollback_session_to(
        std::string_view session, const ContentId& id, std::uint64_t expected_generation);

    [[nodiscard]] bool pin(const ContentId& id);
    void unpin(const ContentId& id);
    [[nodiscard]] CacheStats stats() const;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

[[nodiscard]] ContentId content_id(const ContinuationImage& image);
[[nodiscard]] std::size_t continuation_image_bytes(const ContinuationImage& image);

inline constexpr std::string_view kStableAliasPrefix = "@stable/v1/";

} // namespace ninfer::cache
