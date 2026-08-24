#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "runtime/cache/continuation_cache.h"
#include <ninfer/targets/qwen3_8/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_8 {

// Owned host payload of a decoded continuation image. Opaque here: the engine only moves it from
// the thread that decoded it to the executor thread that imports it, and never inspects it.
struct DecodedContinuation;

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

// One retained lane's complete session image (host bytes) for save/restore persistence. The
// byte layout is a target-private format; callers treat it as opaque and durable only across
// processes serving the identical model and KV configuration.
struct RetainedSessionSnapshot {
    std::vector<std::uint8_t> bytes;
    std::uint32_t tokens = 0;
    std::string session_digest;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct RequestPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
} // namespace detail

template <class Variant>
class SequencePlanner;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    // Family-private construction/storage seam. This header is repository-internal; exact
    // packages expose only the completed alias and never inspect this pointer.
    explicit RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestPlanImpl<Variant>> impl_;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine-internal fixed-lane execution surface. The public Engine owns scheduling; Program
    // owns target state images and executes one immutable decode batch membership.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan<Variant> plan_request_for_lane(std::uint32_t lane,
                                                             const PreparedPrompt& prompt,
                                                             const RequestBasePlan<Variant>& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan<Variant>&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    // Complete a generating lane between rounds, retaining its session. Used when the KV pool
    // cannot be grown to execute one more round, so the request ends at its current length
    // instead of failing mid-stream.
    void retire_lane(std::uint32_t lane);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::size_t retained_lane_resident_bytes(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::size_t retained_lane_reused_bytes(
        std::uint32_t lane, const RequestPlan<Variant>& plan) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] cache::ContinuationImage export_continuation_lane(std::uint32_t lane) const;
    [[nodiscard]] std::optional<std::string>
    stable_prefix_alias(const PreparedPrompt& prompt) const;
    [[nodiscard]] std::optional<cache::ContinuationImage>
    take_stable_continuation_lane(std::uint32_t lane);
    // Metadata preflight is a negative filter and upper bound only. A nonzero result never
    // authorizes import without resolving and exactly preflighting the complete image.
    [[nodiscard]] std::uint32_t preflight_continuation_metadata(
        const cache::SessionCandidateDescriptor& candidate,
        const PreparedPrompt& prompt) const noexcept;
    // Returns the deepest exactly matching reusable frontier: the execution frontier, a saved
    // turn-checkpoint boundary, or zero when the image cannot be reused safely.
    // divergence_tokens, when given, receives the leading tokens the prompt and the image
    // agree on. It is written whether or not the image is reusable, so a rejection reports where
    // the streams parted instead of only that they did.
    [[nodiscard]] std::uint32_t
    preflight_continuation(const cache::ContinuationImage& image, const PreparedPrompt& prompt,
                           std::uint32_t* divergence_tokens = nullptr) const noexcept;
    // Materialises the image as an owned host payload. This is the largest CPU term in a restore
    // and needs no lane, no device and no CUDA call, so the engine may run it on a preparation
    // thread concurrently with GPU execution. Throws on a malformed or incompatible image.
    [[nodiscard]] std::shared_ptr<DecodedContinuation>
    decode_continuation(const cache::ContinuationImage& image) const;
    // Imports an already-decoded image into an empty lane. `decoded` must be the payload produced
    // by decode_continuation() for this exact image; the caller binds the two by content identity.
    // `entitlement` is the caller's planned KV allocation: the restored lane is reserved for it
    // rather than for the bare frontier, so it does not have to grow on its first decode round.
    // Returns ContinuationRestoreFailure::None on success; any other value names the step that
    // refused and leaves the lane empty and the shared KV pool untouched.
    [[nodiscard]] ContinuationRestoreFailure
    import_continuation_lane(std::uint32_t lane, const cache::ContinuationImage& image,
                             const DecodedContinuation& decoded, const PreparedPrompt& prompt,
                             std::int32_t adapter, runtime::KvPageFootprint entitlement) noexcept;
    // Whether the shared paged-KV pools can currently satisfy a reservation of this size. The
    // engine uses this to choose a restore target before it disturbs any retained lane.
    [[nodiscard]] bool kv_reservation_fits(std::uint32_t text_pages,
                                           std::uint32_t backend_pages) const noexcept;
    // Extend a running lane's KV entitlement so it can execute one more decode round, bounded by
    // the ceiling its request was planned with. A request reserves a decode window at admission
    // and acquires the rest of `max_tokens` through this call as it generates. False means the
    // pool is full or the request has reached its ceiling; the lane is left untouched.
    [[nodiscard]] bool try_grow_decode_headroom(std::uint32_t lane);
    // Pages the pools would release if this retained lane were evicted.
    [[nodiscard]] runtime::KvPageFootprint
    retained_lane_kv_footprint(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint32_t retained_lane_depth(std::uint32_t lane) const noexcept;
    // Stable identifier (FNV-1a 64 hex) of the lane's resident token ledger; empty unless the
    // lane holds a retained session.
    [[nodiscard]] std::string retained_lane_digest(std::uint32_t lane) const;
    // Retained turn checkpoints of the lane's resident session, oldest first: the frontiers a
    // diverging prompt can restore from, each with the digest of the ledger prefix it covers.
    [[nodiscard]] std::vector<SlotCheckpoint>
    retained_lane_checkpoints(std::uint32_t lane) const;
    // Session persistence for one idle retained lane. `model_binding` pins the snapshot to the
    // serving weights identity; restore rejects a mismatched binding or configuration. Both
    // synchronize the device before returning and require the lane to hold no active request.
    [[nodiscard]] RetainedSessionSnapshot save_retained_lane(std::uint32_t lane,
                                                             std::string_view model_binding);
    [[nodiscard]] std::uint32_t restore_retained_lane(std::uint32_t lane,
                                                      std::span<const std::uint8_t> snapshot,
                                                      std::string_view model_binding);
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                       typename V::WeightsProfile, SequencePlan<V>&&,
                                                       DeviceContext&, std::string_view,
                                                       std::string_view,
                                                       std::span<const std::uint8_t>);
};

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
                 typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
                 DeviceContext& device, std::string_view model_id, std::string_view weights_id,
                 std::span<const std::uint8_t> artifact_fingerprint);

} // namespace ninfer::targets::qwen3_8
