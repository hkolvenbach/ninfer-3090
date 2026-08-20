// ninfer::ops - lora_delta_add wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/maintainer/op-development.md §2.
#include "ninfer/ops/lora.h"

#include "ops/launcher/lora_delta.h" // detail::lora_delta_add_launch

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool registered_rank(std::int32_t rank) {
    return rank == 8 || rank == 16 || rank == 32 || rank == 64;
}

void require_matrix(const Tensor& tensor, const char* role) {
    if (tensor.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("lora_delta_add: ") + role + " must be BF16");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("lora_delta_add: ") + role + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("lora_delta_add: ") + role + " must be non-null");
    }
    if (tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("lora_delta_add: ") + role + " must be rank two");
    }
}

void validate_group(const LoraGroup& group) {
    if (!registered_rank(group.rank)) {
        throw std::invalid_argument("lora_delta_add: rank must be 8, 16, 32, or 64");
    }
    if (group.adapter_count <= 0) {
        throw std::invalid_argument("lora_delta_add: adapter_count must be positive");
    }
    if (group.site_count <= 0 || group.site_count > kMaximumLoraSites) {
        throw std::invalid_argument("lora_delta_add: site_count must be within 1..4");
    }
}

} // namespace

std::size_t lora_delta_add_workspace_capacity_bytes(std::int32_t rank, std::int32_t site_count,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (!registered_rank(rank)) {
        throw std::invalid_argument(
            "lora_delta_add_workspace_capacity_bytes: rank must be 8, 16, 32, or 64");
    }
    if (site_count <= 0 || site_count > kMaximumLoraSites) {
        throw std::invalid_argument(
            "lora_delta_add_workspace_capacity_bytes: site_count must be within 1..4");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument(
            "lora_delta_add_workspace_capacity_bytes: token interval must be positive and ordered");
    }

    // The split policy is not monotone in T, so the high-water capacity is the
    // maximum over the interval rather than its endpoint.
    std::size_t capacity = 0;
    for (std::int32_t tokens = min_tokens; tokens <= max_tokens; ++tokens) {
        const std::size_t required = detail::lora_partial_bytes(rank, site_count, tokens);
        capacity = required > capacity ? required : capacity;
        if (tokens - min_tokens > 4096) { break; }
    }
    return capacity;
}

void lora_delta_add(const Tensor& x, const LoraGroup& group, const Tensor& adapter_index,
                    std::span<Tensor* const> destinations, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    validate_group(group);
    require_matrix(x, "x");

    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("lora_delta_add: x must carry a positive T"); }

    if (destinations.size() != static_cast<std::size_t>(group.site_count)) {
        throw std::invalid_argument("lora_delta_add: destinations must match site_count");
    }
    if (adapter_index.dtype != DType::I32 || !adapter_index.is_contiguous() ||
        adapter_index.data == nullptr) {
        throw std::invalid_argument("lora_delta_add: adapter_index must be contiguous I32");
    }
    // One entry routes the whole call; otherwise each entry covers an equal, contiguous run of
    // columns. A [width, batch] verify tile therefore routes per sequence with `batch` entries,
    // and an ordinary decode batch routes per column with T entries.
    if (adapter_index.ne[0] < 1 || adapter_index.ne[0] > tokens ||
        tokens % adapter_index.ne[0] != 0) {
        throw std::invalid_argument(
            "lora_delta_add: adapter_index extent must be 1 or an exact divisor of T");
    }
    if (adapter_index.ne[1] != 1 || adapter_index.ne[2] != 1 || adapter_index.ne[3] != 1) {
        throw std::invalid_argument("lora_delta_add: adapter_index must be rank one");
    }

    bool any_active = false;
    for (std::int32_t index = 0; index < group.site_count; ++index) {
        const LoraSite& site = group.sites[index];
        Tensor* destination  = destinations[index];
        if (destination == nullptr) {
            throw std::invalid_argument("lora_delta_add: destination must be non-null");
        }
        if (site.a == nullptr || site.b == nullptr) { continue; }
        any_active = true;

        if (site.k != x.ne[0]) {
            throw std::invalid_argument("lora_delta_add: site K must match the activation rows");
        }
        if (site.n <= 0) {
            throw std::invalid_argument("lora_delta_add: site N must be positive");
        }
        require_matrix(*destination, "destination");
        if (destination->ne[0] != site.n || destination->ne[1] != tokens) {
            throw std::invalid_argument("lora_delta_add: destination shape must be [N,T]");
        }
        const auto element = sizeof(std::uint16_t);
        if (site.a_adapter_stride % element != 0 || site.b_adapter_stride % element != 0) {
            throw std::invalid_argument("lora_delta_add: adapter strides must be BF16 aligned");
        }
        if (group.adapter_count > 1 &&
            (site.a_adapter_stride == 0 || site.b_adapter_stride == 0)) {
            throw std::invalid_argument(
                "lora_delta_add: a banked site must carry non-zero adapter strides");
        }
    }
    if (!any_active) { return; }

    detail::lora_delta_add_launch(x, group, adapter_index, destinations, workspace, stream);
}

} // namespace ninfer::ops
