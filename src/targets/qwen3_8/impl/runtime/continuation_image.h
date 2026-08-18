#pragma once

#include "core/cyclic_kv_cache.h"
#include "artifact/sha256.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "runtime/cache/continuation_cache.h"
#include "targets/qwen3_8/impl/runtime/prefix_identity.h"

#include <bit>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <span>
#include <type_traits>
#include <vector>

namespace ninfer::targets::qwen3_8::detail::continuation {

inline constexpr std::uint32_t kTargetImageVersion = 2;
class Writer {
public:
    void u8(std::uint8_t value) { bytes_.push_back(value); }

    void u32(std::uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void u64(std::uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void i32(std::int32_t value) { u32(std::bit_cast<std::uint32_t>(value)); }
    void f64(double value) { u64(std::bit_cast<std::uint64_t>(value)); }

    void raw(std::span<const std::uint8_t> bytes) {
        bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
    }

    void blob(std::span<const std::uint8_t> bytes) {
        u64(bytes.size());
        raw(bytes);
    }

    void string(std::string_view value) {
        blob(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(value.data()),
                                           value.size()));
    }

    [[nodiscard]] cache::Bytes finish() && { return std::move(bytes_); }

private:
    cache::Bytes bytes_;
};

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t u8() {
        require(1);
        return bytes_[cursor_++];
    }

    [[nodiscard]] std::uint32_t u32() {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift != 32; shift += 8) {
            value |= static_cast<std::uint32_t>(u8()) << shift;
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift != 64; shift += 8) {
            value |= static_cast<std::uint64_t>(u8()) << shift;
        }
        return value;
    }

    [[nodiscard]] std::int32_t i32() { return std::bit_cast<std::int32_t>(u32()); }
    [[nodiscard]] double f64() { return std::bit_cast<double>(u64()); }

    [[nodiscard]] std::size_t count(
        std::size_t maximum, std::uint64_t element_bytes = 1) {
        const std::uint64_t value = u64();
        if (value > maximum ||
            (element_bytes != 0 && value > remaining() / element_bytes) ||
            value > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("continuation image container is out of bounds");
        }
        return static_cast<std::size_t>(value);
    }

    [[nodiscard]] cache::Bytes blob(std::size_t maximum) {
        const std::size_t size = count(maximum);
        require(size);
        cache::Bytes out(bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_),
                         bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_ + size));
        cursor_ += size;
        return out;
    }

    [[nodiscard]] cache::Bytes blob_exact(std::size_t expected) {
        const std::size_t size = count(expected);
        if (size != expected) {
            throw std::invalid_argument("continuation image blob has an incompatible extent");
        }
        require(size);
        cache::Bytes out(bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_),
                         bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_ + size));
        cursor_ += size;
        return out;
    }

    void finish() const {
        if (cursor_ != bytes_.size()) {
            throw std::invalid_argument("continuation image segment has trailing bytes");
        }
    }

private:
    [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - cursor_; }

    void require(std::size_t size) const {
        if (size > remaining()) {
            throw std::invalid_argument("continuation image segment is truncated");
        }
    }

    std::span<const std::uint8_t> bytes_;
    std::size_t cursor_ = 0;
};

inline void write_header(Writer& out, std::string_view kind) {
    out.string(kind);
    out.u32(kTargetImageVersion);
}

inline void read_header(Reader& in, std::string_view kind) {
    const cache::Bytes encoded = in.blob_exact(kind.size());
    if (encoded.size() != kind.size() ||
        !std::equal(encoded.begin(), encoded.end(), kind.begin()) ||
        in.u32() != kTargetImageVersion) {
        throw std::invalid_argument("continuation image segment has an incompatible header");
    }
}

template <class T>
void write_i32_vector(Writer& out, const std::vector<T>& values) {
    static_assert(sizeof(T) == sizeof(std::int32_t));
    out.u64(values.size());
    for (const T value : values) { out.i32(static_cast<std::int32_t>(value)); }
}

template <class T>
std::vector<T> read_i32_vector(Reader& in,
                               std::size_t maximum = std::numeric_limits<std::size_t>::max()) {
    static_assert(sizeof(T) == sizeof(std::int32_t));
    const std::size_t count = in.count(maximum, sizeof(std::int32_t));
    std::vector<T> out(count);
    for (T& value : out) { value = static_cast<T>(in.i32()); }
    return out;
}

inline cache::Bytes encode_prefix(const std::vector<TokenId>& ledger,
                                  const ResidentPrefixIdentitySnapshot& identity) {
    Writer out;
    write_header(out, "qwen-prefix");
    write_i32_vector(out, ledger);
    out.u64(identity.token_types.size());
    out.raw(identity.token_types);
    for (const auto& axis : identity.positions) { write_i32_vector(out, axis); }
    out.u64(identity.vision_items.size());
    for (const VisionItem& item : identity.vision_items) {
        out.u8(static_cast<std::uint8_t>(item.modality));
        out.i32(item.grid.temporal);
        out.i32(item.grid.height);
        out.i32(item.grid.width);
        out.u64(item.patch_begin);
        out.u64(item.patch_count);
        out.raw(item.content_digest);
        out.u64(item.timestamps.size());
        for (const double timestamp : item.timestamps) { out.f64(timestamp); }
        out.u64(item.token_spans.size());
        for (const TokenSpan& span : item.token_spans) {
            out.u64(span.begin);
            out.u64(span.count);
        }
    }
    return std::move(out).finish();
}

struct PrefixData {
    std::vector<TokenId> ledger;
    ResidentPrefixIdentitySnapshot identity;
};

inline PrefixData decode_prefix(
    std::span<const std::uint8_t> bytes,
    std::size_t maximum_tokens = std::numeric_limits<std::size_t>::max()) {
    Reader in(bytes);
    read_header(in, "qwen-prefix");
    PrefixData out;
    out.ledger = read_i32_vector<TokenId>(in, maximum_tokens);
    const std::size_t token_types = in.count(maximum_tokens);
    out.identity.token_types.resize(token_types);
    for (std::uint8_t& type : out.identity.token_types) { type = in.u8(); }
    for (auto& axis : out.identity.positions) {
        axis = read_i32_vector<std::int32_t>(in, maximum_tokens);
    }
    const std::size_t item_count = in.count(maximum_tokens);
    out.identity.vision_items.resize(item_count);
    for (VisionItem& item : out.identity.vision_items) {
        const std::uint8_t modality = in.u8();
        if (modality != static_cast<std::uint8_t>(PromptModality::Image) &&
            modality != static_cast<std::uint8_t>(PromptModality::Video)) {
            throw std::invalid_argument("continuation prefix has an invalid modality");
        }
        item.modality      = static_cast<PromptModality>(modality);
        item.grid.temporal = in.i32();
        item.grid.height   = in.i32();
        item.grid.width    = in.i32();
        item.patch_begin   = in.u64();
        item.patch_count   = in.u64();
        for (std::uint8_t& byte : item.content_digest) { byte = in.u8(); }
        item.timestamps.resize(in.count(maximum_tokens, sizeof(double)));
        for (double& timestamp : item.timestamps) { timestamp = in.f64(); }
        item.token_spans.resize(in.count(maximum_tokens, 2 * sizeof(std::uint64_t)));
        for (TokenSpan& span : item.token_spans) {
            span.begin = in.u64();
            span.count = in.u64();
        }
    }
    in.finish();
    if (out.identity.token_types.size() != out.ledger.size()) {
        throw std::invalid_argument("continuation prefix ledger and identity differ in size");
    }
    for (const auto& axis : out.identity.positions) {
        if (axis.size() != out.ledger.size()) {
            throw std::invalid_argument("continuation prefix positions have an invalid shape");
        }
    }
    return out;
}

inline cache::Bytes prefix_filter_digest(const std::vector<TokenId>& ledger,
                                         ResidentPrefixIdentitySnapshot identity,
                                         std::uint32_t depth) {
    if (depth == 0 || depth > ledger.size()) {
        throw std::invalid_argument("continuation prefix filter depth is out of bounds");
    }
    ResidentPrefixIdentity resident;
    resident.restore(std::move(identity));
    resident.truncate(depth);
    std::vector<TokenId> tokens(ledger.begin(),
                                ledger.begin() + static_cast<std::ptrdiff_t>(depth));
    const cache::Bytes exact = encode_prefix(tokens, resident.export_prefix(depth));
    artifact::Sha256 hash;
    hash.update(std::as_bytes(std::span(exact)));
    const artifact::Sha256Digest digest = hash.finish();
    return cache::Bytes(digest.begin(), digest.end());
}

inline cache::Bytes prefix_filter_digest(const PreparedPromptData& prompt, std::uint32_t depth) {
    if (depth == 0 || depth > prompt.token_ids.size()) {
        throw std::invalid_argument("prompt prefix filter depth is out of bounds");
    }
    ResidentPrefixIdentity identity;
    identity.assign(prompt);
    return prefix_filter_digest(prompt.token_ids, identity.export_prefix(prompt.token_ids.size()),
                                depth);
}

inline std::optional<std::string> stable_alias(
    std::span<const std::uint8_t> compatibility_key, const PreparedPromptData& prompt) {
    if (!prompt.identity.reusable || !prompt.identity.stable_prefix_boundary) return std::nullopt;
    const std::uint32_t boundary = *prompt.identity.stable_prefix_boundary;
    if (boundary == 0 || boundary > prompt.token_ids.size() ||
        prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3 * prompt.token_ids.size()) {
        return std::nullopt;
    }

    ResidentPrefixIdentity identity;
    identity.assign(prompt);
    identity.truncate(boundary);
    std::vector<TokenId> tokens(prompt.token_ids.begin(),
                                prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(boundary));
    const cache::Bytes exact = encode_prefix(tokens, identity.export_prefix(boundary));
    Writer canonical;
    canonical.string("ninfer/qwen3.8/stable-prefix-alias");
    canonical.u32(1);
    canonical.blob(compatibility_key);
    canonical.u32(boundary);
    canonical.blob(exact);
    const cache::Bytes bytes = std::move(canonical).finish();
    artifact::Sha256 hash;
    hash.update(std::as_bytes(std::span(bytes)));
    const artifact::Sha256Digest digest = hash.finish();
    constexpr char hex[] = "0123456789abcdef";
    std::string alias(cache::kStableAliasPrefix);
    alias.reserve(alias.size() + 64);
    for (const std::uint8_t byte : digest) {
        alias.push_back(hex[byte >> 4]);
        alias.push_back(hex[byte & 0x0f]);
    }
    return alias;
}

inline std::optional<std::uint32_t>
next_prefill_checkpoint(std::uint32_t cursor, std::optional<std::uint32_t> stable,
                        std::optional<std::uint32_t> turn) noexcept {
    std::optional<std::uint32_t> next;
    if (stable && *stable > cursor) next = stable;
    if (turn && *turn > cursor && (!next || *turn < *next)) next = turn;
    return next;
}

struct FrontierMetadata {
    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier = 0;
    std::int32_t rope_delta = 0;
    std::uint32_t text_kv_valid = 0;
    std::uint32_t backend_kv_valid = 0;
    std::vector<TokenId> mtp_drafts;
};

inline bool valid_ledger_frontier(std::uint32_t execution_frontier,
                                  std::uint32_t ledger_frontier) noexcept {
    return execution_frontier != 0 &&
           (ledger_frontier == execution_frontier ||
            (execution_frontier != std::numeric_limits<std::uint32_t>::max() &&
             ledger_frontier == execution_frontier + 1U));
}

inline cache::Bytes encode_frontier(const FrontierMetadata& metadata) {
    Writer out;
    write_header(out, "qwen-frontier");
    out.u32(metadata.execution_frontier);
    out.u32(metadata.ledger_frontier);
    out.i32(metadata.rope_delta);
    out.u32(metadata.text_kv_valid);
    out.u32(metadata.backend_kv_valid);
    write_i32_vector(out, metadata.mtp_drafts);
    return std::move(out).finish();
}

inline FrontierMetadata decode_frontier(std::span<const std::uint8_t> bytes) {
    Reader in(bytes);
    read_header(in, "qwen-frontier");
    FrontierMetadata out;
    out.execution_frontier = in.u32();
    out.ledger_frontier    = in.u32();
    out.rope_delta         = in.i32();
    out.text_kv_valid      = in.u32();
    out.backend_kv_valid   = in.u32();
    out.mtp_drafts =
        read_i32_vector<TokenId>(in, qwen3_8::kMtpDecodeMaximumDrafts);
    in.finish();
    return out;
}

struct BoundaryMetadata {
    bool valid = false;
    std::uint32_t frontier = 0;
};

inline cache::Bytes encode_boundary(BoundaryMetadata metadata) {
    Writer out;
    write_header(out, "qwen-boundary");
    out.u8(metadata.valid ? 1 : 0);
    out.u32(metadata.frontier);
    return std::move(out).finish();
}

inline BoundaryMetadata decode_boundary(std::span<const std::uint8_t> bytes) {
    Reader in(bytes);
    read_header(in, "qwen-boundary");
    const std::uint8_t valid = in.u8();
    if (valid > 1) { throw std::invalid_argument("continuation boundary validity is invalid"); }
    BoundaryMetadata out{.valid = valid != 0, .frontier = in.u32()};
    in.finish();
    if (!out.valid && out.frontier != 0) {
        throw std::invalid_argument("invalid continuation boundary has a nonzero frontier");
    }
    return out;
}

inline void write_plane_spec(Writer& out, const PagedKVPlaneSpec& spec) {
    out.u8(static_cast<std::uint8_t>(spec.dtype));
    out.i32(spec.leading_extent);
    out.i32(spec.head_extent);
    out.u64(spec.alignment);
}

inline PagedKVPlaneSpec read_plane_spec(Reader& in) {
    return PagedKVPlaneSpec{.dtype = static_cast<DType>(in.u8()),
                            .leading_extent = in.i32(),
                            .head_extent = in.i32(),
                            .alignment = static_cast<std::size_t>(in.u64())};
}

inline cache::Bytes encode_paged(const PagedKVLogicalImage& image) {
    Writer out;
    write_header(out, "paged-kv");
    out.u32(image.valid_tokens);
    out.u8(static_cast<std::uint8_t>(image.plane_order));
    out.u64(image.planes.size());
    for (const auto& spec : image.planes) { write_plane_spec(out, spec); }
    out.u64(image.payloads.size());
    for (const auto& payload : image.payloads) { out.blob(payload); }
    return std::move(out).finish();
}

inline PagedKVLogicalImage decode_paged(std::span<const std::uint8_t> bytes,
                                        const PagedKVPool& pool,
                                        std::uint32_t expected_valid_tokens) {
    Reader in(bytes);
    read_header(in, "paged-kv");
    PagedKVLogicalImage out;
    out.valid_tokens = in.u32();
    if (out.valid_tokens != expected_valid_tokens) {
        throw std::invalid_argument("paged KV image has an incompatible token extent");
    }
    const std::uint8_t order = in.u8();
    if (order > static_cast<std::uint8_t>(PagedKVPlaneOrder::HeadMajor)) {
        throw std::invalid_argument("paged KV image has an invalid plane order");
    }
    out.plane_order = static_cast<PagedKVPlaneOrder>(order);
    if (out.plane_order != pool.plane_order()) {
        throw std::invalid_argument("paged KV image plane order differs from pool");
    }
    out.planes.resize(in.count(pool.plane_count()));
    if (out.planes.size() != pool.plane_count()) {
        throw std::invalid_argument("paged KV image has an incompatible plane inventory");
    }
    for (std::size_t index = 0; index < out.planes.size(); ++index) {
        out.planes[index] = read_plane_spec(in);
        const Tensor& plane = pool.plane(index);
        const PagedKVPlaneSpec& expected = pool.plane_spec(index);
        const std::int32_t heads = out.plane_order == PagedKVPlaneOrder::PageMajor
                                       ? plane.ne[2]
                                       : plane.ne[3];
        if (out.planes[index].dtype != plane.dtype ||
            out.planes[index].leading_extent != plane.ne[0] ||
            out.planes[index].head_extent != heads ||
            out.planes[index].alignment != expected.alignment) {
            throw std::invalid_argument("paged KV image plane geometry differs from pool");
        }
    }
    out.payloads.resize(in.count(pool.plane_count()));
    if (out.payloads.size() != pool.plane_count()) {
        throw std::invalid_argument("paged KV image has an incompatible payload inventory");
    }
    const std::size_t pages = expected_valid_tokens == 0
                                  ? 0
                                  : 1U + (expected_valid_tokens - 1U) / kPagedKVPageSize;
    for (std::size_t index = 0; index < out.payloads.size(); ++index) {
        const Tensor& plane = pool.plane(index);
        const std::size_t page_bytes =
            out.plane_order == PagedKVPlaneOrder::PageMajor
                ? static_cast<std::size_t>(plane.nb[3])
                : static_cast<std::size_t>(plane.nb[2]) * static_cast<std::size_t>(plane.ne[3]);
        if (pages != 0 && page_bytes > std::numeric_limits<std::size_t>::max() / pages) {
            throw std::invalid_argument("paged KV image payload extent overflows size_t");
        }
        out.payloads[index] = in.blob_exact(page_bytes * pages);
    }
    in.finish();
    if (out.planes.size() != out.payloads.size()) {
        throw std::invalid_argument("paged KV image has inconsistent planes");
    }
    return out;
}

inline cache::Bytes encode_linear(const LinearAttentionStateImage& image) {
    Writer out;
    write_header(out, "linear-state");
    out.u32(image.layers);
    out.i32(image.conv_channels);
    out.i32(image.conv_width);
    out.i32(image.value_heads);
    out.i32(image.value_head_dim);
    out.i32(image.key_head_dim);
    out.u8(static_cast<std::uint8_t>(image.conv_dtype));
    out.u64(image.conv.size());
    for (const auto& payload : image.conv) { out.blob(payload); }
    out.u64(image.recurrent.size());
    for (const auto& payload : image.recurrent) { out.blob(payload); }
    return std::move(out).finish();
}

inline LinearAttentionStateImage decode_linear(std::span<const std::uint8_t> bytes,
                                                const LinearAttentionStatePool& pool) {
    Reader in(bytes);
    read_header(in, "linear-state");
    LinearAttentionStateImage out;
    out.layers         = in.u32();
    out.conv_channels  = in.i32();
    out.conv_width     = in.i32();
    out.value_heads    = in.i32();
    out.value_head_dim = in.i32();
    out.key_head_dim   = in.i32();
    out.conv_dtype     = static_cast<DType>(in.u8());
    if (out.layers != pool.layer_count() || out.conv_channels != pool.spec.conv_channels ||
        out.conv_width != pool.spec.conv_width || out.value_heads != pool.spec.value_heads ||
        out.value_head_dim != pool.spec.value_head_dim ||
        out.key_head_dim != pool.spec.key_head_dim || out.conv_dtype != pool.spec.conv_dtype) {
        throw std::invalid_argument("linear state image geometry differs from pool");
    }
    out.conv.resize(in.count(out.layers));
    if (out.conv.size() != out.layers) {
        throw std::invalid_argument("linear state image has an incompatible layer inventory");
    }
    for (std::uint32_t layer = 0; layer < out.layers; ++layer) {
        out.conv[layer] = in.blob_exact(pool.conv_slot(layer, 0).bytes());
    }
    out.recurrent.resize(in.count(out.layers));
    if (out.recurrent.size() != out.layers) {
        throw std::invalid_argument("linear state image has an incompatible layer inventory");
    }
    for (std::uint32_t layer = 0; layer < out.layers; ++layer) {
        out.recurrent[layer] = in.blob_exact(pool.recurrent_slot(layer, 0).bytes());
    }
    in.finish();
    if (out.layers != out.conv.size() || out.layers != out.recurrent.size()) {
        throw std::invalid_argument("linear state image has inconsistent layers");
    }
    return out;
}

inline cache::Bytes encode_cyclic(const CyclicKVCacheImage& image) {
    Writer out;
    write_header(out, "cyclic-kv");
    out.u32(image.layers);
    out.u32(image.capacity);
    out.u32(image.padded_capacity);
    out.i32(image.num_kv_heads);
    out.i32(image.head_dim);
    out.u64(image.k.size());
    for (const auto& payload : image.k) { out.blob(payload); }
    out.u64(image.v.size());
    for (const auto& payload : image.v) { out.blob(payload); }
    return std::move(out).finish();
}

inline CyclicKVCacheImage decode_cyclic(std::span<const std::uint8_t> bytes,
                                        const CyclicKVCache& cache) {
    Reader in(bytes);
    read_header(in, "cyclic-kv");
    CyclicKVCacheImage out;
    out.layers          = in.u32();
    out.capacity        = in.u32();
    out.padded_capacity = in.u32();
    out.num_kv_heads    = in.i32();
    out.head_dim        = in.i32();
    if (out.layers != cache.layer_count() || out.capacity != cache.capacity() ||
        out.padded_capacity != cache.padded_capacity() ||
        out.num_kv_heads != cache.num_kv_heads() || out.head_dim != cache.head_dim()) {
        throw std::invalid_argument("cyclic KV image geometry differs from cache");
    }
    out.k.resize(in.count(out.layers));
    if (out.k.size() != out.layers) {
        throw std::invalid_argument("cyclic KV image has an incompatible layer inventory");
    }
    for (std::uint32_t layer = 0; layer < out.layers; ++layer) {
        out.k[layer] = in.blob_exact(cache.layer_view(layer).k.slice(3, 0, 1).bytes());
    }
    out.v.resize(in.count(out.layers));
    if (out.v.size() != out.layers) {
        throw std::invalid_argument("cyclic KV image has an incompatible layer inventory");
    }
    for (std::uint32_t layer = 0; layer < out.layers; ++layer) {
        out.v[layer] = in.blob_exact(cache.layer_view(layer).v.slice(3, 0, 1).bytes());
    }
    in.finish();
    if (out.layers != out.k.size() || out.layers != out.v.size()) {
        throw std::invalid_argument("cyclic KV image has inconsistent layers");
    }
    return out;
}

inline cache::Bytes encode_tensor(std::span<const std::uint8_t> payload) {
    Writer out;
    write_header(out, "tensor");
    out.blob(payload);
    return std::move(out).finish();
}

inline cache::Bytes decode_tensor(std::span<const std::uint8_t> bytes, std::size_t expected) {
    Reader in(bytes);
    read_header(in, "tensor");
    cache::Bytes out = in.blob_exact(expected);
    in.finish();
    if (out.size() != expected) {
        throw std::invalid_argument("continuation tensor has an incompatible extent");
    }
    return out;
}

} // namespace ninfer::targets::qwen3_8::detail::continuation
