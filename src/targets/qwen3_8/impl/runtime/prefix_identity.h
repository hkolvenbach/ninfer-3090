#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <ninfer/targets/qwen3_8/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_8::detail {

struct ResidentPrefixIdentitySnapshot {
    std::vector<std::uint8_t> token_types;
    std::array<std::vector<std::int32_t>, 3> positions;
    std::vector<VisionItem> vision_items;
};

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void append_generated(std::size_t count, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] ResidentPrefixIdentitySnapshot export_prefix(std::size_t tokens) const;
    void restore(ResidentPrefixIdentitySnapshot snapshot);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                   const std::vector<TokenId>& resident_tokens,
                                   const ResidentPrefixIdentity& resident_identity,
                                   std::size_t count);

// Returns the deepest planner-usable exact match in a complete continuation image.
[[nodiscard]] std::uint32_t continuation_reuse_depth(
    const PreparedPromptData& prompt, const std::vector<TokenId>& resident_tokens,
    const ResidentPrefixIdentity& resident_identity, std::uint32_t frontier,
    std::optional<std::uint32_t> boundary);

} // namespace ninfer::targets::qwen3_8::detail
