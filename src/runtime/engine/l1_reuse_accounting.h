#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace ninfer::runtime {

struct L1ReuseByteLayout {
    std::size_t main_page_bytes = 0;
    std::size_t backend_page_bytes = 0;
    std::size_t current_state_bytes = 0;
    std::size_t checkpoint_state_bytes = 0;
};

inline std::optional<std::size_t> l1_reused_bytes_at_frontier(
    const L1ReuseByteLayout& layout, std::uint32_t main_tokens, std::uint32_t backend_tokens,
    std::uint32_t page_tokens, bool checkpoint) noexcept {
    if (main_tokens == 0 || page_tokens == 0) return std::size_t{0};
    const auto pages = [page_tokens](std::uint32_t tokens) -> std::size_t {
        return (static_cast<std::size_t>(tokens) + page_tokens - 1U) / page_tokens;
    };
    std::size_t total = checkpoint ? layout.checkpoint_state_bytes : layout.current_state_bytes;
    const auto add_pages = [&](std::size_t count, std::size_t bytes) {
        if (bytes != 0 && count > (std::numeric_limits<std::size_t>::max() - total) / bytes) {
            return false;
        }
        total += count * bytes;
        return true;
    };
    if (!add_pages(pages(main_tokens), layout.main_page_bytes) ||
        !add_pages(pages(backend_tokens), layout.backend_page_bytes)) {
        return std::nullopt;
    }
    return total;
}

} // namespace ninfer::runtime
