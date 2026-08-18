#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace ninfer::runtime {

struct L1RetentionEntry {
    std::uint32_t lane = 0;
    std::size_t resident_bytes = 0;
    std::chrono::steady_clock::time_point last_used;
    bool resident = false;
    bool active   = false;
};

// Selects one inactive victim. Expired entries take precedence; otherwise the least-recently-used
// entry is selected only when inactive residency exceeds the byte budget.
[[nodiscard]] inline std::optional<std::uint32_t>
select_l1_retention_victim(std::span<const L1RetentionEntry> entries, std::size_t byte_budget,
                           std::chrono::steady_clock::duration idle_ttl,
                           std::chrono::steady_clock::time_point now) noexcept {
    std::size_t resident_bytes = 0;
    const L1RetentionEntry* lru = nullptr;
    const L1RetentionEntry* expired_lru = nullptr;
    for (const L1RetentionEntry& entry : entries) {
        if (!entry.resident || entry.active) { continue; }
        if (entry.resident_bytes > std::numeric_limits<std::size_t>::max() - resident_bytes) {
            resident_bytes = std::numeric_limits<std::size_t>::max();
        } else {
            resident_bytes += entry.resident_bytes;
        }
        if (lru == nullptr || entry.last_used < lru->last_used) { lru = &entry; }
        if (idle_ttl > std::chrono::steady_clock::duration::zero() &&
            now - entry.last_used >= idle_ttl) {
            if (expired_lru == nullptr || entry.last_used < expired_lru->last_used) {
                expired_lru = &entry;
            }
        }
    }
    if (expired_lru != nullptr) { return expired_lru->lane; }
    if ((byte_budget == 0 || resident_bytes > byte_budget) && lru != nullptr) { return lru->lane; }
    return std::nullopt;
}

} // namespace ninfer::runtime
