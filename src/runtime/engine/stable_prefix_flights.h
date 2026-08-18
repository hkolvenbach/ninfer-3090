#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace ninfer::runtime {

[[nodiscard]] constexpr bool continuation_lookup_enabled(bool cache_available,
                                                         bool allow_prefix_reuse) noexcept {
    return cache_available && allow_prefix_reuse;
}

class StablePrefixFlights {
public:
    enum class AcquireResult : std::uint8_t { Builder, Follower };

    [[nodiscard]] AcquireResult acquire(const std::string& alias, std::uint64_t request_id) {
        const auto [it, inserted] = builders_.try_emplace(alias, request_id);
        return inserted || it->second == request_id ? AcquireResult::Builder
                                                    : AcquireResult::Follower;
    }

    [[nodiscard]] bool release(const std::string& alias, std::uint64_t request_id) noexcept {
        const auto it = builders_.find(alias);
        if (it == builders_.end() || it->second != request_id) { return false; }
        builders_.erase(it);
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept { return builders_.size(); }

private:
    std::unordered_map<std::string, std::uint64_t> builders_;
};

} // namespace ninfer::runtime
