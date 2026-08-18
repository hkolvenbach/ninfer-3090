#include "runtime/engine/l1_retention_policy.h"

#include <array>
#include <chrono>
#include <cstdio>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        ++failures;
        std::printf("FAIL %s\n", label);
    }
}

} // namespace

int main() {
    using namespace std::chrono_literals;
    using ninfer::runtime::L1RetentionEntry;
    using ninfer::runtime::select_l1_retention_victim;

    const auto now = std::chrono::steady_clock::time_point{} + 100s;
    std::array<L1RetentionEntry, 4> entries{{
        {.lane = 0, .resident_bytes = 40, .last_used = now - 30s, .resident = true},
        {.lane = 1, .resident_bytes = 50, .last_used = now - 20s, .resident = true},
        {.lane = 2, .resident_bytes = 80, .last_used = now - 40s, .resident = true, .active = true},
        {.lane = 3, .resident_bytes = 10, .last_used = now - 50s},
    }};

    check(!select_l1_retention_victim(entries, 90, 60s, now), "entries at budget are retained");
    check(select_l1_retention_victim(entries, 89, 60s, now) == 0,
          "budget pressure selects inactive LRU");
    check(select_l1_retention_victim(entries, 1000, 25s, now) == 0,
          "TTL selects an expired entry without budget pressure");
    check(select_l1_retention_victim(entries, 0, 60s, now) == 0,
          "zero budget disables inactive L1 residency");
    check(select_l1_retention_victim(entries, 0, 1s, now) == 0,
          "active older entry is protected");
    check(!select_l1_retention_victim(entries, 90, 0s, now),
          "zero TTL disables idle expiry");

    entries[0].last_used = now - 10s;
    check(select_l1_retention_victim(entries, 89, 60s, now) == 1,
          "recency update changes LRU selection");

    entries[0].active = true;
    entries[1].active = true;
    check(!select_l1_retention_victim(entries, 0, 0s, now),
          "active entries are protected from zero budget and TTL");
    return failures == 0 ? 0 : 1;
}
