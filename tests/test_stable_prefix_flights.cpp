#include "runtime/engine/stable_prefix_flights.h"

#include <iostream>
#include <string_view>

namespace {

using ninfer::runtime::StablePrefixFlights;
using ninfer::runtime::continuation_lookup_enabled;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

void test_success_releases_followers() {
    StablePrefixFlights flights;
    expect(flights.acquire("stable-a", 1) == StablePrefixFlights::AcquireResult::Builder,
           "first request builds");
    expect(flights.acquire("stable-a", 2) == StablePrefixFlights::AcquireResult::Follower,
           "second request follows");
    expect(flights.acquire("stable-a", 3) == StablePrefixFlights::AcquireResult::Follower,
           "multiple requests follow one builder");
    expect(flights.release("stable-a", 1), "successful publication releases builder");
    expect(flights.size() == 0, "resolved flight is erased");
}

void test_failure_handoff() {
    StablePrefixFlights flights;
    (void)flights.acquire("stable-a", 1);
    expect(!flights.release("stable-a", 2), "follower cannot release builder");
    expect(flights.release("stable-a", 1), "failed or cancelled builder releases flight");
    expect(flights.acquire("stable-a", 2) == StablePrefixFlights::AcquireResult::Builder,
           "follower becomes next builder");
    expect(flights.acquire("stable-a", 3) == StablePrefixFlights::AcquireResult::Follower,
           "remaining follower waits for replacement builder");
}

void test_unrelated_keys() {
    StablePrefixFlights flights;
    expect(flights.acquire("stable-a", 1) == StablePrefixFlights::AcquireResult::Builder,
           "first key has a builder");
    expect(flights.acquire("stable-b", 2) == StablePrefixFlights::AcquireResult::Builder,
           "unrelated key has an independent builder");
    expect(flights.size() == 2, "map contains only active builders");
    expect(flights.release("stable-a", 1), "first key releases independently");
    expect(flights.size() == 1, "release cleans only its key");
    expect(flights.release("stable-b", 2), "second key releases independently");
    expect(flights.size() == 0, "all completed keys are cleaned");
}

void test_lookup_policy() {
    expect(!continuation_lookup_enabled(true, false),
           "disabled prefix reuse skips continuation lookup and stable flights");
    expect(!continuation_lookup_enabled(false, true), "lookup requires an available cache");
    expect(continuation_lookup_enabled(true, true), "enabled reuse may perform cache work");
}

} // namespace

int main() {
    test_success_releases_followers();
    test_failure_handoff();
    test_unrelated_keys();
    test_lookup_policy();
    return failures == 0 ? 0 : 1;
}
