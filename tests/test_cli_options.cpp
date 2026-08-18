#include "options.h"

#include <iostream>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

ninfer::cli::Options parse(std::vector<const char*> arguments) {
    return ninfer::cli::parse_options(static_cast<int>(arguments.size()),
                                      const_cast<char**>(arguments.data()));
}

} // namespace

int main() {
    using ninfer::ContinuationCacheTiers;
    check(parse({"ninfer", "model.ninfer", "--prompt", "hello"}).continuation_cache.tiers ==
              ContinuationCacheTiers::Off,
          "native one-shot continuation cache defaults to off");
    check(parse({"ninfer", "model.ninfer", "--prompt", "hello",
                 "--continuation-cache-l2-mib", "32"})
              .continuation_cache.tiers == ContinuationCacheTiers::L1L2,
          "explicit native memory-cache tuning enables l1-l2");
    check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--continuation-cache-dir",
                 "/tmp/ninfer-cache"})
              .continuation_cache.tiers == ContinuationCacheTiers::L1L2L3,
          "explicit native cache directory enables l1-l2-l3");
    check(ninfer::cli::usage_text("ninfer").find("defaults to off") != std::string::npos,
          "native help documents the off default");
    return failures == 0 ? 0 : 1;
}
