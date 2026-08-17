#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

inline PrefixCheckpointPolicy parse_prefix_checkpoint_policy(std::string_view value) {
    if (value == "stable-turn") { return PrefixCheckpointPolicy::StableTurn; }
    if (value == "rolling-tool") { return PrefixCheckpointPolicy::RollingTool; }
    throw std::invalid_argument("invalid prefix-checkpoint-policy: " + std::string(value));
}

inline const char* prefix_checkpoint_policy_name(PrefixCheckpointPolicy policy) noexcept {
    switch (policy) {
    case PrefixCheckpointPolicy::StableTurn:
        return "stable-turn";
    case PrefixCheckpointPolicy::RollingTool:
        return "rolling-tool";
    }
    return "unknown";
}

} // namespace ninfer::product
