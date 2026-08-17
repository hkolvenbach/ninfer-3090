#pragma once

#include "serve/response_store.h"

#include <nlohmann/json.hpp>

namespace ninfer::serve {

// Replace each item_reference with the exact retained output Item it names.
// This operation deliberately does not inspect or infer previous_response_id.
void resolve_response_item_references(nlohmann::json& body, ResponseStore& store);

} // namespace ninfer::serve
