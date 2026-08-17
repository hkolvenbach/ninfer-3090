#include "serve/responses_state.h"

#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

[[noreturn]] void throw_invalid_reference() {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.param   = "input";
    error.code    = "invalid_value";
    error.message = "item_reference must contain a non-empty string id";
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_item_not_found(const std::string& id) {
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "input";
    error.code    = "item_not_found";
    error.message = "response Item '" + id + "' not found";
    throw ApiException(std::move(error));
}

} // namespace

void resolve_response_item_references(Json& body, ResponseStore& store) {
    if (!body.is_object() || !body.contains("input") || !body.at("input").is_array()) { return; }

    Json& input = body.at("input");
    std::vector<std::size_t> positions;
    std::vector<std::string> ids;
    for (std::size_t index = 0; index < input.size(); ++index) {
        const Json& item = input.at(index);
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string() ||
            item.at("type").get_ref<const std::string&>() != "item_reference") {
            continue;
        }
        if (!item.contains("id") || !item.at("id").is_string() ||
            item.at("id").get_ref<const std::string&>().empty()) {
            throw_invalid_reference();
        }
        positions.push_back(index);
        ids.push_back(item.at("id").get<std::string>());
    }
    if (ids.empty()) { return; }

    std::vector<Json> resolved;
    std::string missing_id;
    if (!store.get_items(ids, resolved, &missing_id)) { throw_item_not_found(missing_id); }
    // Input is mutated only after all references are known to exist.
    for (std::size_t index = 0; index < positions.size(); ++index) {
        input.at(positions[index]) = std::move(resolved[index]);
    }
}

} // namespace ninfer::serve
