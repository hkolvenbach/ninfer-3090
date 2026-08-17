#include "serve/response_store.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ninfer::serve {
namespace {

struct OutputItemId {
    std::string id;
    std::size_t index = 0;
};

std::vector<OutputItemId> output_item_ids(const StoredResponse& record) {
    std::vector<OutputItemId> ids;
    if (!record.response.contains("output") || !record.response.at("output").is_array()) {
        return ids;
    }
    std::size_t index = 0;
    for (const nlohmann::json& item : record.response.at("output")) {
        if (item.is_object() && item.contains("id") && item.at("id").is_string()) {
            ids.push_back(OutputItemId{item.at("id").get<std::string>(), index});
        }
        ++index;
    }
    return ids;
}

std::size_t estimate_turn_bytes(const ChatTurn& turn) {
    std::size_t bytes = sizeof(ChatTurn) + turn.role.size() + turn.tool_call_id.size() +
                        turn.reasoning_content.size();
    for (const ContentPart& part : turn.content) {
        bytes += sizeof(ContentPart) + part.text.size() + part.type_raw.size() +
                 part.source.value.size() + part.source.media_type.size() +
                 part.source.bytes.size();
    }
    for (const ToolCall& call : turn.tool_calls) {
        bytes += sizeof(ToolCall) + call.id.size() + call.name.size() + call.arguments_json.size();
    }
    return bytes;
}

std::size_t record_envelope_bytes(const StoredResponse& record) {
    std::size_t bytes = sizeof(StoredResponse) + record.id.size() + record.response.dump().size();
    for (const nlohmann::json& item : record.input_items) {
        bytes += sizeof(nlohmann::json) + item.dump().size();
    }
    // Each output Item ID is copied into the lookup key and its locator keeps
    // another response ID. Account for those owned strings under the same
    // bounded-store policy as the response payload.
    for (const OutputItemId& item : output_item_ids(record)) {
        bytes += sizeof(std::string) + item.id.size() + sizeof(OutputItemId) + record.id.size();
    }
    return bytes;
}

std::size_t standalone_bytes(const StoredResponse& record) {
    std::size_t bytes = record_envelope_bytes(record);
    for (ResponseContext node = record.context; node != nullptr; node = node->parent) {
        bytes += node->owned_bytes;
    }
    return bytes;
}

[[noreturn]] void throw_store_capacity() {
    ApiError error;
    error.status  = 500;
    error.type    = "server_error";
    error.code    = "response_store_capacity_exceeded";
    error.message = "response exceeds the configured local response store capacity";
    throw ApiException(std::move(error));
}

} // namespace

ResponseContext append_response_context(ResponseContext parent, std::vector<ChatTurn> turns) {
    auto node         = std::make_shared<ResponseContextNode>();
    node->parent      = std::move(parent);
    node->turns       = std::move(turns);
    node->owned_bytes = sizeof(ResponseContextNode);
    for (const ChatTurn& turn : node->turns) { node->owned_bytes += estimate_turn_bytes(turn); }
    return node;
}

std::vector<ChatTurn> flatten_response_context(const ResponseContext& context) {
    std::vector<const ResponseContextNode*> nodes;
    std::size_t turn_count = 0;
    for (ResponseContext node = context; node != nullptr; node = node->parent) {
        nodes.push_back(node.get());
        turn_count += node->turns.size();
    }
    std::vector<ChatTurn> turns;
    turns.reserve(turn_count);
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        turns.insert(turns.end(), (*it)->turns.begin(), (*it)->turns.end());
    }
    return turns;
}

ResponseStore::ResponseStore(std::size_t max_records, std::size_t max_bytes)
    : max_records_(max_records), max_bytes_(max_bytes) {
    if (max_records_ == 0 || max_bytes_ == 0) {
        throw std::invalid_argument("response store limits must be positive");
    }
}

std::shared_ptr<const StoredResponse> ResponseStore::get(const std::string& id) {
    std::lock_guard lock(mutex_);
    const auto found = records_.find(id);
    if (found == records_.end()) { return {}; }
    lru_.splice(lru_.begin(), lru_, found->second.lru);
    return found->second.response;
}

bool ResponseStore::get_items(const std::vector<std::string>& ids,
                              std::vector<nlohmann::json>& items, std::string* missing_id) {
    std::lock_guard lock(mutex_);
    std::vector<std::unordered_map<std::string, Entry>::iterator> owners;
    owners.reserve(ids.size());
    std::vector<nlohmann::json> resolved;
    resolved.reserve(ids.size());
    for (const std::string& id : ids) {
        const auto indexed = item_records_.find(id);
        if (indexed == item_records_.end()) {
            if (missing_id != nullptr) { *missing_id = id; }
            return false;
        }
        const auto found = records_.find(indexed->second.response_id);
        if (found == records_.end()) {
            throw std::logic_error("response item index references a missing response");
        }
        const nlohmann::json& output = found->second.response->response.at("output");
        if (!output.is_array() || indexed->second.output_index >= output.size() ||
            output.at(indexed->second.output_index).value("id", std::string()) != id) {
            throw std::logic_error("response item index is inconsistent");
        }
        owners.push_back(found);
        resolved.push_back(output.at(indexed->second.output_index));
    }
    // No externally visible state changes until every requested ID resolves.
    for (const auto owner : owners) { lru_.splice(lru_.begin(), lru_, owner->second.lru); }
    items = std::move(resolved);
    return true;
}

void ResponseStore::put(StoredResponse response) {
    if (response.id.empty() || !response.response.is_object()) {
        throw std::invalid_argument("stored response must have an id and object body");
    }
    if (standalone_bytes(response) > max_bytes_) { throw_store_capacity(); }

    const std::vector<OutputItemId> item_ids = output_item_ids(response);
    std::unordered_set<std::string> unique_item_ids;
    for (const OutputItemId& item : item_ids) {
        if (item.id.empty() || !unique_item_ids.insert(item.id).second) {
            throw std::logic_error("duplicate or empty response item id in response store");
        }
    }
    auto owned = std::make_shared<const StoredResponse>(std::move(response));
    std::lock_guard lock(mutex_);
    if (records_.contains(owned->id)) {
        throw std::logic_error("duplicate response id in response store");
    }
    for (const OutputItemId& item : item_ids) {
        if (item_records_.contains(item.id)) {
            throw std::logic_error("duplicate response item id in response store");
        }
    }
    lru_.push_front(owned->id);
    records_.emplace(owned->id, Entry{owned, lru_.begin()});
    for (const OutputItemId& item : item_ids) {
        item_records_.emplace(item.id, ItemLocator{owned->id, item.index});
    }
    current_bytes_ = recompute_bytes_locked();

    while (records_.size() > max_records_ || current_bytes_ > max_bytes_) {
        if (lru_.empty()) { throw std::logic_error("response store LRU is empty"); }
        auto victim = std::prev(lru_.end());
        if (*victim == owned->id) {
            if (victim == lru_.begin()) { throw_store_capacity(); }
            --victim;
        }
        const std::string victim_id = *victim;
        erase_locked(victim_id);
        current_bytes_ = recompute_bytes_locked();
    }
}

bool ResponseStore::erase(const std::string& id) {
    std::lock_guard lock(mutex_);
    if (!records_.contains(id)) { return false; }
    erase_locked(id);
    current_bytes_ = recompute_bytes_locked();
    return true;
}

std::size_t ResponseStore::size() const {
    std::lock_guard lock(mutex_);
    return records_.size();
}

std::size_t ResponseStore::bytes() const {
    std::lock_guard lock(mutex_);
    return current_bytes_;
}

std::size_t ResponseStore::recompute_bytes_locked() const {
    std::size_t bytes = 0;
    std::unordered_set<const ResponseContextNode*> seen;
    for (const auto& [id, entry] : records_) {
        (void)id;
        bytes += record_envelope_bytes(*entry.response);
        for (ResponseContext node = entry.response->context; node != nullptr; node = node->parent) {
            if (seen.insert(node.get()).second) { bytes += node->owned_bytes; }
        }
    }
    return bytes;
}

void ResponseStore::erase_locked(const std::string& id) {
    const auto found = records_.find(id);
    if (found == records_.end()) { return; }
    for (const OutputItemId& item : output_item_ids(*found->second.response)) {
        item_records_.erase(item.id);
    }
    lru_.erase(found->second.lru);
    records_.erase(found);
}

} // namespace ninfer::serve
