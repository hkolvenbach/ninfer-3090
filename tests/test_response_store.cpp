#include "serve/response_store.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ChatTurn text_turn(std::string role, std::string text) {
    ChatTurn turn;
    turn.role = std::move(role);
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "input_text";
    part.text     = std::move(text);
    turn.content.push_back(std::move(part));
    return turn;
}

StoredResponse record(std::string id, ResponseContext context) {
    StoredResponse value;
    value.id       = std::move(id);
    value.response = nlohmann::json{
        {"id", value.id},
        {"object", "response"},
        {"status", "completed"},
        {"output",
         nlohmann::json::array({nlohmann::json{{"id", "out_" + value.id}, {"type", "message"}}})}};
    value.input_items.push_back(nlohmann::json{{"id", "msg_" + value.id}, {"type", "message"}});
    value.context = std::move(context);
    return value;
}

void set_output(StoredResponse& value, nlohmann::json output) {
    value.response["output"] = std::move(output);
}

int test_context_dag() {
    const ResponseContext first =
        append_response_context({}, {text_turn("user", "one"), text_turn("assistant", "a")});
    const ResponseContext second =
        append_response_context(first, {text_turn("user", "two"), text_turn("assistant", "b")});
    const std::vector<ChatTurn> flattened = flatten_response_context(second);
    int failures                          = 0;
    failures += check(flattened.size() == 4, "context chain flattened all turns");
    failures += check(flattened[0].content[0].text == "one" && flattened[3].content[0].text == "b",
                      "context chain preserves chronological order");
    failures += check(second->parent.get() == first.get(), "context nodes share their parent");
    return failures;
}

int test_lru_and_delete() {
    ResponseStore store(2, 1ULL << 20);
    const ResponseContext root = append_response_context({}, {text_turn("user", "root")});
    store.put(record("resp_1", root));
    const ResponseContext child = append_response_context(root, {text_turn("assistant", "child")});
    store.put(record("resp_2", child));
    (void)store.get("resp_1"); // resp_2 becomes the least-recently used entry.
    store.put(record("resp_3", append_response_context(root, {text_turn("user", "fork")})));

    int failures = 0;
    failures += check(store.get("resp_1") != nullptr, "get refreshes LRU recency");
    std::vector<nlohmann::json> items;
    failures += check(store.get_items({"out_resp_1"}, items) && items.size() == 1,
                      "stored output Item is addressable by id");
    failures += check(store.get("resp_2") == nullptr, "least-recent response evicted");
    failures += check(!store.get_items({"out_resp_2"}, items),
                      "eviction removes output Item index entries");
    failures += check(store.get("resp_3") != nullptr, "new response retained");
    failures += check(store.size() == 2 && store.bytes() != 0, "store reports bounded usage");
    failures += check(store.erase("resp_1"), "stored response deleted");
    failures += check(!store.get_items({"out_resp_1"}, items),
                      "deletion removes output Item index entries");
    failures += check(!store.erase("resp_1") && store.get("resp_1") == nullptr,
                      "deleted response is no longer addressable");
    // The child/fork context owns a shared parent even when the parent's public
    // response entry is deleted.
    const std::shared_ptr<const StoredResponse> fork = store.get("resp_3");
    failures += check(fork && flatten_response_context(fork->context).size() == 2,
                      "descendant context survives parent response deletion");
    return failures;
}

int test_exact_batch_lookup() {
    ResponseStore store(4, 1ULL << 20);
    StoredResponse value = record("resp_items", {});
    set_output(value,
               nlohmann::json::array({nlohmann::json{{"id", "item_first"}, {"marker", 1}},
                                      nlohmann::json{{"id", "item_second"}, {"marker", 2}}}));
    store.put(std::move(value));

    std::vector<nlohmann::json> items;
    int failures = 0;
    failures += check(store.get_items({"item_second", "item_first", "item_second"}, items),
                      "batch lookup resolves multiple ordered references");
    failures += check(items.size() == 3 && items[0].at("marker") == 2 &&
                          items[1].at("marker") == 1 && items[2].at("marker") == 2,
                      "item locators preserve exact output indexes and request order");
    return failures;
}

int test_item_index_counts_toward_capacity() {
    StoredResponse without_index = record("resp_same", {});
    set_output(without_index, nlohmann::json::array({nlohmann::json{{"id", 0}}}));
    const std::size_t unindexed_json_bytes = without_index.response.dump().size();
    ResponseStore baseline(1, 1ULL << 20);
    baseline.put(std::move(without_index));

    StoredResponse with_index = record("resp_same", {});
    set_output(with_index, nlohmann::json::array({nlohmann::json{{"id", "0"}}}));
    const std::size_t indexed_json_bytes = with_index.response.dump().size();
    ResponseStore indexed(1, 1ULL << 20);
    indexed.put(std::move(with_index));

    const std::size_t payload_delta = indexed_json_bytes - unindexed_json_bytes;
    return check(indexed.bytes() - baseline.bytes() > payload_delta,
                 "response-store bytes include the output Item lookup index");
}

int test_failed_batch_does_not_refresh_lru() {
    ResponseStore store(2, 1ULL << 20);
    store.put(record("resp_1", {}));
    store.put(record("resp_2", {}));
    std::vector<nlohmann::json> items = {nlohmann::json{{"sentinel", true}}};
    std::string missing;
    const bool found = store.get_items({"out_resp_1", "missing"}, items, &missing);
    store.put(record("resp_3", {}));

    int failures = 0;
    failures += check(!found && missing == "missing", "failed batch reports its first missing id");
    failures += check(items.size() == 1 && items[0].contains("sentinel"),
                      "failed batch leaves its output unchanged");
    failures += check(store.get("resp_1") == nullptr && store.get("resp_2") != nullptr,
                      "failed batch leaves LRU recency unchanged");
    return failures;
}

int test_logical_put_failures_are_atomic() {
    ResponseStore store(3, 1ULL << 20);
    store.put(record("resp_live", {}));
    const std::size_t bytes = store.bytes();

    StoredResponse duplicate_items = record("resp_bad", {});
    set_output(duplicate_items,
               nlohmann::json::array({nlohmann::json{{"id", "same"}, {"marker", 1}},
                                      nlohmann::json{{"id", "same"}, {"marker", 2}}}));
    bool duplicate_items_failed = false;
    try {
        store.put(std::move(duplicate_items));
    } catch (const std::logic_error&) { duplicate_items_failed = true; }

    StoredResponse colliding_item = record("resp_collision", {});
    set_output(colliding_item, nlohmann::json::array({nlohmann::json{{"id", "out_resp_live"}}}));
    bool collision_failed = false;
    try {
        store.put(std::move(colliding_item));
    } catch (const std::logic_error&) { collision_failed = true; }

    bool duplicate_response_failed = false;
    try {
        store.put(record("resp_live", {}));
    } catch (const std::logic_error&) { duplicate_response_failed = true; }

    std::vector<nlohmann::json> live_item;
    int failures = 0;
    failures += check(duplicate_items_failed && collision_failed && duplicate_response_failed,
                      "all duplicate insertion forms fail logically");
    failures += check(store.size() == 1 && store.bytes() == bytes &&
                          store.get_items({"out_resp_live"}, live_item),
                      "logical insertion failures leave live records and indexes unchanged");
    failures += check(!store.get("resp_bad") && !store.get("resp_collision"),
                      "failed records are not partially inserted");
    return failures;
}

int test_oversized_record() {
    ResponseStore store(4, 256);
    StoredResponse large = record(
        "resp_large", append_response_context({}, {text_turn("user", std::string(1024, 'x'))}));
    std::string code;
    try {
        store.put(std::move(large));
    } catch (const ApiException& exception) { code = exception.error().code; }
    int failures = 0;
    failures += check(code == "response_store_capacity_exceeded",
                      "oversized response fails deterministically");
    failures += check(store.size() == 0, "oversized insertion does not mutate store");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_context_dag();
    failures += test_lru_and_delete();
    failures += test_exact_batch_lookup();
    failures += test_item_index_counts_toward_capacity();
    failures += test_failed_batch_does_not_refresh_lru();
    failures += test_logical_put_failures_are_atomic();
    failures += test_oversized_record();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
