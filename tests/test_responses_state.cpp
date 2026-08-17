#include "serve/responses_state.h"
#include "serve/responses_schema.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

StoredResponse record(std::string response_id, Json output) {
    StoredResponse value;
    value.id       = std::move(response_id);
    value.response = Json{{"id", value.id}, {"output", std::move(output)}};
    return value;
}

Json message(std::string id, std::string text) {
    return Json{{"id", std::move(id)},
                {"type", "message"},
                {"status", "completed"},
                {"role", "assistant"},
                {"content", Json::array({Json{{"type", "output_text"},
                                              {"text", std::move(text)},
                                              {"annotations", Json::array()}}})}};
}

std::string api_code(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const ApiException& exception) { return exception.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

int test_exact_in_place_substitution() {
    ResponseStore store(4, 1ULL << 20);
    store.put(record("resp_a", Json::array({message("msg_a", "A")})));
    store.put(record("resp_b", Json::array({message("msg_b", "B")})));
    Json body = {{"model", "m"},
                 {"previous_response_id", "resp_independent"},
                 {"input", Json::array({Json{{"role", "user"}, {"content", "before"}},
                                        Json{{"type", "item_reference"}, {"id", "msg_b"}},
                                        Json{{"role", "user"}, {"content", "middle"}},
                                        Json{{"type", "item_reference"}, {"id", "msg_a"}}})}};

    resolve_response_item_references(body, store);
    const Json& input = body.at("input");
    int failures      = 0;
    failures += check(input.size() == 4 && input[0].at("content") == "before" &&
                          input[1].at("id") == "msg_b" && input[2].at("content") == "middle" &&
                          input[3].at("id") == "msg_a",
                      "references substitute exact Items without removing neighboring input");
    failures += check(input[1].at("content")[0].at("text") == "B" &&
                          input[3].at("content")[0].at("text") == "A",
                      "multiple references preserve caller order");
    failures += check(body.at("previous_response_id") == "resp_independent",
                      "item references do not imply or rewrite previous_response_id");

    RequestLimits limits;
    limits.default_max_tokens = 32;
    ResponsesRequest request  = parse_responses_request(body, limits);
    ChatTurn previous;
    previous.role = "assistant";
    ContentPart previous_text;
    previous_text.kind = ContentKind::Text;
    previous_text.text = "previous";
    previous.content.push_back(std::move(previous_text));
    compose_responses_generation_messages(request, {previous});
    const std::vector<ChatTurn>& turns = request.generation.messages;
    failures += check(turns.size() == 5 && turns[0].content[0].text == "previous" &&
                          turns[1].content[0].text == "before" && turns[2].content[0].text == "B" &&
                          turns[3].content[0].text == "middle" && turns[4].content[0].text == "A",
                      "explicit previous context composes independently before substituted input");
    return failures;
}

int test_reference_only_input_and_atomic_failure() {
    ResponseStore store(2, 1ULL << 20);
    store.put(record("resp_a", Json::array({message("msg_a", "A")})));
    Json reference_only = {
        {"input", Json::array({Json{{"type", "item_reference"}, {"id", "msg_a"}}})}};
    resolve_response_item_references(reference_only, store);

    Json missing        = {{"input", Json::array({Json{{"type", "item_reference"}, {"id", "msg_a"}},
                                                  Json{{"type", "item_reference"}, {"id", "missing"}}})}};
    const Json original = missing;
    const std::string code = api_code([&] { resolve_response_item_references(missing, store); });

    int failures = 0;
    failures += check(reference_only.at("input").size() == 1 &&
                          reference_only.at("input")[0].at("id") == "msg_a",
                      "a reference does not require a new-input suffix");
    failures +=
        check(code == "item_not_found" && missing == original,
              "missing batch reference reports item_not_found without partial substitution");
    return failures;
}

int test_invalid_reference() {
    ResponseStore store(2, 1ULL << 20);
    Json body = {{"input", Json::array({Json{{"type", "item_reference"}, {"id", ""}}})}};
    return check(api_code([&] { resolve_response_item_references(body, store); }) ==
                     "invalid_value",
                 "empty item_reference id is rejected");
}

} // namespace

int main() {
    int failures = 0;
    failures += test_exact_in_place_substitution();
    failures += test_reference_only_input_and_atomic_failure();
    failures += test_invalid_reference();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
