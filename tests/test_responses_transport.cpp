#include "serve/responses_transport.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer::serve;

namespace {

int check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; }
    return condition ? 0 : 1;
}

std::string error_code(RetrievalRoute route,
                       std::vector<std::pair<std::string, std::string>> parameters) {
    try {
        (void)parse_retrieval_query(route, std::move(parameters));
    } catch (const ApiException& exception) { return exception.error().code; }
    return {};
}

std::string error_param(RetrievalRoute route,
                        std::vector<std::pair<std::string, std::string>> parameters) {
    try {
        (void)parse_retrieval_query(route, std::move(parameters));
    } catch (const ApiException& exception) { return exception.error().param; }
    return {};
}

int test_queries() {
    int failures                = 0;
    const RetrievalQuery parsed = parse_retrieval_query(
        RetrievalRoute::InputItems, {{"order", "asc"}, {"limit", "7"}, {"after", "item_1"}});
    failures += check(parsed.limit == 7 && parsed.order == "asc" && parsed.after == "item_1",
                      "input_items query parses supported parameters");
    failures += check(error_code(RetrievalRoute::InputItems, {{"limit", "1"}, {"limit", "2"}}) ==
                          "duplicate_parameter",
                      "duplicate parameters are rejected");
    failures +=
        check(error_code(RetrievalRoute::InputItems, {{"z", ""}, {"a", ""}}) == "unknown_parameter",
              "unknown parameter rejection is deterministic");
    failures += check(error_param(RetrievalRoute::InputItems, {{"z", ""}, {"a", ""}}) == "a",
                      "unknown parameters are rejected in lexical order");
    failures += check(error_code(RetrievalRoute::Response, {{"stream", "true"}}) ==
                          "parameter_not_supported",
                      "known response retrieval options are explicitly unsupported");
    failures +=
        check(error_code(RetrievalRoute::DeleteResponse, {{"ignored", "1"}}) == "query_not_allowed",
              "DELETE rejects every query parameter");
    failures +=
        check(error_code(RetrievalRoute::CancelResponse, {{"ignored", "1"}}) == "query_not_allowed",
              "cancel rejects every query parameter");
    return failures;
}

int test_commit_decisions() {
    int failures = 0;
    failures += check(response_commit_action(false, ninfer::FinishReason::StopToken, true) ==
                          ResponseCommitAction::StoreThenSend,
                      "successful stored response commits before send");
    failures += check(response_commit_action(false, ninfer::FinishReason::StopToken, false) ==
                          ResponseCommitAction::Send,
                      "store=false sends without commit");
    failures += check(response_commit_action(true, ninfer::FinishReason::StopToken, true) ==
                          ResponseCommitAction::FailCancelled,
                      "transport cancellation prevents storage");
    failures += check(response_commit_action(false, ninfer::FinishReason::Cancelled, true) ==
                          ResponseCommitAction::FailCancelled,
                      "cancelled generation prevents storage");
    return failures;
}

} // namespace

int main() {
    const int failures = test_queries() + test_commit_decisions();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
