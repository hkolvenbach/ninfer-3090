#include "serve/responses_transport.h"

#include <algorithm>
#include <charconv>
#include <string_view>
#include <system_error>
#include <utility>

namespace ninfer::serve {
namespace {

[[noreturn]] void query_error(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

bool is_known(RetrievalRoute route, std::string_view key) {
    switch (route) {
    case RetrievalRoute::Response:
        return key == "include" || key == "include_obfuscation" || key == "starting_after" ||
               key == "stream";
    case RetrievalRoute::InputItems:
        return key == "after" || key == "include" || key == "limit" || key == "order";
    case RetrievalRoute::DeleteResponse:
    case RetrievalRoute::CancelResponse:
        return false;
    }
    return false;
}

} // namespace

RetrievalQuery parse_retrieval_query(RetrievalRoute route,
                                     std::vector<std::pair<std::string, std::string>> parameters) {
    std::sort(parameters.begin(), parameters.end());

    if (route == RetrievalRoute::DeleteResponse || route == RetrievalRoute::CancelResponse) {
        if (!parameters.empty()) {
            query_error("query parameters are not allowed for this endpoint",
                        parameters.front().first, "query_not_allowed");
        }
        return {};
    }

    for (std::size_t index = 1; index < parameters.size(); ++index) {
        if (parameters[index - 1].first == parameters[index].first) {
            query_error("duplicate query parameter: " + parameters[index].first,
                        parameters[index].first, "duplicate_parameter");
        }
    }
    for (const auto& [key, value] : parameters) {
        (void)value;
        if (!is_known(route, key)) {
            query_error("unknown query parameter: " + key, key, "unknown_parameter");
        }
    }

    RetrievalQuery parsed;
    if (route == RetrievalRoute::Response) {
        if (!parameters.empty()) {
            query_error("query parameter is not supported: " + parameters.front().first,
                        parameters.front().first, "parameter_not_supported");
        }
        return parsed;
    }

    for (const auto& [key, value] : parameters) {
        if (key == "after") {
            if (value.empty()) {
                query_error("after must be a non-empty Item id", key, "invalid_pagination");
            }
            parsed.after = value;
        } else if (key == "include") {
            if (!value.empty()) {
                query_error("additional input Item fields are not supported", key,
                            "include_not_supported");
            }
        } else if (key == "limit") {
            int limit         = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), limit);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
                limit < 1 || limit > 100) {
                query_error("limit must be an integer in [1,100]", key, "invalid_pagination");
            }
            parsed.limit = limit;
        } else if (key == "order") {
            if (value != "asc" && value != "desc") {
                query_error("order must be 'asc' or 'desc'", key, "invalid_pagination");
            }
            parsed.order = value;
        }
    }
    return parsed;
}

ResponseCommitAction response_commit_action(bool transport_cancelled,
                                            ninfer::FinishReason finish_reason,
                                            bool store_requested) noexcept {
    if (transport_cancelled || finish_reason == ninfer::FinishReason::Cancelled) {
        return ResponseCommitAction::FailCancelled;
    }
    return store_requested ? ResponseCommitAction::StoreThenSend : ResponseCommitAction::Send;
}

} // namespace ninfer::serve
