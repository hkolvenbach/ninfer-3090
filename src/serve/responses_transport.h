#pragma once

#include "serve/request.h"

#include <ninfer/types.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {

enum class RetrievalRoute { Response, InputItems, DeleteResponse, CancelResponse };

struct RetrievalQuery {
    int limit         = 20;
    std::string order = "desc";
    std::optional<std::string> after;
};

// Parses an already URL-decoded query independently of httplib so duplicate,
// unknown, and unsupported parameters have deterministic error precedence.
RetrievalQuery parse_retrieval_query(RetrievalRoute route,
                                     std::vector<std::pair<std::string, std::string>> parameters);

enum class ResponseCommitAction { FailCancelled, StoreThenSend, Send };

// This decision must be made immediately before storage. A cancelled outcome is
// a transport failure, never a durable Responses record.
ResponseCommitAction response_commit_action(bool transport_cancelled,
                                            ninfer::FinishReason finish_reason,
                                            bool store_requested) noexcept;

} // namespace ninfer::serve
