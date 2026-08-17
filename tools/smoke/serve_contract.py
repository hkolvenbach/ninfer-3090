"""Exercise the implemented NInfer HTTP product contract with the standard library."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# A one-pixel PNG. The target frontend performs its normal resize/patch expansion.
_IMAGE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    headers: dict[str, str]
    body: bytes


def request(
    base_url: str,
    method: str,
    path: str,
    payload: Any | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    body = None
    request_headers = {"Accept": "application/json"}
    if headers is not None:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        base_url + path, data=body, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = Response(
                status=response.status,
                content_type=response.headers.get_content_type(),
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read(),
            )
    except urllib.error.HTTPError as error:
        result = Response(
            status=error.code,
            content_type=error.headers.get_content_type(),
            headers={key.lower(): value for key, value in error.headers.items()},
            body=error.read(),
        )
    # Request IDs are the cross-protocol handle for logs and support, including errors and SSE.
    if not result.headers.get("x-request-id"):
        raise ContractError(f"{method} {path} did not return x-request-id")
    return result


def json_response(
    base_url: str, method: str, path: str, payload: Any | None = None
) -> dict[str, Any]:
    response = request(base_url, method, path, payload)
    if response.status != 200 or response.content_type != "application/json":
        detail = response.body.decode("utf-8", errors="replace")
        raise ContractError(
            f"{method} {path} returned status={response.status} "
            f"content-type={response.content_type}: {detail}"
        )
    try:
        value = json.loads(response.body)
    except json.JSONDecodeError as error:
        raise ContractError(f"{method} {path} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ContractError(f"{method} {path} did not return a JSON object")
    return value


def require_error(
    base_url: str,
    method: str,
    path: str,
    code: str,
    payload: Any | None = None,
    *,
    status: int = 400,
) -> Response:
    response = request(base_url, method, path, payload)
    if response.status != status or response.content_type != "application/json":
        raise ContractError(
            f"{method} {path} did not return the expected HTTP {status} JSON error"
        )
    try:
        value = json.loads(response.body)
    except json.JSONDecodeError as error:
        raise ContractError(f"{method} {path} returned an invalid error body") from error
    if not isinstance(value, dict) or not isinstance(value.get("error"), dict):
        raise ContractError(f"{method} {path} returned the wrong error envelope")
    if value["error"].get("code") != code:
        raise ContractError(
            f"{method} {path} returned error {value['error'].get('code')!r}, expected {code!r}"
        )
    return response


def wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if json_response(base_url, "GET", "/health") == {"status": "ok"}:
                return
        except (ContractError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise ContractError(f"server did not become healthy within {timeout:g}s: {last_error}")


def require_usage(usage: Any, prompt_key: str, completion_key: str) -> tuple[int, int]:
    if not isinstance(usage, dict):
        raise ContractError("response usage is not an object")
    prompt = usage.get(prompt_key)
    completion = usage.get(completion_key)
    if not isinstance(prompt, int) or prompt <= 0:
        raise ContractError(f"invalid {prompt_key}: {prompt!r}")
    if not isinstance(completion, int) or completion < 0:
        raise ContractError(f"invalid {completion_key}: {completion!r}")
    return prompt, completion


def openai_nonstream(base_url: str, model: str, messages: list[dict[str, Any]], *, max_tokens: int,
                     stop: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
    }
    if stop is not None:
        payload["stop"] = stop
    response = json_response(base_url, "POST", "/v1/chat/completions", payload)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ContractError("OpenAI response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ContractError("OpenAI response is missing assistant content")
    if choice.get("finish_reason") not in {"stop", "length", "tool_calls"}:
        raise ContractError(f"invalid OpenAI finish_reason: {choice.get('finish_reason')!r}")
    prompt, completion = require_usage(response.get("usage"), "prompt_tokens", "completion_tokens")
    usage = response["usage"]
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError("OpenAI total_tokens does not equal prompt + completion")
    return response


def parse_openai_stream(response: Response) -> tuple[str, str, str, dict[str, Any]]:
    if response.status != 200 or response.content_type != "text/event-stream":
        raise ContractError(
            f"OpenAI stream returned status={response.status} content-type={response.content_type}"
        )
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    saw_role = False
    saw_done = False
    for block in response.body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        if not block:
            continue
        lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(lines) != 1:
            raise ContractError(f"malformed OpenAI SSE block: {block!r}")
        if lines[0] == "[DONE]":
            if saw_done:
                raise ContractError("OpenAI stream emitted [DONE] more than once")
            saw_done = True
            continue
        if saw_done:
            raise ContractError("OpenAI stream emitted data after [DONE]")
        try:
            event = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise ContractError("OpenAI stream contained invalid JSON") from error
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise ContractError("OpenAI stream event is missing choices")
        event_usage = event.get("usage")
        if event_usage is not None:
            if choices:
                raise ContractError("OpenAI usage event unexpectedly contains choices")
            if usage is not None:
                raise ContractError("OpenAI stream emitted usage more than once")
            if finish_reason is None:
                raise ContractError("OpenAI stream emitted usage before its finish event")
            usage = event_usage
            continue
        if usage is not None:
            raise ContractError("OpenAI stream emitted an event after usage")
        if len(choices) != 1:
            raise ContractError("ordinary OpenAI stream event must contain one choice")
        if finish_reason is not None:
            raise ContractError("OpenAI stream emitted an event after its finish event")
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ContractError("OpenAI stream choice is missing delta")
        if "role" in delta:
            if saw_role or delta.get("role") != "assistant":
                raise ContractError("invalid or duplicate assistant-role event")
            saw_role = True
        if "content" in delta:
            if not isinstance(delta["content"], str):
                raise ContractError("OpenAI content delta is not a string")
            content.append(delta["content"])
        if "reasoning_content" in delta:
            if not isinstance(delta["reasoning_content"], str):
                raise ContractError("OpenAI reasoning delta is not a string")
            reasoning.append(delta["reasoning_content"])
        reason = choice.get("finish_reason")
        if reason is not None:
            if finish_reason is not None or reason not in {"stop", "length", "tool_calls"}:
                raise ContractError(f"invalid or duplicate OpenAI finish reason: {reason!r}")
            finish_reason = reason
    if not saw_role or not saw_done or finish_reason is None or usage is None:
        raise ContractError("OpenAI stream did not complete its role/finish/usage/[DONE] contract")
    prompt, completion = require_usage(usage, "prompt_tokens", "completion_tokens")
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError("OpenAI streamed total_tokens does not equal prompt + completion")
    return "".join(content), "".join(reasoning), finish_reason, usage


def response_text(response: dict[str, Any]) -> tuple[str, str]:
    output = response.get("output")
    if not isinstance(output, list):
        raise ContractError("Responses output is not an array")
    content: list[str] = []
    reasoning: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise ContractError("Responses output contains a non-object Item")
        item_type = item.get("type")
        if item_type == "reasoning":
            summary = item.get("summary")
            if not isinstance(summary, list):
                raise ContractError("Responses reasoning summary is not an array")
            for part in summary:
                if (
                    not isinstance(part, dict)
                    or part.get("type") != "summary_text"
                    or not isinstance(part.get("text"), str)
                ):
                    raise ContractError("Responses reasoning summary part has the wrong shape")
                reasoning.append(part["text"])
        elif item_type == "message":
            parts = item.get("content")
            if not isinstance(parts, list):
                raise ContractError("Responses message content is not an array")
            for part in parts:
                if (
                    not isinstance(part, dict)
                    or part.get("type") != "output_text"
                    or not isinstance(part.get("text"), str)
                ):
                    raise ContractError("Responses output_text part has the wrong shape")
                content.append(part["text"])
        elif item_type != "function_call":
            raise ContractError(f"unexpected Responses output Item type: {item_type!r}")
    return "".join(content), "".join(reasoning)


def require_responses_usage(usage: Any) -> tuple[int, int]:
    prompt, completion = require_usage(usage, "input_tokens", "output_tokens")
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError("Responses total_tokens does not equal input + output")
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, dict) or not isinstance(
        input_details.get("cached_tokens"), int
    ):
        raise ContractError("Responses cached-token usage is missing")
    if input_details.get("cache_write_tokens") != 0:
        raise ContractError("Responses cache_write_tokens must be zero")
    if not isinstance(output_details, dict) or not isinstance(
        output_details.get("reasoning_tokens"), int
    ):
        raise ContractError("Responses reasoning-token usage is missing")
    return prompt, completion


def responses_nonstream(
    base_url: str,
    model: str,
    input_value: Any,
    *,
    store: bool,
    previous_response_id: str | None = None,
    reasoning_summary: str | None = None,
    max_output_tokens: int = 16,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_value,
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
        "store": store,
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    if reasoning_summary is not None:
        payload["reasoning"] = {"summary": reasoning_summary}
    response = json_response(base_url, "POST", "/v1/responses", payload)
    status = response.get("status")
    if status == "cancelled" or status not in {"completed", "incomplete", "failed"}:
        raise ContractError(f"Responses returned an invalid terminal status: {status!r}")
    if response.get("object") != "response" or status not in {"completed", "incomplete"}:
        raise ContractError("Responses non-streaming envelope has the wrong shape")
    if not isinstance(response.get("id"), str) or not response["id"].startswith("resp_"):
        raise ContractError("Responses non-streaming envelope has an invalid id")
    if response.get("model") != model or response.get("store") is not store:
        raise ContractError("Responses non-streaming request fields were not echoed")
    require_responses_usage(response.get("usage"))
    response_text(response)
    return response


def parse_responses_stream(response: Response) -> tuple[str, str, dict[str, Any]]:
    if response.status != 200 or response.content_type != "text/event-stream":
        raise ContractError(
            "Responses stream returned "
            f"status={response.status} content-type={response.content_type}"
        )
    content: list[str] = []
    reasoning: list[str] = []
    terminal: dict[str, Any] | None = None
    expected_sequence = 0
    added_items: dict[int, str] = {}
    done_items: set[int] = set()
    reasoning_parts: set[tuple[int, int]] = set()
    done_reasoning_parts: set[tuple[int, int]] = set()
    content_parts: set[tuple[int, int]] = set()
    done_content_parts: set[tuple[int, int]] = set()
    item_types: dict[int, str] = {}
    function_arguments: dict[int, list[str]] = {}
    done_function_arguments: set[int] = set()
    lifecycle: list[str] = []
    response_id: str | None = None
    allowed_events = {
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
        "response.incomplete",
        "response.failed",
    }
    # This parser is intentionally a protocol state machine: accepting unknown events or merely
    # checking the terminal frame would let lifecycle and Item-order regressions pass unnoticed.
    for block in response.body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        if not block:
            continue
        event_lines = [line[7:] for line in block.splitlines() if line.startswith("event: ")]
        data_lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(event_lines) != 1 or len(data_lines) != 1 or data_lines[0] == "[DONE]":
            raise ContractError(f"malformed Responses SSE block: {block!r}")
        try:
            event = json.loads(data_lines[0])
        except json.JSONDecodeError as error:
            raise ContractError("Responses stream contained invalid JSON") from error
        event_type = event.get("type")
        if event_type != event_lines[0]:
            raise ContractError("Responses SSE event name differs from JSON type")
        if event_type not in allowed_events:
            raise ContractError(f"Responses stream emitted unknown event: {event_type!r}")
        if event.get("sequence_number") != expected_sequence:
            raise ContractError("Responses sequence_number is not contiguous")
        expected_sequence += 1
        if terminal is not None:
            raise ContractError("Responses stream emitted data after its terminal event")
        if event_type in {"response.created", "response.in_progress"}:
            expected_type = (
                ["response.created", "response.in_progress"][len(lifecycle)]
                if len(lifecycle) < 2
                else None
            )
            value = event.get("response")
            if event_type != expected_type or not isinstance(value, dict):
                raise ContractError("Responses stream did not start created -> in_progress")
            if value.get("status") != "in_progress" or not isinstance(value.get("id"), str):
                raise ContractError("Responses lifecycle event has the wrong Response state")
            if response_id is not None and value["id"] != response_id:
                raise ContractError("Responses lifecycle changed Response id")
            response_id = value["id"]
            lifecycle.append(event_type)
        elif len(lifecycle) != 2:
            raise ContractError("Responses output preceded created -> in_progress")
        elif event_type == "response.output_item.added":
            index = event.get("output_index")
            item = event.get("item")
            if not isinstance(index, int) or not isinstance(item, dict) or not isinstance(
                item.get("id"), str
            ) or index in added_items:
                raise ContractError("Responses output_item.added has invalid identity")
            added_items[index] = item["id"]
            item_type = item.get("type")
            if item_type not in {"reasoning", "message", "function_call"}:
                raise ContractError(f"Responses added unsupported output Item: {item_type!r}")
            item_types[index] = item_type
            if item_type == "function_call":
                function_arguments[index] = []
        elif event_type == "response.output_item.done":
            index = event.get("output_index")
            item = event.get("item")
            if (
                not isinstance(index, int)
                or not isinstance(item, dict)
                or added_items.get(index) != item.get("id")
                or index in done_items
                or item.get("type") != item_types.get(index)
            ):
                raise ContractError("Responses output_item.done does not match its added Item")
            if any(
                key[0] == index and key not in done_reasoning_parts
                for key in reasoning_parts
            ) or any(key[0] == index and key not in done_content_parts for key in content_parts):
                raise ContractError("Responses output Item finished before one of its parts")
            if item_types[index] == "function_call":
                arguments = "".join(function_arguments[index])
                if index not in done_function_arguments or item.get("arguments") != arguments:
                    raise ContractError("function argument deltas do not reconstruct done Item")
            done_items.add(index)
        elif event_type == "response.reasoning_summary_part.added":
            key = (event.get("output_index"), event.get("summary_index"))
            if (
                item_types.get(key[0]) != "reasoning"
                or event.get("item_id") != added_items.get(key[0])
                or key in reasoning_parts
            ):
                raise ContractError("Responses reasoning summary part was added out of order")
            reasoning_parts.add(key)
        elif event_type == "response.reasoning_summary_part.done":
            key = (event.get("output_index"), event.get("summary_index"))
            if (
                event.get("item_id") != added_items.get(key[0])
                or key not in reasoning_parts
                or key in done_reasoning_parts
            ):
                raise ContractError("Responses reasoning summary part done has no matching add")
            done_reasoning_parts.add(key)
        elif event_type == "response.content_part.added":
            key = (event.get("output_index"), event.get("content_index"))
            if (
                item_types.get(key[0]) != "message"
                or event.get("item_id") != added_items.get(key[0])
                or key in content_parts
            ):
                raise ContractError("Responses content part was added out of order")
            content_parts.add(key)
        elif event_type == "response.content_part.done":
            key = (event.get("output_index"), event.get("content_index"))
            if (
                event.get("item_id") != added_items.get(key[0])
                or key not in content_parts
                or key in done_content_parts
            ):
                raise ContractError("Responses content part done has no matching add")
            done_content_parts.add(key)
        elif event_type == "response.output_text.delta":
            if not isinstance(event.get("delta"), str):
                raise ContractError("Responses output_text delta is not a string")
            key = (event.get("output_index"), event.get("content_index"))
            if key not in content_parts or event.get("item_id") != added_items.get(key[0]):
                raise ContractError("Responses output_text delta preceded its content part")
            content.append(event["delta"])
        elif event_type == "response.output_text.done":
            if (
                event.get("item_id") != added_items.get(event.get("output_index"))
                or not isinstance(event.get("text"), str)
                or event["text"] != "".join(content)
            ):
                raise ContractError("Responses output_text.done differs from prior deltas")
        elif event_type == "response.reasoning_summary_text.delta":
            if not isinstance(event.get("delta"), str):
                raise ContractError("Responses reasoning delta is not a string")
            key = (event.get("output_index"), event.get("summary_index"))
            if key not in reasoning_parts or event.get("item_id") != added_items.get(key[0]):
                raise ContractError("Responses reasoning delta preceded its summary part")
            reasoning.append(event["delta"])
        elif event_type == "response.reasoning_summary_text.done":
            if (
                event.get("item_id") != added_items.get(event.get("output_index"))
                or not isinstance(event.get("text"), str)
                or event["text"] != "".join(reasoning)
            ):
                raise ContractError("Responses reasoning done differs from prior deltas")
        elif event_type == "response.function_call_arguments.delta":
            index = event.get("output_index")
            if (
                item_types.get(index) != "function_call"
                or event.get("item_id") != added_items.get(index)
                or not isinstance(event.get("delta"), str)
                or index in done_function_arguments
            ):
                raise ContractError("Responses function argument delta has invalid state")
            function_arguments[index].append(event["delta"])
        elif event_type == "response.function_call_arguments.done":
            index = event.get("output_index")
            if (
                item_types.get(index) != "function_call"
                or event.get("item_id") != added_items.get(index)
                or index in done_function_arguments
                or event.get("arguments") != "".join(function_arguments[index])
            ):
                raise ContractError("Responses function argument completion differs from deltas")
            done_function_arguments.add(index)
        elif event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            value = event.get("response")
            if not isinstance(value, dict):
                raise ContractError("Responses terminal event is missing its Response")
            if value.get("id") != response_id:
                raise ContractError("Responses terminal event changed Response id")
            if value.get("status") == "cancelled":
                raise ContractError("Responses must represent cancellation as failed, not cancelled")
            if event_type != f"response.{value.get('status')}":
                raise ContractError("Responses terminal event type differs from Response status")
            terminal = value
    if terminal is None or terminal.get("status") not in {"completed", "incomplete", "failed"}:
        raise ContractError("Responses stream has an unrecognized terminal status")
    if terminal.get("status") not in {"completed", "incomplete"}:
        raise ContractError("Responses stream did not terminate successfully")
    if set(added_items) != done_items:
        raise ContractError("Responses stream did not finish every added output Item")
    if reasoning_parts != done_reasoning_parts or content_parts != done_content_parts:
        raise ContractError("Responses stream did not finish every added content part")
    terminal_content, terminal_reasoning = response_text(terminal)
    if "".join(content) != terminal_content or "".join(reasoning) != terminal_reasoning:
        raise ContractError("Responses deltas do not reconstruct terminal output Items")
    require_responses_usage(terminal.get("usage"))
    return terminal_content, terminal_reasoning, terminal


def exercise(base_url: str, model: str, *, vision_disabled: bool = False) -> dict[str, Any]:
    preflight = request(
        base_url,
        "OPTIONS",
        "/v1/responses",
        headers={
            "Origin": "https://example.invalid",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization",
        },
    )
    # CORS is startup-optional. If enabled, verify the browser-visible contract completely.
    if "access-control-allow-origin" in preflight.headers:
        if preflight.status != 204:
            raise ContractError("CORS preflight did not return HTTP 204")
        exposed = preflight.headers.get("access-control-expose-headers", "").lower()
        allowed = preflight.headers.get("access-control-allow-headers", "").lower()
        if "x-request-id" not in exposed or "retry-after" not in exposed:
            raise ContractError("CORS does not expose x-request-id and Retry-After")
        if "content-type" not in allowed or "authorization" not in allowed:
            raise ContractError("CORS does not allow SDK authentication/content headers")

    models = json_response(base_url, "GET", "/v1/models")
    entries = models.get("data")
    if models.get("object") != "list" or not isinstance(entries, list) or len(entries) != 1:
        raise ContractError("model-list response has the wrong shape")
    if entries[0].get("id") != model or entries[0].get("owned_by") != "ninfer":
        raise ContractError("model-list response does not identify the configured NInfer model")
    single_model = json_response(base_url, "GET", f"/v1/models/{model}")
    if single_model.get("id") != model:
        raise ContractError("single-model response has the wrong id")

    anthropic_prompt = {
        "model": model,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "Reply briefly."}],
    }
    counted = json_response(base_url, "POST", "/v1/messages/count_tokens", anthropic_prompt)
    input_tokens = counted.get("input_tokens")
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        raise ContractError("count_tokens returned a non-positive input_tokens value")

    messages = [{"role": "user", "content": "Reply with a single short word."}]
    stops = ["__NINFER_SMOKE_UNLIKELY_STOP__"]
    nonstream = openai_nonstream(base_url, model, messages, max_tokens=4, stop=stops)
    expected_message = nonstream["choices"][0]["message"]
    expected_content = expected_message.get("content", "")
    expected_reasoning = expected_message.get("reasoning_content", "")
    if nonstream["usage"]["completion_tokens"] <= 0:
        raise ContractError("OpenAI request completed without producing a token")
    if not expected_content and not expected_reasoning:
        raise ContractError("OpenAI request completed without publishing output bytes")
    stream_payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 4,
        "temperature": 0,
        "stop": stops,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    streamed = request(base_url, "POST", "/v1/chat/completions", stream_payload)
    content, reasoning, stream_finish, stream_usage = parse_openai_stream(streamed)
    if content != expected_content:
        raise ContractError("streamed content differs from the non-streaming greedy response")
    if reasoning != expected_reasoning:
        raise ContractError("streamed reasoning differs from the non-streaming greedy response")
    if stream_finish != nonstream["choices"][0]["finish_reason"]:
        raise ContractError("streamed and non-streaming finish reasons differ")
    if stream_usage != nonstream["usage"]:
        raise ContractError("streamed and non-streaming usage differs")

    responses_input = "Reply with a single short word."
    response_count = json_response(
        base_url,
        "POST",
        "/v1/responses/input_tokens",
        {"model": model, "input": responses_input},
    )
    if response_count.get("object") != "response.input_tokens" or not isinstance(
        response_count.get("input_tokens"), int
    ):
        raise ContractError("Responses input_tokens returned the wrong shape")
    response_sync = responses_nonstream(
        base_url, model, responses_input, store=False
    )
    response_sync_text, response_sync_reasoning = response_text(response_sync)
    if response_sync.get("reasoning", {}).get("summary") != "auto":
        raise ContractError("omitted reasoning.summary did not use the auto default")
    response_prompt_tokens, response_output_tokens = require_responses_usage(
        response_sync.get("usage")
    )
    if response_prompt_tokens != response_count["input_tokens"]:
        raise ContractError("Responses input_tokens differs from generation usage")
    reasoning_tokens = response_sync["usage"]["output_tokens_details"]["reasoning_tokens"]
    if reasoning_tokens > 0 and not response_sync_reasoning:
        raise ContractError("default reasoning summary omitted generated reasoning")
    summarized = responses_nonstream(
        base_url, model, responses_input, store=False, reasoning_summary="auto"
    )
    summarized_text, summarized_reasoning = response_text(summarized)
    if summarized_text != response_sync_text:
        raise ContractError("requesting a reasoning summary changed greedy answer text")
    if summarized_reasoning != response_sync_reasoning:
        raise ContractError("explicit auto changed the default public reasoning summary")

    response_stream_payload: dict[str, Any] = {
        "model": model,
        "input": responses_input,
        "max_output_tokens": 16,
        "temperature": 0,
        "store": False,
        "stream": True,
    }
    default_stream = request(base_url, "POST", "/v1/responses", response_stream_payload)
    default_text, default_reasoning, _ = parse_responses_stream(default_stream)
    if (default_text, default_reasoning) != (response_sync_text, response_sync_reasoning):
        raise ContractError("default Responses SSE differs from non-streaming public output")

    response_stream_payload["reasoning"] = {"summary": "auto"}
    response_stream = request(
        base_url, "POST", "/v1/responses", response_stream_payload
    )
    streamed_text, streamed_reasoning, response_stream_terminal = parse_responses_stream(
        response_stream
    )
    if (streamed_text, streamed_reasoning) != (
        summarized_text,
        summarized_reasoning,
    ):
        raise ContractError("Responses streaming output differs from greedy non-streaming output")
    _, streamed_output_tokens = require_responses_usage(response_stream_terminal.get("usage"))
    if streamed_output_tokens != response_output_tokens:
        raise ContractError("Responses streaming output-token usage differs")

    stored_response = responses_nonstream(
        base_url,
        model,
        "Remember the code word ORCHID. Reply briefly.",
        store=True,
        max_output_tokens=64,
    )
    stored_id = stored_response.get("id")
    if not isinstance(stored_id, str) or not stored_id.startswith("resp_"):
        raise ContractError("stored Response has an invalid id")
    retrieved = json_response(base_url, "GET", f"/v1/responses/{stored_id}")
    if retrieved != stored_response:
        raise ContractError("retrieved Response differs from the created Response")
    input_items = json_response(
        base_url, "GET", f"/v1/responses/{stored_id}/input_items?order=asc&limit=1"
    )
    if (
        input_items.get("object") != "list"
        or len(input_items.get("data", [])) != 1
        or input_items.get("first_id") != input_items.get("last_id")
    ):
        raise ContractError("Responses input_items list has the wrong shape")
    continuation_input = "What code word was given? Reply with only that word."
    continuation = responses_nonstream(
        base_url,
        model,
        continuation_input,
        store=True,
        previous_response_id=stored_id,
        max_output_tokens=64,
    )
    standalone_continuation_count = json_response(
        base_url,
        "POST",
        "/v1/responses/input_tokens",
        {"model": model, "input": continuation_input},
    )["input_tokens"]
    continuation_prompt_tokens, _ = require_responses_usage(continuation.get("usage"))
    if continuation.get("previous_response_id") != stored_id or (
        continuation_prompt_tokens <= standalone_continuation_count
    ):
        raise ContractError("previous_response_id did not reconstruct stored context")
    continuation_text, _ = response_text(continuation)
    if "ORCHID" not in continuation_text.upper():
        raise ContractError("previous_response_id continuation did not recall ORCHID")

    output_items = stored_response.get("output", [])
    referenced_item = next(
        (item for item in output_items if isinstance(item, dict) and item.get("type") == "message"),
        None,
    )
    if not isinstance(referenced_item, dict) or not isinstance(referenced_item.get("id"), str):
        raise ContractError("stored Response has no referenceable message Item")
    if referenced_item.get("status") != "completed":
        raise ContractError("stored Response did not complete a referenceable message Item")
    reference_input = [
        {"role": "user", "content": "Prefix before the exact reference."},
        {"type": "item_reference", "id": referenced_item["id"]},
        {"role": "user", "content": "Reply with one short word."},
    ]
    referenced = responses_nonstream(
        base_url,
        model,
        reference_input,
        store=True,
        previous_response_id=continuation.get("id"),
        max_output_tokens=64,
    )
    if referenced.get("previous_response_id") != continuation.get("id"):
        raise ContractError("item_reference rewrote previous_response_id")
    referenced_inputs = json_response(
        base_url,
        "GET",
        f"/v1/responses/{referenced['id']}/input_items?order=asc&limit=100",
    ).get("data")
    if (
        not isinstance(referenced_inputs, list)
        or len(referenced_inputs) != 3
        or not isinstance(referenced_inputs[1], dict)
        or referenced_inputs[1].get("id") != referenced_item["id"]
        or referenced_inputs[1].get("content") != referenced_item.get("content")
    ):
        raise ContractError("item_reference was not substituted exactly in place")

    require_error(
        base_url,
        "POST",
        f"/v1/responses/{stored_id}/cancel",
        "background_not_supported",
        {},
    )
    require_error(
        base_url,
        "GET",
        f"/v1/responses/{stored_id}?stream=true",
        "parameter_not_supported",
    )

    # Item references are exact in-place substitutions and depend on the owning record's index.
    deleted = json_response(base_url, "DELETE", f"/v1/responses/{stored_id}")
    if deleted != {"id": stored_id, "object": "response.deleted", "deleted": True}:
        raise ContractError("Responses delete returned the wrong object")
    require_error(
        base_url,
        "POST",
        "/v1/responses",
        "item_not_found",
        {
            "model": model,
            "input": [{"type": "item_reference", "id": referenced_item["id"]}],
            "max_output_tokens": 16,
            "store": False,
        },
        status=404,
    )
    continuation_id = continuation.get("id")
    for response_id in (referenced.get("id"), continuation_id):
        deleted = json_response(base_url, "DELETE", f"/v1/responses/{response_id}")
        if deleted != {"id": response_id, "object": "response.deleted", "deleted": True}:
            raise ContractError("Responses delete returned the wrong object")

    tool = {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "No operation",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        },
    }
    require_error(
        base_url,
        "POST",
        "/v1/chat/completions",
        "strict_tools_not_supported",
        {"model": model, "messages": messages, "tools": [tool]},
    )
    tool["function"]["strict"] = False
    require_error(
        base_url,
        "POST",
        "/v1/chat/completions",
        "tool_choice_not_supported",
        {"model": model, "messages": messages, "tools": [tool], "tool_choice": "required"},
    )
    require_error(
        base_url,
        "POST",
        "/v1/chat/completions",
        "logit_bias_not_supported",
        {"model": model, "messages": messages, "logit_bias": {"1": 1}},
    )
    require_error(
        base_url,
        "POST",
        "/v1/responses/compact",
        "compaction_not_supported",
        {},
    )

    image_prompt = "What is visible? Answer briefly."
    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URI}},
                {"type": "text", "text": image_prompt},
            ],
        }
    ]
    if vision_disabled:
        require_error(
            base_url,
            "POST",
            "/v1/chat/completions",
            "vision_disabled",
            {
                "model": model,
                "messages": image_messages,
                "max_completion_tokens": 2,
                "temperature": 0,
            },
        )
        image_prompt_tokens = 0
    else:
        image_baseline = openai_nonstream(
            base_url,
            model,
            [{"role": "user", "content": image_prompt}],
            max_tokens=2,
        )
        baseline_prompt_tokens, _ = require_usage(
            image_baseline.get("usage"), "prompt_tokens", "completion_tokens"
        )
        image_response = openai_nonstream(base_url, model, image_messages, max_tokens=2)
        image_prompt_tokens, _ = require_usage(
            image_response.get("usage"), "prompt_tokens", "completion_tokens"
        )
        if image_prompt_tokens <= baseline_prompt_tokens:
            raise ContractError(
                "image request did not exceed its identical no-image prompt baseline"
            )

    anthropic = json_response(base_url, "POST", "/v1/messages", anthropic_prompt)
    if anthropic.get("type") != "message" or anthropic.get("role") != "assistant":
        raise ContractError("Anthropic response has the wrong envelope")
    if anthropic.get("stop_reason") not in {"end_turn", "max_tokens", "tool_use"}:
        raise ContractError(f"invalid Anthropic stop_reason: {anthropic.get('stop_reason')!r}")
    blocks = anthropic.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise ContractError("Anthropic response content is empty")
    anthropic_input_tokens, _ = require_usage(
        anthropic.get("usage"), "input_tokens", "output_tokens"
    )
    if anthropic_input_tokens != input_tokens:
        raise ContractError("Anthropic usage input_tokens differs from count_tokens")

    return {
        "format": "ninfer_serve_contract_v2",
        "model": model,
        "count_tokens": input_tokens,
        "openai_finish_reason": stream_finish,
        "openai_completion_tokens": stream_usage["completion_tokens"],
        "responses_input_tokens": response_prompt_tokens,
        "responses_output_tokens": response_output_tokens,
        "responses_reasoning_tokens": reasoning_tokens,
        "image_prompt_tokens": image_prompt_tokens,
        "anthropic_stop_reason": anthropic["stop_reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--health-timeout", type=float, default=300.0)
    parser.add_argument(
        "--vision-disabled",
        action="store_true",
        help="expect image requests to fail with vision_disabled instead of exercising Vision",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_for_health(base_url, args.health_timeout)
    print(
        json.dumps(
            exercise(base_url, args.model, vision_disabled=args.vision_disabled),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
