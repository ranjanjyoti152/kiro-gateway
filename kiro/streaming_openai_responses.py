# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Streaming logic for converting a Kiro stream to OpenAI Responses API SSE events.

Built on ``kiro/streaming_core.py`` so it inherits first-token timeout handling,
thinking-block parsing and tool-call parsing from the shared layer, exactly like
``streaming_openai.py`` and ``streaming_anthropic.py`` do for their surfaces.

Event contract
--------------
The Responses API streams *named* SSE events. Every event carries a strictly
increasing ``sequence_number``; item-scoped events additionally carry
``output_index``, ``item_id`` and (for content parts) ``content_index``.

The emitted order for a response containing reasoning, text and tool calls is::

    response.created
    response.in_progress
    response.output_item.added             (reasoning, output_index 0)
    response.reasoning_summary_part.added  (summary_index 0)
    response.reasoning_summary_text.delta  ... (repeated)
    response.reasoning_summary_text.done
    response.reasoning_summary_part.done
    response.output_item.done              (reasoning)
    response.output_item.added             (message, output_index 1)
    response.content_part.added            (content_index 0)
    response.output_text.delta             ... (repeated)
    response.output_text.done
    response.content_part.done
    response.output_item.done              (message)
    response.output_item.added             (function_call, output_index 2)
    response.function_call_arguments.delta
    response.function_call_arguments.done
    response.output_item.done              (function_call)
    response.completed

Ordering is not cosmetic: the Codex CLI decoder reports
``OutputTextDelta without active item`` if a text delta arrives before the
matching ``response.output_item.added``, so the item envelope is always opened
first.

Kiro API reports tool calls only once the upstream stream has finished, so
``function_call`` items are always emitted after the assistant message and their
arguments are delivered as a single delta.

Freeform tools (declared by the client as ``{"type": "custom"}``) are reported as
``custom_tool_call`` items instead, using
``response.custom_tool_call_input.delta`` / ``.done`` in place of the
``function_call_arguments`` events. Codex CLI routes freeform tools strictly on
that item type: its code-mode ``exec`` runtime accepts only
``ToolPayload::Custom`` and answers a ``function_call`` for the same tool with
"expects raw JavaScript source text".
"""

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import (
    FAKE_REASONING_HANDLING,
    FIRST_TOKEN_MAX_RETRIES,
    FIRST_TOKEN_TIMEOUT,
    TRUNCATION_RECOVERY,
)
from kiro.converters_openai_responses import ToolRegistry, extract_custom_tool_input
from kiro.models_openai_responses import (
    ResponsesCustomToolCallItem,
    ResponsesFunctionCallItem,
    ResponsesIncompleteDetails,
    ResponsesInputTokensDetails,
    ResponsesMessageItem,
    ResponsesOutputTextPart,
    ResponsesOutputTokensDetails,
    ResponsesReasoningItem,
    ResponsesResponse,
    ResponsesSummaryPart,
    ResponsesUsage,
)
from kiro.network_errors import build_http_error_detail, classify_network_error
from kiro.parsers import deduplicate_tool_calls, parse_bracket_tool_calls
from kiro.streaming_core import (
    FirstTokenTimeoutError,
    calculate_tokens_from_context_usage,
    parse_kiro_stream,
    stream_with_first_token_retry as stream_with_first_token_retry_core,
)
from kiro.tokenizer import count_message_tokens, count_tokens, count_tools_tokens

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager
    from kiro.cache import ModelInfoCache

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


__all__ = [
    "SequenceCounter",
    "build_failed_event",
    "collect_responses_response",
    "format_responses_sse",
    "generate_response_id",
    "stream_kiro_to_responses",
    "stream_responses_with_first_token_retry",
]


# Chunk size used when emitting emulated web-search output as text deltas.
WEB_SEARCH_CHUNK_SIZE: int = 100

# Content index of the single ``output_text`` part of an assistant message. The
# gateway never produces multi-part assistant messages.
TEXT_CONTENT_INDEX: int = 0

# Summary index of the single reasoning summary part produced per response.
REASONING_SUMMARY_INDEX: int = 0


# ==================================================================================================
# Event plumbing
# ==================================================================================================

class SequenceCounter:
    """
    Monotonic counter for the ``sequence_number`` field of Responses SSE events.

    The Responses API requires ``sequence_number`` to increase strictly over the
    lifetime of one response stream, including terminal error events.

    Examples:
        >>> counter = SequenceCounter()
        >>> counter.next()
        0
        >>> counter.next()
        1
        >>> counter.current
        2
    """

    def __init__(self) -> None:
        """Initialize the counter at zero."""
        self._value: int = 0

    def next(self) -> int:
        """
        Hand out the next sequence number.

        Returns:
            The current value; internal state is then incremented.
        """
        value = self._value
        self._value += 1
        return value

    @property
    def current(self) -> int:
        """
        Peek at the next value without consuming it.

        Returns:
            The next unissued sequence number.
        """
        return self._value


def generate_response_id() -> str:
    """
    Generate a Responses API response identifier.

    Returns:
        ID in the form ``resp_{uuid_hex}``.
    """
    return f"resp_{uuid.uuid4().hex}"


def generate_item_id(prefix: str) -> str:
    """
    Generate an output item identifier.

    Args:
        prefix: Short type prefix (``msg`` for messages, ``rs`` for reasoning,
            ``fc`` for function calls, ``call`` for synthesized call IDs).

    Returns:
        ID in the form ``{prefix}_{uuid_hex}``.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def format_responses_sse(event: Dict[str, Any]) -> str:
    """
    Serialize one Responses event as an SSE frame.

    Both the ``event:`` name line and the ``data:`` payload are emitted. The name
    line mirrors the official OpenAI wire format; clients that dispatch purely on
    the ``type`` field inside ``data`` also work.

    Args:
        event: Event payload; must contain a ``type`` key.

    Returns:
        SSE frame terminated by a blank line.

    Examples:
        >>> format_responses_sse({"type": "response.created"})
        'event: response.created\\ndata: {"type": "response.created"}\\n\\n'
    """
    event_name = event.get("type", "message")
    payload = json.dumps(event, ensure_ascii=False)
    frame = f"event: {event_name}\ndata: {payload}\n\n"

    if debug_logger:
        debug_logger.log_modified_chunk(frame.encode("utf-8"))

    return frame


def build_failed_event(
    response_id: str,
    model: str,
    created_at: int,
    sequence_number: int,
    message: str,
    code: str = "server_error",
) -> str:
    """
    Build a terminal ``response.failed`` SSE frame.

    Used when the upstream stream breaks after the response has already started,
    so the client sees a typed failure instead of a silently truncated stream.

    Args:
        response_id: ID of the response being failed.
        model: Model name to echo.
        created_at: Creation timestamp of the response.
        sequence_number: Sequence number for this event.
        message: Human-readable failure message.
        code: Machine-readable error code.

    Returns:
        SSE frame for the ``response.failed`` event.
    """
    response = ResponsesResponse(
        id=response_id,
        model=model,
        created_at=created_at,
        status="failed",
        error={"code": code, "message": message},
    )
    return format_responses_sse(
        {
            "type": "response.failed",
            "sequence_number": sequence_number,
            "response": response.model_dump(),
        }
    )


def build_usage(
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    total_tokens: int,
    metering_data: Optional[Dict[str, Any]] = None,
) -> ResponsesUsage:
    """
    Assemble the Responses usage block.

    Args:
        prompt_tokens: Input token count.
        completion_tokens: Output token count (reasoning tokens included).
        reasoning_tokens: Tokens attributed to thinking content.
        total_tokens: Combined token count.
        metering_data: Kiro metering payload, or ``None``.

    Returns:
        Populated ResponsesUsage.
    """
    return ResponsesUsage(
        input_tokens=prompt_tokens,
        input_tokens_details=ResponsesInputTokensDetails(cached_tokens=0),
        output_tokens=completion_tokens,
        output_tokens_details=ResponsesOutputTokensDetails(reasoning_tokens=reasoning_tokens),
        total_tokens=total_tokens,
        credits_used=metering_data,
    )


# ==================================================================================================
# Streaming
# ==================================================================================================

async def stream_kiro_to_responses(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    tool_registry: ToolRegistry,
    response_id: str,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[List[Dict[str, Any]]] = None,
    request_tools: Optional[List[Dict[str, Any]]] = None,
    request_echo: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert a Kiro stream into OpenAI Responses SSE events.

    Parses the Kiro AWS event stream through ``parse_kiro_stream`` and re-emits it
    as named Responses events with consistent ``output_index`` / ``item_id`` /
    ``content_index`` and a strictly increasing ``sequence_number``.

    Thinking content becomes reasoning events when
    ``FAKE_REASONING_HANDLING == "as_reasoning_content"``; in every other mode it
    is streamed as ordinary assistant text, matching the Chat Completions surface.

    Args:
        client: HTTP client (kept for symmetry with the other streaming modules;
            connection lifetime is owned by the route).
        response: Upstream HTTP response carrying the Kiro stream.
        model: Model name to echo back to the client.
        model_cache: Model cache used to turn context usage into token counts.
        auth_manager: Authentication manager (used by the web search emulation).
        tool_registry: Routing table mapping Kiro tool names back to the client's
            ``name``/``namespace`` pairs.
        response_id: Pre-allocated response ID, so the caller can reference the
            same ID in a failure event.
        first_token_timeout: First token wait timeout in seconds.
        request_messages: Original request messages for the tiktoken fallback.
        request_tools: Original request tools for the tiktoken fallback.
        request_echo: Request fields echoed onto the response object
            (``instructions``, ``metadata``, ``tools``, ...).

    Yields:
        SSE frames, one per Responses event.

    Raises:
        FirstTokenTimeoutError: If the first token does not arrive in time, so the
            retry wrapper can open a new upstream request.
    """
    counter = SequenceCounter()
    created_at = int(time.time())
    echo: Dict[str, Any] = dict(request_echo or {})
    reasoning_as_items = FAKE_REASONING_HANDLING == "as_reasoning_content"

    def build_envelope(status: str, **overrides: Any) -> Dict[str, Any]:
        """Build the response envelope embedded in lifecycle events."""
        payload: Dict[str, Any] = {
            "id": response_id,
            "model": model,
            "created_at": created_at,
            "status": status,
        }
        payload.update(echo)
        payload.update(overrides)
        return ResponsesResponse(**payload).model_dump()

    # Text streamed to the client as ``output_text`` deltas. Equals ``content``
    # plus, when thinking is inlined, the thinking text as well.
    message_text = ""
    # Model content only, used for bracket tool-call detection and truncation
    # detection, exactly like the Chat Completions surface.
    full_content = ""
    full_thinking_content = ""

    metering_data: Optional[Dict[str, Any]] = None
    context_usage_percentage: Optional[float] = None
    tool_calls_from_stream: List[Dict[str, Any]] = []

    next_output_index = 0
    reasoning_item_id: Optional[str] = None
    reasoning_output_index: Optional[int] = None
    message_item_id: Optional[str] = None
    message_output_index: Optional[int] = None

    output_items: List[Dict[str, Any]] = []
    streaming_error_occurred = False

    def open_reasoning_item() -> List[str]:
        """Open a reasoning output item and return the frames to emit."""
        nonlocal reasoning_item_id, reasoning_output_index, next_output_index
        reasoning_item_id = generate_item_id("rs")
        reasoning_output_index = next_output_index
        next_output_index += 1
        return [
            format_responses_sse(
                {
                    "type": "response.output_item.added",
                    "sequence_number": counter.next(),
                    "output_index": reasoning_output_index,
                    "item": ResponsesReasoningItem(id=reasoning_item_id).model_dump(),
                }
            ),
            format_responses_sse(
                {
                    "type": "response.reasoning_summary_part.added",
                    "sequence_number": counter.next(),
                    "item_id": reasoning_item_id,
                    "output_index": reasoning_output_index,
                    "summary_index": REASONING_SUMMARY_INDEX,
                    "part": {"type": "summary_text", "text": ""},
                }
            ),
        ]

    def close_reasoning_item() -> List[str]:
        """Close the open reasoning output item and return the frames to emit."""
        nonlocal reasoning_item_id
        item_id = reasoning_item_id
        if item_id is None:
            return []
        frames = [
            format_responses_sse(
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": counter.next(),
                    "item_id": item_id,
                    "output_index": reasoning_output_index,
                    "summary_index": REASONING_SUMMARY_INDEX,
                    "text": full_thinking_content,
                }
            ),
            format_responses_sse(
                {
                    "type": "response.reasoning_summary_part.done",
                    "sequence_number": counter.next(),
                    "item_id": item_id,
                    "output_index": reasoning_output_index,
                    "summary_index": REASONING_SUMMARY_INDEX,
                    "part": {"type": "summary_text", "text": full_thinking_content},
                }
            ),
        ]
        reasoning_item = ResponsesReasoningItem(
            id=item_id,
            summary=(
                [ResponsesSummaryPart(type="summary_text", text=full_thinking_content)]
                if full_thinking_content
                else []
            ),
        )
        frames.append(
            format_responses_sse(
                {
                    "type": "response.output_item.done",
                    "sequence_number": counter.next(),
                    "output_index": reasoning_output_index,
                    "item": reasoning_item.model_dump(),
                }
            )
        )
        output_items.append(reasoning_item.model_dump())
        reasoning_item_id = None
        return frames

    def open_message_item() -> List[str]:
        """Open the assistant message output item and return the frames to emit."""
        nonlocal message_item_id, message_output_index, next_output_index
        message_item_id = generate_item_id("msg")
        message_output_index = next_output_index
        next_output_index += 1
        return [
            format_responses_sse(
                {
                    "type": "response.output_item.added",
                    "sequence_number": counter.next(),
                    "output_index": message_output_index,
                    "item": ResponsesMessageItem(
                        id=message_item_id, status="in_progress"
                    ).model_dump(),
                }
            ),
            format_responses_sse(
                {
                    "type": "response.content_part.added",
                    "sequence_number": counter.next(),
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": TEXT_CONTENT_INDEX,
                    "part": ResponsesOutputTextPart().model_dump(),
                }
            ),
        ]

    def text_delta(text: str) -> str:
        """Build one ``response.output_text.delta`` frame."""
        return format_responses_sse(
            {
                "type": "response.output_text.delta",
                "sequence_number": counter.next(),
                "item_id": message_item_id,
                "output_index": message_output_index,
                "content_index": TEXT_CONTENT_INDEX,
                "delta": text,
            }
        )

    try:
        yield format_responses_sse(
            {
                "type": "response.created",
                "sequence_number": counter.next(),
                "response": build_envelope("in_progress"),
            }
        )
        yield format_responses_sse(
            {
                "type": "response.in_progress",
                "sequence_number": counter.next(),
                "response": build_envelope("in_progress"),
            }
        )

        async for event in parse_kiro_stream(response, first_token_timeout):
            # ----------------------------------------------------------------
            # Thinking
            # ----------------------------------------------------------------
            if event.type == "thinking" and event.thinking_content:
                full_thinking_content += event.thinking_content

                if reasoning_as_items:
                    if reasoning_item_id is None:
                        for frame in open_reasoning_item():
                            yield frame
                    yield format_responses_sse(
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "sequence_number": counter.next(),
                            "item_id": reasoning_item_id,
                            "output_index": reasoning_output_index,
                            "summary_index": REASONING_SUMMARY_INDEX,
                            "delta": event.thinking_content,
                        }
                    )
                    continue

                # Thinking is part of the visible answer in every other mode.
                if message_item_id is None:
                    for frame in open_message_item():
                        yield frame
                message_text += event.thinking_content
                yield text_delta(event.thinking_content)
                continue

            # ----------------------------------------------------------------
            # Regular content
            # ----------------------------------------------------------------
            if event.type == "content" and event.content:
                for frame in close_reasoning_item():
                    yield frame
                if message_item_id is None:
                    for frame in open_message_item():
                        yield frame
                full_content += event.content
                message_text += event.content
                yield text_delta(event.content)
                continue

            # ----------------------------------------------------------------
            # Tool calls
            # ----------------------------------------------------------------
            if event.type == "tool_use" and event.tool_use:
                tool = event.tool_use
                function = tool.get("function") or {}
                tool_name = function.get("name") or tool.get("name") or ""

                if tool_name == "web_search":
                    summary = await run_emulated_web_search(tool, auth_manager)
                    if summary is not None:
                        for frame in close_reasoning_item():
                            yield frame
                        if message_item_id is None:
                            for frame in open_message_item():
                                yield frame
                        for offset in range(0, len(summary), WEB_SEARCH_CHUNK_SIZE):
                            chunk = summary[offset:offset + WEB_SEARCH_CHUNK_SIZE]
                            full_content += chunk
                            message_text += chunk
                            yield text_delta(chunk)
                        continue

                tool_calls_from_stream.append(tool)
                continue

            # ----------------------------------------------------------------
            # Usage
            # ----------------------------------------------------------------
            if event.type == "usage" and event.usage:
                metering_data = event.usage
                continue

            if event.type == "context_usage" and event.context_usage_percentage is not None:
                context_usage_percentage = event.context_usage_percentage
                continue

        # ------------------------------------------------------------------
        # Stream finished: close open items
        # ------------------------------------------------------------------
        for frame in close_reasoning_item():
            yield frame

        if message_item_id is not None:
            yield format_responses_sse(
                {
                    "type": "response.output_text.done",
                    "sequence_number": counter.next(),
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": TEXT_CONTENT_INDEX,
                    "text": message_text,
                }
            )
            yield format_responses_sse(
                {
                    "type": "response.content_part.done",
                    "sequence_number": counter.next(),
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": TEXT_CONTENT_INDEX,
                    "part": ResponsesOutputTextPart(text=message_text).model_dump(),
                }
            )
            message_item = ResponsesMessageItem(
                id=message_item_id,
                status="completed",
                content=[ResponsesOutputTextPart(text=message_text)],
            )
            yield format_responses_sse(
                {
                    "type": "response.output_item.done",
                    "sequence_number": counter.next(),
                    "output_index": message_output_index,
                    "item": message_item.model_dump(),
                }
            )
            output_items.append(message_item.model_dump())

        # ------------------------------------------------------------------
        # Tool calls and truncation detection
        # ------------------------------------------------------------------
        received_completion_signal = (
            metering_data is not None or context_usage_percentage is not None
        )

        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        all_tool_calls = deduplicate_tool_calls(tool_calls_from_stream + bracket_tool_calls)

        content_was_truncated = (
            not received_completion_signal and len(full_content) > 0 and not all_tool_calls
        )
        if content_was_truncated:
            logger.error(
                f"Content truncated by Kiro API: stream ended without completion signals, "
                f"length={len(full_content)} chars. "
                + (
                    "Model will be notified automatically about truncation."
                    if TRUNCATION_RECOVERY
                    else "Set TRUNCATION_RECOVERY=true in .env to auto-notify model about truncation."
                )
            )

        for tool_call in all_tool_calls:
            function = tool_call.get("function") or {}
            exposed_name = function.get("name") or tool_call.get("name") or ""
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            if not arguments:
                arguments = "{}"

            route = tool_registry.resolve(exposed_name)
            call_id = tool_call.get("id") or generate_item_id("call")
            output_index = next_output_index
            next_output_index += 1

            if route.is_custom:
                # Freeform tool: the client dispatches strictly on the
                # ``custom_tool_call`` item type and expects raw text in
                # ``input``, so a ``function_call`` would be rejected.
                custom_input = extract_custom_tool_input(arguments)
                item_id = generate_item_id("ctc")

                logger.debug(
                    f"Emitting custom_tool_call '{route.client_name}' "
                    f"(namespace={route.namespace}, call_id={call_id}, "
                    f"input_length={len(custom_input)})"
                )

                yield format_responses_sse(
                    {
                        "type": "response.output_item.added",
                        "sequence_number": counter.next(),
                        "output_index": output_index,
                        "item": ResponsesCustomToolCallItem(
                            id=item_id,
                            status="in_progress",
                            name=route.client_name,
                            namespace=route.namespace,
                            input="",
                            call_id=call_id,
                        ).model_dump(),
                    }
                )
                yield format_responses_sse(
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "sequence_number": counter.next(),
                        "item_id": item_id,
                        "output_index": output_index,
                        "call_id": call_id,
                        "delta": custom_input,
                    }
                )
                yield format_responses_sse(
                    {
                        "type": "response.custom_tool_call_input.done",
                        "sequence_number": counter.next(),
                        "item_id": item_id,
                        "output_index": output_index,
                        "call_id": call_id,
                        "input": custom_input,
                    }
                )
                custom_done_item = ResponsesCustomToolCallItem(
                    id=item_id,
                    status="completed",
                    name=route.client_name,
                    namespace=route.namespace,
                    input=custom_input,
                    call_id=call_id,
                )
                yield format_responses_sse(
                    {
                        "type": "response.output_item.done",
                        "sequence_number": counter.next(),
                        "output_index": output_index,
                        "item": custom_done_item.model_dump(),
                    }
                )
                output_items.append(custom_done_item.model_dump())
                continue

            item_id = generate_item_id("fc")

            logger.debug(
                f"Emitting function_call '{route.client_name}' "
                f"(namespace={route.namespace}, call_id={call_id}, "
                f"args_length={len(arguments)})"
            )

            yield format_responses_sse(
                {
                    "type": "response.output_item.added",
                    "sequence_number": counter.next(),
                    "output_index": output_index,
                    "item": ResponsesFunctionCallItem(
                        id=item_id,
                        status="in_progress",
                        name=route.client_name,
                        namespace=route.namespace,
                        arguments="",
                        call_id=call_id,
                    ).model_dump(),
                }
            )
            yield format_responses_sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": counter.next(),
                    "item_id": item_id,
                    "output_index": output_index,
                    "call_id": call_id,
                    "delta": arguments,
                }
            )
            yield format_responses_sse(
                {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": counter.next(),
                    "item_id": item_id,
                    "output_index": output_index,
                    "call_id": call_id,
                    "arguments": arguments,
                }
            )
            done_item = ResponsesFunctionCallItem(
                id=item_id,
                status="completed",
                name=route.client_name,
                namespace=route.namespace,
                arguments=arguments,
                call_id=call_id,
            )
            yield format_responses_sse(
                {
                    "type": "response.output_item.done",
                    "sequence_number": counter.next(),
                    "output_index": output_index,
                    "item": done_item.model_dump(),
                }
            )
            output_items.append(done_item.model_dump())

        save_truncation_state(all_tool_calls, full_content, content_was_truncated)

        # ------------------------------------------------------------------
        # Usage and terminal event
        # ------------------------------------------------------------------
        completion_tokens = count_tokens(full_content + full_thinking_content)
        reasoning_tokens = count_tokens(full_thinking_content) if full_thinking_content else 0

        prompt_tokens, total_tokens, prompt_source, total_source = (
            calculate_tokens_from_context_usage(
                context_usage_percentage, completion_tokens, model_cache, model
            )
        )

        if prompt_source == "unknown" and request_messages:
            prompt_tokens = count_message_tokens(request_messages, apply_claude_correction=False)
            if request_tools:
                prompt_tokens += count_tools_tokens(request_tools, apply_claude_correction=False)
            total_tokens = prompt_tokens + completion_tokens
            prompt_source = "tiktoken"
            total_source = "tiktoken"

        usage = build_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            metering_data=metering_data,
        )

        logger.debug(
            f"[Usage] {model}: input_tokens={prompt_tokens} ({prompt_source}), "
            f"output_tokens={completion_tokens} (tiktoken), "
            f"reasoning_tokens={reasoning_tokens}, "
            f"total_tokens={total_tokens} ({total_source})"
        )

        # Truncated output is reported through the response ``status`` and
        # ``incomplete_details`` rather than a ``response.incomplete`` event:
        # clients treat that event as a hard failure, while the gateway's
        # truncation-recovery system already notifies the model on the next
        # request. The fields stay present so nothing is hidden.
        final_status = "incomplete" if content_was_truncated else "completed"
        incomplete_details = (
            ResponsesIncompleteDetails(reason="max_output_tokens").model_dump()
            if content_was_truncated
            else None
        )

        yield format_responses_sse(
            {
                "type": "response.completed",
                "sequence_number": counter.next(),
                "response": build_envelope(
                    final_status,
                    output=output_items,
                    usage=usage.model_dump(),
                    incomplete_details=incomplete_details,
                ),
            }
        )

    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        logger.debug("Client disconnected (GeneratorExit)")
        streaming_error_occurred = True
    except Exception as e:
        streaming_error_occurred = True
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "(empty message)"
        logger.error(
            f"Error during Responses streaming: [{error_type}] {error_msg}", exc_info=True
        )
        raise
    finally:
        try:
            await response.aclose()
        except (httpx.HTTPError, RuntimeError) as close_error:
            logger.debug(f"Error closing response: {close_error}")

        if streaming_error_occurred:
            logger.debug("Responses streaming completed with error")
        else:
            logger.debug("Responses streaming completed successfully")


# ==================================================================================================
# Built-in tool emulation
# ==================================================================================================

async def run_emulated_web_search(
    tool: Dict[str, Any],
    auth_manager: "KiroAuthManager",
) -> Optional[str]:
    """
    Service a ``web_search`` tool call through the Kiro MCP endpoint.

    Mirrors the Chat Completions surface: the gateway performs the search itself
    and streams a human-readable summary as assistant text, so the client never
    has to implement the built-in tool.

    Args:
        tool: Tool call payload emitted by the Kiro stream.
        auth_manager: Authentication manager used for the MCP call.

    Returns:
        The search summary, or ``None`` when the call could not be serviced. The
        caller then falls back to reporting a normal ``function_call`` item.
    """
    from kiro.mcp_tools import call_kiro_mcp_api, generate_search_summary

    logger.info("Intercepted web_search tool call (Responses API, MCP emulation)")

    function = tool.get("function") or {}
    tool_input = function.get("arguments") or tool.get("input") or {}
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            logger.warning("web_search arguments were not valid JSON, skipping MCP call")
            return None

    query = tool_input.get("query", "") if isinstance(tool_input, dict) else ""
    if not query:
        logger.warning("web_search called without a query, skipping MCP call")
        return None

    logger.debug(f"WebSearch query (Responses API): {query}")
    _mcp_tool_use_id, results = await call_kiro_mcp_api(query, auth_manager)

    if results is None:
        logger.error("MCP API call failed for web_search")
        return None

    return generate_search_summary(query, results)


def save_truncation_state(
    tool_calls: List[Dict[str, Any]],
    full_content: str,
    content_was_truncated: bool,
) -> None:
    """
    Record truncation information for the recovery system.

    Identical behaviour to the Chat Completions surface: truncated tool calls are
    tracked by ``tool_call_id`` and truncated content by content hash, so the
    model is told about the truncation on the next request.

    Args:
        tool_calls: Tool calls extracted from the stream.
        full_content: Full assistant text produced by the stream.
        content_was_truncated: Whether the stream ended without completion
            signals.
    """
    from kiro.truncation_recovery import should_inject_recovery
    from kiro.truncation_state import save_content_truncation, save_tool_truncation

    if not should_inject_recovery():
        return

    truncated_count = 0
    for tool_call in tool_calls:
        if tool_call.get("_truncation_detected"):
            save_tool_truncation(
                tool_call_id=tool_call["id"],
                tool_name=(tool_call.get("function") or {}).get("name", ""),
                truncation_info=tool_call["_truncation_info"],
            )
            truncated_count += 1

    if content_was_truncated:
        save_content_truncation(full_content)

    if truncated_count > 0 or content_was_truncated:
        logger.info(
            f"Truncation detected: {truncated_count} tool(s), content={content_was_truncated}. "
            f"Will be handled when client sends next request."
        )


# ==================================================================================================
# Retry wrapper
# ==================================================================================================

async def stream_responses_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    client: httpx.AsyncClient,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    tool_registry: ToolRegistry,
    response_id: str,
    initial_response: Optional[httpx.Response] = None,
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[List[Dict[str, Any]]] = None,
    request_tools: Optional[List[Dict[str, Any]]] = None,
    request_echo: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream Responses SSE with automatic retry on first-token timeout.

    Thin adapter over ``streaming_core.stream_with_first_token_retry`` so the
    Responses surface inherits the exact retry semantics used by the other two
    API surfaces.

    Args:
        make_request: Factory that opens a new upstream request.
        client: HTTP client.
        model: Model name.
        model_cache: Model cache.
        auth_manager: Authentication manager.
        tool_registry: Tool routing table.
        response_id: Pre-allocated response ID.
        initial_response: Pre-validated response reused on the first attempt.
        max_retries: Maximum number of attempts.
        first_token_timeout: First token wait timeout in seconds.
        request_messages: Original request messages for the tiktoken fallback.
        request_tools: Original request tools for the tiktoken fallback.
        request_echo: Request fields echoed onto the response object.

    Yields:
        SSE frames.

    Raises:
        HTTPException: On upstream HTTP errors, or 504 after every first-token
            attempt has timed out.
    """

    def create_http_error(status_code: int, error_text: str) -> HTTPException:
        """Build the HTTPException raised for an upstream HTTP error."""
        return HTTPException(
            status_code=status_code,
            detail=f"Upstream API error: {error_text}",
        )

    def create_timeout_error(retries: int, timeout: float) -> HTTPException:
        """Build the HTTPException raised when every attempt timed out."""
        return HTTPException(
            status_code=504,
            detail=(
                f"Model did not respond within {timeout}s after {retries} attempts. "
                f"Please try again."
            ),
        )

    async def stream_processor(upstream: httpx.Response) -> AsyncGenerator[str, None]:
        """Convert one upstream response into Responses SSE frames."""
        async for frame in stream_kiro_to_responses(
            client=client,
            response=upstream,
            model=model,
            model_cache=model_cache,
            auth_manager=auth_manager,
            tool_registry=tool_registry,
            response_id=response_id,
            first_token_timeout=first_token_timeout,
            request_messages=request_messages,
            request_tools=request_tools,
            request_echo=request_echo,
        ):
            yield frame

    async for frame in stream_with_first_token_retry_core(
        make_request=make_request,
        stream_processor=stream_processor,
        initial_response=initial_response,
        max_retries=max_retries,
        first_token_timeout=first_token_timeout,
        on_http_error=create_http_error,
        on_all_retries_failed=create_timeout_error,
    ):
        yield frame


# ==================================================================================================
# Non-streaming collection
# ==================================================================================================

async def collect_responses_response(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    tool_registry: ToolRegistry,
    response_id: str,
    request_messages: Optional[List[Dict[str, Any]]] = None,
    request_tools: Optional[List[Dict[str, Any]]] = None,
    request_echo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Collect a complete non-streaming Responses response.

    Consumes the same SSE generator used for streaming and keeps the payload of
    the terminal ``response.completed`` event, so streaming and non-streaming can
    never drift apart.

    Args:
        client: HTTP client.
        response: Upstream HTTP response carrying the Kiro stream.
        model: Model name.
        model_cache: Model cache.
        auth_manager: Authentication manager.
        tool_registry: Tool routing table.
        response_id: Pre-allocated response ID.
        request_messages: Original request messages for the tiktoken fallback.
        request_tools: Original request tools for the tiktoken fallback.
        request_echo: Request fields echoed onto the response object.

    Returns:
        The assembled Responses response object.

    Raises:
        HTTPException: 502/504 when the upstream connection breaks while the body
            is being read. ``request_with_retry`` has already returned HTTP 200 at
            that point, so the failure is classified here instead of surfacing as
            a generic 500 — which also lets the account system fail over.
    """
    final_response: Optional[Dict[str, Any]] = None
    failure_message: Optional[str] = None

    try:
        async for frame in stream_kiro_to_responses(
            client=client,
            response=response,
            model=model,
            model_cache=model_cache,
            auth_manager=auth_manager,
            tool_registry=tool_registry,
            response_id=response_id,
            request_messages=request_messages,
            request_tools=request_tools,
            request_echo=request_echo,
        ):
            for line in frame.splitlines():
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed Responses SSE frame while collecting")
                    continue
                event_type = event.get("type")
                if event_type == "response.completed":
                    final_response = event.get("response")
                elif event_type == "response.failed":
                    failure_message = (
                        (event.get("response") or {}).get("error", {}).get("message")
                        or "Upstream response failed"
                    )
    except httpx.RequestError as e:
        # Upstream closed the connection mid-response (for example
        # RemoteProtocolError: "incomplete chunked read"). Classify it so the
        # route returns 502/504, which the account system treats as recoverable.
        error_info = classify_network_error(e)
        logger.error(
            f"Upstream connection error during non-streaming collection (Responses): "
            f"{error_info.technical_details}"
        )
        raise HTTPException(
            status_code=error_info.suggested_http_code,
            detail=build_http_error_detail(error_info),
        )

    if final_response is None and failure_message:
        raise HTTPException(status_code=502, detail=failure_message)

    if final_response is None:
        logger.error(
            "Kiro stream ended without a completion event; returning an empty Responses object"
        )
        empty: Dict[str, Any] = {
            "id": response_id,
            "model": model,
            "usage": build_usage(0, 0, 0, 0).model_dump(),
            "incomplete_details": ResponsesIncompleteDetails(
                reason="upstream_closed_without_output"
            ).model_dump(),
        }
        empty.update(request_echo or {})
        empty["status"] = "incomplete"
        empty["output"] = []
        return ResponsesResponse(**empty).model_dump()

    return final_response
