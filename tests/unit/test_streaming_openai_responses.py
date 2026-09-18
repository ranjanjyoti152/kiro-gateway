# -*- coding: utf-8 -*-
"""
Unit tests for the OpenAI Responses API streaming layer (streaming_openai_responses.py).

Covers:
- SequenceCounter and SSE frame formatting
- Event ordering for text-only, reasoning and tool-call responses
- Strictly increasing sequence_number and consistent output_index / item_id /
  content_index
- Terminal response.completed carrying the assembled response and usage
- Non-streaming collection (full response shape, output items, usage, status)
- Failure modes: first-token timeout, mid-stream httpx.RemoteProtocolError,
  upstream non-200, client disconnect
- Edge cases: empty stream, truncation, unicode, unknown tool names
"""
import json

import httpx
import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch

from kiro.converters_openai_responses import ToolRegistry
from kiro.streaming_core import FirstTokenTimeoutError, KiroEvent
from kiro.streaming_openai_responses import (
    SequenceCounter,
    build_failed_event,
    build_usage,
    collect_responses_response,
    format_responses_sse,
    generate_item_id,
    generate_response_id,
    stream_kiro_to_responses,
    stream_responses_with_first_token_retry,
)


# =============================================================================
# Fixtures and helpers
# =============================================================================

@pytest.fixture
def mock_model_cache():
    """Mock ModelInfoCache with a known input limit."""
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


@pytest.fixture
def mock_auth_manager():
    """Mock KiroAuthManager."""
    return MagicMock()


@pytest.fixture
def mock_http_client():
    """Mock httpx.AsyncClient."""
    return AsyncMock()


@pytest.fixture
def mock_response():
    """Mock httpx.Response for the upstream Kiro stream."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


@pytest.fixture
def registry_with_namespaced_tool():
    """ToolRegistry holding one plain and one namespaced tool."""
    registry = ToolRegistry()
    registry.register("exec_command", None)
    registry.register("spawn_agent", "multi_agent_v1")
    return registry


@pytest.fixture
def registry_with_custom_tool():
    """
    ToolRegistry mirroring the real Codex code-mode declaration.

    ``exec`` is a freeform tool inside the ``functions`` namespace, ``wait`` is a
    plain function tool in the same namespace, so tests can prove the two are
    reported with different item types.
    """
    registry = ToolRegistry()
    registry.register("exec", "functions", is_custom=True)
    registry.register("wait", "functions")
    return registry


def kiro_events(*events):
    """Build a parse_kiro_stream replacement yielding the given KiroEvents."""
    async def _generator(*args, **kwargs):
        for event in events:
            yield event

    return _generator


async def collect_events(
    parse_stream,
    mock_http_client,
    mock_response,
    mock_model_cache,
    mock_auth_manager,
    tool_registry=None,
    bracket_tool_calls=None,
    **kwargs,
):
    """
    Run stream_kiro_to_responses and return the parsed event payloads.

    Args:
        parse_stream: Replacement for parse_kiro_stream.
        mock_http_client: Mock HTTP client.
        mock_response: Mock upstream response.
        mock_model_cache: Mock model cache.
        mock_auth_manager: Mock auth manager.
        tool_registry: Optional tool registry (empty when omitted).
        bracket_tool_calls: Value returned by parse_bracket_tool_calls.
        **kwargs: Extra keyword arguments for stream_kiro_to_responses.

    Returns:
        Tuple of (list of raw SSE frames, list of parsed event dicts).
    """
    frames = []
    with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse_stream):
        with patch(
            "kiro.streaming_openai_responses.parse_bracket_tool_calls",
            return_value=bracket_tool_calls or [],
        ):
            async for frame in stream_kiro_to_responses(
                client=mock_http_client,
                response=mock_response,
                model="claude-sonnet-4.5",
                model_cache=mock_model_cache,
                auth_manager=mock_auth_manager,
                tool_registry=tool_registry or ToolRegistry(),
                response_id="resp_test",
                **kwargs,
            ):
                frames.append(frame)

    events = []
    for frame in frames:
        for line in frame.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return frames, events


# =============================================================================
# Primitives
# =============================================================================

class TestSequenceCounter:
    """Tests for SequenceCounter."""

    def test_starts_at_zero_and_increments(self):
        """
        What it does: Hands out 0, 1, 2 ...
        Purpose: The Responses API requires strictly increasing numbers.
        """
        counter = SequenceCounter()

        assert [counter.next() for _ in range(3)] == [0, 1, 2]

    def test_current_peeks_without_consuming(self):
        """
        What it does: Exposes the next unissued number.
        Purpose: Error paths need a number without disturbing the stream.
        """
        counter = SequenceCounter()
        counter.next()

        assert counter.current == 1
        assert counter.current == 1


class TestFrameFormatting:
    """Tests for format_responses_sse() and the ID generators."""

    def test_frame_has_event_name_and_data(self):
        """
        What it does: Emits both the event name line and the data line.
        Purpose: Matches the official OpenAI wire format.
        """
        frame = format_responses_sse({"type": "response.created", "sequence_number": 0})

        assert frame.startswith("event: response.created\n")
        assert "\ndata: " in frame
        assert frame.endswith("\n\n")

    def test_unicode_is_not_escaped(self):
        """
        What it does: Writes non-ASCII deltas literally.
        Purpose: ensure_ascii=False keeps frames small and readable.
        """
        frame = format_responses_sse({"type": "response.output_text.delta", "delta": "Привет"})

        assert "Привет" in frame

    def test_response_id_prefix(self):
        """
        What it does: Prefixes response IDs with resp_.
        Purpose: Clients and logs rely on the prefix convention.
        """
        assert generate_response_id().startswith("resp_")

    def test_item_id_prefixes(self):
        """
        What it does: Prefixes item IDs with the given type prefix.
        Purpose: Makes streams readable when debugging.
        """
        assert generate_item_id("msg").startswith("msg_")
        assert generate_item_id("fc").startswith("fc_")

    def test_failed_event_carries_error_and_status(self):
        """
        What it does: Builds a response.failed frame.
        Purpose: A broken stream must terminate with a typed failure.
        """
        frame = build_failed_event(
            response_id="resp_1",
            model="m",
            created_at=123,
            sequence_number=7,
            message="boom",
        )
        payload = json.loads(frame.split("data: ", 1)[1])

        assert payload["type"] == "response.failed"
        assert payload["sequence_number"] == 7
        assert payload["response"]["status"] == "failed"
        assert payload["response"]["error"]["message"] == "boom"

    def test_usage_block_shape(self):
        """
        What it does: Builds the usage block.
        Purpose: All five OpenAI usage fields must be present.
        """
        usage = build_usage(10, 4, 3, 14, metering_data=1.5).model_dump()

        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 4
        assert usage["output_tokens_details"]["reasoning_tokens"] == 3
        assert usage["total_tokens"] == 14
        assert usage["credits_used"] == 1.5


# =============================================================================
# Text-only streaming
# =============================================================================

class TestTextStreaming:
    """Tests for text-only responses."""

    @pytest.mark.asyncio
    async def test_event_order_for_text_only_response(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Emits the documented event order for a plain answer.
        Purpose: Codex reports "OutputTextDelta without active item" if the item
                 envelope does not open before the first delta.
        """
        print("Setup: content events only...")
        parse = kiro_events(
            KiroEvent(type="content", content="Hel"),
            KiroEvent(type="content", content="lo"),
            KiroEvent(type="context_usage", context_usage_percentage=2.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]

    @pytest.mark.asyncio
    async def test_sequence_numbers_strictly_increase(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Checks sequence_number monotonicity across every event.
        Purpose: A repeated or missing number breaks strict SSE consumers.
        """
        parse = kiro_events(
            KiroEvent(type="thinking", thinking_content="why"),
            KiroEvent(type="content", content="because"),
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": "{}"},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        numbers = [event["sequence_number"] for event in events]
        assert numbers == list(range(len(numbers)))

    @pytest.mark.asyncio
    async def test_item_and_content_indices_are_consistent(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Checks every item-scoped event references the same item.
        Purpose: A mismatched item_id makes the client drop the delta.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="hi"),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        added = next(e for e in events if e["type"] == "response.output_item.added")
        item_id = added["item"]["id"]
        assert added["output_index"] == 0

        for event in events:
            if "item_id" in event:
                assert event["item_id"] == item_id
                assert event["output_index"] == 0
            if "content_index" in event:
                assert event["content_index"] == 0

    @pytest.mark.asyncio
    async def test_completed_event_carries_output_and_usage(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Checks the terminal event carries the assembled response.
        Purpose: Codex reads the final response id, output and token usage from
                 response.completed and nowhere else.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="Answer"),
            KiroEvent(type="context_usage", context_usage_percentage=10.0),
            KiroEvent(type="usage", usage=2.5),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        completed = events[-1]
        response = completed["response"]
        assert completed["type"] == "response.completed"
        assert response["id"] == "resp_test"
        assert response["status"] == "completed"
        assert response["output"][0]["content"][0]["text"] == "Answer"
        # 10% of 200000 = 20000 total tokens reported by Kiro
        assert response["usage"]["total_tokens"] == 20000
        assert response["usage"]["input_tokens"] == 20000 - response["usage"]["output_tokens"]
        assert response["usage"]["credits_used"] == 2.5

    @pytest.mark.asyncio
    async def test_output_text_done_matches_concatenated_deltas(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Compares output_text.done against the deltas.
        Purpose: Clients that rebuild text from deltas must agree with the final
                 text, otherwise the transcript diverges.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="one "),
            KiroEvent(type="content", content="two"),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        done = next(e for e in events if e["type"] == "response.output_text.done")
        item_done = next(
            e for e in events
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "message"
        )

        assert deltas == "one two"
        assert done["text"] == "one two"
        assert item_done["item"]["content"][0]["text"] == "one two"

    @pytest.mark.asyncio
    async def test_unicode_content_round_trips(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Streams non-ASCII content.
        Purpose: Multi-byte text must survive SSE encoding exactly.
        """
        text = "Привет 你好 🎉"
        parse = kiro_events(
            KiroEvent(type="content", content=text),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert next(
            e for e in events if e["type"] == "response.output_text.done"
        )["text"] == text

    @pytest.mark.asyncio
    async def test_empty_stream_still_completes(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Emits created/in_progress/completed for an empty stream.
        Purpose: A client waiting for response.completed must never hang.
        """
        _, events = await collect_events(
            kiro_events(), mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.completed",
        ]
        assert events[-1]["response"]["output"] == []
        assert events[-1]["response"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_request_echo_is_reflected_on_the_response(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Echoes request configuration onto every envelope.
        Purpose: store must always read False and previous_response_id None,
                 because the gateway keeps no server-side conversation.
        """
        parse = kiro_events(KiroEvent(type="content", content="x"), KiroEvent(type="usage", usage=1.0))

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            request_echo={
                "instructions": "SYS",
                "store": False,
                "previous_response_id": None,
                "metadata": {"k": "v"},
            },
        )

        for event in events:
            if "response" in event:
                assert event["response"]["instructions"] == "SYS"
                assert event["response"]["store"] is False
                assert event["response"]["previous_response_id"] is None
                assert event["response"]["metadata"] == {"k": "v"}


# =============================================================================
# Reasoning
# =============================================================================

class TestReasoningStreaming:
    """Tests for thinking content mapped to reasoning events."""

    @pytest.mark.asyncio
    async def test_reasoning_events_precede_the_message(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, monkeypatch
    ):
        """
        What it does: Emits the reasoning item, then the message item.
        Purpose: Reasoning must be closed before text starts, and it occupies
                 output_index 0.
        """
        monkeypatch.setattr(
            "kiro.streaming_openai_responses.FAKE_REASONING_HANDLING", "as_reasoning_content"
        )
        parse = kiro_events(
            KiroEvent(type="thinking", thinking_content="step 1 "),
            KiroEvent(type="thinking", thinking_content="step 2"),
            KiroEvent(type="content", content="Answer"),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]

        reasoning_added = events[2]
        assert reasoning_added["item"]["type"] == "reasoning"
        assert reasoning_added["output_index"] == 0

        message_added = events[9]
        assert message_added["item"]["type"] == "message"
        assert message_added["output_index"] == 1

    @pytest.mark.asyncio
    async def test_reasoning_summary_index_is_always_zero(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, monkeypatch
    ):
        """
        What it does: Uses summary_index 0 for every reasoning event.
        Purpose: Codex requires summary_index on reasoning deltas; the gateway
                 produces a single summary part per response.
        """
        monkeypatch.setattr(
            "kiro.streaming_openai_responses.FAKE_REASONING_HANDLING", "as_reasoning_content"
        )
        parse = kiro_events(
            KiroEvent(type="thinking", thinking_content="t"),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        reasoning_events = [e for e in events if "summary_index" in e]
        assert reasoning_events
        assert all(e["summary_index"] == 0 for e in reasoning_events)

    @pytest.mark.asyncio
    async def test_reasoning_item_reaches_the_completed_output(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, monkeypatch
    ):
        """
        What it does: Adds the reasoning item to the final output list.
        Purpose: The assembled response must describe everything that streamed.
        """
        monkeypatch.setattr(
            "kiro.streaming_openai_responses.FAKE_REASONING_HANDLING", "as_reasoning_content"
        )
        parse = kiro_events(
            KiroEvent(type="thinking", thinking_content="deep thought"),
            KiroEvent(type="content", content="42"),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        output = events[-1]["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", "message"]
        assert output[0]["summary"][0] == {"type": "summary_text", "text": "deep thought"}
        assert output[0]["encrypted_content"] is None
        assert events[-1]["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] > 0

    @pytest.mark.asyncio
    async def test_thinking_is_inlined_as_text_in_other_modes(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, monkeypatch
    ):
        """
        What it does: Streams thinking as ordinary text when the handling mode is
                      not as_reasoning_content.
        Purpose: Matches the Chat Completions surface, where the same setting
                 inlines thinking into the visible answer.
        """
        monkeypatch.setattr("kiro.streaming_openai_responses.FAKE_REASONING_HANDLING", "pass")
        parse = kiro_events(
            KiroEvent(type="thinking", thinking_content="[think] "),
            KiroEvent(type="content", content="Answer"),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert not any("reasoning" in event["type"] for event in events)
        done = next(e for e in events if e["type"] == "response.output_text.done")
        assert done["text"] == "[think] Answer"
        assert [item["type"] for item in events[-1]["response"]["output"]] == ["message"]


# =============================================================================
# Tool calls
# =============================================================================

class TestToolCallStreaming:
    """Tests for function_call output items."""

    @pytest.mark.asyncio
    async def test_tool_call_event_order_and_payload(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_namespaced_tool,
    ):
        """
        What it does: Emits added / arguments.delta / arguments.done / done.
        Purpose: Codex executes the tool from response.output_item.done, and the
                 argument events drive its live diff view.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "tooluse_ABC",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_namespaced_tool,
        )

        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]

        added, delta, done, item_done = events[2], events[3], events[4], events[5]
        assert added["item"]["type"] == "function_call"
        assert added["item"]["name"] == "exec_command"
        assert added["item"]["call_id"] == "tooluse_ABC"
        assert added["item"]["arguments"] == ""
        assert delta["delta"] == '{"cmd":"ls"}'
        assert done["arguments"] == '{"cmd":"ls"}'
        assert item_done["item"]["arguments"] == '{"cmd":"ls"}'
        assert item_done["item"]["status"] == "completed"
        assert item_done["item"]["id"] == added["item"]["id"]

    @pytest.mark.asyncio
    async def test_namespaced_tool_call_is_routed_back(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_namespaced_tool,
    ):
        """
        What it does: Restores the client's name and namespace.
        Purpose: The model answers with the flattened Kiro name; the client only
                 knows name+namespace and would not recognize the flat name.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "multi_agent_v1__spawn_agent", "arguments": "{}"},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_namespaced_tool,
        )

        item = events[-1]["response"]["output"][0]
        assert item["name"] == "spawn_agent"
        assert item["namespace"] == "multi_agent_v1"

    @pytest.mark.asyncio
    async def test_unknown_tool_name_passes_through(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Reports an undeclared tool name unchanged.
        Purpose: The client must be able to reject it; silently rewriting it
                 would hide the model's mistake.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={"id": "t1", "type": "function",
                          "function": {"name": "invented", "arguments": "{}"}},
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert events[-1]["response"]["output"][0]["name"] == "invented"

    @pytest.mark.asyncio
    async def test_tool_calls_follow_the_message_item(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Places function_call items after the assistant message.
        Purpose: Kiro reports tool calls only at the end of the stream, so the
                 output indices must reflect that order.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="Running it."),
            KiroEvent(
                type="tool_use",
                tool_use={"id": "t1", "type": "function",
                          "function": {"name": "run", "arguments": "{}"}},
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        output = events[-1]["response"]["output"]
        assert [item["type"] for item in output] == ["message", "function_call"]

        added_events = [e for e in events if e["type"] == "response.output_item.added"]
        assert [e["output_index"] for e in added_events] == [0, 1]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_get_distinct_indices(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Assigns a unique output_index and item_id per tool call.
        Purpose: Parallel tool calls must not overwrite each other.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={"id": "t1", "type": "function",
                          "function": {"name": "a", "arguments": "{}"}},
            ),
            KiroEvent(
                type="tool_use",
                tool_use={"id": "t2", "type": "function",
                          "function": {"name": "b", "arguments": "{}"}},
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        added = [e for e in events if e["type"] == "response.output_item.added"]
        assert [e["output_index"] for e in added] == [0, 1]
        assert added[0]["item"]["id"] != added[1]["item"]["id"]
        assert len({e["item"]["call_id"] for e in added}) == 2

    @pytest.mark.asyncio
    async def test_missing_arguments_become_an_empty_object(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Substitutes "{}" for absent tool arguments.
        Purpose: Clients parse 'arguments' as JSON and would crash on "".
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={"id": "t1", "type": "function", "function": {"name": "a"}},
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert events[-1]["response"]["output"][0]["arguments"] == "{}"

    @pytest.mark.asyncio
    async def test_dict_arguments_are_json_encoded(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Serializes dict arguments to a JSON string.
        Purpose: The Responses API always transports arguments as a string.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={"id": "t1", "type": "function",
                          "function": {"name": "a", "arguments": {"x": 1}}},
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert json.loads(events[-1]["response"]["output"][0]["arguments"]) == {"x": 1}

    @pytest.mark.asyncio
    async def test_tool_call_without_id_gets_a_generated_call_id(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Generates a call_id when Kiro provides none.
        Purpose: The client cannot correlate a tool result without one.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={"type": "function", "function": {"name": "a", "arguments": "{}"}},
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert events[-1]["response"]["output"][0]["call_id"].startswith("call_")

    @pytest.mark.asyncio
    async def test_bracket_style_tool_calls_are_recovered(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Emits tool calls recovered from bracket-format content.
        Purpose: Kiro sometimes returns tool calls inside the text body; the
                 shared parser recovers them and this surface must use it.
        """
        parse = kiro_events(
            KiroEvent(type="content", content='[{"name":"run","arguments":{}}]'),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            bracket_tool_calls=[
                {"id": "b1", "type": "function",
                 "function": {"name": "run", "arguments": "{}"}}
            ],
        )

        output_types = [item["type"] for item in events[-1]["response"]["output"]]
        assert "function_call" in output_types


# =============================================================================
# Web search emulation
# =============================================================================

class TestWebSearchEmulation:
    """Tests for the built-in web_search emulation."""

    @pytest.mark.asyncio
    async def test_web_search_result_is_streamed_as_text(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Streams the MCP search summary as assistant text.
        Purpose: The client never has to implement the built-in tool, matching
                 the Chat Completions surface.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"kiro"}'},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        with patch(
            "kiro.streaming_openai_responses.run_emulated_web_search",
            new=AsyncMock(return_value="SEARCH SUMMARY"),
        ):
            _, events = await collect_events(
                parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
            )

        assert not any(e["type"].startswith("response.function_call") for e in events)
        done = next(e for e in events if e["type"] == "response.output_text.done")
        assert done["text"] == "SEARCH SUMMARY"

    @pytest.mark.asyncio
    async def test_failed_web_search_falls_back_to_a_tool_call(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Reports a normal function_call when the MCP call fails.
        Purpose: Never swallow the model's intent; let the client decide.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"kiro"}'},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        with patch(
            "kiro.streaming_openai_responses.run_emulated_web_search",
            new=AsyncMock(return_value=None),
        ):
            _, events = await collect_events(
                parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
            )

        assert events[-1]["response"]["output"][0]["name"] == "web_search"


# =============================================================================
# Truncation
# =============================================================================

class TestTruncationHandling:
    """Tests for streams that end without Kiro completion signals."""

    @pytest.mark.asyncio
    async def test_missing_completion_signal_marks_the_response_incomplete(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Marks a truncated response incomplete but still completes.
        Purpose: response.incomplete is treated as a hard failure by clients, so
                 the truncation is reported through status + incomplete_details
                 while the stream terminates normally.
        """
        parse = kiro_events(KiroEvent(type="content", content="half an ans"))

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        completed = events[-1]
        assert completed["type"] == "response.completed"
        assert completed["response"]["status"] == "incomplete"
        assert completed["response"]["incomplete_details"] == {"reason": "max_output_tokens"}

    @pytest.mark.asyncio
    async def test_tool_call_without_usage_is_not_treated_as_truncation(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Keeps status 'completed' when the stream ended on a tool call.
        Purpose: A tool call is a legitimate stopping point, not truncation.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="calling"),
            KiroEvent(
                type="tool_use",
                tool_use={"id": "t1", "type": "function",
                          "function": {"name": "a", "arguments": "{}"}},
            ),
        )

        _, events = await collect_events(
            parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
        )

        assert events[-1]["response"]["status"] == "completed"
        assert events[-1]["response"]["incomplete_details"] is None


# =============================================================================
# Usage fallback
# =============================================================================

class TestUsageFallback:
    """Tests for token counting when Kiro reports no context usage."""

    @pytest.mark.asyncio
    async def test_tiktoken_fallback_counts_request_messages(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Counts prompt tokens locally when Kiro sends no percentage.
        Purpose: Clients that budget context need a non-zero input_tokens.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="short"),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            request_messages=[{"role": "user", "content": "a much longer prompt " * 20}],
            request_tools=[{"name": "run", "description": "d", "input_schema": {}}],
        )

        usage = events[-1]["response"]["usage"]
        assert usage["input_tokens"] > 0
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]

    @pytest.mark.asyncio
    async def test_context_usage_percentage_wins_over_tiktoken(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Prefers Kiro's context usage percentage.
        Purpose: Upstream data is more accurate than a local estimate.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="x"),
            KiroEvent(type="context_usage", context_usage_percentage=50.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            request_messages=[{"role": "user", "content": "tiny"}],
        )

        assert events[-1]["response"]["usage"]["total_tokens"] == 100000


# =============================================================================
# Failure modes
# =============================================================================

class TestStreamingFailureModes:
    """Tests for streaming error paths."""

    @pytest.mark.asyncio
    async def test_first_token_timeout_propagates(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Lets FirstTokenTimeoutError escape the generator.
        Purpose: The shared retry wrapper must see it to open a new request.
        """
        async def parse(*args, **kwargs):
            raise FirstTokenTimeoutError("no response within 15 seconds")
            yield  # pragma: no cover - makes this an async generator

        with pytest.raises(FirstTokenTimeoutError):
            await collect_events(
                parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
            )

    @pytest.mark.asyncio
    async def test_mid_stream_protocol_error_propagates(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Lets a mid-stream RemoteProtocolError escape.
        Purpose: The route classifies it; swallowing it would leave the client
                 with a silently truncated answer.
        """
        async def parse(*args, **kwargs):
            yield KiroEvent(type="content", content="partial")
            raise httpx.RemoteProtocolError("incomplete chunked read")

        with pytest.raises(httpx.RemoteProtocolError):
            await collect_events(
                parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
            )

    @pytest.mark.asyncio
    async def test_upstream_response_is_always_closed(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Closes the upstream response after an error.
        Purpose: Leaked connections show up as CLOSE_WAIT sockets.
        """
        async def parse(*args, **kwargs):
            yield KiroEvent(type="content", content="partial")
            raise httpx.RemoteProtocolError("boom")

        with pytest.raises(httpx.RemoteProtocolError):
            await collect_events(
                parse, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
            )

        mock_response.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_client_disconnect_is_not_an_error(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Treats GeneratorExit as a normal disconnect.
        Purpose: Aborted requests are routine and must not be logged as failures
                 or leak the upstream response.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="one"),
            KiroEvent(type="content", content="two"),
            KiroEvent(type="usage", usage=1.0),
        )

        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                generator = stream_kiro_to_responses(
                    client=mock_http_client,
                    response=mock_response,
                    model="m",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=ToolRegistry(),
                    response_id="resp_test",
                )
                await generator.__anext__()
                await generator.aclose()

        mock_response.aclose.assert_awaited()


class TestFirstTokenRetryWrapper:
    """Tests for stream_responses_with_first_token_retry()."""

    @pytest.mark.asyncio
    async def test_upstream_non_200_raises_http_exception(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Converts an upstream non-200 into an HTTPException.
        Purpose: The route turns it into an OpenAI-shaped error response.
        """
        error_response = AsyncMock()
        error_response.status_code = 429
        error_response.aread = AsyncMock(return_value=b'{"message":"slow down"}')
        error_response.aclose = AsyncMock()

        async def make_request():
            return error_response

        with pytest.raises(HTTPException) as exc_info:
            async for _ in stream_responses_with_first_token_retry(
                make_request=make_request,
                client=mock_http_client,
                model="m",
                model_cache=mock_model_cache,
                auth_manager=mock_auth_manager,
                tool_registry=ToolRegistry(),
                response_id="resp_test",
                initial_response=error_response,
            ):
                pass

        assert exc_info.value.status_code == 429
        assert "slow down" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_retries_after_first_token_timeout(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Opens a new upstream request after a first-token timeout.
        Purpose: The user should see a delay, not an error.
        """
        attempts = {"count": 0}

        async def parse(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise FirstTokenTimeoutError("timeout")
            yield KiroEvent(type="content", content="second try")
            yield KiroEvent(type="usage", usage=1.0)

        responses = []

        async def make_request():
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            responses.append(response)
            return response

        first = await make_request()

        frames = []
        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                async for frame in stream_responses_with_first_token_retry(
                    make_request=make_request,
                    client=mock_http_client,
                    model="m",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=ToolRegistry(),
                    response_id="resp_test",
                    initial_response=first,
                    max_retries=2,
                ):
                    frames.append(frame)

        assert attempts["count"] == 2
        assert any("second try" in frame for frame in frames)
        assert frames[-1].startswith("event: response.completed")

    @pytest.mark.asyncio
    async def test_all_attempts_timing_out_raises_504(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Raises HTTP 504 after every attempt timed out.
        Purpose: The client gets an actionable gateway timeout, not a hang.
        """
        async def parse(*args, **kwargs):
            raise FirstTokenTimeoutError("timeout")
            yield  # pragma: no cover

        async def make_request():
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response

        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with pytest.raises(HTTPException) as exc_info:
                async for _ in stream_responses_with_first_token_retry(
                    make_request=make_request,
                    client=mock_http_client,
                    model="m",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=ToolRegistry(),
                    response_id="resp_test",
                    max_retries=2,
                ):
                    pass

        assert exc_info.value.status_code == 504
        assert "did not respond" in exc_info.value.detail


# =============================================================================
# Non-streaming collection
# =============================================================================

class TestCollectResponsesResponse:
    """Tests for collect_responses_response()."""

    @pytest.mark.asyncio
    async def test_returns_the_assembled_response_object(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Returns the payload of the terminal completed event.
        Purpose: Streaming and non-streaming must never drift apart.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="Hello"),
            KiroEvent(type="context_usage", context_usage_percentage=5.0),
            KiroEvent(type="usage", usage=1.25),
        )

        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                result = await collect_responses_response(
                    client=mock_http_client,
                    response=mock_response,
                    model="claude-sonnet-4.5",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=ToolRegistry(),
                    response_id="resp_nonstream",
                )

        assert result["id"] == "resp_nonstream"
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["model"] == "claude-sonnet-4.5"
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["content"][0]["text"] == "Hello"
        assert result["usage"]["total_tokens"] == 10000
        assert result["usage"]["credits_used"] == 1.25
        assert result["store"] is False

    @pytest.mark.asyncio
    async def test_tool_call_appears_in_the_output_list(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_namespaced_tool,
    ):
        """
        What it does: Returns function_call items in the non-streaming body.
        Purpose: stream:false with tools is a supported combination.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="Working"),
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "tooluse_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                result = await collect_responses_response(
                    client=mock_http_client,
                    response=mock_response,
                    model="m",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=registry_with_namespaced_tool,
                    response_id="resp_1",
                )

        assert [item["type"] for item in result["output"]] == ["message", "function_call"]
        assert result["output"][1]["call_id"] == "tooluse_1"
        assert result["output"][1]["arguments"] == '{"cmd":"ls"}'

    @pytest.mark.asyncio
    async def test_truncated_stream_reports_incomplete_status(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Returns status 'incomplete' for a truncated stream.
        Purpose: The client must be able to tell a full answer from a cut one.
        """
        parse = kiro_events(KiroEvent(type="content", content="cut off"))

        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                result = await collect_responses_response(
                    client=mock_http_client,
                    response=mock_response,
                    model="m",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=ToolRegistry(),
                    response_id="resp_1",
                )

        assert result["status"] == "incomplete"
        assert result["incomplete_details"]["reason"] == "max_output_tokens"

    @pytest.mark.asyncio
    async def test_empty_stream_returns_an_empty_completed_object(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Returns an empty but well-formed response object.
        Purpose: A silent upstream must not produce a malformed body.
        """
        with patch("kiro.streaming_openai_responses.parse_kiro_stream", kiro_events()):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                result = await collect_responses_response(
                    client=mock_http_client,
                    response=mock_response,
                    model="m",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=ToolRegistry(),
                    response_id="resp_1",
                )

        assert result["status"] == "completed"
        assert result["output"] == []
        assert result["usage"]["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_mid_stream_protocol_error_becomes_a_classified_502(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Converts a mid-body httpx.RemoteProtocolError into 502/504.
        Purpose: request_with_retry already returned HTTP 200, so this failure
                 must be classified here; a generic 500 would also stop the
                 account system from failing over.
        """
        async def parse(*args, **kwargs):
            yield KiroEvent(type="content", content="partial")
            raise httpx.RemoteProtocolError("incomplete chunked read")

        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await collect_responses_response(
                        client=mock_http_client,
                        response=mock_response,
                        model="m",
                        model_cache=mock_model_cache,
                        auth_manager=mock_auth_manager,
                        tool_registry=ToolRegistry(),
                        response_id="resp_1",
                    )

        assert exc_info.value.status_code in (502, 504)
        assert exc_info.value.detail

    @pytest.mark.asyncio
    async def test_request_echo_is_present_on_the_body(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: Reflects echoed request fields on the non-streaming body.
        Purpose: Parity with the streaming envelopes.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="hi"),
            KiroEvent(type="usage", usage=1.0),
        )

        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                result = await collect_responses_response(
                    client=mock_http_client,
                    response=mock_response,
                    model="m",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=ToolRegistry(),
                    response_id="resp_1",
                    request_echo={"instructions": "SYS", "metadata": {"a": 1}},
                )

        assert result["instructions"] == "SYS"
        assert result["metadata"] == {"a": 1}


# =============================================================================
# Freeform ("custom") tool calls
# =============================================================================

class TestCustomToolCallStreaming:
    """Tests for custom_tool_call output items (freeform tools)."""

    @pytest.mark.asyncio
    async def test_custom_tool_call_event_order_and_payload(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_custom_tool,
    ):
        """
        What it does: Emits added / input.delta / input.done / done.
        Purpose: Codex dispatches freeform tools strictly on the
                 custom_tool_call item type and reads the payload from ``input``;
                 a function_call for the same tool is rejected with "expects raw
                 JavaScript source text".
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "call_ABC",
                    "type": "function",
                    "function": {
                        "name": "functions__exec",
                        "arguments": '{"input": "text(1);"}',
                    },
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_custom_tool,
        )

        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.output_item.done",
            "response.completed",
        ]
        added, delta, done, item_done = events[2], events[3], events[4], events[5]
        assert added["item"]["type"] == "custom_tool_call"
        assert added["item"]["name"] == "exec"
        assert added["item"]["namespace"] == "functions"
        assert added["item"]["call_id"] == "call_ABC"
        assert added["item"]["input"] == ""
        assert "arguments" not in added["item"]
        assert delta["delta"] == "text(1);"
        assert delta["item_id"] == added["item"]["id"]
        assert delta["call_id"] == "call_ABC"
        assert done["input"] == "text(1);"
        assert item_done["item"]["input"] == "text(1);"
        assert item_done["item"]["status"] == "completed"
        assert item_done["item"]["id"] == added["item"]["id"]

    @pytest.mark.asyncio
    async def test_custom_tool_call_lands_in_the_final_output(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_custom_tool,
    ):
        """
        What it does: Includes the custom_tool_call in response.completed.
        Purpose: Clients that read the terminal envelope instead of the item
                 events must still see the call.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="Running it."),
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "functions__exec", "arguments": '{"input": "x"}'},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_custom_tool,
        )

        output = events[-1]["response"]["output"]
        assert [item["type"] for item in output] == ["message", "custom_tool_call"]
        assert output[1]["input"] == "x"

    @pytest.mark.asyncio
    async def test_function_and_custom_tools_use_their_own_item_types(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_custom_tool,
    ):
        """
        What it does: Streams one freeform and one normal tool call.
        Purpose: The item type must follow the declaration, not the request.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "functions__exec", "arguments": '{"input": "js"}'},
                },
            ),
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "f1",
                    "type": "function",
                    "function": {
                        "name": "functions__wait",
                        "arguments": '{"cell_id": "1"}',
                    },
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_custom_tool,
        )

        output = events[-1]["response"]["output"]
        assert [item["type"] for item in output] == ["custom_tool_call", "function_call"]
        assert output[0]["name"] == "exec"
        assert output[1]["name"] == "wait"
        assert output[1]["arguments"] == '{"cell_id": "1"}'

    @pytest.mark.asyncio
    async def test_custom_tool_call_output_index_and_sequence_stay_consistent(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_custom_tool,
    ):
        """
        What it does: Checks sequence_number and output_index across the burst.
        Purpose: Codex rejects an out-of-order stream, and a repeated
                 output_index would overwrite the previous item.
        """
        parse = kiro_events(
            KiroEvent(type="content", content="hi"),
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "functions__exec", "arguments": '{"input": "a"}'},
                },
            ),
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "functions__exec", "arguments": '{"input": "b"}'},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_custom_tool,
        )

        numbers = [event["sequence_number"] for event in events]
        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers))

        added = [
            event
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "custom_tool_call"
        ]
        assert [event["output_index"] for event in added] == [1, 2]
        assert added[0]["item"]["id"] != added[1]["item"]["id"]

    @pytest.mark.asyncio
    async def test_raw_text_arguments_are_forwarded_verbatim(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
    ):
        """
        What it does: Streams a freeform call whose arguments are not JSON.
        Purpose: For a freeform tool the raw text is the payload, so it must
                 reach the client unchanged rather than being dropped.
        """
        registry = ToolRegistry()
        registry.register("apply_patch", None, is_custom=True)
        patch_text = "*** Begin Patch\n*** Add File: a.txt\n+1\n*** End Patch"

        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "apply_patch", "arguments": patch_text},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry,
        )

        assert events[-1]["response"]["output"][0]["input"] == patch_text

    @pytest.mark.asyncio
    async def test_empty_arguments_produce_an_empty_input(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_custom_tool,
    ):
        """
        What it does: Streams a freeform call with no arguments at all.
        Purpose: The item must still be emitted so the client can answer, and
                 ``input`` must be a string rather than ``None``.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "functions__exec", "arguments": ""},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_custom_tool,
        )

        item = events[-1]["response"]["output"][0]
        assert item["type"] == "custom_tool_call"
        assert item["input"] == ""

    @pytest.mark.asyncio
    async def test_unicode_freeform_payload_survives(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_custom_tool,
    ):
        """
        What it does: Streams a freeform payload containing non-ASCII text.
        Purpose: JSON round-tripping must not mangle the source the client runs.
        """
        source = 'text("Привет 🌍");'
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "functions__exec",
                        "arguments": json.dumps({"input": source}),
                    },
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_custom_tool,
        )

        assert events[-1]["response"]["output"][0]["input"] == source

    @pytest.mark.asyncio
    async def test_undeclared_tool_falls_back_to_function_call(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_custom_tool,
    ):
        """
        What it does: Streams a call to a name the client never declared.
        Purpose: The gateway cannot know it is freeform, so it stays a
                 function_call and the client rejects it with its own error.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "invented_tool", "arguments": "{}"},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        _, events = await collect_events(
            parse,
            mock_http_client,
            mock_response,
            mock_model_cache,
            mock_auth_manager,
            tool_registry=registry_with_custom_tool,
        )

        assert events[-1]["response"]["output"][0]["type"] == "function_call"

    @pytest.mark.asyncio
    async def test_non_streaming_collection_returns_custom_tool_calls(
        self,
        mock_http_client,
        mock_response,
        mock_model_cache,
        mock_auth_manager,
        registry_with_custom_tool,
    ):
        """
        What it does: Collects a non-streaming body containing a freeform call.
        Purpose: Streaming and non-streaming must never drift apart, so the
                 non-streaming path needs the same item type.
        """
        parse = kiro_events(
            KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "functions__exec", "arguments": '{"input": "y"}'},
                },
            ),
            KiroEvent(type="usage", usage=1.0),
        )

        with patch("kiro.streaming_openai_responses.parse_kiro_stream", parse):
            with patch(
                "kiro.streaming_openai_responses.parse_bracket_tool_calls", return_value=[]
            ):
                result = await collect_responses_response(
                    client=mock_http_client,
                    response=mock_response,
                    model="gpt-5.6-terra",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    tool_registry=registry_with_custom_tool,
                    response_id="resp_test",
                )

        assert [item["type"] for item in result["output"]] == ["custom_tool_call"]
        assert result["output"][0]["name"] == "exec"
        assert result["output"][0]["namespace"] == "functions"
        assert result["output"][0]["input"] == "y"
