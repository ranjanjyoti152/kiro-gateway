# -*- coding: utf-8 -*-
"""
Unit tests for the OpenAI Responses API converter (converters_openai_responses.py).

Covers:
- Tool flattening: flat function tools, namespace containers, built-in web_search,
  unsupported tool types, name collisions and over-long namespaced names
- Input flattening: instructions folded into the system prompt, multi-turn order,
  images, tool round-trip via call_id, unsupported item types
- Codex CLI code mode: tools declared inside an ``additional_tools`` input item,
  freeform ``{"type": "custom"}`` tools, and the ``custom_tool_call`` /
  ``custom_tool_call_output`` round trip
- Thinking configuration derived from reasoning.effort
- Unsupported field handling: previous_response_id rejected, store:true warned
- build_kiro_payload_responses end to end, including the real Codex CLI payload
"""
import json

import pytest

from codex_responses_payload import (
    CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST,
    CODEX_RESPONSES_CODE_MODE_REQUEST,
    CODEX_RESPONSES_FOLLOW_UP_REQUEST,
    CODEX_RESPONSES_REQUEST,
)
from kiro.converters_core import ThinkingConfig
from kiro.converters_openai_responses import (
    CUSTOM_TOOL_INPUT_PROPERTY,
    EmptyResponsesInputError,
    ToolRegistry,
    UnsupportedResponsesFeatureError,
    build_custom_tool_description,
    build_custom_tool_schema,
    build_kiro_payload_responses,
    convert_responses_input_to_unified,
    convert_responses_tools_to_unified,
    extract_additional_tools,
    extract_custom_tool_input,
    extract_function_call_output,
    extract_images_from_parts,
    extract_text_from_parts,
    extract_thinking_config_from_responses,
    log_ignored_responses_fields,
    validate_responses_request,
)
from kiro.models_openai_responses import ResponsesRequest, ResponsesTool
from kiro.tool_sanitizer import ToolSpecValidationError

VALID_PROFILE_ARN = "arn:aws:codewhisperer:us-east-1:123456789012:profile/TESTPROFILE"


def make_request(**kwargs) -> ResponsesRequest:
    """Build a ResponsesRequest with a default model."""
    kwargs.setdefault("model", "claude-sonnet-4.5")
    return ResponsesRequest(**kwargs)


# =============================================================================
# Content extraction helpers
# =============================================================================

class TestContentExtraction:
    """Tests for the low-level content extraction helpers."""

    def test_string_content_passes_through(self):
        """
        What it does: Returns a plain string unchanged.
        Purpose: 'content' may be a bare string.
        """
        assert extract_text_from_parts("hello") == "hello"

    def test_none_content_becomes_empty_string(self):
        """
        What it does: Maps None content to "".
        Purpose: Missing content must never raise.
        """
        assert extract_text_from_parts(None) == ""

    def test_multiple_text_parts_are_concatenated(self):
        """
        What it does: Joins consecutive text parts.
        Purpose: Multi-part prompts must not lose text.
        """
        parts = [
            {"type": "input_text", "text": "a"},
            {"type": "output_text", "text": "b"},
            {"type": "summary_text", "text": "c"},
        ]
        assert extract_text_from_parts(parts) == "abc"

    def test_image_and_audio_parts_contribute_no_text(self):
        """
        What it does: Ignores non-text parts when extracting text.
        Purpose: Binary payloads must not leak into the prompt text.
        """
        parts = [
            {"type": "input_image", "image_url": "data:image/png;base64,AA"},
            {"type": "input_audio", "audio_url": "data:audio/wav;base64,BB"},
            {"type": "input_text", "text": "only this"},
        ]
        assert extract_text_from_parts(parts) == "only this"

    def test_refusal_part_is_preserved(self):
        """
        What it does: Keeps refusal text.
        Purpose: A refusal is content the model produced and must survive.
        """
        assert extract_text_from_parts([{"type": "refusal", "refusal": "no"}]) == "no"

    def test_untyped_part_with_text_is_accepted(self):
        """
        What it does: Reads 'text' from a part with no 'type'.
        Purpose: Robustness against loosely-typed clients.
        """
        assert extract_text_from_parts([{"text": "loose"}]) == "loose"

    def test_unicode_is_preserved_exactly(self):
        """
        What it does: Preserves non-ASCII text.
        Purpose: The gateway must never mangle unicode.
        """
        text = "Привет 你好 مرحبا 🎉 café"
        assert extract_text_from_parts([{"type": "input_text", "text": text}]) == text

    def test_data_url_image_is_extracted(self):
        """
        What it does: Splits a data URL into media type and base64 payload.
        Purpose: Kiro API needs inline bytes plus a format.
        """
        images = extract_images_from_parts(
            [{"type": "input_image", "image_url": "data:image/png;base64,QUJD"}]
        )
        assert images == [{"media_type": "image/png", "data": "QUJD"}]

    def test_remote_url_image_is_skipped(self):
        """
        What it does: Skips http(s) image URLs.
        Purpose: Kiro API cannot fetch remote images; silently sending a URL as
                 bytes would be wrong.
        """
        images = extract_images_from_parts(
            [{"type": "input_image", "image_url": "https://example.com/a.png"}]
        )
        assert images == []

    def test_malformed_data_url_is_skipped(self):
        """
        What it does: Skips a data URL with no comma separator.
        Purpose: Malformed input must not raise.
        """
        assert extract_images_from_parts(
            [{"type": "input_image", "image_url": "data:image/png;base64"}]
        ) == []

    def test_function_call_output_string(self):
        """
        What it does: Returns a string tool output unchanged.
        Purpose: This is the shape Codex CLI sends.
        """
        assert extract_function_call_output("2 data.txt\n") == ("2 data.txt\n", [])

    def test_function_call_output_content_items(self):
        """
        What it does: Flattens structured tool output.
        Purpose: Text and images must both be recovered.
        """
        text, images = extract_function_call_output(
            [
                {"type": "input_text", "text": "shot:"},
                {"type": "input_image", "image_url": "data:image/png;base64,QQ"},
            ]
        )
        assert text == "shot:"
        assert images == [{"media_type": "image/png", "data": "QQ"}]

    def test_function_call_output_envelope(self):
        """
        What it does: Unwraps a {"body": ...} envelope.
        Purpose: Some SDKs serialize the payload wrapper.
        """
        assert extract_function_call_output({"body": "done", "success": True}) == ("done", [])

    def test_function_call_output_unknown_dict_is_json_encoded(self):
        """
        What it does: JSON-encodes an unrecognized dict output.
        Purpose: Never drop a tool result the gateway does not understand.
        """
        text, images = extract_function_call_output({"exit_code": 0})
        assert json.loads(text) == {"exit_code": 0}
        assert images == []

    def test_function_call_output_none(self):
        """
        What it does: Maps a missing output to "".
        Purpose: The converter substitutes "(empty result)" later.
        """
        assert extract_function_call_output(None) == ("", [])


# =============================================================================
# Tool conversion
# =============================================================================

class TestToolConversion:
    """Tests for convert_responses_tools_to_unified()."""

    def test_no_tools_returns_none_and_empty_registry(self):
        """
        What it does: Handles a request without tools.
        Purpose: Tool-free requests must not create tool specs.
        """
        tools, registry = convert_responses_tools_to_unified(None)

        assert tools is None
        assert registry.routes == {}

    def test_flat_function_tool_is_converted(self):
        """
        What it does: Converts a flat Responses function tool.
        Purpose: name/description/parameters sit at the top level, not under
                 "function" as in Chat Completions.
        """
        tools, registry = convert_responses_tools_to_unified(
            [
                ResponsesTool(
                    type="function",
                    name="exec_command",
                    description="Run a command",
                    parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
                )
            ]
        )

        assert len(tools) == 1
        assert tools[0].name == "exec_command"
        assert tools[0].description == "Run a command"
        assert tools[0].input_schema["properties"]["cmd"]["type"] == "string"
        assert registry.routes["exec_command"].namespace is None

    def test_namespace_tool_is_flattened_with_prefix(self):
        """
        What it does: Flattens a namespace container into prefixed tool names.
        Purpose: Kiro API has no namespaces, so the namespace is encoded in the
                 exposed name and recovered from the registry.
        """
        tools, registry = convert_responses_tools_to_unified(
            [
                ResponsesTool(
                    type="namespace",
                    name="multi_agent_v1",
                    tools=[
                        {"type": "function", "name": "spawn_agent", "parameters": {}},
                        {"type": "function", "name": "close_agent", "parameters": {}},
                    ],
                )
            ]
        )

        names = [tool.name for tool in tools]
        assert names == ["multi_agent_v1__spawn_agent", "multi_agent_v1__close_agent"]

        route = registry.routes["multi_agent_v1__spawn_agent"]
        assert route.client_name == "spawn_agent"
        assert route.namespace == "multi_agent_v1"

    def test_namespace_tool_without_name_is_skipped(self):
        """
        What it does: Skips a namespace container with no name.
        Purpose: There is no way to route calls back without a namespace name.
        """
        tools, registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="namespace", tools=[{"type": "function", "name": "x"}])]
        )

        assert tools is None
        assert registry.routes == {}

    def test_namespace_tool_without_nested_tools_is_skipped(self):
        """
        What it does: Skips an empty namespace container.
        Purpose: Nothing to expose.
        """
        tools, _ = convert_responses_tools_to_unified(
            [ResponsesTool(type="namespace", name="ns", tools=[])]
        )

        assert tools is None

    def test_over_long_namespaced_name_falls_back_to_bare_name(self):
        """
        What it does: Uses the bare tool name when namespace__name exceeds the
                      Kiro 64-character tool-name limit.
        Purpose: The request must still succeed, and the route must still carry
                 the namespace so the tool call is routed correctly.
        """
        long_namespace = "n" * 60
        tools, registry = convert_responses_tools_to_unified(
            [
                ResponsesTool(
                    type="namespace",
                    name=long_namespace,
                    tools=[{"type": "function", "name": "run", "parameters": {}}],
                )
            ]
        )

        assert [tool.name for tool in tools] == ["run"]
        assert registry.routes["run"].namespace == long_namespace

    def test_duplicate_exposed_names_are_disambiguated(self):
        """
        What it does: Suffixes a second tool that would collide with the first.
        Purpose: Two distinct client tools must never collapse into one Kiro tool.
        """
        tools, registry = convert_responses_tools_to_unified(
            [
                ResponsesTool(type="function", name="run", parameters={}),
                ResponsesTool(
                    type="namespace",
                    name="x" * 62,
                    tools=[{"type": "function", "name": "run", "parameters": {}}],
                ),
            ]
        )

        assert [tool.name for tool in tools] == ["run", "run_2"]
        assert registry.routes["run"].namespace is None
        assert registry.routes["run_2"].namespace == "x" * 62

    def test_builtin_web_search_is_emulated_when_enabled(self, monkeypatch):
        """
        What it does: Turns {"type": "web_search"} into a function tool.
        Purpose: The gateway services web search itself through the Kiro MCP
                 endpoint, so the client's built-in tool keeps working.
        """
        monkeypatch.setattr(
            "kiro.converters_openai_responses.WEB_SEARCH_ENABLED", True, raising=False
        )
        tools, registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="web_search", external_web_access=True)]
        )

        assert [tool.name for tool in tools] == ["web_search"]
        assert tools[0].input_schema["required"] == ["query"]
        assert registry.routes["web_search"].namespace is None

    def test_builtin_web_search_is_skipped_when_disabled(self, monkeypatch):
        """
        What it does: Skips the built-in web_search tool when the feature is off.
        Purpose: Offering a tool the gateway cannot service would break the turn.
        """
        monkeypatch.setattr(
            "kiro.converters_openai_responses.WEB_SEARCH_ENABLED", False, raising=False
        )
        tools, registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="web_search")]
        )

        assert tools is None
        assert registry.routes == {}

    def test_duplicate_web_search_declaration_is_not_registered_twice(self, monkeypatch):
        """
        What it does: Skips the built-in descriptor when a function tool already
                      declares web_search.
        Purpose: Duplicate Kiro tool names are rejected upstream.
        """
        monkeypatch.setattr(
            "kiro.converters_openai_responses.WEB_SEARCH_ENABLED", True, raising=False
        )
        tools, registry = convert_responses_tools_to_unified(
            [
                ResponsesTool(type="function", name="web_search", parameters={}),
                ResponsesTool(type="web_search"),
            ]
        )

        assert [tool.name for tool in tools] == ["web_search"]
        assert len(registry.routes) == 1

    def test_unsupported_tool_type_is_skipped(self):
        """
        What it does: Skips a server-side tool type the gateway cannot execute.
        Purpose: Kiro API only executes client-side function tools.
        """
        tools, registry = convert_responses_tools_to_unified(
            [
                ResponsesTool(type="code_interpreter"),
                ResponsesTool(type="function", name="ok", parameters={}),
            ]
        )

        assert [tool.name for tool in tools] == ["ok"]
        assert "code_interpreter" not in registry.routes

    def test_function_tool_without_name_is_skipped(self):
        """
        What it does: Skips a function tool with no name.
        Purpose: An unnamed tool can never be called or routed back.
        """
        tools, _ = convert_responses_tools_to_unified(
            [ResponsesTool(type="function", description="anonymous")]
        )

        assert tools is None

    def test_tool_without_parameters_keeps_none_schema(self):
        """
        What it does: Passes a missing schema through as None.
        Purpose: The shared sanitizer is the single place that repairs schemas.
        """
        tools, _ = convert_responses_tools_to_unified(
            [ResponsesTool(type="function", name="get_goal")]
        )

        assert tools[0].input_schema is None


class TestToolRegistry:
    """Tests for the ToolRegistry routing table."""

    def test_resolve_returns_registered_route(self):
        """
        What it does: Resolves a registered exposed name.
        Purpose: Namespaced calls must be reported under the client's identity.
        """
        registry = ToolRegistry()
        registry.register("spawn_agent", "multi_agent_v1")

        route = registry.resolve("multi_agent_v1__spawn_agent")
        assert route.client_name == "spawn_agent"
        assert route.namespace == "multi_agent_v1"

    def test_resolve_unknown_name_passes_through(self):
        """
        What it does: Returns an identity route for an unknown tool name.
        Purpose: A hallucinated tool name must reach the client unchanged so the
                 client can reject it, instead of being silently rewritten.
        """
        registry = ToolRegistry()

        route = registry.resolve("invented_tool")
        assert route.client_name == "invented_tool"
        assert route.namespace is None

    def test_exposed_name_for_matches_namespace(self):
        """
        What it does: Finds the exposed name for a (name, namespace) pair.
        Purpose: Replayed function_call history items must reference the same
                 tool name that was sent to Kiro.
        """
        registry = ToolRegistry()
        registry.register("run", None)
        registry.register("run", "ns")

        assert registry.exposed_name_for("run", "ns") == "ns__run"
        assert registry.exposed_name_for("run", None) == "run"

    def test_exposed_name_for_falls_back_to_name_only_match(self):
        """
        What it does: Matches on name when the history item omits the namespace.
        Purpose: Clients do not always echo the namespace back.
        """
        registry = ToolRegistry()
        registry.register("spawn_agent", "ns")

        assert registry.exposed_name_for("spawn_agent", None) == "ns__spawn_agent"

    def test_exposed_name_for_unregistered_tool_returns_input(self):
        """
        What it does: Returns the client name when the tool is not registered.
        Purpose: A request may replay a call to a tool it no longer declares.
        """
        assert ToolRegistry().exposed_name_for("gone", None) == "gone"


# =============================================================================
# Input conversion
# =============================================================================

class TestInputConversion:
    """Tests for convert_responses_input_to_unified()."""

    def test_instructions_become_the_system_prompt(self):
        """
        What it does: Folds 'instructions' into the system prompt.
        Purpose: Kiro API has no separate instructions field.
        """
        system, messages = convert_responses_input_to_unified(
            "hello", "You are terse.", ToolRegistry()
        )

        assert system == "You are terse."
        assert [(m.role, m.content) for m in messages] == [("user", "hello")]

    def test_developer_and_system_items_append_to_the_system_prompt(self):
        """
        What it does: Folds developer/system items into the system prompt in order.
        Purpose: Both roles are system-level in the Responses API; Codex sends its
                 skills block as a developer message.
        """
        system, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "DEV"}]},
                {"type": "message", "role": "system", "content": "SYS"},
                {"type": "message", "role": "user", "content": "hi"},
            ],
            "HEAD",
            ToolRegistry(),
        )

        assert system == "HEAD\n\nDEV\n\nSYS"
        assert [m.role for m in messages] == ["user"]

    def test_empty_system_level_item_is_ignored(self):
        """
        What it does: Drops a developer item with no text.
        Purpose: An empty block must not add blank lines to the system prompt.
        """
        system, _ = convert_responses_input_to_unified(
            [{"type": "message", "role": "developer", "content": []},
             {"type": "message", "role": "user", "content": "hi"}],
            "HEAD",
            ToolRegistry(),
        )

        assert system == "HEAD"

    def test_multi_turn_order_is_preserved(self):
        """
        What it does: Keeps user/assistant turns in wire order.
        Purpose: Reordering the conversation would change the model's answer.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "one"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "two"}]},
                {"type": "message", "role": "user", "content": "three"},
            ],
            None,
            ToolRegistry(),
        )

        assert [(m.role, m.content) for m in messages] == [
            ("user", "one"),
            ("assistant", "two"),
            ("user", "three"),
        ]

    def test_unknown_role_is_treated_as_user(self):
        """
        What it does: Maps an unknown role onto 'user'.
        Purpose: Kiro API only understands user and assistant.
        """
        _, messages = convert_responses_input_to_unified(
            [{"type": "message", "role": "critic", "content": "hm"}], None, ToolRegistry()
        )

        assert messages[0].role == "user"

    def test_images_are_attached_to_the_user_turn(self):
        """
        What it does: Attaches input_image parts to the message they belong to.
        Purpose: Images must travel with their prompt, not separately.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this"},
                        {"type": "input_image", "image_url": "data:image/jpeg;base64,Zm9v"},
                    ],
                }
            ],
            None,
            ToolRegistry(),
        )

        assert messages[0].content == "what is this"
        assert messages[0].images == [{"media_type": "image/jpeg", "data": "Zm9v"}]

    def test_function_call_attaches_to_preceding_assistant_turn(self):
        """
        What it does: Adds a function_call to the assistant turn before it.
        Purpose: Kiro expects toolUses inside the assistant response.
        """
        registry = ToolRegistry()
        registry.register("exec_command", None)

        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "run ls"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "sure"}]},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                    "call_id": "tooluse_1",
                },
            ],
            None,
            registry,
        )

        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].content == "sure"
        assert messages[1].tool_calls == [
            {
                "id": "tooluse_1",
                "type": "function",
                "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'},
            }
        ]

    def test_function_call_without_assistant_turn_opens_one(self):
        """
        What it does: Creates an assistant turn when a function_call has none.
        Purpose: Kiro rejects toolResults without a preceding assistant toolUse.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "name": "run", "arguments": "{}", "call_id": "c1"},
            ],
            None,
            ToolRegistry(),
        )

        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].tool_calls[0]["id"] == "c1"

    def test_function_call_uses_namespaced_exposed_name(self):
        """
        What it does: Rewrites a namespaced function_call to its exposed name.
        Purpose: The assistant turn must reference the same name that was sent to
                 Kiro in the tool specifications.
        """
        registry = ToolRegistry()
        registry.register("spawn_agent", "multi_agent_v1")

        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "go"},
                {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "namespace": "multi_agent_v1",
                    "arguments": "{}",
                    "call_id": "c1",
                },
            ],
            None,
            registry,
        )

        assert messages[1].tool_calls[0]["function"]["name"] == "multi_agent_v1__spawn_agent"

    def test_function_call_with_dict_arguments_is_json_encoded(self):
        """
        What it does: JSON-encodes non-string arguments.
        Purpose: The unified tool_call format always carries a string.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "name": "run", "arguments": {"a": 1}, "call_id": "c1"},
            ],
            None,
            ToolRegistry(),
        )

        assert json.loads(messages[1].tool_calls[0]["function"]["arguments"]) == {"a": 1}

    def test_function_call_with_empty_arguments_defaults_to_empty_object(self):
        """
        What it does: Substitutes "{}" for empty arguments.
        Purpose: Kiro rejects tool uses with no input object.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "name": "run", "arguments": "", "call_id": "c1"},
            ],
            None,
            ToolRegistry(),
        )

        assert messages[1].tool_calls[0]["function"]["arguments"] == "{}"

    def test_tool_round_trip_is_keyed_by_call_id(self):
        """
        What it does: Pairs function_call and function_call_output on call_id.
        Purpose: This is the whole tool round trip; a mismatch breaks the turn.
        """
        registry = ToolRegistry()
        registry.register("exec_command", None)

        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "count lines"},
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"wc -l"}',
                    "call_id": "tooluse_ABC",
                },
                {"type": "function_call_output", "call_id": "tooluse_ABC", "output": "2\n"},
            ],
            None,
            registry,
        )

        assert [m.role for m in messages] == ["user", "assistant", "user"]
        assert messages[1].tool_calls[0]["id"] == "tooluse_ABC"
        assert messages[2].tool_results == [
            {"type": "tool_result", "tool_use_id": "tooluse_ABC", "content": "2\n"}
        ]

    def test_consecutive_tool_outputs_merge_into_one_turn(self):
        """
        What it does: Groups parallel tool outputs into a single user turn.
        Purpose: Kiro expects all toolResults of one round in one message.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c1"},
                {"type": "function_call", "name": "b", "arguments": "{}", "call_id": "c2"},
                {"type": "function_call_output", "call_id": "c1", "output": "r1"},
                {"type": "function_call_output", "call_id": "c2", "output": "r2"},
            ],
            None,
            ToolRegistry(),
        )

        assert len(messages[1].tool_calls) == 2
        assert [r["tool_use_id"] for r in messages[2].tool_results] == ["c1", "c2"]

    def test_empty_tool_output_gets_a_placeholder(self):
        """
        What it does: Replaces an empty tool result with "(empty result)".
        Purpose: Kiro API rejects empty toolResult content.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c1"},
                {"type": "function_call_output", "call_id": "c1", "output": ""},
            ],
            None,
            ToolRegistry(),
        )

        assert messages[2].tool_results[0]["content"] == "(empty result)"

    def test_tool_output_images_are_attached(self):
        """
        What it does: Attaches images returned by a tool to the tool-result turn.
        Purpose: MCP tools return screenshots alongside text.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c1"},
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [
                        {"type": "input_text", "text": "shot"},
                        {"type": "input_image", "image_url": "data:image/png;base64,QQ"},
                    ],
                },
            ],
            None,
            ToolRegistry(),
        )

        assert messages[2].images == [{"media_type": "image/png", "data": "QQ"}]

    def test_reasoning_items_are_dropped(self):
        """
        What it does: Drops replayed reasoning items.
        Purpose: Kiro API has no slot for them and the gateway issued no
                 encrypted reasoning content.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "reasoning", "id": "rs_1",
                 "summary": [{"type": "summary_text", "text": "t"}]},
                {"type": "message", "role": "user", "content": "hi"},
            ],
            None,
            ToolRegistry(),
        )

        assert [(m.role, m.content) for m in messages] == [("user", "hi")]

    def test_unknown_item_types_are_dropped(self):
        """
        What it does: Drops item types the gateway cannot represent.
        Purpose: Kiro API cannot consume server-side call records.
        """
        _, messages = convert_responses_input_to_unified(
            [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {"type": "message", "role": "user", "content": "hi"},
            ],
            None,
            ToolRegistry(),
        )

        assert len(messages) == 1

    def test_empty_input_produces_no_messages(self):
        """
        What it does: Returns no messages for an empty input list.
        Purpose: The caller turns this into an actionable HTTP 400.
        """
        system, messages = convert_responses_input_to_unified([], "sys", ToolRegistry())

        assert system == "sys"
        assert messages == []

    def test_empty_string_input_produces_no_messages(self):
        """
        What it does: Returns no messages for an empty string input.
        Purpose: Same actionable HTTP 400 path as an empty list.
        """
        _, messages = convert_responses_input_to_unified("", None, ToolRegistry())

        assert messages == []


# =============================================================================
# Thinking configuration
# =============================================================================

class TestThinkingConfig:
    """Tests for extract_thinking_config_from_responses()."""

    def test_no_reasoning_block_uses_defaults(self):
        """
        What it does: Enables thinking with the default budget.
        Purpose: Matches the Chat Completions and Messages surfaces.
        """
        assert extract_thinking_config_from_responses(make_request()) == ThinkingConfig(
            enabled=True, budget_tokens=None
        )

    def test_summary_only_reasoning_block_uses_defaults(self):
        """
        What it does: Ignores 'summary' when no effort is given.
        Purpose: Codex sends {"summary": "auto"} with no effort by default.
        """
        request = make_request(reasoning={"summary": "auto"})

        assert extract_thinking_config_from_responses(request) == ThinkingConfig(
            enabled=True, budget_tokens=None
        )

    def test_effort_none_disables_thinking(self):
        """
        What it does: Disables thinking for effort='none'.
        Purpose: The client explicitly opted out.
        """
        request = make_request(reasoning={"effort": "none"})

        assert extract_thinking_config_from_responses(request).enabled is False

    @pytest.mark.parametrize(
        "effort,expected",
        [
            ("minimal", 400),
            ("low", 800),
            ("medium", 2000),
            ("high", 3200),
            ("xhigh", 3800),
        ],
    )
    def test_effort_maps_to_percentage_of_max_output_tokens(self, effort, expected):
        """
        What it does: Derives the thinking budget from max_output_tokens.
        Purpose: Same percentage mapping the Chat Completions surface uses.
        """
        request = make_request(reasoning={"effort": effort}, max_output_tokens=4000)

        assert extract_thinking_config_from_responses(request).budget_tokens == expected

    def test_missing_max_output_tokens_falls_back_to_4096(self):
        """
        What it does: Uses 4096 output tokens when the client sets no limit.
        Purpose: Codex never sends max_output_tokens.
        """
        request = make_request(reasoning={"effort": "medium"})

        assert extract_thinking_config_from_responses(request).budget_tokens == 2048


# =============================================================================
# Unsupported fields
# =============================================================================

class TestUnsupportedFields:
    """Tests for how unsupported request fields surface."""

    def test_previous_response_id_is_rejected(self):
        """
        What it does: Raises for previous_response_id.
        Purpose: The gateway stores no responses, so answering would silently
                 drop the hidden conversation history.
        """
        request = make_request(input="hi", previous_response_id="resp_123")

        with pytest.raises(UnsupportedResponsesFeatureError) as exc_info:
            validate_responses_request(request)

        message = str(exc_info.value)
        assert "previous_response_id" in message
        assert "input" in message

    def test_store_true_is_accepted_not_rejected(self):
        """
        What it does: Accepts store=true.
        Purpose: The client still sends the full input, so the answer is correct;
                 only the server-side copy is missing.
        """
        request = make_request(input="hi", store=True)

        validate_responses_request(request)

    def test_store_true_is_named_in_the_ignored_warning(self):
        """
        What it does: Logs a WARNING naming store=true.
        Purpose: Accepted-but-ignored fields must never be silent.
        """
        from loguru import logger

        records = []
        sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
        try:
            log_ignored_responses_fields(make_request(input="hi", store=True))
        finally:
            logger.remove(sink_id)

        assert any("store=true" in record["message"] for record in records)

    def test_all_ignored_fields_are_reported_in_one_warning(self):
        """
        What it does: Names every ignored field in a single WARNING.
        Purpose: One actionable line instead of a flood of messages.
        """
        from loguru import logger

        records = []
        sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
        try:
            log_ignored_responses_fields(
                make_request(
                    input="hi",
                    store=True,
                    temperature=0.5,
                    top_p=0.9,
                    truncation="auto",
                    service_tier="flex",
                    max_output_tokens=100,
                    tool_choice="required",
                    parallel_tool_calls=False,
                    text={"format": {"type": "json_object"}},
                    include=["reasoning.encrypted_content", "message.output_text.logprobs"],
                )
            )
        finally:
            logger.remove(sink_id)

        assert len(records) == 1
        message = records[0]["message"]
        for field in (
            "store=true",
            "temperature",
            "top_p",
            "truncation",
            "service_tier",
            "max_output_tokens",
            "tool_choice",
            "parallel_tool_calls",
            "text.format",
            "reasoning.encrypted_content",
            "message.output_text.logprobs",
        ):
            assert field in message

    def test_clean_request_logs_no_warning(self):
        """
        What it does: Logs nothing when every field is supported.
        Purpose: No noise on the happy path.
        """
        from loguru import logger

        records = []
        sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
        try:
            log_ignored_responses_fields(make_request(input="hi", store=False))
        finally:
            logger.remove(sink_id)

        assert records == []


# =============================================================================
# build_kiro_payload_responses
# =============================================================================

class TestBuildKiroPayloadResponses:
    """Tests for the module entry point."""

    def test_simple_string_input_builds_a_payload(self):
        """
        What it does: Builds a Kiro payload from a string prompt.
        Purpose: The simplest possible request must work end to end.
        """
        result = build_kiro_payload_responses(
            make_request(input="Hello"), "conv-1", VALID_PROFILE_ARN
        )

        state = result.payload["conversationState"]
        assert state["conversationId"] == "conv-1"
        assert state["currentMessage"]["userInputMessage"]["content"].endswith("Hello")
        assert result.payload["profileArn"] == VALID_PROFILE_ARN
        assert "history" not in state

    def test_instructions_are_prepended_to_the_prompt(self):
        """
        What it does: Puts the system prompt ahead of the only user message.
        Purpose: Kiro API has no system field, so the core converter inlines it.
        """
        result = build_kiro_payload_responses(
            make_request(input="Hi", instructions="BE BRIEF"), "c", VALID_PROFILE_ARN
        )

        content = result.payload["conversationState"]["currentMessage"]["userInputMessage"]["content"]
        assert content.startswith("BE BRIEF")
        assert content.endswith("Hi")

    def test_multi_turn_input_builds_history(self):
        """
        What it does: Puts earlier turns into Kiro 'history'.
        Purpose: Only the last turn is the current message.
        """
        result = build_kiro_payload_responses(
            make_request(
                input=[
                    {"type": "message", "role": "user", "content": "one"},
                    {"type": "message", "role": "assistant", "content": [
                        {"type": "output_text", "text": "two"}]},
                    {"type": "message", "role": "user", "content": "three"},
                ]
            ),
            "c",
            VALID_PROFILE_ARN,
        )

        history = result.payload["conversationState"]["history"]
        assert len(history) == 2
        # The core converter may append opt-in system additions (thinking mode,
        # truncation recovery) to the first history message, so match on the tail.
        assert history[0]["userInputMessage"]["content"].endswith("one")
        assert history[1]["assistantResponseMessage"]["content"] == "two"
        assert (
            result.payload["conversationState"]["currentMessage"]["userInputMessage"]["content"]
            == "three"
        )

    def test_tools_are_sanitized_before_reaching_kiro(self):
        """
        What it does: Strips schema annotation keywords from tool schemas.
        Purpose: The shared sanitizer must run for this surface too, otherwise
                 Kiro answers "Improperly formed request".
        """
        result = build_kiro_payload_responses(
            make_request(
                input="Hi",
                tools=[
                    {
                        "type": "function",
                        "name": "run",
                        "description": "Run",
                        "parameters": {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "additionalProperties": False,
                        },
                    }
                ],
            ),
            "c",
            VALID_PROFILE_ARN,
        )

        context = result.payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]
        schema = context["tools"][0]["toolSpecification"]["inputSchema"]["json"]
        assert "$schema" not in schema
        assert "additionalProperties" not in schema
        assert schema["type"] == "object"

    def test_tool_round_trip_reaches_kiro_as_tool_uses_and_results(self):
        """
        What it does: Translates a full tool round trip into Kiro structures.
        Purpose: toolUses must sit on the assistant turn and toolResults on the
                 following user turn, keyed by the same ID.
        """
        result = build_kiro_payload_responses(
            make_request(
                input=[
                    {"type": "message", "role": "user", "content": "count lines"},
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": '{"cmd":"wc -l"}',
                        "call_id": "tooluse_ABC",
                    },
                    {"type": "function_call_output", "call_id": "tooluse_ABC", "output": "2\n"},
                ],
                tools=[
                    {"type": "function", "name": "exec_command", "description": "Run",
                     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
                ],
            ),
            "c",
            VALID_PROFILE_ARN,
        )

        state = result.payload["conversationState"]
        assistant = state["history"][1]["assistantResponseMessage"]
        assert assistant["toolUses"][0]["toolUseId"] == "tooluse_ABC"
        assert assistant["toolUses"][0]["name"] == "exec_command"
        assert assistant["toolUses"][0]["input"] == {"cmd": "wc -l"}

        current_context = state["currentMessage"]["userInputMessage"]["userInputMessageContext"]
        assert current_context["toolResults"][0]["toolUseId"] == "tooluse_ABC"
        assert current_context["toolResults"][0]["content"][0]["text"] == "2\n"

    def test_images_reach_kiro_in_the_user_input_message(self):
        """
        What it does: Puts images directly on userInputMessage.
        Purpose: Kiro IDE format requires images there, not in the context block.
        """
        result = build_kiro_payload_responses(
            make_request(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "what"},
                            {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
                        ],
                    }
                ]
            ),
            "c",
            VALID_PROFILE_ARN,
        )

        message = result.payload["conversationState"]["currentMessage"]["userInputMessage"]
        assert message["images"] == [{"format": "png", "source": {"bytes": "QUJD"}}]

    def test_empty_input_raises_actionable_error(self):
        """
        What it does: Raises EmptyResponsesInputError for an empty input.
        Purpose: The route turns this into an actionable HTTP 400 instead of
                 sending an empty conversation upstream.
        """
        with pytest.raises(EmptyResponsesInputError) as exc_info:
            build_kiro_payload_responses(make_request(input=[]), "c", VALID_PROFILE_ARN)

        assert "input" in str(exc_info.value)
        assert "input_text" in str(exc_info.value)

    def test_instructions_only_request_still_needs_input(self):
        """
        What it does: Raises when only 'instructions' is present.
        Purpose: A system prompt alone is not a conversation Kiro can answer.
        """
        with pytest.raises(EmptyResponsesInputError):
            build_kiro_payload_responses(
                make_request(instructions="only system"), "c", VALID_PROFILE_ARN
            )

    def test_previous_response_id_raises_before_any_conversion(self):
        """
        What it does: Rejects previous_response_id at the entry point.
        Purpose: Fail fast, before contacting Kiro.
        """
        with pytest.raises(UnsupportedResponsesFeatureError):
            build_kiro_payload_responses(
                make_request(input="hi", previous_response_id="resp_1"), "c", VALID_PROFILE_ARN
            )

    def test_illegal_tool_name_raises_tool_spec_validation_error(self):
        """
        What it does: Rejects a tool name Kiro API cannot accept.
        Purpose: A clear 400 beats a silently broken tool call; the shared
                 sanitizer is the single source of truth for the rule.
        """
        with pytest.raises(ToolSpecValidationError):
            build_kiro_payload_responses(
                make_request(
                    input="hi",
                    tools=[{"type": "function", "name": "bad name!", "parameters": {}}],
                ),
                "c",
                VALID_PROFILE_ARN,
            )

    def test_tokenizer_payloads_include_system_prompt_and_tools(self):
        """
        What it does: Returns tokenizer-ready copies of messages and tools.
        Purpose: The streaming layer needs them for the tiktoken fallback when
                 Kiro reports no context usage.
        """
        result = build_kiro_payload_responses(
            make_request(
                input="Hi",
                instructions="SYS",
                tools=[{"type": "function", "name": "run", "parameters": {}}],
            ),
            "c",
            VALID_PROFILE_ARN,
        )

        assert result.messages_for_tokenizer[0] == {"role": "system", "content": "SYS"}
        assert result.messages_for_tokenizer[1]["role"] == "user"
        assert result.tools_for_tokenizer[0]["name"] == "run"

    def test_tokenizer_tools_are_none_without_tools(self):
        """
        What it does: Leaves tools_for_tokenizer as None when no tools are sent.
        Purpose: Avoid counting tokens for tools that do not exist.
        """
        result = build_kiro_payload_responses(
            make_request(input="Hi"), "c", VALID_PROFILE_ARN
        )

        assert result.tools_for_tokenizer is None

    def test_very_long_input_is_forwarded(self):
        """
        What it does: Builds a payload from a very large prompt.
        Purpose: Large prompts must reach the shared payload-size guard rather
                 than failing in the converter.
        """
        long_text = "x" * 200_000
        result = build_kiro_payload_responses(
            make_request(input=long_text), "c", VALID_PROFILE_ARN
        )

        content = result.payload["conversationState"]["currentMessage"]["userInputMessage"]["content"]
        assert len(content) >= 100_000

    def test_unicode_survives_conversion(self):
        """
        What it does: Keeps non-ASCII prompt text byte-identical.
        Purpose: The gateway must not transliterate or escape user text.
        """
        text = "Проверка 测试 اختبار 🚀"
        result = build_kiro_payload_responses(
            make_request(input=text), "c", VALID_PROFILE_ARN
        )

        content = result.payload["conversationState"]["currentMessage"]["userInputMessage"]["content"]
        assert content.endswith(text)


# =============================================================================
# Real captured Codex CLI payload
# =============================================================================

class TestRealCodexPayloadConversion:
    """Converter tests driven by the real Codex CLI 0.153.4 payloads."""

    def test_captured_first_turn_converts(self):
        """
        What it does: Converts the real first-turn Codex request.
        Purpose: End-to-end proof that the captured contract is handled.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_REQUEST)
        result = build_kiro_payload_responses(request, "conv-codex", VALID_PROFILE_ARN)

        message = result.payload["conversationState"]["currentMessage"]["userInputMessage"]
        assert message["modelId"] == "claude-sonnet-4.5"
        assert message["origin"] == "AI_EDITOR"
        assert message["content"].startswith("You are a coding agent running in the Codex CLI")
        assert "Say hello in exactly three words." in message["content"]

    def test_captured_tools_are_flattened_with_namespaces(self):
        """
        What it does: Flattens Codex's mixed tool list.
        Purpose: 4 top-level tools + 5 namespaced + 3 goal tools + web_search all
                 have to land as flat, uniquely named Kiro tools.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_REQUEST)
        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        context = result.payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]
        names = [spec["toolSpecification"]["name"] for spec in context["tools"]]

        assert "exec_command" in names
        assert "multi_agent_v1__spawn_agent" in names
        assert all(len(name) <= 64 for name in names)
        assert len(names) == len(set(names))

        route = result.tool_registry.routes["multi_agent_v1__spawn_agent"]
        assert route.client_name == "spawn_agent"
        assert route.namespace == "multi_agent_v1"

    def test_captured_developer_item_goes_into_the_system_prompt(self):
        """
        What it does: Folds Codex's developer skills block into the system prompt.
        Purpose: It is system-level context, not a conversation turn.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_REQUEST)
        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        content = result.payload["conversationState"]["currentMessage"]["userInputMessage"]["content"]
        assert "<skills_instructions>" in content

    def test_captured_follow_up_turn_converts_tool_round_trip(self):
        """
        What it does: Converts the real follow-up turn Codex sends after a tool run.
        Purpose: The assistant toolUse and the user toolResult must both appear
                 with the same Kiro tool use ID.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_FOLLOW_UP_REQUEST)
        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        state = result.payload["conversationState"]
        serialized = json.dumps(state)
        call_id = next(
            item["call_id"]
            for item in CODEX_RESPONSES_FOLLOW_UP_REQUEST["input"]
            if item.get("type") == "function_call"
        )

        assert serialized.count(call_id) >= 2

        tool_uses = [
            entry["assistantResponseMessage"]["toolUses"]
            for entry in state["history"]
            if "assistantResponseMessage" in entry
            and entry["assistantResponseMessage"].get("toolUses")
        ]
        assert tool_uses[0][0]["name"] == "exec_command"
        assert tool_uses[0][0]["toolUseId"] == call_id

        current_context = state["currentMessage"]["userInputMessage"]["userInputMessageContext"]
        assert current_context["toolResults"][0]["toolUseId"] == call_id


# =============================================================================
# additional_tools input item
# =============================================================================

class TestExtractAdditionalTools:
    """Tests for harvesting tool declarations out of the ``input`` array."""

    def test_extracts_tools_in_declaration_order(self):
        """
        What it does: Reads tool specs out of an additional_tools item.
        Purpose: Codex code mode declares every tool here, so losing them means
                 the model is offered no tools at all.
        """
        items = [
            {
                "type": "additional_tools",
                "id": "at_1",
                "role": "developer",
                "tools": [
                    {"type": "function", "name": "first"},
                    {"type": "function", "name": "second"},
                ],
            }
        ]

        assert [tool["name"] for tool in extract_additional_tools(items)] == ["first", "second"]

    def test_merges_several_additional_tools_items_in_order(self):
        """
        What it does: Concatenates the payloads of two additional_tools items.
        Purpose: Nothing in the wire format forbids more than one item.
        """
        items = [
            {"type": "additional_tools", "tools": [{"type": "function", "name": "a"}]},
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "additional_tools", "tools": [{"type": "function", "name": "b"}]},
        ]

        assert [tool["name"] for tool in extract_additional_tools(items)] == ["a", "b"]

    def test_string_input_yields_no_tools(self):
        """
        What it does: Accepts a plain-string ``input``.
        Purpose: ``input`` is legally a string; that must not raise.
        """
        assert extract_additional_tools("just a prompt") == []

    def test_none_input_yields_no_tools(self):
        """
        What it does: Accepts a missing ``input``.
        Purpose: ``input`` is optional on the request model.
        """
        assert extract_additional_tools(None) == []

    def test_item_without_tools_field_is_ignored(self):
        """
        What it does: Ignores an additional_tools item with no ``tools`` field.
        Purpose: Malformed items must not crash the conversion.
        """
        assert extract_additional_tools([{"type": "additional_tools", "id": "at_1"}]) == []

    def test_non_list_tools_payload_is_ignored(self):
        """
        What it does: Ignores a ``tools`` field that is not a list.
        Purpose: A dict or string payload carries no readable declaration.
        """
        assert extract_additional_tools([{"type": "additional_tools", "tools": "nope"}]) == []
        assert extract_additional_tools([{"type": "additional_tools", "tools": {"a": 1}}]) == []
        assert extract_additional_tools([{"type": "additional_tools", "tools": 7}]) == []

    def test_empty_tools_payload_is_ignored(self):
        """
        What it does: Ignores an empty ``tools`` list.
        Purpose: Nothing to merge, and no exception either.
        """
        assert extract_additional_tools([{"type": "additional_tools", "tools": []}]) == []

    def test_junk_entries_are_passed_through_to_the_tool_pipeline(self):
        """
        What it does: Returns entries verbatim, including junk.
        Purpose: Validation belongs to the tool pipeline, which reports each
                 unusable entry by type; this helper must not silently filter.
        """
        items = [{"type": "additional_tools", "tools": [None, 42, {"type": "mystery"}]}]

        assert extract_additional_tools(items) == [None, 42, {"type": "mystery"}]

    def test_report_flag_suppresses_logging_only(self):
        """
        What it does: Returns the same result with ``report=False``.
        Purpose: The route layer uses the quiet mode purely for its log line.
        """
        items = [{"type": "additional_tools", "tools": [{"type": "function", "name": "a"}]}]

        assert extract_additional_tools(items, report=False) == extract_additional_tools(items)

    def test_pydantic_input_items_are_supported(self):
        """
        What it does: Reads additional_tools from parsed ResponsesRequest items.
        Purpose: The route passes Pydantic models, not raw dicts.
        """
        request = make_request(
            input=[
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "function", "name": "parsed"}],
                }
            ]
        )

        assert [t["name"] for t in extract_additional_tools(request.input)] == ["parsed"]


class TestAdditionalToolsMerging:
    """Tests for merging additional_tools into the unified tool pipeline."""

    def test_additional_tools_alone_produce_a_non_empty_tool_list(self):
        """
        What it does: Converts a request with no ``tools`` array at all.
        Purpose: This is exactly the code-mode shape; it must still yield tools.
        """
        additional = [{"type": "function", "name": "solo", "parameters": {"type": "object"}}]

        tools, registry = convert_responses_tools_to_unified(None, additional)

        assert tools is not None
        assert [tool.name for tool in tools] == ["solo"]
        assert registry.routes["solo"].client_name == "solo"

    def test_empty_tools_array_plus_additional_tools_yields_tools(self):
        """
        What it does: Converts ``tools: []`` combined with additional_tools.
        Purpose: An explicitly empty array must not short-circuit the merge.
        """
        tools, _registry = convert_responses_tools_to_unified(
            [], [{"type": "function", "name": "solo"}]
        )

        assert tools is not None and len(tools) == 1

    def test_top_level_tools_come_before_additional_tools(self):
        """
        What it does: Checks the merged declaration order.
        Purpose: Tool order influences model behaviour, so it is preserved.
        """
        tools, _registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="function", name="top_a"), ResponsesTool(type="function", name="top_b")],
            [{"type": "function", "name": "extra_a"}, {"type": "function", "name": "extra_b"}],
        )

        assert [tool.name for tool in tools] == ["top_a", "top_b", "extra_a", "extra_b"]

    def test_duplicate_between_sources_is_registered_once(self):
        """
        What it does: Declares the same tool top-level and in additional_tools.
        Purpose: Kiro API rejects duplicate tool names, so exactly one entry.
        """
        tools, registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="function", name="dup", description="first")],
            [{"type": "function", "name": "dup", "description": "second"}],
        )

        assert [tool.name for tool in tools] == ["dup"]
        assert tools[0].description == "first"
        assert len(registry.routes) == 1

    def test_duplicate_inside_one_source_is_registered_once(self):
        """
        What it does: Declares the same tool twice in the same array.
        Purpose: Deduplication is a property of the pipeline, not of the source.
        """
        tools, _registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="function", name="dup"), ResponsesTool(type="function", name="dup")]
        )

        assert [tool.name for tool in tools] == ["dup"]

    def test_same_name_in_different_namespaces_stays_distinct(self):
        """
        What it does: Declares ``wait`` top-level and inside a namespace.
        Purpose: Deduplication keys on (name, namespace), so both survive under
                 distinct Kiro-facing names.
        """
        tools, registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="function", name="wait")],
            [
                {
                    "type": "namespace",
                    "name": "functions",
                    "tools": [{"type": "function", "name": "wait"}],
                }
            ],
        )

        assert [tool.name for tool in tools] == ["wait", "functions__wait"]
        assert registry.routes["functions__wait"].namespace == "functions"
        assert registry.routes["wait"].namespace is None

    def test_namespaced_tools_inside_additional_tools_are_routed(self):
        """
        What it does: Flattens namespace containers found in additional_tools.
        Purpose: Code mode delivers every tool inside a namespace container.
        """
        additional = [
            {
                "type": "namespace",
                "name": "collaboration",
                "description": "Tools in the collaboration namespace.",
                "tools": [
                    {"type": "function", "name": "spawn_agent"},
                    {"type": "function", "name": "list_agents"},
                ],
            }
        ]

        tools, registry = convert_responses_tools_to_unified(None, additional)

        assert [tool.name for tool in tools] == [
            "collaboration__spawn_agent",
            "collaboration__list_agents",
        ]
        route = registry.routes["collaboration__spawn_agent"]
        assert route.client_name == "spawn_agent"
        assert route.namespace == "collaboration"
        assert route.is_custom is False

    def test_junk_additional_tools_entries_are_reported_not_fatal(self):
        """
        What it does: Feeds unusable entries through the merge.
        Purpose: Unsupported declarations must be skipped, never crash, and the
                 usable ones must still reach the model.
        """
        additional = [
            {"type": "mystery", "name": "weird"},
            {"type": "function"},
            {"type": "namespace", "tools": [{"type": "function", "name": "orphan"}]},
            {"type": "namespace", "name": "empty_ns", "tools": []},
            {"type": "function", "name": "usable"},
        ]

        tools, registry = convert_responses_tools_to_unified(None, additional)

        assert [tool.name for tool in tools] == ["usable"]
        assert list(registry.routes) == ["usable"]

    def test_additional_tools_are_merged_by_build_kiro_payload(self):
        """
        What it does: Runs the full conversion with tools only in the input.
        Purpose: The Kiro payload, not just the helper, must carry the tools.
        """
        request = make_request(
            input=[
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "exec_command",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"cmd": {"type": "string"}},
                                    },
                                }
                            ],
                        }
                    ],
                },
                {"type": "message", "role": "user", "content": "run ls"},
            ]
        )

        result = build_kiro_payload_responses(request, "conv", VALID_PROFILE_ARN)

        context = result.payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]
        assert [spec["toolSpecification"]["name"] for spec in context["tools"]] == [
            "functions__exec_command"
        ]


# =============================================================================
# Freeform ("custom") tools
# =============================================================================

class TestCustomToolSchema:
    """Tests for exposing a freeform tool with a Kiro-valid JSON schema."""

    def test_schema_has_a_single_required_string_property(self):
        """
        What it does: Builds the schema of a freeform tool.
        Purpose: Kiro API only accepts JSON-schema tools, so the freeform text
                 needs exactly one string slot.
        """
        schema = build_custom_tool_schema({"type": "custom", "name": "exec"})

        assert schema["type"] == "object"
        assert schema["required"] == [CUSTOM_TOOL_INPUT_PROPERTY]
        assert schema["properties"][CUSTOM_TOOL_INPUT_PROPERTY]["type"] == "string"

    def test_schema_property_description_names_the_declared_syntax(self):
        """
        What it does: Mentions the declared syntax in the property description.
        Purpose: The syntax is the only hint the model gets about the payload.
        """
        schema = build_custom_tool_schema(
            {
                "type": "custom",
                "name": "exec",
                "format": {"type": "grammar", "syntax": "lark", "definition": "start: X"},
            }
        )

        assert "lark" in schema["properties"][CUSTOM_TOOL_INPUT_PROPERTY]["description"]

    def test_schema_survives_a_malformed_format_block(self):
        """
        What it does: Builds a schema when ``format`` is not a dict.
        Purpose: A junk format must not break tool registration.
        """
        for bad_format in ("grammar", 7, [], None):
            schema = build_custom_tool_schema(
                {"type": "custom", "name": "x", "format": bad_format}
            )
            assert schema["required"] == [CUSTOM_TOOL_INPUT_PROPERTY]

    def test_description_keeps_client_text_and_adds_the_contract(self):
        """
        What it does: Assembles a freeform tool description.
        Purpose: The client's own description must be preserved, and the model
                 must be told the payload goes into the ``input`` property.
        """
        description = build_custom_tool_description(
            {
                "type": "custom",
                "name": "exec",
                "description": "Run JavaScript code",
                "format": {"type": "grammar", "syntax": "lark", "definition": "start: SOURCE"},
            }
        )

        assert "Run JavaScript code" in description
        assert CUSTOM_TOOL_INPUT_PROPERTY in description
        assert "start: SOURCE" in description

    def test_description_is_never_empty(self):
        """
        What it does: Builds a description for a tool that declares none.
        Purpose: The calling contract alone still has to reach the model.
        """
        description = build_custom_tool_description({"type": "custom", "name": "exec"})

        assert CUSTOM_TOOL_INPUT_PROPERTY in description

    def test_oversized_grammar_definition_is_capped(self):
        """
        What it does: Inlines a pathologically long grammar.
        Purpose: The payload-size budget must not be blown by one tool.
        """
        description = build_custom_tool_description(
            {
                "type": "custom",
                "name": "exec",
                "format": {"syntax": "lark", "definition": "x" * 20000},
            }
        )

        assert len(description) < 20000

    def test_custom_tool_is_registered_and_flagged(self):
        """
        What it does: Registers a top-level freeform tool.
        Purpose: The route must be marked custom so the streaming layer emits a
                 custom_tool_call rather than a function_call.
        """
        tools, registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="custom", name="apply_patch", description="Edit files")]
        )

        assert [tool.name for tool in tools] == ["apply_patch"]
        assert registry.routes["apply_patch"].is_custom is True
        assert tools[0].input_schema["required"] == [CUSTOM_TOOL_INPUT_PROPERTY]

    def test_namespaced_custom_tool_is_registered_and_flagged(self):
        """
        What it does: Registers a freeform tool nested in a namespace.
        Purpose: This is the real code-mode shape (``functions.exec``).
        """
        tools, registry = convert_responses_tools_to_unified(
            None,
            [
                {
                    "type": "namespace",
                    "name": "functions",
                    "tools": [{"type": "custom", "name": "exec", "description": "Run JS"}],
                }
            ],
        )

        assert [tool.name for tool in tools] == ["functions__exec"]
        route = registry.routes["functions__exec"]
        assert route.is_custom is True
        assert route.client_name == "exec"
        assert route.namespace == "functions"

    def test_custom_tool_without_a_name_is_skipped(self):
        """
        What it does: Feeds a freeform tool with no name.
        Purpose: An unnamed tool cannot be called or routed back, so it is
                 skipped instead of producing a nameless Kiro tool.
        """
        tools, registry = convert_responses_tools_to_unified(
            [ResponsesTool(type="custom", description="no name here")]
        )

        assert tools is None
        assert registry.routes == {}

    def test_namespaced_custom_tool_without_a_name_is_skipped(self):
        """
        What it does: Feeds a nameless freeform tool inside a namespace.
        Purpose: Same guarantee one level down.
        """
        tools, registry = convert_responses_tools_to_unified(
            None,
            [
                {
                    "type": "namespace",
                    "name": "functions",
                    "tools": [
                        {"type": "custom", "description": "no name"},
                        {"type": "custom", "name": "exec"},
                    ],
                }
            ],
        )

        assert [tool.name for tool in tools] == ["functions__exec"]

    def test_custom_and_function_tool_of_the_same_name_are_deduplicated(self):
        """
        What it does: Declares ``apply_patch`` as both function and custom.
        Purpose: Kiro rejects duplicate names; the first declaration wins.
        """
        tools, registry = convert_responses_tools_to_unified(
            [
                ResponsesTool(type="function", name="apply_patch"),
                ResponsesTool(type="custom", name="apply_patch"),
            ]
        )

        assert [tool.name for tool in tools] == ["apply_patch"]
        assert registry.routes["apply_patch"].is_custom is False


class TestExtractCustomToolInput:
    """Tests for recovering the freeform text out of model-emitted arguments."""

    def test_well_formed_arguments(self):
        """
        What it does: Reads the ``input`` property of a JSON argument string.
        Purpose: This is the shape the synthesized schema asks for.
        """
        assert extract_custom_tool_input('{"input": "console.log(1)"}') == "console.log(1)"

    def test_already_parsed_dict(self):
        """
        What it does: Accepts a pre-parsed arguments dict.
        Purpose: Kiro sometimes hands back parsed arguments.
        """
        assert extract_custom_tool_input({"input": "let x = 1"}) == "let x = 1"

    def test_raw_text_arguments_are_used_verbatim(self):
        """
        What it does: Treats unparseable arguments as the payload itself.
        Purpose: For a freeform tool the raw text IS the argument, so losing it
                 would break the call.
        """
        patch = "*** Begin Patch\n*** Add File: a.txt\n+1\n*** End Patch"

        assert extract_custom_tool_input(patch) == patch

    def test_json_string_arguments(self):
        """
        What it does: Unwraps arguments that are a bare JSON string.
        Purpose: ``"\\"hello\\""`` means the payload is ``hello``.
        """
        assert extract_custom_tool_input('"hello"') == "hello"

    def test_single_string_property_fallback(self):
        """
        What it does: Uses the only string argument when ``input`` is missing.
        Purpose: Models occasionally rename the property; the call still works.
        """
        assert extract_custom_tool_input('{"code": "let x = 1"}') == "let x = 1"

    def test_non_string_input_value_is_serialized(self):
        """
        What it does: Serializes a non-string ``input`` value.
        Purpose: Better to forward JSON than to drop the call.
        """
        assert extract_custom_tool_input('{"input": {"a": 1}}') == '{"a": 1}'

    def test_ambiguous_object_is_forwarded_as_json(self):
        """
        What it does: Forwards an object with several string properties.
        Purpose: Nothing is silently dropped when the shape is ambiguous.
        """
        result = extract_custom_tool_input('{"a": "one", "b": "two"}')

        assert json.loads(result) == {"a": "one", "b": "two"}

    def test_empty_and_missing_arguments(self):
        """
        What it does: Handles ``None``, ``""``, whitespace and ``{}``.
        Purpose: A freeform tool may legitimately be called with nothing.
        """
        assert extract_custom_tool_input(None) == ""
        assert extract_custom_tool_input("") == ""
        assert extract_custom_tool_input("   ") == ""
        assert extract_custom_tool_input("{}") == ""

    def test_scalar_arguments_are_serialized(self):
        """
        What it does: Handles numeric / boolean / null arguments.
        Purpose: The helper must never raise, whatever the model produced.
        """
        assert extract_custom_tool_input("42") == "42"
        assert extract_custom_tool_input("true") == "true"
        assert extract_custom_tool_input("null") == "null"
        assert extract_custom_tool_input([1, 2]) == "[1, 2]"


class TestCustomToolCallRoundTrip:
    """Tests for replaying custom_tool_call / custom_tool_call_output items."""

    def test_custom_tool_call_becomes_an_assistant_tool_call(self):
        """
        What it does: Converts a replayed custom_tool_call input item.
        Purpose: Dropping it would delete the assistant's tool use from history
                 and leave Kiro with an orphaned tool result.
        """
        _tools, registry = convert_responses_tools_to_unified(
            None,
            [{"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "exec"}]}],
        )

        _system, messages = convert_responses_input_to_unified(
            [
                {
                    "type": "custom_tool_call",
                    "id": "ctc_1",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "exec",
                    "namespace": "functions",
                    "input": "text('hi');",
                }
            ],
            None,
            registry,
        )

        assert len(messages) == 1
        assert messages[0].role == "assistant"
        call = messages[0].tool_calls[0]
        assert call["id"] == "call_1"
        assert call["function"]["name"] == "functions__exec"
        assert json.loads(call["function"]["arguments"]) == {
            CUSTOM_TOOL_INPUT_PROPERTY: "text('hi');"
        }

    def test_custom_tool_call_output_becomes_a_tool_result(self):
        """
        What it does: Converts a replayed custom_tool_call_output item.
        Purpose: Kiro needs the toolResult to follow the toolUse.
        """
        _system, messages = convert_responses_input_to_unified(
            [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "exec",
                    "input": "text('hi');",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "input_text", "text": "Script completed"}],
                },
            ],
            None,
            ToolRegistry(),
        )

        assert [m.role for m in messages] == ["assistant", "user"]
        assert messages[1].tool_results[0]["tool_use_id"] == "call_1"
        assert messages[1].tool_results[0]["content"] == "Script completed"

    def test_custom_tool_call_output_accepts_a_plain_string(self):
        """
        What it does: Converts a string-shaped custom tool output.
        Purpose: The wire format allows both string and content-item payloads.
        """
        _system, messages = convert_responses_input_to_unified(
            [{"type": "custom_tool_call_output", "call_id": "c", "output": "done"}],
            None,
            ToolRegistry(),
        )

        assert messages[0].tool_results[0]["content"] == "done"

    def test_custom_tool_call_without_input_is_still_replayed(self):
        """
        What it does: Replays a custom_tool_call with no ``input`` field.
        Purpose: The tool use must survive so the round trip stays balanced.
        """
        _system, messages = convert_responses_input_to_unified(
            [{"type": "custom_tool_call", "call_id": "c", "name": "exec"}],
            None,
            ToolRegistry(),
        )

        assert json.loads(messages[0].tool_calls[0]["function"]["arguments"]) == {
            CUSTOM_TOOL_INPUT_PROPERTY: ""
        }

    def test_custom_tool_call_with_non_string_input_is_serialized(self):
        """
        What it does: Replays a custom_tool_call whose ``input`` is not a string.
        Purpose: A malformed replay must not lose the tool use.
        """
        _system, messages = convert_responses_input_to_unified(
            [{"type": "custom_tool_call", "call_id": "c", "name": "exec", "input": {"a": 1}}],
            None,
            ToolRegistry(),
        )

        arguments = json.loads(messages[0].tool_calls[0]["function"]["arguments"])
        assert json.loads(arguments[CUSTOM_TOOL_INPUT_PROPERTY]) == {"a": 1}

    def test_additional_tools_item_produces_no_conversation_turn(self):
        """
        What it does: Runs the message pass over an additional_tools item.
        Purpose: It carries declarations, not content, so it must not become a
                 user turn (which would leak tool JSON into the prompt).
        """
        system, messages = convert_responses_input_to_unified(
            [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "function", "name": "a"}],
                },
                {"type": "message", "role": "user", "content": "hello"},
            ],
            None,
            ToolRegistry(),
        )

        assert system == ""
        assert [m.content for m in messages] == ["hello"]

    def test_mixed_function_and_custom_round_trips_interleave_correctly(self):
        """
        What it does: Replays a function_call and a custom_tool_call in order.
        Purpose: Both item families must share the same turn-building logic.
        """
        _system, messages = convert_responses_input_to_unified(
            [
                {"type": "function_call", "call_id": "f1", "name": "wait", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "f1", "output": "ok"},
                {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": "x"},
                {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
            ],
            None,
            ToolRegistry(),
        )

        assert [m.role for m in messages] == ["assistant", "user", "assistant", "user"]
        assert messages[0].tool_calls[0]["id"] == "f1"
        assert messages[2].tool_calls[0]["id"] == "c1"


# =============================================================================
# Real captured Codex CLI code-mode payload
# =============================================================================

class TestRealCodexCodeModePayloadConversion:
    """Converter tests driven by the real Codex CLI 0.153.4 code-mode payloads."""

    def test_captured_payload_has_no_top_level_tools(self):
        """
        What it does: Asserts the fixture really is the code-mode shape.
        Purpose: The whole point of the fix is that ``tools`` is absent while
                 tools are declared in ``input``; if the fixture drifted, the
                 rest of this class would be testing nothing.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_REQUEST)

        assert not request.tools
        assert request.input[0].type == "additional_tools"

    def test_captured_additional_tools_reach_the_kiro_payload(self):
        """
        What it does: Converts the real code-mode first turn.
        Purpose: Every tool Codex declared inside the input item must be offered
                 to the model, or tool calling cannot work at all.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_REQUEST)

        result = build_kiro_payload_responses(request, "conv-code-mode", VALID_PROFILE_ARN)

        context = result.payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]
        names = [spec["toolSpecification"]["name"] for spec in context["tools"]]
        assert names == [
            "functions__exec",
            "functions__wait",
            "functions__request_user_input",
            "collaboration__followup_task",
            "collaboration__interrupt_agent",
            "collaboration__list_agents",
            "collaboration__send_message",
            "collaboration__spawn_agent",
            "collaboration__wait_agent",
        ]
        assert len(names) == len(set(names))
        assert all(len(name) <= 64 for name in names)

    def test_captured_exec_tool_is_exposed_as_a_freeform_tool(self):
        """
        What it does: Inspects the converted ``exec`` tool specification.
        Purpose: ``exec`` is freeform, so it needs a Kiro-valid single-string
                 schema and a route flagged as custom.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_REQUEST)

        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        context = result.payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]
        spec = next(
            entry["toolSpecification"]
            for entry in context["tools"]
            if entry["toolSpecification"]["name"] == "functions__exec"
        )
        schema = spec["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert schema["required"] == [CUSTOM_TOOL_INPUT_PROPERTY]
        assert schema["properties"][CUSTOM_TOOL_INPUT_PROPERTY]["type"] == "string"
        assert "lark" in spec["description"]

        route = result.tool_registry.routes["functions__exec"]
        assert route.is_custom is True
        assert route.client_name == "exec"
        assert route.namespace == "functions"

    def test_captured_non_freeform_tools_keep_their_own_schema(self):
        """
        What it does: Checks a plain function tool from the same item.
        Purpose: The freeform handling must not leak onto normal tools.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_REQUEST)

        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        context = result.payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]
        spec = next(
            entry["toolSpecification"]
            for entry in context["tools"]
            if entry["toolSpecification"]["name"] == "functions__wait"
        )
        assert "cell_id" in spec["inputSchema"]["json"]["properties"]
        assert result.tool_registry.routes["functions__wait"].is_custom is False

    def test_captured_tool_declarations_do_not_leak_into_the_prompt(self):
        """
        What it does: Checks the system prompt and current message.
        Purpose: The additional_tools item must not be rendered as text.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_REQUEST)

        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        message = result.payload["conversationState"]["currentMessage"]["userInputMessage"]
        assert "additional_tools" not in message["content"]

    def test_captured_code_mode_follow_up_replays_the_custom_round_trip(self):
        """
        What it does: Converts the real turn Codex sends after ``exec`` ran.
        Purpose: The freeform toolUse and its toolResult must share one ID, so
                 Kiro accepts the conversation and the loop can continue.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST)

        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        state = result.payload["conversationState"]
        call_id = next(
            item["call_id"]
            for item in CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST["input"]
            if item.get("type") == "custom_tool_call"
        )
        tool_uses = [
            entry["assistantResponseMessage"]["toolUses"]
            for entry in state["history"]
            if "assistantResponseMessage" in entry
            and entry["assistantResponseMessage"].get("toolUses")
        ]
        assert tool_uses[0][0]["name"] == "functions__exec"
        assert tool_uses[0][0]["toolUseId"] == call_id
        assert CUSTOM_TOOL_INPUT_PROPERTY in tool_uses[0][0]["input"]

        current_context = state["currentMessage"]["userInputMessage"]["userInputMessageContext"]
        assert current_context["toolResults"][0]["toolUseId"] == call_id

    def test_captured_code_mode_follow_up_still_declares_every_tool(self):
        """
        What it does: Converts the follow-up turn and counts the tools.
        Purpose: Codex resends additional_tools on every turn; the tools must be
                 offered again, not just on the first request.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST)

        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        context = result.payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]
        assert len(context["tools"]) == 9


class TestRealCodexPayloadRegression:
    """Guards the previously captured non-code-mode payload against drift."""

    def test_nine_tool_payload_is_unchanged(self):
        """
        What it does: Re-converts the original 9-tool Codex payload.
        Purpose: The additional_tools / custom work must not alter the shape
                 that already worked.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_REQUEST)

        result = build_kiro_payload_responses(request, "c", VALID_PROFILE_ARN)

        context = result.payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]
        assert [spec["toolSpecification"]["name"] for spec in context["tools"]] == [
            "exec_command",
            "write_stdin",
            "request_user_input",
            "view_image",
            "multi_agent_v1__close_agent",
            "multi_agent_v1__resume_agent",
            "multi_agent_v1__send_input",
            "multi_agent_v1__spawn_agent",
            "multi_agent_v1__wait_agent",
            "get_goal",
            "create_goal",
            "update_goal",
            "web_search",
        ]
        assert all(route.is_custom is False for route in result.tool_registry.routes.values())

    def test_nine_tool_payload_declares_no_freeform_tools(self):
        """
        What it does: Asserts no route is flagged custom.
        Purpose: A false positive here would switch the streaming layer to
                 custom_tool_call items and break the working path.
        """
        request = ResponsesRequest(**CODEX_RESPONSES_REQUEST)

        _tools, registry = convert_responses_tools_to_unified(request.tools)

        assert registry.routes
        assert not any(route.is_custom for route in registry.routes.values())
