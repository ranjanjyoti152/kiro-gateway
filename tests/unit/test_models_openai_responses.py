# -*- coding: utf-8 -*-
"""
Unit tests for the OpenAI Responses API Pydantic models (models_openai_responses.py).

Covers:
- ResponsesRequest validation (string input, item-array input, missing/extra fields)
- Input item shapes: message/text, input_image, function_call, function_call_output,
  reasoning
- Tool shapes: flat function, namespace container, built-in web_search
- reasoning configuration variants
- Response/output/usage models used to assemble the reply
- A test built from the REAL payload captured from OpenAI Codex CLI 0.153.4
"""
import json

import pytest
from pydantic import ValidationError

from kiro.models_openai_responses import (
    ResponsesContentPart,
    ResponsesCustomToolCallItem,
    ResponsesFunctionCallItem,
    ResponsesIncompleteDetails,
    ResponsesInputItem,
    ResponsesInputTokensDetails,
    ResponsesMessageItem,
    ResponsesOutputTextPart,
    ResponsesOutputTokensDetails,
    ResponsesReasoningConfig,
    ResponsesReasoningItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesSummaryPart,
    ResponsesTextConfig,
    ResponsesTool,
    ResponsesUsage,
)
from codex_responses_payload import CODEX_RESPONSES_REQUEST


# =============================================================================
# ResponsesRequest — happy paths
# =============================================================================

class TestResponsesRequestSuccess:
    """Tests for valid ResponsesRequest payloads."""

    def test_minimal_request_requires_only_model(self):
        """
        What it does: Parses a request that carries nothing but a model name.
        Purpose: 'input' is optional on the wire (instructions may carry the whole
                 prompt), so validation must not reject it.
        """
        print("Action: Parsing a model-only request...")
        request = ResponsesRequest(model="claude-sonnet-4.5")

        assert request.model == "claude-sonnet-4.5"
        assert request.input is None
        assert request.stream is False
        assert request.store is None
        assert request.previous_response_id is None

    def test_input_as_plain_string(self):
        """
        What it does: Accepts 'input' as a plain string.
        Purpose: The Responses API allows a bare string prompt.
        """
        print("Action: Parsing string input...")
        request = ResponsesRequest(model="m", input="Hello there")

        assert request.input == "Hello there"
        assert isinstance(request.input, str)

    def test_input_as_item_array_with_text_part(self):
        """
        What it does: Accepts 'input' as an array of message items.
        Purpose: This is what Codex CLI actually sends.
        """
        print("Action: Parsing item-array input...")
        request = ResponsesRequest(
            model="m",
            input=[
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
        )

        assert len(request.input) == 1
        item = request.input[0]
        assert isinstance(item, ResponsesInputItem)
        assert item.type == "message"
        assert item.role == "user"
        assert item.content[0].type == "input_text"
        assert item.content[0].text == "hi"

    def test_stream_defaults_to_false(self):
        """
        What it does: Leaves 'stream' at False when absent.
        Purpose: Non-streaming must be the default, matching OpenAI.
        """
        print("Action: Parsing request without 'stream'...")
        assert ResponsesRequest(model="m").stream is False

    def test_unknown_fields_are_preserved_not_rejected(self):
        """
        What it does: Accepts unknown top-level fields.
        Purpose: extra="allow" keeps new upstream fields from becoming HTTP 422.
        """
        print("Action: Parsing request with an unknown field...")
        request = ResponsesRequest(model="m", brand_new_field={"a": 1})

        assert request.model_extra["brand_new_field"] == {"a": 1}

    def test_codex_specific_fields_are_typed(self):
        """
        What it does: Parses prompt_cache_key and client_metadata.
        Purpose: Both are sent by Codex CLI on every request.
        """
        print("Action: Parsing Codex-specific fields...")
        request = ResponsesRequest(
            model="m",
            prompt_cache_key="session-1",
            client_metadata={"session_id": "abc"},
        )

        assert request.prompt_cache_key == "session-1"
        assert request.client_metadata == {"session_id": "abc"}


class TestResponsesRequestErrors:
    """Tests for invalid ResponsesRequest payloads."""

    def test_missing_model_is_rejected(self):
        """
        What it does: Rejects a request without 'model'.
        Purpose: The model name is the one field the gateway cannot infer.
        """
        print("Action: Parsing request without 'model'...")
        with pytest.raises(ValidationError) as exc_info:
            ResponsesRequest(input="hi")

        assert "model" in str(exc_info.value)

    def test_input_of_wrong_scalar_type_is_rejected(self):
        """
        What it does: Rejects a numeric 'input'.
        Purpose: 'input' must be a string or an item list, never a number.
        """
        print("Action: Parsing request with numeric input...")
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input=42)

    def test_invalid_reasoning_effort_is_rejected(self):
        """
        What it does: Rejects an unknown reasoning effort level.
        Purpose: Only the documented effort levels map to a thinking budget.
        """
        print("Action: Parsing reasoning.effort='turbo'...")
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", reasoning={"effort": "turbo"})

    def test_tools_must_be_a_list(self):
        """
        What it does: Rejects a dict 'tools' value.
        Purpose: The Responses API always sends tools as a list.
        """
        print("Action: Parsing tools as a dict...")
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", tools={"type": "function"})


# =============================================================================
# Input item shapes
# =============================================================================

class TestResponsesInputItemShapes:
    """Tests for every input item shape the gateway must accept."""

    def test_message_without_type_is_accepted(self):
        """
        What it does: Accepts a bare {"role": ..., "content": ...} item.
        Purpose: 'type' is optional in the OpenAI schema and defaults to message.
        """
        print("Action: Parsing item without 'type'...")
        item = ResponsesInputItem(role="user", content="hi")

        assert item.type is None
        assert item.content == "hi"

    def test_input_image_part(self):
        """
        What it does: Parses an input_image content part.
        Purpose: Multimodal prompts must survive validation.
        """
        print("Action: Parsing input_image part...")
        item = ResponsesInputItem(
            type="message",
            role="user",
            content=[
                {"type": "input_text", "text": "look"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "high",
                },
            ],
        )

        assert item.content[1].image_url == "data:image/png;base64,AAAA"
        assert item.content[1].detail == "high"

    def test_function_call_item(self):
        """
        What it does: Parses a function_call item with call_id and namespace.
        Purpose: Tool round-trips are keyed by call_id; namespace routes the call.
        """
        print("Action: Parsing function_call item...")
        item = ResponsesInputItem(
            type="function_call",
            id="fc_1",
            name="spawn_agent",
            namespace="multi_agent_v1",
            arguments='{"task":"x"}',
            call_id="tooluse_1",
            status="completed",
        )

        assert item.name == "spawn_agent"
        assert item.namespace == "multi_agent_v1"
        assert item.call_id == "tooluse_1"
        assert json.loads(item.arguments) == {"task": "x"}

    def test_function_call_output_item_with_string_output(self):
        """
        What it does: Parses a function_call_output whose output is a string.
        Purpose: This is the exact shape Codex CLI sends back after a tool run.
        """
        print("Action: Parsing function_call_output with string output...")
        item = ResponsesInputItem(
            type="function_call_output",
            id="fco_1",
            call_id="tooluse_1",
            output="2 data.txt\n",
        )

        assert item.output == "2 data.txt\n"

    def test_function_call_output_item_with_content_items(self):
        """
        What it does: Parses a function_call_output whose output is a content list.
        Purpose: The Responses API allows structured tool output.
        """
        print("Action: Parsing function_call_output with content items...")
        item = ResponsesInputItem(
            type="function_call_output",
            call_id="tooluse_1",
            output=[{"type": "input_text", "text": "ok"}],
        )

        assert item.output == [{"type": "input_text", "text": "ok"}]

    def test_reasoning_item(self):
        """
        What it does: Parses a replayed reasoning item.
        Purpose: Codex replays reasoning items it received previously.
        """
        print("Action: Parsing reasoning item...")
        item = ResponsesInputItem(
            type="reasoning",
            id="rs_1",
            summary=[{"type": "summary_text", "text": "thought"}],
            encrypted_content=None,
        )

        assert isinstance(item.summary[0], ResponsesSummaryPart)
        assert item.summary[0].text == "thought"
        assert item.encrypted_content is None

    def test_unknown_item_type_is_accepted(self):
        """
        What it does: Accepts an item type the gateway does not know.
        Purpose: Validation must never be the layer that rejects new item types;
                 the converter reports and skips them instead.
        """
        print("Action: Parsing unknown item type...")
        item = ResponsesInputItem(type="web_search_call", id="ws_1", status="completed")

        assert item.type == "web_search_call"

    def test_content_part_with_unknown_type_is_accepted(self):
        """
        What it does: Accepts an unknown content part type.
        Purpose: Forward compatibility for new part types.
        """
        print("Action: Parsing unknown content part...")
        part = ResponsesContentPart(type="input_video", video_url="data:video/mp4;base64,AA")

        assert part.type == "input_video"
        assert part.model_extra["video_url"] == "data:video/mp4;base64,AA"


# =============================================================================
# Tool shapes
# =============================================================================

class TestResponsesToolShapes:
    """Tests for the three tool shapes observed on the wire."""

    def test_flat_function_tool_is_not_nested_under_function(self):
        """
        What it does: Parses a flat function tool.
        Purpose: The Responses API puts name/parameters at the top level, unlike
                 Chat Completions which nests them under "function".
        """
        print("Action: Parsing flat function tool...")
        tool = ResponsesTool(
            type="function",
            name="exec_command",
            description="Run a command",
            strict=False,
            parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
        )

        assert tool.name == "exec_command"
        assert tool.parameters["properties"]["cmd"]["type"] == "string"
        assert tool.strict is False

    def test_namespace_tool_holds_nested_function_tools(self):
        """
        What it does: Parses a namespace container tool.
        Purpose: Codex groups sub-agent tools inside a namespace.
        """
        print("Action: Parsing namespace tool...")
        tool = ResponsesTool(
            type="namespace",
            name="multi_agent_v1",
            description="Sub-agents",
            tools=[
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )

        assert tool.type == "namespace"
        assert isinstance(tool.tools[0], ResponsesTool)
        assert tool.tools[0].name == "spawn_agent"

    def test_builtin_tool_without_name_or_parameters(self):
        """
        What it does: Parses the built-in web_search tool descriptor.
        Purpose: Built-in tools carry no name or schema.
        """
        print("Action: Parsing built-in web_search tool...")
        tool = ResponsesTool(type="web_search", external_web_access=True)

        assert tool.type == "web_search"
        assert tool.name is None
        assert tool.parameters is None
        assert tool.model_extra["external_web_access"] is True

    def test_tool_without_parameters_is_accepted(self):
        """
        What it does: Parses a function tool that declares no parameters.
        Purpose: Zero-argument tools are legal and must reach the sanitizer.
        """
        print("Action: Parsing tool without parameters...")
        tool = ResponsesTool(type="function", name="get_goal")

        assert tool.name == "get_goal"
        assert tool.parameters is None


# =============================================================================
# reasoning / text configuration
# =============================================================================

class TestResponsesReasoningAndTextConfig:
    """Tests for the reasoning and text configuration blocks."""

    @pytest.mark.parametrize(
        "effort", ["none", "minimal", "low", "medium", "high", "xhigh"]
    )
    def test_all_documented_effort_levels_are_accepted(self, effort):
        """
        What it does: Accepts every documented reasoning effort level.
        Purpose: Each level maps to a thinking budget percentage.
        """
        print(f"Action: Parsing reasoning.effort={effort!r}...")
        config = ResponsesReasoningConfig(effort=effort)

        assert config.effort == effort

    def test_summary_only_reasoning_block(self):
        """
        What it does: Parses {"summary": "auto"} with no effort.
        Purpose: This is exactly what Codex CLI sends by default.
        """
        print("Action: Parsing summary-only reasoning block...")
        config = ResponsesReasoningConfig(summary="auto")

        assert config.effort is None
        assert config.summary == "auto"

    def test_legacy_generate_summary_alias(self):
        """
        What it does: Parses the legacy generate_summary field.
        Purpose: Older SDKs still send it.
        """
        print("Action: Parsing generate_summary...")
        assert ResponsesReasoningConfig(generate_summary="concise").generate_summary == "concise"

    def test_text_format_block(self):
        """
        What it does: Parses the structured-output text.format block.
        Purpose: The field must be accepted even though Kiro ignores it.
        """
        print("Action: Parsing text.format...")
        config = ResponsesTextConfig(format={"type": "json_object"}, verbosity="low")

        assert config.format == {"type": "json_object"}
        assert config.verbosity == "low"


# =============================================================================
# Response / output / usage models
# =============================================================================

class TestResponsesOutputModels:
    """Tests for the response-side models."""

    def test_message_item_defaults(self):
        """
        What it does: Builds a message output item from just an ID.
        Purpose: Defaults must match the OpenAI shape (assistant role, empty
                 content list, completed status).
        """
        print("Action: Building message item...")
        item = ResponsesMessageItem(id="msg_1")

        assert item.type == "message"
        assert item.role == "assistant"
        assert item.status == "completed"
        assert item.content == []

    def test_output_text_part_always_has_annotations(self):
        """
        What it does: Builds an output_text part.
        Purpose: Codex and the OpenAI SDK expect an 'annotations' key to exist.
        """
        print("Action: Building output_text part...")
        part = ResponsesOutputTextPart(text="hello")

        dumped = part.model_dump()
        assert dumped == {"type": "output_text", "text": "hello", "annotations": []}

    def test_reasoning_item_has_null_encrypted_content(self):
        """
        What it does: Builds a reasoning output item.
        Purpose: The gateway reconstructs reasoning from plain text, so
                 encrypted_content must be explicitly null, never fabricated.
        """
        print("Action: Building reasoning item...")
        item = ResponsesReasoningItem(
            id="rs_1", summary=[ResponsesSummaryPart(type="summary_text", text="t")]
        )

        dumped = item.model_dump()
        assert dumped["type"] == "reasoning"
        assert dumped["encrypted_content"] is None
        assert dumped["summary"][0]["text"] == "t"

    def test_function_call_item_shape(self):
        """
        What it does: Builds a function_call output item.
        Purpose: name/arguments/call_id are the fields the client correlates on.
        """
        print("Action: Building function_call item...")
        item = ResponsesFunctionCallItem(
            id="fc_1",
            name="exec_command",
            namespace=None,
            arguments='{"cmd":"ls"}',
            call_id="tooluse_1",
        )

        dumped = item.model_dump()
        assert dumped["type"] == "function_call"
        assert dumped["call_id"] == "tooluse_1"
        assert dumped["arguments"] == '{"cmd":"ls"}'
        assert dumped["namespace"] is None

    def test_usage_shape_matches_openai(self):
        """
        What it does: Builds the usage block.
        Purpose: Codex parses input_tokens/output_tokens/total_tokens plus the
                 details sub-objects; all must be present.
        """
        print("Action: Building usage block...")
        usage = ResponsesUsage(
            input_tokens=10,
            input_tokens_details=ResponsesInputTokensDetails(cached_tokens=0),
            output_tokens=4,
            output_tokens_details=ResponsesOutputTokensDetails(reasoning_tokens=3),
            total_tokens=14,
        )

        dumped = usage.model_dump()
        assert dumped["input_tokens"] == 10
        assert dumped["output_tokens"] == 4
        assert dumped["total_tokens"] == 14
        assert dumped["input_tokens_details"]["cached_tokens"] == 0
        assert dumped["output_tokens_details"]["reasoning_tokens"] == 3

    def test_response_envelope_defaults_are_stateless(self):
        """
        What it does: Builds a response envelope.
        Purpose: store must default to False and previous_response_id to None,
                 because the gateway keeps no server-side conversation.
        """
        print("Action: Building response envelope...")
        response = ResponsesResponse(id="resp_1", model="m")

        dumped = response.model_dump()
        assert dumped["object"] == "response"
        assert dumped["status"] == "in_progress"
        assert dumped["store"] is False
        assert dumped["previous_response_id"] is None
        assert dumped["output"] == []
        assert dumped["usage"] is None

    def test_incomplete_details_requires_reason(self):
        """
        What it does: Rejects incomplete_details without a reason.
        Purpose: A truncated response must always say why.
        """
        print("Action: Building incomplete_details without reason...")
        with pytest.raises(ValidationError):
            ResponsesIncompleteDetails()

        assert ResponsesIncompleteDetails(reason="max_output_tokens").reason == "max_output_tokens"


# =============================================================================
# Real captured Codex CLI payload
# =============================================================================

class TestRealCodexPayload:
    """Tests built from the payload captured from OpenAI Codex CLI 0.153.4."""

    def test_captured_codex_request_validates(self):
        """
        What it does: Parses the real captured Codex CLI request.
        Purpose: The endpoint is only useful if the exact bytes Codex sends
                 validate without modification.
        """
        print("Action: Parsing the captured Codex CLI request...")
        request = ResponsesRequest(**CODEX_RESPONSES_REQUEST)

        assert request.model == "claude-sonnet-4.5"
        assert request.stream is True
        assert request.store is False
        assert request.previous_response_id is None
        assert request.tool_choice == "auto"
        assert request.parallel_tool_calls is True
        assert request.include == ["reasoning.encrypted_content"]
        assert request.reasoning.summary == "auto"
        assert request.reasoning.effort is None
        assert request.instructions.startswith("You are a coding agent running in the Codex CLI")

    def test_captured_codex_input_items(self):
        """
        What it does: Checks the input item shapes Codex actually sends.
        Purpose: A developer message plus two user messages, each with
                 input_text parts.
        """
        print("Action: Inspecting captured input items...")
        request = ResponsesRequest(**CODEX_RESPONSES_REQUEST)

        roles = [item.role for item in request.input]
        assert roles == ["developer", "user", "user"]
        assert all(item.type == "message" for item in request.input)
        assert all(item.content[0].type == "input_text" for item in request.input)
        assert request.input[-1].content[0].text == "Say hello in exactly three words."

    def test_captured_codex_tool_shapes(self):
        """
        What it does: Checks the tool shapes Codex actually sends.
        Purpose: Flat function tools, one namespace container and one built-in
                 web_search descriptor must all survive validation.
        """
        print("Action: Inspecting captured tool shapes...")
        request = ResponsesRequest(**CODEX_RESPONSES_REQUEST)

        types = [tool.type for tool in request.tools]
        assert types.count("function") >= 4
        assert "namespace" in types
        assert "web_search" in types

        namespace_tool = next(tool for tool in request.tools if tool.type == "namespace")
        assert namespace_tool.name == "multi_agent_v1"
        assert len(namespace_tool.tools) == 5

        web_search_tool = next(tool for tool in request.tools if tool.type == "web_search")
        assert web_search_tool.name is None

    def test_captured_codex_follow_up_turn_validates(self):
        """
        What it does: Parses the follow-up turn Codex sends after a tool run.
        Purpose: function_call and function_call_output items must round-trip.
        """
        print("Action: Parsing the captured follow-up turn...")
        from codex_responses_payload import CODEX_RESPONSES_FOLLOW_UP_REQUEST

        request = ResponsesRequest(**CODEX_RESPONSES_FOLLOW_UP_REQUEST)

        types = [item.type for item in request.input]
        assert types == [
            "message",
            "message",
            "message",
            "message",
            "function_call",
            "function_call_output",
        ]

        call = next(item for item in request.input if item.type == "function_call")
        output = next(item for item in request.input if item.type == "function_call_output")
        assert call.call_id == output.call_id
        assert call.name == "exec_command"
        assert isinstance(output.output, str)


# =============================================================================
# Codex CLI code-mode shapes
# =============================================================================

class TestCodeModeModels:
    """Tests for the freeform-tool and additional_tools model fields."""

    def test_additional_tools_input_item_parses(self):
        """
        What it does: Parses an additional_tools input item.
        Purpose: Its ``tools`` payload must survive validation untouched so the
                 converter can read the declarations out of it.
        """
        print("Action: Parsing an additional_tools item...")
        item = ResponsesInputItem(
            type="additional_tools",
            id="at_1",
            role="developer",
            tools=[{"type": "namespace", "name": "functions", "tools": []}],
        )

        assert item.type == "additional_tools"
        assert item.role == "developer"
        assert item.tools[0]["name"] == "functions"

    def test_additional_tools_accepts_a_junk_payload_without_422(self):
        """
        What it does: Parses an additional_tools item with a non-list ``tools``.
        Purpose: A malformed upstream payload must be reportable by the
                 converter, not rejected as HTTP 422 before it is seen.
        """
        print("Action: Parsing a malformed additional_tools item...")
        item = ResponsesInputItem(type="additional_tools", tools="nonsense")

        assert item.tools == "nonsense"

    def test_custom_tool_call_input_item_parses(self):
        """
        What it does: Parses a custom_tool_call input item.
        Purpose: Codex replays these on every follow-up turn of a code-mode
                 session; losing them breaks the tool round trip.
        """
        print("Action: Parsing a custom_tool_call item...")
        item = ResponsesInputItem(
            type="custom_tool_call",
            id="ctc_1",
            status="completed",
            call_id="call_1",
            name="exec",
            namespace="functions",
            input="text(1);",
        )

        assert item.input == "text(1);"
        assert item.namespace == "functions"

    def test_custom_tool_call_output_input_item_parses(self):
        """
        What it does: Parses a custom_tool_call_output item with content items.
        Purpose: The freeform output uses the same encoding as
                 function_call_output: either a string or a content-item list.
        """
        print("Action: Parsing a custom_tool_call_output item...")
        item = ResponsesInputItem(
            type="custom_tool_call_output",
            call_id="call_1",
            output=[{"type": "input_text", "text": "Script completed"}],
        )

        assert item.call_id == "call_1"
        assert item.output[0]["text"] == "Script completed"

    def test_custom_tool_declaration_parses_with_its_format(self):
        """
        What it does: Parses a ``{"type": "custom"}`` tool.
        Purpose: The ``format`` block is the only description of the expected
                 payload syntax, so it must be typed and preserved.
        """
        print("Action: Parsing a custom tool declaration...")
        tool = ResponsesTool(
            type="custom",
            name="exec",
            description="Run JavaScript",
            format={"type": "grammar", "syntax": "lark", "definition": "start: SOURCE"},
        )

        assert tool.type == "custom"
        assert tool.format["syntax"] == "lark"
        assert tool.parameters is None

    def test_namespace_can_nest_a_custom_tool(self):
        """
        What it does: Parses a namespace containing a custom tool.
        Purpose: This is the exact code-mode shape (``functions.exec``).
        """
        print("Action: Parsing a namespace with a nested custom tool...")
        tool = ResponsesTool(
            type="namespace",
            name="functions",
            tools=[{"type": "custom", "name": "exec", "format": {"syntax": "lark"}}],
        )

        assert tool.tools[0].type == "custom"
        assert tool.tools[0].format["syntax"] == "lark"

    def test_custom_tool_call_output_item_serializes(self):
        """
        What it does: Serializes a ResponsesCustomToolCallItem.
        Purpose: Codex matches on ``type`` and reads ``input``; an ``arguments``
                 field or a wrong type would make it ignore the call.
        """
        print("Action: Serializing a custom_tool_call output item...")
        item = ResponsesCustomToolCallItem(
            id="ctc_1",
            name="exec",
            namespace="functions",
            input="text(1);",
            call_id="call_1",
        )

        payload = item.model_dump()
        assert payload["type"] == "custom_tool_call"
        assert payload["input"] == "text(1);"
        assert payload["status"] == "completed"
        assert "arguments" not in payload
        assert json.loads(json.dumps(payload))["call_id"] == "call_1"

    def test_custom_tool_call_output_item_defaults(self):
        """
        What it does: Builds the item with only an ID.
        Purpose: Defaults must never emit ``None`` where Codex expects a string.
        """
        print("Action: Building a minimal custom_tool_call item...")
        payload = ResponsesCustomToolCallItem(id="ctc_1").model_dump()

        assert payload["input"] == ""
        assert payload["name"] == ""
        assert payload["call_id"] == ""

    def test_captured_code_mode_request_parses(self):
        """
        What it does: Parses the real captured code-mode request.
        Purpose: No field of the live Codex body may cause HTTP 422.
        """
        print("Action: Parsing the captured code-mode request...")
        from codex_responses_payload import CODEX_RESPONSES_CODE_MODE_REQUEST

        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_REQUEST)

        assert not request.tools
        assert request.input[0].type == "additional_tools"
        namespaces = [entry["name"] for entry in request.input[0].tools]
        assert namespaces == ["functions", "collaboration"]

    def test_captured_code_mode_follow_up_parses(self):
        """
        What it does: Parses the real captured code-mode follow-up request.
        Purpose: The custom_tool_call round trip must survive validation with a
                 matching call_id on both items.
        """
        print("Action: Parsing the captured code-mode follow-up...")
        from codex_responses_payload import CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST

        request = ResponsesRequest(**CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST)

        call = next(item for item in request.input if item.type == "custom_tool_call")
        output = next(
            item for item in request.input if item.type == "custom_tool_call_output"
        )
        assert call.call_id == output.call_id
        assert call.name == "exec"
        assert call.namespace == "functions"
        assert isinstance(call.input, str) and call.input
