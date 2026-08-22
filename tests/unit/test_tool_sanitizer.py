# -*- coding: utf-8 -*-
"""
Unit tests for the tool_sanitizer module.

Tests for the rule-based Kiro tool specification sanitizer:
- Tool name rules (length boundaries, charset, empty names)
- Actionable violation messages
- Schema key rules (annotations, additionalProperties, empty required)
- Object schema root normalization
- Per-specification repairs (description, inputSchema)
- Duplicate tool collapsing
- Pass-through guarantee for already-valid specifications

Every constraint asserted here was verified against the live Kiro runtime:
names longer than 64 characters, names containing characters outside
[A-Za-z0-9_-], empty names, empty descriptions, a missing inputSchema, a
non-object schema root and duplicate tool names are all rejected upstream with
an opaque HTTP 400. Constructs verified as accepted ($ref/$defs, oneOf/anyOf/
allOf, const, format, pattern, propertyNames, long descriptions) must pass
through unchanged.
"""

import copy

import pytest

from kiro.tool_sanitizer import (
    DISALLOWED_SCHEMA_KEYS,
    KIRO_TOOL_NAME_MAX_LENGTH,
    REASON_NAME_EMPTY,
    REASON_NAME_INVALID_CHARS,
    REASON_NAME_TOO_LONG,
    SCHEMA_KEY_RULES,
    SPEC_REPAIR_RULES,
    TOOL_NAME_RULES,
    ToolSpecValidationError,
    build_violation_message,
    check_tool_names,
    enforce_tool_names,
    ensure_object_schema_root,
    sanitize_schema_node,
    sanitize_tool_specs,
)


def make_spec(name="TestTool", description="A test tool", schema=None):
    """
    Build a Kiro tool specification for tests.

    Args:
        name: Tool name.
        description: Tool description.
        schema: Input schema placed under inputSchema.json.

    Returns:
        A dict in Kiro toolSpecification format.
    """
    if schema is None:
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    return {
        "toolSpecification": {
            "name": name,
            "description": description,
            "inputSchema": {"json": schema},
        }
    }


# ==================================================================================================
# Tool name rules
# ==================================================================================================

class TestToolNameRules:
    """Tests for the tool name constraints enforced before sending to Kiro API."""

    def test_accepts_typical_names(self):
        """
        What it does: Verifies that ordinary tool names produce no violations.
        Purpose: Ensure valid input is never flagged.
        """
        names = ["Read", "get_weather", "Task-Create", "mcp__llm-memory__search_nodes", "1tool"]
        assert check_tool_names(names) == []

    def test_accepts_name_at_limit(self):
        """
        What it does: Verifies the exact 64-character boundary is accepted.
        Purpose: Lock the upper boundary confirmed against the live API.
        """
        assert check_tool_names(["a" * KIRO_TOOL_NAME_MAX_LENGTH]) == []

    def test_accepts_name_one_under_limit(self):
        """
        What it does: Verifies a 63-character name is accepted.
        Purpose: Guard the boundary from the other side.
        """
        assert check_tool_names(["a" * (KIRO_TOOL_NAME_MAX_LENGTH - 1)]) == []

    def test_rejects_name_one_over_limit(self):
        """
        What it does: Verifies a 65-character name is reported as too long.
        Purpose: Lock the exact point where Kiro API starts rejecting names.
        """
        violations = check_tool_names(["a" * (KIRO_TOOL_NAME_MAX_LENGTH + 1)])
        assert [v.reason for v in violations] == [REASON_NAME_TOO_LONG]
        assert "65 characters" in violations[0].detail

    @pytest.mark.parametrize("name,bad_char", [
        ("my.tool", "."),
        ("my tool", " "),
        ("my/tool", "/"),
        ("my:tool", ":"),
        ("my@tool", "@"),
        ("my#tool", "#"),
        ("tool!", "!"),
        ("инструмент", "и"),
        ("tool\u00e9", "\u00e9"),
    ])
    def test_rejects_invalid_characters(self, name, bad_char):
        """
        What it does: Verifies that names with unsupported characters are flagged.
        Purpose: Cover the charset constraint the upstream API enforces silently.
        """
        violations = check_tool_names([name])
        assert [v.reason for v in violations] == [REASON_NAME_INVALID_CHARS]
        assert repr(bad_char) in violations[0].detail

    @pytest.mark.parametrize("name", ["", "   ", "\t", "\n"])
    def test_rejects_empty_names(self, name):
        """
        What it does: Verifies empty and whitespace-only names are flagged.
        Purpose: Empty names are rejected upstream and cannot be repaired.
        """
        violations = check_tool_names([name])
        assert REASON_NAME_EMPTY in [v.reason for v in violations]

    def test_reports_all_violations_of_one_name(self):
        """
        What it does: Verifies a name that is both too long and malformed reports both.
        Purpose: Users should see every reason at once, not one per request.
        """
        name = "a." * 40  # 80 chars, contains dots
        reasons = {v.reason for v in check_tool_names([name])}
        assert reasons == {REASON_NAME_TOO_LONG, REASON_NAME_INVALID_CHARS}

    def test_reports_violations_for_every_tool(self):
        """
        What it does: Verifies violations from multiple tools are all collected.
        Purpose: Ensure the report is complete across the tool array.
        """
        violations = check_tool_names(["ok_name", "a" * 70, "bad.name", ""])
        assert len(violations) == 3

    def test_handles_non_string_name(self):
        """
        What it does: Verifies a None name does not crash the checker.
        Purpose: Malformed client input must produce a violation, not a TypeError.
        """
        violations = check_tool_names([None])
        assert REASON_NAME_EMPTY in [v.reason for v in violations]

    def test_rule_registry_is_not_empty(self):
        """
        What it does: Verifies the name rule registry is populated.
        Purpose: The rules engine must not silently degrade to a no-op.
        """
        assert len(TOOL_NAME_RULES) == 3


class TestEnforceToolNames:
    """Tests for enforce_tool_names raising an actionable error."""

    def test_passes_for_valid_names(self):
        """
        What it does: Verifies no exception is raised for valid names.
        Purpose: Valid requests must not be blocked.
        """
        enforce_tool_names(["Read", "Write"])

    def test_raises_value_error_subclass(self):
        """
        What it does: Verifies the raised error is a ValueError subclass.
        Purpose: Route handlers catch ValueError to return HTTP 400.
        """
        with pytest.raises(ToolSpecValidationError) as exc_info:
            enforce_tool_names(["bad.name"])
        assert isinstance(exc_info.value, ValueError)
        assert exc_info.value.violations[0].reason == REASON_NAME_INVALID_CHARS

    def test_error_message_lists_every_offender(self):
        """
        What it does: Verifies each offending tool appears in the message.
        Purpose: Actionable errors must name the tools to fix.
        """
        with pytest.raises(ToolSpecValidationError) as exc_info:
            enforce_tool_names(["a" * 65, "bad name", "ok"])
        message = str(exc_info.value)
        assert "a" * 65 in message
        assert "bad name" in message
        assert "Solution:" in message

    def test_logs_warning_per_violation(self):
        """
        What it does: Verifies a WARNING naming the tool is logged per violation.
        Purpose: Operators must see the cause in server logs (AGENTS.md section 9).
        """
        from loguru import logger

        records = []
        handler_id = logger.add(lambda message: records.append(message), level="WARNING")
        try:
            with pytest.raises(ToolSpecValidationError):
                enforce_tool_names(["bad.name", "a" * 70])
        finally:
            logger.remove(handler_id)

        text = "".join(records)
        assert "bad.name" in text
        assert "a" * 70 in text


class TestViolationMessage:
    """Tests for the user-facing violation message."""

    def test_empty_for_no_violations(self):
        """
        What it does: Verifies an empty message for an empty violation list.
        Purpose: No error text should be produced when nothing is wrong.
        """
        assert build_violation_message([]) == ""

    def test_length_section_wording(self):
        """
        What it does: Verifies the length section keeps its established wording.
        Purpose: The message is the documented contract for issue #41.
        """
        message = build_violation_message(check_tool_names(["a" * 68]))
        assert "exceed Kiro API limit of 64 characters" in message
        assert "68 characters" in message
        assert "Example:" in message

    def test_charset_section_names_allowed_characters(self):
        """
        What it does: Verifies the charset section explains what is allowed.
        Purpose: Users need to know how to rename their tools.
        """
        message = build_violation_message(check_tool_names(["bad.name"]))
        assert "underscores" in message and "hyphens" in message

    def test_empty_name_section_counts_tools(self):
        """
        What it does: Verifies empty names are reported with a count.
        Purpose: Unnamed tools cannot be identified by name, only counted.
        """
        message = build_violation_message(check_tool_names(["", ""]))
        assert "2 tool definition(s) have an empty name" in message

    def test_message_explains_why_names_are_not_rewritten(self):
        """
        What it does: Verifies the message explains the no-rename decision.
        Purpose: Users must understand why the gateway does not fix names itself.
        """
        message = build_violation_message(check_tool_names(["bad.name"]))
        assert "would not recognize" in message


# ==================================================================================================
# Schema key rules
# ==================================================================================================

class TestSanitizeSchemaNode:
    """Tests for recursive schema key sanitization."""

    def test_returns_empty_dict_for_none(self):
        """
        What it does: Verifies None input yields an empty dict.
        Purpose: Tools without arguments must not crash the sanitizer.
        """
        assert sanitize_schema_node(None) == {}

    def test_removes_schema_annotations(self):
        """
        What it does: Verifies annotation keywords are stripped.
        Purpose: Claude Code sends $schema on every tool definition.
        """
        schema = {"type": "object", "properties": {}}
        for key in DISALLOWED_SCHEMA_KEYS:
            schema[key] = "value"
        result = sanitize_schema_node(schema)
        for key in DISALLOWED_SCHEMA_KEYS:
            assert key not in result
        assert result["type"] == "object"

    def test_removes_empty_required_but_keeps_populated(self):
        """
        What it does: Verifies only an empty required array is removed.
        Purpose: Boundary between an invalid empty list and valid content.
        """
        assert "required" not in sanitize_schema_node({"type": "object", "required": []})
        assert sanitize_schema_node({"type": "object", "required": ["a"]})["required"] == ["a"]

    def test_removes_additional_properties_at_any_depth(self):
        """
        What it does: Verifies additionalProperties is stripped recursively.
        Purpose: Nested objects must be sanitized too.
        """
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "nested": {"type": "object", "additionalProperties": True, "properties": {}}
            },
        }
        result = sanitize_schema_node(schema)
        assert "additionalProperties" not in result
        assert "additionalProperties" not in result["properties"]["nested"]

    def test_sanitizes_inside_combinators(self):
        """
        What it does: Verifies list-valued combinators are sanitized element-wise.
        Purpose: anyOf/oneOf/allOf branches can carry disallowed keys.
        """
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "anyOf": [
                        {"type": "string", "$comment": "note", "enum": ["a"]},
                        {"type": "string", "const": "deleted"},
                    ]
                }
            },
        }
        branches = sanitize_schema_node(schema)["properties"]["status"]["anyOf"]
        assert "$comment" not in branches[0]
        assert branches[0]["enum"] == ["a"]
        assert branches[1]["const"] == "deleted"

    @pytest.mark.parametrize("key,value", [
        ("$ref", "#/$defs/Item"),
        ("format", "uri"),
        ("pattern", "^wf_[a-z0-9-]{6,}$"),
        ("const", "deleted"),
        ("enum", ["a", "b"]),
        ("minLength", 2),
        ("maxLength", 524288),
        ("minimum", 0),
        ("maximum", 9007199254740991),
        ("exclusiveMinimum", 0),
        ("minItems", 1),
        ("maxItems", 4),
        ("default", False),
        ("propertyNames", {"type": "string"}),
        ("x-vendor-extension", "kept"),
    ])
    def test_preserves_accepted_constructs(self, key, value):
        """
        What it does: Verifies constructs accepted upstream are not touched.
        Purpose: No gratuitous mutation of user input (AGENTS.md section 2).
        """
        schema = {"type": "object", "properties": {"a": {"type": "string", key: value}}}
        result = sanitize_schema_node(schema)
        assert result["properties"]["a"][key] == value

    def test_does_not_mutate_input(self):
        """
        What it does: Verifies the input schema object is left untouched.
        Purpose: Callers may reuse the original request objects.
        """
        schema = {"type": "object", "$schema": "https://x", "properties": {}}
        snapshot = copy.deepcopy(schema)
        sanitize_schema_node(schema)
        assert schema == snapshot

    def test_deeply_nested_schema_is_sanitized(self):
        """
        What it does: Verifies recursion reaches deeply nested nodes.
        Purpose: Real MCP schemas nest several levels deep.
        """
        node = {"type": "string", "$id": "leaf"}
        for _ in range(8):
            node = {"type": "object", "$comment": "x", "properties": {"n": node}}
        result = sanitize_schema_node(node)
        for _ in range(8):
            assert "$comment" not in result
            result = result["properties"]["n"]
        assert "$id" not in result

    def test_rule_registry_is_not_empty(self):
        """
        What it does: Verifies the schema rule registry is populated.
        Purpose: The rules engine must not degrade to a no-op.
        """
        assert len(SCHEMA_KEY_RULES) == 3


class TestEnsureObjectSchemaRoot:
    """Tests for object root normalization required by Bedrock."""

    @pytest.mark.parametrize("schema", [None, {}, {"properties": {"a": {"type": "string"}}}])
    def test_adds_missing_object_type(self, schema):
        """
        What it does: Verifies a missing root type becomes "object".
        Purpose: Bedrock rejects any other root type for tool schemas.
        """
        assert ensure_object_schema_root(schema)["type"] == "object"

    def test_overrides_non_object_root_type(self):
        """
        What it does: Verifies a non-object root type is replaced.
        Purpose: A scalar root makes no sense for an argument container.
        """
        result = ensure_object_schema_root({"type": "string", "minLength": 1})
        assert result["type"] == "object"
        assert result["properties"] == {}

    def test_adds_missing_properties_map(self):
        """
        What it does: Verifies a properties map is always present.
        Purpose: Bedrock expects the key for object schemas.
        """
        assert ensure_object_schema_root({"type": "object"})["properties"] == {}

    def test_replaces_non_dict_properties(self):
        """
        What it does: Verifies a malformed properties value is replaced.
        Purpose: Defend against clients sending a list or string.
        """
        assert ensure_object_schema_root({"type": "object", "properties": []})["properties"] == {}

    def test_preserves_existing_properties_and_required(self):
        """
        What it does: Verifies valid object schemas pass through unchanged.
        Purpose: Regression guard against gratuitous mutation.
        """
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        assert ensure_object_schema_root(schema) == schema

    def test_does_not_mutate_input(self):
        """
        What it does: Verifies the input dict is not modified in place.
        Purpose: Callers may reuse the original schema.
        """
        schema = {"type": "string"}
        ensure_object_schema_root(schema)
        assert schema == {"type": "string"}


# ==================================================================================================
# Specification-level sanitization
# ==================================================================================================

class TestSanitizeToolSpecs:
    """Tests for the full tool specification pipeline."""

    def test_empty_input_returns_empty_result(self):
        """
        What it does: Verifies empty input produces empty output.
        Purpose: Requests without tools must be unaffected.
        """
        result = sanitize_tool_specs([])
        assert result.specs == []
        assert result.repairs == []

    def test_valid_specs_pass_through_unchanged(self):
        """
        What it does: Verifies already-valid specs are returned untouched.
        Purpose: Regression guard - no gratuitous mutation of user input.
        """
        specs = [
            make_spec("Read", "Read a file", {
                "type": "object",
                "properties": {"path": {"type": "string", "format": "uri"}},
                "required": ["path"],
            }),
            make_spec("Write", "Write a file"),
        ]
        snapshot = copy.deepcopy(specs)
        result = sanitize_tool_specs(specs)
        assert result.specs == snapshot
        assert result.repairs == []

    def test_replaces_empty_description(self):
        """
        What it does: Verifies an empty description is replaced with a placeholder.
        Purpose: Kiro API rejects tools whose description is an empty string.
        """
        result = sanitize_tool_specs([make_spec("Read", "")])
        assert result.specs[0]["toolSpecification"]["description"] == "Tool: Read"
        assert [r.rule for r in result.repairs] == ["non_empty_description"]

    @pytest.mark.parametrize("description", ["", "   ", "\n\t", None])
    def test_replaces_blank_or_missing_description(self, description):
        """
        What it does: Verifies blank and missing descriptions are replaced.
        Purpose: Cover every falsy description variant clients send.
        """
        result = sanitize_tool_specs([make_spec("Read", description)])
        assert result.specs[0]["toolSpecification"]["description"] == "Tool: Read"

    def test_keeps_relocation_placeholder_description(self):
        """
        What it does: Verifies the long-description relocation placeholder survives.
        Purpose: process_tools_with_long_descriptions leaves this text behind and
                 it must not be overwritten.
        """
        placeholder = "[Full documentation in system prompt under '## Tool: Workflow']"
        result = sanitize_tool_specs([make_spec("Workflow", placeholder)])
        assert result.specs[0]["toolSpecification"]["description"] == placeholder
        assert result.repairs == []

    def test_keeps_very_long_description(self):
        """
        What it does: Verifies long descriptions are not truncated.
        Purpose: Descriptions of 100k characters were accepted upstream.
        """
        long_description = "x" * 100000
        result = sanitize_tool_specs([make_spec("Read", long_description)])
        assert result.specs[0]["toolSpecification"]["description"] == long_description

    def test_repairs_missing_input_schema(self):
        """
        What it does: Verifies a missing inputSchema is replaced with an object schema.
        Purpose: A missing inputSchema is rejected with "Improperly formed request".
        """
        result = sanitize_tool_specs([{"toolSpecification": {"name": "NoSchema", "description": "d"}}])
        assert result.specs[0]["toolSpecification"]["inputSchema"] == {
            "json": {"type": "object", "properties": {}}
        }
        assert [r.rule for r in result.repairs] == ["object_input_schema"]

    def test_repairs_none_input_schema(self):
        """
        What it does: Verifies inputSchema.json set to None is repaired.
        Purpose: Tools without arguments often arrive with a null schema.
        """
        result = sanitize_tool_specs([make_spec("NoArgs", "d", None)])
        assert result.specs[0]["toolSpecification"]["inputSchema"]["json"]["type"] == "object"

    def test_repairs_non_object_schema_root(self):
        """
        What it does: Verifies a scalar schema root is normalized to object.
        Purpose: Bedrock requires inputSchema.json.type == "object".
        """
        result = sanitize_tool_specs([make_spec("BadRoot", "d", {"type": "string"})])
        json_schema = result.specs[0]["toolSpecification"]["inputSchema"]["json"]
        assert json_schema == {"type": "object", "properties": {}}

    def test_strips_schema_annotations_from_spec(self):
        """
        What it does: Verifies schema key rules are applied through the pipeline.
        Purpose: The spec path and the schema path must share one rule set.
        """
        result = sanitize_tool_specs([make_spec("Read", "d", {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
            "required": [],
        })])
        json_schema = result.specs[0]["toolSpecification"]["inputSchema"]["json"]
        assert json_schema == {"type": "object", "properties": {"a": {"type": "string"}}}
        assert [r.rule for r in result.repairs] == ["object_input_schema"]

    def test_collapses_duplicate_names(self):
        """
        What it does: Verifies duplicate tool names are collapsed to the first.
        Purpose: Bedrock rejects duplicates with TOOL_DUPLICATE.
        """
        result = sanitize_tool_specs([
            make_spec("Dupe", "First"),
            make_spec("Other", "Other"),
            make_spec("Dupe", "Second"),
        ])
        names = [s["toolSpecification"]["name"] for s in result.specs]
        assert names == ["Dupe", "Other"]
        assert result.specs[0]["toolSpecification"]["description"] == "First"
        assert [r.rule for r in result.repairs] == ["unique_names"]

    def test_duplicate_collapse_is_reported_as_notable(self):
        """
        What it does: Verifies dropping a duplicate is flagged for a WARNING.
        Purpose: Users must never lose a tool definition silently.
        """
        result = sanitize_tool_specs([make_spec("Dupe"), make_spec("Dupe")])
        assert result.repairs[0].notable is True
        assert "Dupe" == result.repairs[0].tool_name

    def test_collapses_three_identical_names(self):
        """
        What it does: Verifies more than one duplicate is handled.
        Purpose: Ensure the counter, not a boolean, drives the logic.
        """
        result = sanitize_tool_specs([make_spec("Dupe") for _ in range(3)])
        assert len(result.specs) == 1
        assert len(result.repairs) == 2

    def test_raises_for_invalid_name(self):
        """
        What it does: Verifies name violations abort sanitization.
        Purpose: An unrepairable spec must produce an actionable error.
        """
        with pytest.raises(ToolSpecValidationError):
            sanitize_tool_specs([make_spec("bad.name")])

    def test_raises_before_applying_repairs(self):
        """
        What it does: Verifies input is untouched when a name is invalid.
        Purpose: Failure must not leave partially mutated specs behind.
        """
        specs = [make_spec("Read", ""), make_spec("a" * 65)]
        snapshot = copy.deepcopy(specs)
        with pytest.raises(ToolSpecValidationError):
            sanitize_tool_specs(specs)
        assert specs == snapshot

    def test_does_not_mutate_input_specs(self):
        """
        What it does: Verifies repairs are applied to copies.
        Purpose: The caller's tool list must remain usable and unchanged.
        """
        specs = [make_spec("Read", "", {"type": "string"})]
        snapshot = copy.deepcopy(specs)
        sanitize_tool_specs(specs)
        assert specs == snapshot

    def test_handles_large_tool_arrays(self):
        """
        What it does: Verifies a large tool array is processed intact.
        Purpose: 200 tools were accepted upstream; there is no count limit to enforce.
        """
        specs = [make_spec(f"Tool{i:03d}", f"Tool {i}") for i in range(200)]
        result = sanitize_tool_specs(specs)
        assert len(result.specs) == 200
        assert result.repairs == []

    def test_repair_rule_registry_is_not_empty(self):
        """
        What it does: Verifies the spec repair registry is populated.
        Purpose: The rules engine must not degrade to a no-op.
        """
        assert len(SPEC_REPAIR_RULES) == 2
