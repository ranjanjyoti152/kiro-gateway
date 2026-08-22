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
Rule-based sanitizer for Kiro tool specifications.

Kiro API (and the Bedrock runtime behind it) validates tool specifications far
more strictly than the JSON Schema / OpenAI / Anthropic tool formats that
clients send, and it reports every violation with the same opaque HTTP 400:

    {"message": "Improperly formed request.", "reason": "REQUEST_BODY_INVALID"}
    {"message": "Invalid tool use format.",   "reason": "REQUEST_BODY_INVALID"}
    {"message": "Bedrock error message: ... must be one of the following: object.", ...}
    {"message": "Bedrock error message: The tool X is already defined ...", "reason": "TOOL_DUPLICATE"}

Because the upstream error never names the offending tool or field, every new
client integration risks re-discovering the same constraints by trial and
error. This module therefore centralizes the constraints as an explicit,
extensible rule set with two categories:

**Repair rules** — violations that can be fixed without changing the meaning of
the user's tool set (empty descriptions, a non-object schema root, schema
annotation keywords, duplicate definitions). These are applied silently at
DEBUG level, or with a WARNING when a tool definition is affected in a way the
user should know about.

**Rejection rules** — violations of the tool *name* contract. A name cannot be
repaired without breaking the round trip: the model would emit a tool call
under the rewritten name and the client, which never declared that name, could
not match it back. Renaming would therefore trade a clear error for silent,
hard-to-debug tool failures. Instead, these are collected and surfaced as one
actionable error listing every offending tool and how to fix it.

All constraints encoded here were verified empirically against
``runtime.us-east-1.kiro.dev/generateAssistantResponse``:

======================================  ==========================================
Constraint                              Observed upstream behaviour
======================================  ==========================================
name length <= 64                       65+ chars -> "Invalid tool use format."
name charset [A-Za-z0-9_-]              '.', ' ', '/', ':', non-ASCII -> rejected
name non-empty                          "" -> "Invalid tool use format."
description non-empty                   "" -> "Invalid tool use format."
                                        (whitespace-only and >100k are accepted,
                                        but are normalized for safety)
inputSchema present                     missing key -> "Improperly formed request."
inputSchema.json.type == "object"        missing/other type -> Bedrock type error
unique names                            duplicates -> TOOL_DUPLICATE
======================================  ==========================================

Constructs that were verified as ACCEPTED and are therefore left untouched (no
gratuitous mutation of user input): ``$ref``/``$defs``, ``oneOf``/``anyOf``/
``allOf``, ``const``, ``format``, ``pattern``, ``propertyNames``, vendor
extension keywords, ``type`` arrays, deep nesting, long descriptions, and large
tool counts (200 tools were accepted).

Adding a new constraint means appending a rule to
:data:`SCHEMA_KEY_RULES`, :data:`SPEC_REPAIR_RULES` or
:data:`TOOL_NAME_RULES` — no changes to the conversion pipeline are required.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


# ==================================================================================================
# Kiro Tool Specification Constraints
# ==================================================================================================

#: Maximum number of characters Kiro API accepts in a tool name.
KIRO_TOOL_NAME_MAX_LENGTH: int = 64

#: Characters Kiro API accepts in a tool name (letters, digits, underscore, hyphen).
KIRO_TOOL_NAME_PATTERN: re.Pattern = re.compile(r"^[A-Za-z0-9_-]+$")

#: Human-readable description of the accepted charset, used in error messages.
KIRO_TOOL_NAME_CHARSET_TEXT: str = "letters, digits, underscores ('_') and hyphens ('-')"

# JSON Schema meta/annotation keywords that are stripped from tool inputSchema.
#
# These are pure metadata: they carry no structural meaning for argument
# validation, so removing them preserves the user's intent. Claude Code sends
# "$schema" on every tool definition
# (e.g. "https://json-schema.org/draft/2020-12/schema").
#
# NOTE: Structural "$"-keywords like "$ref"/"$defs" are intentionally NOT listed
# here - they are accepted upstream and removing them would corrupt the schema.
DISALLOWED_SCHEMA_KEYS: frozenset = frozenset({
    "$schema",
    "$id",
    "$anchor",
    "$comment",
})


# ==================================================================================================
# Violations and Repairs
# ==================================================================================================

#: Violation reason: tool name is empty or whitespace-only.
REASON_NAME_EMPTY: str = "empty_name"

#: Violation reason: tool name exceeds KIRO_TOOL_NAME_MAX_LENGTH.
REASON_NAME_TOO_LONG: str = "name_too_long"

#: Violation reason: tool name contains characters outside the accepted charset.
REASON_NAME_INVALID_CHARS: str = "name_invalid_characters"


@dataclass(frozen=True)
class ToolViolation:
    """
    An unrepairable tool specification violation.

    Attributes:
        tool_name: The offending tool name (may be empty for unnamed tools).
        reason: One of the ``REASON_*`` constants.
        detail: Human-readable explanation, shown to the user.
    """
    tool_name: str
    reason: str
    detail: str


@dataclass(frozen=True)
class ToolRepair:
    """
    A repair applied to a tool specification.

    Attributes:
        tool_name: Name of the tool that was repaired.
        rule: Identifier of the rule that performed the repair.
        detail: Human-readable explanation of what was changed and why.
        notable: True when the user should be told about it (logged as WARNING),
            False for routine normalization (logged as DEBUG).
    """
    tool_name: str
    rule: str
    detail: str
    notable: bool = False


@dataclass
class ToolSanitizationResult:
    """
    Outcome of sanitizing a list of Kiro tool specifications.

    Attributes:
        specs: Sanitized tool specifications, ready for the Kiro payload.
        repairs: Repairs that were applied (for logging / diagnostics).
    """
    specs: List[Dict[str, Any]] = field(default_factory=list)
    repairs: List[ToolRepair] = field(default_factory=list)


class ToolSpecValidationError(ValueError):
    """
    Raised when tool specifications contain violations that cannot be repaired.

    Subclasses ``ValueError`` so the existing route error handling (OpenAI and
    Anthropic, streaming and non-streaming) converts it into an HTTP 400
    ``invalid_request_error`` carrying the actionable message.

    Attributes:
        violations: All detected violations, in input order.
    """

    def __init__(self, violations: List[ToolViolation]):
        """
        Initialize the error with the full list of violations.

        Args:
            violations: Detected unrepairable violations.
        """
        self.violations: List[ToolViolation] = list(violations)
        super().__init__(build_violation_message(self.violations))


# ==================================================================================================
# Tool Name Rules (rejection rules)
# ==================================================================================================

@dataclass(frozen=True)
class ToolNameRule:
    """
    A single tool name constraint.

    Attributes:
        reason: One of the ``REASON_*`` constants, used to group violations.
        check: Returns True when the name VIOLATES this rule.
        describe: Builds the per-tool detail text shown to the user.
    """
    reason: str
    check: Callable[[str], bool]
    describe: Callable[[str], str]


def _find_invalid_name_characters(name: str) -> List[str]:
    """
    Collect the distinct characters of a tool name that Kiro API rejects.

    Args:
        name: Tool name to inspect.

    Returns:
        Distinct offending characters, in order of first appearance.

    Examples:
        >>> _find_invalid_name_characters("my.tool name")
        ['.', ' ']
        >>> _find_invalid_name_characters("valid_name-1")
        []
    """
    invalid: List[str] = []
    for char in name:
        if not KIRO_TOOL_NAME_PATTERN.match(char) and char not in invalid:
            invalid.append(char)
    return invalid


def _describe_invalid_characters(name: str) -> str:
    """
    Build the detail text for a tool name with rejected characters.

    Args:
        name: Offending tool name.

    Returns:
        Detail text listing the rejected characters.
    """
    invalid = _find_invalid_name_characters(name)
    rendered = ", ".join(repr(char) for char in invalid)
    return f"contains character(s) Kiro API rejects: {rendered}"


#: Ordered tool name constraints. Each name is checked against every rule, so a
#: name that is both too long and malformed reports both problems.
TOOL_NAME_RULES: Tuple[ToolNameRule, ...] = (
    ToolNameRule(
        reason=REASON_NAME_EMPTY,
        check=lambda name: not name.strip(),
        describe=lambda name: "is empty (Kiro API requires a non-empty tool name)",
    ),
    ToolNameRule(
        reason=REASON_NAME_TOO_LONG,
        check=lambda name: len(name) > KIRO_TOOL_NAME_MAX_LENGTH,
        describe=lambda name: f"{len(name)} characters",
    ),
    ToolNameRule(
        reason=REASON_NAME_INVALID_CHARS,
        check=lambda name: bool(name.strip()) and not KIRO_TOOL_NAME_PATTERN.match(name),
        describe=_describe_invalid_characters,
    ),
)


def check_tool_names(names: List[str]) -> List[ToolViolation]:
    """
    Validate tool names against every rule in :data:`TOOL_NAME_RULES`.

    Args:
        names: Tool names to validate, in request order.

    Returns:
        All detected violations. Empty list when every name is acceptable.

    Examples:
        >>> check_tool_names(["get_weather"])
        []
        >>> [v.reason for v in check_tool_names(["bad.name"])]
        ['name_invalid_characters']
        >>> [v.reason for v in check_tool_names(["a" * 65])]
        ['name_too_long']
    """
    violations: List[ToolViolation] = []

    for name in names:
        safe_name = name if isinstance(name, str) else str(name or "")
        for rule in TOOL_NAME_RULES:
            if rule.check(safe_name):
                violations.append(ToolViolation(
                    tool_name=safe_name,
                    reason=rule.reason,
                    detail=rule.describe(safe_name),
                ))

    return violations


def build_violation_message(violations: List[ToolViolation]) -> str:
    """
    Build a single actionable error message from tool name violations.

    Violations are grouped by reason so the user sees every offending tool at
    once instead of fixing them one request at a time (AGENTS.md section 9).

    Args:
        violations: Violations to render.

    Returns:
        Multi-line, user-facing error message. Empty string when there are no
        violations.

    Examples:
        >>> msg = build_violation_message(check_tool_names(["a" * 65]))
        >>> "exceed Kiro API limit of 64 characters" in msg
        True
        >>> "Solution:" in msg
        True
    """
    if not violations:
        return ""

    sections: List[str] = []

    too_long = [v for v in violations if v.reason == REASON_NAME_TOO_LONG]
    if too_long:
        listed = "\n".join(f"  - '{v.tool_name}' ({v.detail})" for v in too_long)
        sections.append(
            f"Tool name(s) exceed Kiro API limit of {KIRO_TOOL_NAME_MAX_LENGTH} characters:\n{listed}"
        )

    invalid_chars = [v for v in violations if v.reason == REASON_NAME_INVALID_CHARS]
    if invalid_chars:
        listed = "\n".join(f"  - '{v.tool_name}' ({v.detail})" for v in invalid_chars)
        sections.append(
            f"Tool name(s) use characters Kiro API does not accept "
            f"(allowed: {KIRO_TOOL_NAME_CHARSET_TEXT}):\n{listed}"
        )

    empty_names = [v for v in violations if v.reason == REASON_NAME_EMPTY]
    if empty_names:
        sections.append(
            f"{len(empty_names)} tool definition(s) have an empty name, "
            f"which Kiro API rejects."
        )

    solution = (
        f"Solution: Use tool names of at most {KIRO_TOOL_NAME_MAX_LENGTH} characters "
        f"built only from {KIRO_TOOL_NAME_CHARSET_TEXT}.\n"
        f"Example: 'get_user_data' instead of "
        f"'get.authenticated.user profile data with extended information about it'\n"
        f"Tool names cannot be rewritten automatically: the model would answer with the "
        f"rewritten name and your client would not recognize the tool call."
    )

    return "\n\n".join(sections) + "\n\n" + solution


def enforce_tool_names(names: List[str]) -> None:
    """
    Validate tool names and raise when any violation is found.

    Each violation is logged as a WARNING naming the tool and the reason, so
    operators can see the cause in the server log even when the client only
    shows the HTTP error.

    Args:
        names: Tool names to validate, in request order.

    Raises:
        ToolSpecValidationError: If any name violates a rule in
            :data:`TOOL_NAME_RULES`.

    Examples:
        >>> enforce_tool_names(["get_weather", "Read"])
    """
    violations = check_tool_names(names)
    if not violations:
        return

    for violation in violations:
        logger.warning(
            f"Tool '{violation.tool_name}' cannot be sent to Kiro API: {violation.detail} "
            f"(rule: {violation.reason})"
        )

    raise ToolSpecValidationError(violations)


# ==================================================================================================
# Schema Key Rules (repair rules)
# ==================================================================================================

@dataclass(frozen=True)
class SchemaKeyRule:
    """
    A rule that removes one class of keys from a tool input schema node.

    Attributes:
        rule_id: Stable identifier used in debug logs.
        matches: Returns True when the (key, value) pair must be removed.
        reason: Why Kiro API cannot receive this key.
    """
    rule_id: str
    matches: Callable[[str, Any], bool]
    reason: str


#: Ordered schema key constraints, applied recursively to every schema node.
SCHEMA_KEY_RULES: Tuple[SchemaKeyRule, ...] = (
    SchemaKeyRule(
        rule_id="empty_required",
        matches=lambda key, value: key == "required" and isinstance(value, list) and len(value) == 0,
        reason="Kiro API rejects an empty 'required' array",
    ),
    SchemaKeyRule(
        rule_id="additional_properties",
        matches=lambda key, value: key == "additionalProperties",
        reason="Kiro API does not support 'additionalProperties'",
    ),
    SchemaKeyRule(
        rule_id="schema_annotations",
        matches=lambda key, value: key in DISALLOWED_SCHEMA_KEYS,
        reason="JSON Schema annotation keywords are metadata Kiro API does not accept",
    ),
)


def sanitize_schema_node(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recursively strip keys rejected by Kiro API from a tool input schema.

    Every :data:`SCHEMA_KEY_RULES` entry is applied to each node, including
    nested ``properties`` values and list-valued combinators
    (``anyOf``/``oneOf``/``allOf``). Everything else is copied through
    unchanged, so legitimate constructs such as ``$ref``, ``format``,
    ``pattern`` and ``const`` are preserved.

    Args:
        schema: JSON Schema to sanitize (may be ``None`` or empty).

    Returns:
        Sanitized copy of the schema. Empty dict for empty input.

    Examples:
        >>> sanitize_schema_node({"type": "object", "$schema": "https://x", "required": []})
        {'type': 'object'}
        >>> sanitize_schema_node({"type": "string", "format": "uri"})
        {'type': 'string', 'format': 'uri'}
    """
    if not schema:
        return {}

    result: Dict[str, Any] = {}

    for key, value in schema.items():
        if any(rule.matches(key, value) for rule in SCHEMA_KEY_RULES):
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {
                prop_name: sanitize_schema_node(prop_value) if isinstance(prop_value, dict) else prop_value
                for prop_name, prop_value in value.items()
            }
        elif isinstance(value, dict):
            result[key] = sanitize_schema_node(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_schema_node(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def ensure_object_schema_root(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Guarantee a tool root input schema is a valid JSON Schema object.

    Bedrock (the engine behind Kiro's runtime) requires every tool's
    ``inputSchema.json.type`` to be exactly ``"object"`` and to carry a
    ``properties`` map. Real clients sometimes send tool schemas that violate
    this, which surfaces as:

        "The value at toolConfig.tools.N.toolSpec.inputSchema.json.type
         must be one of the following: object."

    Common violations this normalizes:
    - Missing ``type`` (e.g. a bare ``{}`` or only ``{"properties": {...}}``)
    - A non-object ``type`` (e.g. ``"string"``) at the schema root, which makes
      no sense for a tool's argument container
    - Missing ``properties`` for an object schema (Bedrock expects the key)

    This applies ONLY to the root of a tool input schema, never recursively, so
    nested property schemas keep their own legitimate non-object types.

    Args:
        schema: A sanitized tool input schema (may be empty or partial).

    Returns:
        A schema dict whose root is ``type: "object"`` with a ``properties`` map.

    Examples:
        >>> ensure_object_schema_root({})
        {'type': 'object', 'properties': {}}
        >>> ensure_object_schema_root({"properties": {"x": {"type": "string"}}})
        {'properties': {'x': {'type': 'string'}}, 'type': 'object'}
        >>> ensure_object_schema_root({"type": "string"})
        {'type': 'object', 'properties': {}}
    """
    if not schema or not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    result = dict(schema)

    if result.get("type") != "object":
        if "type" in result:
            logger.debug(
                f"Tool input schema root type was {result.get('type')!r}; "
                f"normalizing to 'object' for Bedrock compatibility"
            )
        result["type"] = "object"

    if not isinstance(result.get("properties"), dict):
        result["properties"] = {}

    return result


# ==================================================================================================
# Specification Repair Rules
# ==================================================================================================

@dataclass(frozen=True)
class SpecRepairRule:
    """
    A rule that repairs one field of a single Kiro tool specification.

    Attributes:
        rule_id: Stable identifier used in logs and tests.
        repair: Mutates the ``toolSpecification`` dict in place and returns a
            detail string when a repair was applied, or ``None`` when the spec
            already satisfied the rule.
        notable: True when an applied repair deserves a WARNING rather than a
            DEBUG log line.
    """
    rule_id: str
    repair: Callable[[Dict[str, Any]], Optional[str]]
    notable: bool = False


def _repair_description(spec: Dict[str, Any]) -> Optional[str]:
    """
    Ensure the tool description is a non-empty string.

    Kiro API rejects a tool whose description is an empty string
    ("Invalid tool use format."). A whitespace-only description is accepted
    upstream but carries no information for the model, so it is normalized too.

    Args:
        spec: The ``toolSpecification`` dict to repair in place.

    Returns:
        Detail string when a placeholder was substituted, otherwise ``None``.
    """
    description = spec.get("description")
    if isinstance(description, str) and description.strip():
        return None

    spec["description"] = f"Tool: {spec.get('name', '')}"
    return "empty description replaced with a placeholder"


def _repair_input_schema(spec: Dict[str, Any]) -> Optional[str]:
    """
    Ensure ``inputSchema.json`` exists, is sanitized and has an object root.

    Args:
        spec: The ``toolSpecification`` dict to repair in place.

    Returns:
        Detail string when the schema had to be changed, otherwise ``None``.
    """
    input_schema = spec.get("inputSchema")
    original = input_schema.get("json") if isinstance(input_schema, dict) else None

    sanitized = sanitize_schema_node(original if isinstance(original, dict) else None)
    sanitized = ensure_object_schema_root(sanitized)

    spec["inputSchema"] = {"json": sanitized}

    if not isinstance(original, dict):
        return "missing or malformed inputSchema replaced with an empty object schema"
    if sanitized != original:
        return "inputSchema normalized for Kiro API (annotations stripped / object root enforced)"
    return None


#: Ordered per-specification repair rules.
SPEC_REPAIR_RULES: Tuple[SpecRepairRule, ...] = (
    SpecRepairRule(rule_id="non_empty_description", repair=_repair_description),
    SpecRepairRule(rule_id="object_input_schema", repair=_repair_input_schema),
)


# ==================================================================================================
# Collection-Level Sanitization
# ==================================================================================================

def sanitize_tool_specs(specs: List[Dict[str, Any]]) -> ToolSanitizationResult:
    """
    Apply the full rule set to a list of Kiro tool specifications.

    Pipeline:
    1. Validate every tool name (raises on unrepairable violations).
    2. Apply :data:`SPEC_REPAIR_RULES` to each specification.
    3. Collapse duplicate names, which Bedrock rejects with ``TOOL_DUPLICATE``.
       The first definition wins; a duplicate is unreachable for the model
       anyway because a tool call can only reference one name.

    The same result is produced for both API surfaces (OpenAI and Anthropic)
    and both request modes, because every path builds its payload through this
    function.

    Args:
        specs: Tool specifications in Kiro format
            (``[{"toolSpecification": {...}}, ...]``).

    Returns:
        ToolSanitizationResult with the sanitized specs and applied repairs.

    Raises:
        ToolSpecValidationError: If any tool name violates a name rule.

    Examples:
        >>> result = sanitize_tool_specs([
        ...     {"toolSpecification": {"name": "Read", "description": "", "inputSchema": {"json": {}}}}
        ... ])
        >>> result.specs[0]["toolSpecification"]["description"]
        'Tool: Read'
        >>> result.specs[0]["toolSpecification"]["inputSchema"]
        {'json': {'type': 'object', 'properties': {}}}
    """
    if not specs:
        return ToolSanitizationResult(specs=[], repairs=[])

    names = [
        (spec.get("toolSpecification") or {}).get("name", "") or ""
        for spec in specs
    ]
    enforce_tool_names(names)

    repairs: List[ToolRepair] = []
    sanitized_specs: List[Dict[str, Any]] = []
    seen_names: Dict[str, int] = {}

    for spec in specs:
        tool_spec = dict(spec.get("toolSpecification") or {})
        name = tool_spec.get("name", "")

        if name in seen_names:
            repairs.append(ToolRepair(
                tool_name=name,
                rule="unique_names",
                detail=(
                    f"duplicate definition dropped (Kiro API rejects duplicate tool names; "
                    f"keeping the first of {seen_names[name] + 1} definitions)"
                ),
                notable=True,
            ))
            seen_names[name] += 1
            continue

        seen_names[name] = 1

        for rule in SPEC_REPAIR_RULES:
            detail = rule.repair(tool_spec)
            if detail:
                repairs.append(ToolRepair(
                    tool_name=name,
                    rule=rule.rule_id,
                    detail=detail,
                    notable=rule.notable,
                ))

        sanitized_specs.append({"toolSpecification": tool_spec})

    _log_repairs(repairs, total_tools=len(specs))

    return ToolSanitizationResult(specs=sanitized_specs, repairs=repairs)


def _log_repairs(repairs: List[ToolRepair], total_tools: int) -> None:
    """
    Log applied repairs at the appropriate level.

    Notable repairs (those that change what the user asked for, such as
    dropping a duplicate definition) are logged as WARNING naming the tool and
    the reason. Routine normalization is logged at DEBUG to keep production
    logs quiet.

    Args:
        repairs: Repairs applied by :func:`sanitize_tool_specs`.
        total_tools: Number of tool specifications that were processed.

    Returns:
        None
    """
    if not repairs:
        return

    for repair in repairs:
        message = f"Tool '{repair.tool_name}': {repair.detail} (rule: {repair.rule})"
        if repair.notable:
            logger.warning(message)
        else:
            logger.debug(message)

    logger.debug(f"Sanitized {len(repairs)} tool spec issue(s) across {total_tools} tool(s)")
