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
Converters for transforming the OpenAI Responses API format to Kiro format.

This module is the adapter layer for ``POST /v1/responses``. It converts the
Responses request shape into the API-agnostic unified format consumed by
``kiro/converters_core.py``, exactly like ``converters_openai.py`` does for Chat
Completions and ``converters_anthropic.py`` for the Messages API.

Responsibilities:

- Fold ``instructions`` (and any ``system``/``developer`` input items) into the
  Kiro system prompt.
- Flatten the ordered ``input`` item list into alternating conversation turns,
  mapping ``function_call`` / ``function_call_output`` items onto the unified
  ``tool_calls`` / ``tool_results`` structures so tool round-trips survive.
- Flatten flat, namespaced and freeform (``custom``) tool specifications into
  ``UnifiedTool`` and keep a routing table so tool calls can be reported back
  under the exact ``name``/``namespace`` pair the client declared, and as the
  item type the client dispatches on.
- Merge tool declarations that arrive inside the ``input`` array as an
  ``additional_tools`` item into the same tool pipeline. Codex CLI code mode
  sends an empty top-level ``tools`` array and declares everything there, so
  ignoring the item leaves the model with no tools at all.
- Map ``reasoning.effort`` onto ``ThinkingConfig``.
- Name every accepted-but-ignored request field in a single WARNING, and reject
  ``previous_response_id`` with an actionable error instead of answering from a
  conversation the gateway never stored.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import HIDDEN_MODELS, WEB_SEARCH_ENABLED
from kiro.converters_core import (
    ThinkingConfig,
    UnifiedMessage,
    UnifiedTool,
    build_kiro_payload as core_build_kiro_payload,
)
from kiro.converters_openai import reasoning_effort_to_budget
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_openai_responses import ResponsesRequest, ResponsesTool
from kiro.tool_sanitizer import KIRO_TOOL_NAME_MAX_LENGTH


# Separator used to expose a namespaced tool under a single flat Kiro tool name.
# Kiro API accepts letters, digits, underscore and hyphen in tool names, so a
# double underscore is both legal and unlikely to appear by accident.
NAMESPACE_SEPARATOR: str = "__"

# Roles whose content is system-level and therefore folded into the Kiro system
# prompt instead of becoming a conversation turn.
SYSTEM_LEVEL_ROLES: frozenset = frozenset({"system", "developer"})

# Message content part types that carry plain text.
TEXT_PART_TYPES: frozenset = frozenset({"input_text", "output_text", "summary_text", "text"})

# Built-in (server-side) tool types the gateway can emulate. ``web_search`` is
# routed through the Kiro MCP endpoint by the streaming layer, exactly like the
# Chat Completions surface does.
EMULATED_BUILTIN_TOOL_TYPES: frozenset = frozenset({"web_search", "web_search_preview"})

# Name the emulated built-in web search tool is exposed under.
WEB_SEARCH_TOOL_NAME: str = "web_search"

# Description used for the emulated built-in web search tool.
WEB_SEARCH_TOOL_DESCRIPTION: str = (
    "Search the web for current information. Use when you need up-to-date data "
    "from the internet."
)

# Input schema used for the emulated built-in web search tool.
WEB_SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query",
        }
    },
    "required": ["query"],
}

# Responses tool type of a freeform tool: it takes raw text instead of JSON
# arguments. Codex CLI declares its code-mode ``exec`` tool this way, and also
# ``apply_patch`` when the model preset sets ``apply_patch_tool_type =
# "freeform"``.
CUSTOM_TOOL_TYPE: str = "custom"

# Responses tool type of a container grouping several tools under one namespace.
NAMESPACE_TOOL_TYPE: str = "namespace"

# Input item type Codex CLI uses to declare tools inside the ``input`` array
# instead of the top-level ``tools`` array (observed with code-mode models such
# as ``gpt-5.6-terra``, where ``tools`` is sent empty).
ADDITIONAL_TOOLS_ITEM_TYPE: str = "additional_tools"

# Input item type of an assistant call to a freeform tool.
CUSTOM_TOOL_CALL_ITEM_TYPE: str = "custom_tool_call"

# Input item type of a client result for a freeform tool call.
CUSTOM_TOOL_CALL_OUTPUT_ITEM_TYPE: str = "custom_tool_call_output"

# Kiro API only accepts JSON-schema tools, so a freeform tool is exposed with a
# single required string property. Its value is forwarded to the client verbatim
# as the ``input`` of a ``custom_tool_call`` item.
CUSTOM_TOOL_INPUT_PROPERTY: str = "input"

# Upper bound on how much of a custom tool's grammar definition is inlined into
# its description. Grammars are normally a few hundred characters; the cap stops
# a pathological one from dominating the payload-size budget.
CUSTOM_TOOL_GRAMMAR_MAX_CHARS: int = 4000

# Instructions appended to a custom tool's description so the model knows the
# single ``input`` property carries the tool's raw text payload.
CUSTOM_TOOL_CALLING_CONTRACT: str = (
    "CALLING CONVENTION: this is a freeform tool. Put the tool's complete raw "
    f"text payload in the single '{CUSTOM_TOOL_INPUT_PROPERTY}' string property. "
    "Do not wrap it in extra JSON, do not escape it into a nested object and do "
    "not add markdown code fences: the string is delivered to the client exactly "
    "as written."
)


# ==================================================================================================
# Errors
# ==================================================================================================

class UnsupportedResponsesFeatureError(ValueError):
    """
    Raised when a Responses request needs a feature this gateway cannot provide.

    Subclasses ``ValueError`` so the route handlers turn it into an HTTP 400
    ``invalid_request_error``, matching how ``ToolSpecValidationError`` is
    surfaced on the other API surfaces.
    """
    pass


class EmptyResponsesInputError(ValueError):
    """
    Raised when a Responses request contains nothing that can be sent to Kiro.

    Subclasses ``ValueError`` so route handlers turn it into an HTTP 400
    ``invalid_request_error``.
    """
    pass


# ==================================================================================================
# Tool routing
# ==================================================================================================

@dataclass(frozen=True)
class ToolRoute:
    """
    Mapping between a Kiro-facing tool name and the client's tool identity.

    Kiro API only understands a flat list of uniquely named tools, while the
    Responses API lets clients group tools inside a ``namespace`` container. The
    gateway therefore exposes ``namespace__name`` to Kiro and uses this route to
    restore the original ``name``/``namespace`` pair when reporting the tool call
    back to the client.

    Attributes:
        exposed_name: Tool name sent to Kiro API (and returned by the model).
        client_name: Tool name as declared by the client.
        namespace: Namespace the tool was declared in, or ``None`` for
            top-level tools.
        is_custom: True when the client declared the tool as a freeform
            (``{"type": "custom"}``) tool. The streaming layer then reports the
            call as a ``custom_tool_call`` item instead of a ``function_call``,
            because Codex CLI dispatches freeform tools strictly on that item
            type.
    """
    exposed_name: str
    client_name: str
    namespace: Optional[str] = None
    is_custom: bool = False


@dataclass
class ToolRegistry:
    """
    Bidirectional registry of tool routes for a single Responses request.

    Attributes:
        routes: Routes keyed by the Kiro-facing ``exposed_name``.
    """
    routes: Dict[str, ToolRoute] = field(default_factory=dict)

    def find(self, client_name: str, namespace: Optional[str]) -> Optional[ToolRoute]:
        """
        Look up an already registered tool by its client identity.

        Used to deduplicate a tool that is declared both in the top-level
        ``tools`` array and inside an ``additional_tools`` input item: Kiro API
        rejects a payload containing two tools with the same name.

        Args:
            client_name: Tool name as declared by the client.
            namespace: Namespace the tool was declared in, or ``None``.

        Returns:
            The matching ToolRoute, or ``None`` when the tool is not registered.
        """
        for route in self.routes.values():
            if route.client_name == client_name and route.namespace == namespace:
                return route
        return None

    def register(
        self,
        client_name: str,
        namespace: Optional[str],
        is_custom: bool = False,
    ) -> ToolRoute:
        """
        Register a client tool and allocate its Kiro-facing name.

        Namespaced tools are exposed as ``{namespace}__{name}``. When that would
        exceed the Kiro tool-name limit the bare name is used instead, and a
        numeric suffix is appended if the bare name is already taken, so two
        distinct client tools never collapse into one Kiro tool.

        Args:
            client_name: Tool name as declared by the client.
            namespace: Namespace the tool was declared in, or ``None``.
            is_custom: True when the client declared the tool as a freeform
                (``{"type": "custom"}``) tool.

        Returns:
            The registered ToolRoute.
        """
        if namespace:
            exposed_name = f"{namespace}{NAMESPACE_SEPARATOR}{client_name}"
            if len(exposed_name) > KIRO_TOOL_NAME_MAX_LENGTH:
                logger.warning(
                    f"Namespaced tool name '{exposed_name}' exceeds the Kiro API limit of "
                    f"{KIRO_TOOL_NAME_MAX_LENGTH} characters. Exposing it to the model as "
                    f"'{client_name}' instead; tool calls are still routed back to namespace "
                    f"'{namespace}'."
                )
                exposed_name = client_name
        else:
            exposed_name = client_name

        unique_name = exposed_name
        suffix = 2
        while unique_name in self.routes and (
            self.routes[unique_name].client_name != client_name
            or self.routes[unique_name].namespace != namespace
        ):
            unique_name = f"{exposed_name}_{suffix}"
            suffix += 1

        if unique_name != exposed_name:
            logger.warning(
                f"Tool name '{exposed_name}' is declared more than once with different "
                f"identities. Exposing this one as '{unique_name}' so tool calls stay "
                f"unambiguous."
            )

        route = ToolRoute(
            exposed_name=unique_name,
            client_name=client_name,
            namespace=namespace,
            is_custom=is_custom,
        )
        self.routes[unique_name] = route
        return route

    def resolve(self, exposed_name: str) -> ToolRoute:
        """
        Resolve a model-emitted tool name back to the client's tool identity.

        Args:
            exposed_name: Tool name as returned by the model.

        Returns:
            The registered ToolRoute, or an identity route (no namespace) when
            the model invented a name the client never declared. Passing the
            name through unchanged lets the client reject it with its own error
            instead of the gateway silently rewriting it.
        """
        route = self.routes.get(exposed_name)
        if route is not None:
            return route

        logger.warning(
            f"Model returned tool call '{exposed_name}' which was not declared in the "
            f"request. Passing the name through unchanged."
        )
        return ToolRoute(exposed_name=exposed_name, client_name=exposed_name, namespace=None)

    def exposed_name_for(self, client_name: str, namespace: Optional[str]) -> str:
        """
        Find the Kiro-facing name for a client tool identity.

        Used when replaying a ``function_call`` history item: the assistant turn
        must reference the same tool name that was sent to Kiro in the tool
        specifications, otherwise Kiro rejects the conversation.

        Args:
            client_name: Tool name as declared by the client.
            namespace: Namespace of the tool, or ``None``.

        Returns:
            The matching exposed name, or ``client_name`` when the tool is not
            registered (for example when the request replays a call to a tool it
            no longer declares).
        """
        for route in self.routes.values():
            if route.client_name == client_name and route.namespace == namespace:
                return route.exposed_name

        # Namespace may be absent on the history item even though the tool is
        # namespaced; fall back to a name-only match before giving up.
        for route in self.routes.values():
            if route.client_name == client_name:
                return route.exposed_name

        return client_name


# ==================================================================================================
# Content extraction
# ==================================================================================================

def _part_type(part: Any) -> Optional[str]:
    """
    Read the ``type`` discriminator of a content part.

    Args:
        part: Content part as a Pydantic model or a plain dict.

    Returns:
        The part type, or ``None`` when absent.
    """
    if isinstance(part, dict):
        return part.get("type")
    return getattr(part, "type", None)


def _part_field(part: Any, name: str) -> Any:
    """
    Read a named field from a content part.

    Args:
        part: Content part as a Pydantic model or a plain dict.
        name: Field name to read.

    Returns:
        The field value, or ``None`` when absent.
    """
    if isinstance(part, dict):
        return part.get(name)
    return getattr(part, name, None)


def extract_text_from_parts(content: Any) -> str:
    """
    Extract the concatenated text of a Responses content payload.

    Args:
        content: ``None``, a plain string, or a list of content parts
            (``input_text``, ``output_text``, ``summary_text``, ``refusal``).

    Returns:
        Concatenated text, or an empty string when there is none.

    Examples:
        >>> extract_text_from_parts("hello")
        'hello'
        >>> extract_text_from_parts([{"type": "input_text", "text": "hi"}])
        'hi'
        >>> extract_text_from_parts([{"type": "input_image", "image_url": "data:..."}])
        ''
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    text_parts: List[str] = []
    for part in content:
        part_type = _part_type(part)
        if part_type in TEXT_PART_TYPES:
            text = _part_field(part, "text")
            if text:
                text_parts.append(str(text))
        elif part_type == "refusal":
            refusal = _part_field(part, "refusal")
            if refusal:
                text_parts.append(str(refusal))
        elif part_type is None:
            # Untyped part: accept a bare ``text`` field for robustness.
            text = _part_field(part, "text")
            if text:
                text_parts.append(str(text))

    return "".join(text_parts)


def extract_images_from_parts(content: Any) -> List[Dict[str, Any]]:
    """
    Extract images from a Responses content payload in unified format.

    Only base64 data URLs can be forwarded: the Kiro API accepts inline image
    bytes only, so remote URLs are reported and skipped.

    Args:
        content: A list of content parts, or any other value (ignored).

    Returns:
        List of ``{"media_type": ..., "data": ...}`` entries, possibly empty.

    Examples:
        >>> extract_images_from_parts(
        ...     [{"type": "input_image", "image_url": "data:image/png;base64,abc"}]
        ... )
        [{'media_type': 'image/png', 'data': 'abc'}]
    """
    if not isinstance(content, list):
        return []

    images: List[Dict[str, Any]] = []
    for part in content:
        if _part_type(part) not in ("input_image", "image"):
            continue

        url = _part_field(part, "image_url") or ""
        if not isinstance(url, str):
            continue

        if url.startswith("data:"):
            try:
                header, data = url.split(",", 1)
            except ValueError:
                logger.warning("Skipping input_image with a malformed data URL")
                continue
            media_type = header.split(";")[0].replace("data:", "") or "image/jpeg"
            if data:
                images.append({"media_type": media_type, "data": data})
        elif url:
            logger.warning(
                f"URL-based images are not supported by Kiro API, skipping: {url[:80]}"
            )

    if images:
        logger.debug(f"Extracted {len(images)} image(s) from Responses input item")

    return images


def extract_function_call_output(output: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Normalize the ``output`` payload of a ``function_call_output`` input item.

    Codex CLI sends either a plain string or a list of content items. Both are
    accepted, as is a ``{"body": ..., "success": ...}`` envelope, so that tool
    results are never silently dropped.

    Args:
        output: Raw ``output`` value from the input item.

    Returns:
        Tuple of (text, images) where ``images`` is in unified format.

    Examples:
        >>> extract_function_call_output("done")
        ('done', [])
        >>> extract_function_call_output([{"type": "input_text", "text": "ok"}])
        ('ok', [])
    """
    if output is None:
        return "", []

    if isinstance(output, str):
        return output, []

    if isinstance(output, list):
        return extract_text_from_parts(output), extract_images_from_parts(output)

    if isinstance(output, dict):
        if "body" in output:
            return extract_function_call_output(output["body"])
        for key in ("content", "text", "output"):
            if key in output:
                return extract_function_call_output(output[key])
        return json.dumps(output, ensure_ascii=False), []

    return str(output), []


# ==================================================================================================
# Tools
# ==================================================================================================

def _tool_field(tool: Any, name: str) -> Any:
    """
    Read a named field from a tool specification.

    Args:
        tool: Tool spec as a Pydantic model or a plain dict.
        name: Field name to read.

    Returns:
        The field value, or ``None`` when absent.
    """
    if isinstance(tool, dict):
        return tool.get(name)
    return getattr(tool, name, None)


def build_custom_tool_schema(tool: Any) -> Dict[str, Any]:
    """
    Build a Kiro-valid input schema for a freeform (``custom``) Responses tool.

    Kiro API only accepts JSON-schema tools, so the freeform payload is carried
    by one required string property. The property description repeats the tool's
    declared syntax when it has one, because that is the only place the model
    can learn what text to produce.

    Args:
        tool: Tool spec as a Pydantic model or a plain dict.

    Returns:
        JSON Schema object with a single required string property.

    Examples:
        >>> build_custom_tool_schema({"type": "custom", "name": "exec"})["required"]
        ['input']
    """
    fmt = _tool_field(tool, "format")
    syntax = ""
    if isinstance(fmt, dict):
        syntax = str(fmt.get("syntax") or fmt.get("type") or "")

    description = "The complete raw text payload for this freeform tool, passed through verbatim."
    if syntax:
        description = (
            f"The complete raw {syntax} text payload for this freeform tool, "
            f"passed through verbatim."
        )

    return {
        "type": "object",
        "properties": {
            CUSTOM_TOOL_INPUT_PROPERTY: {
                "type": "string",
                "description": description,
            }
        },
        "required": [CUSTOM_TOOL_INPUT_PROPERTY],
    }


def build_custom_tool_description(tool: Any) -> str:
    """
    Build the description of a freeform (``custom``) Responses tool.

    The client's own description is kept first, then the calling convention for
    the synthesized ``input`` property is spelled out, then the declared grammar
    is inlined so the model can satisfy it. Without the grammar the model has no
    way to produce text the client will accept.

    Args:
        tool: Tool spec as a Pydantic model or a plain dict.

    Returns:
        Description string, never empty.
    """
    parts: List[str] = []

    description = _tool_field(tool, "description")
    if description:
        parts.append(str(description))

    parts.append(CUSTOM_TOOL_CALLING_CONTRACT)

    fmt = _tool_field(tool, "format")
    if isinstance(fmt, dict):
        definition = fmt.get("definition")
        if definition:
            definition_text = str(definition)
            if len(definition_text) > CUSTOM_TOOL_GRAMMAR_MAX_CHARS:
                logger.warning(
                    f"Freeform tool '{_tool_field(tool, 'name')}' declares a "
                    f"{len(definition_text)}-character grammar definition; only the first "
                    f"{CUSTOM_TOOL_GRAMMAR_MAX_CHARS} characters are shown to the model to "
                    f"stay inside the Kiro API payload budget."
                )
                definition_text = definition_text[:CUSTOM_TOOL_GRAMMAR_MAX_CHARS]
            syntax = str(fmt.get("syntax") or fmt.get("type") or "grammar")
            parts.append(
                f"The payload must conform to this {syntax} grammar:\n{definition_text}"
            )

    return "\n\n".join(parts)


def extract_custom_tool_input(arguments: Any) -> str:
    """
    Recover the freeform text a model passed to a ``custom`` tool.

    The tool was exposed to Kiro API as a JSON-schema function with a single
    ``input`` string property, so the well-behaved case is
    ``{"input": "<raw text>"}``. Models occasionally deviate, and the raw text is
    the whole point of a freeform tool, so several fallbacks are accepted rather
    than losing the call.

    Args:
        arguments: Tool arguments as emitted by the model: a JSON string, an
            already-parsed value, or ``None``.

    Returns:
        The freeform text, possibly empty.

    Examples:
        >>> extract_custom_tool_input('{"input": "console.log(1)"}')
        'console.log(1)'
        >>> extract_custom_tool_input('*** Begin Patch')
        '*** Begin Patch'
        >>> extract_custom_tool_input({"code": "let x = 1"})
        'let x = 1'
    """
    if arguments is None:
        return ""

    parsed: Any = arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # A freeform tool's payload is raw text, so unparseable arguments are
            # very likely the payload itself rather than a mistake.
            logger.debug(
                "Freeform tool arguments were not JSON; treating them as the raw payload"
            )
            return arguments

    if isinstance(parsed, str):
        return parsed

    if isinstance(parsed, dict):
        value = parsed.get(CUSTOM_TOOL_INPUT_PROPERTY)
        if isinstance(value, str):
            return value
        if value is not None:
            logger.warning(
                f"Freeform tool argument '{CUSTOM_TOOL_INPUT_PROPERTY}' was "
                f"{type(value).__name__}, not a string; serializing it to JSON so the call "
                f"still reaches the client."
            )
            return json.dumps(value, ensure_ascii=False)

        string_values = [v for v in parsed.values() if isinstance(v, str)]
        if len(string_values) == 1:
            logger.warning(
                f"Freeform tool call omitted the '{CUSTOM_TOOL_INPUT_PROPERTY}' property; "
                f"using the only string argument as the payload."
            )
            return string_values[0]

        if not parsed:
            return ""

        logger.warning(
            f"Freeform tool call did not provide a '{CUSTOM_TOOL_INPUT_PROPERTY}' string; "
            f"forwarding the raw arguments object so nothing is lost."
        )
        return json.dumps(parsed, ensure_ascii=False)

    return json.dumps(parsed, ensure_ascii=False)


def extract_additional_tools(input_data: Any, report: bool = True) -> List[Any]:
    """
    Collect tool declarations carried by ``additional_tools`` input items.

    Codex CLI code mode (observed with ``gpt-5.6-terra`` on CLI 0.153.4) sends an
    empty top-level ``tools`` array and delivers every tool inside an
    ``additional_tools`` input item::

        {"type": "additional_tools", "id": "at_...", "role": "developer",
         "tools": [{"type": "namespace", "name": "functions", "tools": [...]}, ...]}

    Ignoring the item leaves the model with no tools at all, so its payload is
    merged into the same tool pipeline as the top-level ``tools`` array.

    Args:
        input_data: The request ``input`` payload (list, string or ``None``).
        report: Whether to log what was found. The route layer calls this a
            second time purely to size its request log line, and passes ``False``
            so the same finding is not reported twice per request.

    Returns:
        Tool specifications in declaration order, possibly empty.

    Examples:
        >>> extract_additional_tools(
        ...     [{"type": "additional_tools", "tools": [{"type": "function", "name": "a"}]}]
        ... )
        [{'type': 'function', 'name': 'a'}]
        >>> extract_additional_tools("just a prompt")
        []
    """
    if not isinstance(input_data, list):
        return []

    collected: List[Any] = []
    item_count = 0

    for item in input_data:
        if (_item_field(item, "type") or "") != ADDITIONAL_TOOLS_ITEM_TYPE:
            continue

        item_count += 1
        payload = _item_field(item, "tools")

        if payload is None:
            if report:
                logger.warning(
                    f"Responses '{ADDITIONAL_TOOLS_ITEM_TYPE}' input item carries no 'tools' "
                    f"field, so it declares nothing the model can call."
                )
            continue
        if not isinstance(payload, list):
            if report:
                logger.warning(
                    f"Responses '{ADDITIONAL_TOOLS_ITEM_TYPE}' input item has a "
                    f"{type(payload).__name__} 'tools' field instead of a list; it is ignored "
                    f"because no tool declaration can be read from it."
                )
            continue
        if not payload:
            if report:
                logger.warning(
                    f"Responses '{ADDITIONAL_TOOLS_ITEM_TYPE}' input item declares an empty "
                    f"tool list, so the model is offered no additional tools."
                )
            continue

        collected.extend(payload)

    if item_count and report:
        logger.info(
            f"Found {item_count} '{ADDITIONAL_TOOLS_ITEM_TYPE}' input item(s) carrying "
            f"{len(collected)} tool declaration(s); merging them with the top-level 'tools' "
            f"array so the model can actually call them."
        )

    return collected


def convert_responses_tools_to_unified(
    tools: Optional[List[ResponsesTool]],
    additional_tools: Optional[List[Any]] = None,
) -> Tuple[Optional[List[UnifiedTool]], ToolRegistry]:
    """
    Convert Responses tool specifications to unified format.

    Handles every shape observed on the wire:

    1. Flat function tools: ``{"type": "function", "name": ..., "parameters": ...}``
    2. Namespace containers: ``{"type": "namespace", "name": ..., "tools": [...]}``
       — flattened, with each nested tool exposed as ``namespace__name``.
    3. Built-in server-side tools such as ``{"type": "web_search"}`` — exposed as
       a regular function tool when ``WEB_SEARCH_ENABLED`` so the streaming layer
       can service it through the Kiro MCP endpoint, otherwise reported and
       skipped.
    4. Freeform tools: ``{"type": "custom", "name": ..., "format": {...}}`` —
       exposed with a single required ``input`` string property, both at the top
       level and nested inside a ``namespace`` container.

    ``additional_tools`` (tool declarations that arrived inside the ``input``
    array) go through the exact same pipeline and are appended after the
    top-level tools, preserving declaration order. A tool declared in both places
    is registered once, because Kiro API rejects duplicate tool names.

    Any other tool type is named in a WARNING and skipped, because Kiro API can
    only execute client-side tools.

    Args:
        tools: Tool specifications from the top-level ``tools`` array, or ``None``.
        additional_tools: Tool specifications collected from ``additional_tools``
            input items, or ``None``.

    Returns:
        Tuple of (unified tools or ``None``, routing registry).
    """
    registry = ToolRegistry()
    if not tools and not additional_tools:
        return None, registry

    unified_tools: List[UnifiedTool] = []
    skipped_types: List[str] = []
    custom_count = 0
    duplicate_count = 0

    def add_callable_tool(tool: Any, namespace: Optional[str], is_custom: bool) -> None:
        """Register one function or freeform tool and append its unified form."""
        nonlocal custom_count, duplicate_count

        name = _tool_field(tool, "name")
        if not name:
            kind = "freeform" if is_custom else "function"
            logger.warning(
                f"Skipping Responses {kind} tool without a 'name' field: an unnamed tool "
                f"cannot be called or routed back to the client."
            )
            return

        client_name = str(name)
        existing = registry.find(client_name=client_name, namespace=namespace)
        if existing is not None:
            duplicate_count += 1
            logger.debug(
                f"Tool '{client_name}' (namespace={namespace}) is declared more than once; "
                f"keeping the first declaration because Kiro API rejects duplicate tool names"
            )
            return

        route = registry.register(
            client_name=client_name, namespace=namespace, is_custom=is_custom
        )

        if is_custom:
            custom_count += 1
            unified_tools.append(
                UnifiedTool(
                    name=route.exposed_name,
                    description=build_custom_tool_description(tool),
                    input_schema=build_custom_tool_schema(tool),
                )
            )
            return

        unified_tools.append(
            UnifiedTool(
                name=route.exposed_name,
                description=_tool_field(tool, "description"),
                input_schema=_tool_field(tool, "parameters"),
            )
        )

    def merge(specs: List[Any]) -> None:
        """Fold one list of tool specifications into the registry."""
        for tool in specs:
            tool_type = _tool_field(tool, "type") or "function"

            if tool_type == "function":
                add_callable_tool(tool, namespace=None, is_custom=False)
                continue

            if tool_type == CUSTOM_TOOL_TYPE:
                add_callable_tool(tool, namespace=None, is_custom=True)
                continue

            if tool_type == NAMESPACE_TOOL_TYPE:
                namespace = _tool_field(tool, "name")
                nested = _tool_field(tool, "tools") or []
                if not namespace:
                    logger.warning("Skipping Responses namespace tool without a 'name' field")
                    continue
                if not nested:
                    logger.warning(
                        f"Responses namespace tool '{namespace}' declares no nested tools, "
                        f"skipping"
                    )
                    continue
                for nested_tool in nested:
                    nested_type = _tool_field(nested_tool, "type") or "function"
                    if nested_type == "function":
                        add_callable_tool(
                            nested_tool, namespace=str(namespace), is_custom=False
                        )
                        continue
                    if nested_type == CUSTOM_TOOL_TYPE:
                        add_callable_tool(
                            nested_tool, namespace=str(namespace), is_custom=True
                        )
                        continue
                    skipped_types.append(f"{namespace}.{nested_type}")
                continue

            if tool_type in EMULATED_BUILTIN_TOOL_TYPES:
                if not WEB_SEARCH_ENABLED:
                    skipped_types.append(tool_type)
                    continue
                if any(
                    route.client_name == WEB_SEARCH_TOOL_NAME
                    for route in registry.routes.values()
                ):
                    logger.debug(
                        f"Built-in tool '{tool_type}' requested but '{WEB_SEARCH_TOOL_NAME}' is "
                        f"already declared, skipping duplicate"
                    )
                    continue
                route = registry.register(client_name=WEB_SEARCH_TOOL_NAME, namespace=None)
                unified_tools.append(
                    UnifiedTool(
                        name=route.exposed_name,
                        description=WEB_SEARCH_TOOL_DESCRIPTION,
                        input_schema=dict(WEB_SEARCH_TOOL_SCHEMA),
                    )
                )
                logger.info(
                    f"Built-in '{tool_type}' tool emulated via the Kiro MCP web search endpoint"
                )
                continue

            skipped_types.append(str(tool_type))

    merge(list(tools or []))
    merge(list(additional_tools or []))

    if skipped_types:
        logger.warning(
            f"Ignoring {len(skipped_types)} unsupported Responses tool type(s): "
            f"{', '.join(sorted(set(skipped_types)))}. Kiro API can only execute client-side "
            f"function and freeform tools, so the model will not be offered these."
        )

    if custom_count:
        logger.info(
            f"Exposed {custom_count} freeform ('{CUSTOM_TOOL_TYPE}') tool(s) to Kiro API with a "
            f"single '{CUSTOM_TOOL_INPUT_PROPERTY}' string property; calls are reported back as "
            f"'{CUSTOM_TOOL_CALL_ITEM_TYPE}' items"
        )

    if duplicate_count:
        logger.info(
            f"Deduplicated {duplicate_count} repeated tool declaration(s) between the "
            f"top-level 'tools' array and '{ADDITIONAL_TOOLS_ITEM_TYPE}' input items"
        )

    logger.debug(
        f"Converted Responses tools: {len(unified_tools)} tool(s), "
        f"{sum(1 for r in registry.routes.values() if r.namespace)} namespaced, "
        f"{custom_count} freeform"
    )

    return (unified_tools or None), registry


# ==================================================================================================
# Input items
# ==================================================================================================

def _item_field(item: Any, name: str) -> Any:
    """
    Read a named field from an input item.

    Args:
        item: Input item as a Pydantic model or a plain dict.
        name: Field name to read.

    Returns:
        The field value, or ``None`` when absent.
    """
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def convert_responses_input_to_unified(
    input_data: Any,
    instructions: Optional[str],
    registry: ToolRegistry,
) -> Tuple[str, List[UnifiedMessage]]:
    """
    Convert the Responses ``input`` payload to unified messages.

    The item list is walked in order so the conversation shape is preserved:

    - ``instructions`` becomes the head of the system prompt, followed by every
      ``system``/``developer`` message in the order it appeared.
    - ``user``/``assistant`` messages become conversation turns.
    - ``function_call`` items attach to the preceding assistant turn (a new
      assistant turn is opened when there is none), so Kiro sees the tool
      invocation as part of the assistant response.
    - ``function_call_output`` items become user turns carrying ``tool_results``
      keyed by ``call_id``, which is what Kiro expects to follow ``toolUses``.
    - ``custom_tool_call`` / ``custom_tool_call_output`` items are the freeform
      equivalents and are mapped exactly the same way, with the raw ``input``
      text re-wrapped into the synthesized ``input`` argument so the replayed
      call matches the schema the tool was declared with.
    - ``additional_tools`` items are skipped here: their tool declarations are
      consumed by ``extract_additional_tools`` before this function runs.
    - ``reasoning`` items are dropped: Kiro API has no slot for replayed
      reasoning, and the gateway never issued encrypted reasoning content.

    Args:
        input_data: ``None``, a plain string, or a list of input items.
        instructions: Request-level instructions, or ``None``.
        registry: Tool registry used to map ``function_call`` names onto the
            tool names actually sent to Kiro.

    Returns:
        Tuple of (system_prompt, unified messages).
    """
    system_parts: List[str] = []
    if instructions:
        system_parts.append(instructions.strip())

    messages: List[UnifiedMessage] = []

    if isinstance(input_data, str):
        if input_data:
            messages.append(UnifiedMessage(role="user", content=input_data))
        return "\n\n".join(part for part in system_parts if part), messages

    items: List[Any] = list(input_data or [])

    total_tool_calls = 0
    total_tool_results = 0
    total_images = 0
    skipped_types: List[str] = []

    def attach_tool_call(tool_call: Dict[str, Any]) -> None:
        """Attach a tool call to the trailing assistant turn, opening one if needed."""
        nonlocal total_tool_calls
        if messages and messages[-1].role == "assistant":
            if messages[-1].tool_calls is None:
                messages[-1].tool_calls = []
            messages[-1].tool_calls.append(tool_call)
        else:
            messages.append(
                UnifiedMessage(role="assistant", content="", tool_calls=[tool_call])
            )
        total_tool_calls += 1

    def attach_tool_result(tool_result: Dict[str, Any], images: List[Dict[str, Any]]) -> None:
        """Attach a tool result to the trailing user turn, opening one if needed."""
        nonlocal total_tool_results, total_images
        if images:
            total_images += len(images)
        if messages and messages[-1].role == "user" and messages[-1].tool_results is not None:
            messages[-1].tool_results.append(tool_result)
            if images:
                messages[-1].images = (messages[-1].images or []) + images
        else:
            messages.append(
                UnifiedMessage(
                    role="user",
                    content="",
                    tool_results=[tool_result],
                    images=images or None,
                )
            )
        total_tool_results += 1

    for item in items:
        item_type = _item_field(item, "type") or "message"

        if item_type in ("message", "input_text", "output_text"):
            role = (_item_field(item, "role") or "user").lower()
            content = _item_field(item, "content")
            text = extract_text_from_parts(content)

            if role in SYSTEM_LEVEL_ROLES:
                if text:
                    system_parts.append(text)
                else:
                    logger.debug(f"Skipping empty '{role}' input item")
                continue

            images = extract_images_from_parts(content) or None
            if images:
                total_images += len(images)

            messages.append(
                UnifiedMessage(
                    role="assistant" if role == "assistant" else "user",
                    content=text,
                    images=images,
                )
            )
            continue

        if item_type == "function_call":
            client_name = str(_item_field(item, "name") or "")
            namespace = _item_field(item, "namespace")
            exposed_name = registry.exposed_name_for(client_name, namespace)
            arguments = _item_field(item, "arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)

            attach_tool_call(
                {
                    "id": str(_item_field(item, "call_id") or _item_field(item, "id") or ""),
                    "type": "function",
                    "function": {"name": exposed_name, "arguments": arguments or "{}"},
                }
            )
            continue

        if item_type == CUSTOM_TOOL_CALL_ITEM_TYPE:
            # Freeform call replayed by the client. Kiro API knows the tool only
            # through its synthesized ``input`` string property, so the raw text
            # is wrapped back into that shape; otherwise Kiro sees a toolUse
            # whose arguments do not match the tool it was given.
            client_name = str(_item_field(item, "name") or "")
            namespace = _item_field(item, "namespace")
            exposed_name = registry.exposed_name_for(client_name, namespace)
            raw_input = _item_field(item, "input")
            if raw_input is None:
                raw_input = ""
            elif not isinstance(raw_input, str):
                logger.warning(
                    f"Replayed '{CUSTOM_TOOL_CALL_ITEM_TYPE}' for tool '{client_name}' carries a "
                    f"{type(raw_input).__name__} 'input' instead of a string; serializing it so "
                    f"the tool round-trip is preserved."
                )
                raw_input = json.dumps(raw_input, ensure_ascii=False)

            attach_tool_call(
                {
                    "id": str(_item_field(item, "call_id") or _item_field(item, "id") or ""),
                    "type": "function",
                    "function": {
                        "name": exposed_name,
                        "arguments": json.dumps(
                            {CUSTOM_TOOL_INPUT_PROPERTY: raw_input}, ensure_ascii=False
                        ),
                    },
                }
            )
            continue

        if item_type in ("function_call_output", CUSTOM_TOOL_CALL_OUTPUT_ITEM_TYPE):
            text, images = extract_function_call_output(_item_field(item, "output"))
            attach_tool_result(
                {
                    "type": "tool_result",
                    "tool_use_id": str(_item_field(item, "call_id") or ""),
                    "content": text or "(empty result)",
                },
                images,
            )
            continue

        if item_type == ADDITIONAL_TOOLS_ITEM_TYPE:
            # Tool declarations, not conversation content. Already merged into the
            # tool registry by ``extract_additional_tools``.
            logger.debug(
                f"Skipping '{ADDITIONAL_TOOLS_ITEM_TYPE}' input item in the message pass "
                f"(its tools were already merged into the tool specifications)"
            )
            continue

        if item_type == "reasoning":
            logger.debug("Dropping replayed 'reasoning' input item (no Kiro API equivalent)")
            continue

        skipped_types.append(str(item_type))

    if skipped_types:
        logger.warning(
            f"Ignoring {len(skipped_types)} unsupported Responses input item type(s): "
            f"{', '.join(sorted(set(skipped_types)))}. These carry no content Kiro API can "
            f"consume, so they are not forwarded."
        )

    if total_tool_calls or total_tool_results or total_images:
        logger.debug(
            f"Converted {len(items)} Responses input items: {total_tool_calls} function_call, "
            f"{total_tool_results} function_call_output, {total_images} image(s)"
        )

    system_prompt = "\n\n".join(part for part in system_parts if part)
    return system_prompt, messages


# ==================================================================================================
# Thinking configuration
# ==================================================================================================

def extract_thinking_config_from_responses(request: ResponsesRequest) -> ThinkingConfig:
    """
    Extract thinking configuration from a Responses request.

    Mirrors ``extract_thinking_config_from_openai`` / ``..._from_anthropic``:

    - No ``reasoning`` block, or no ``effort`` → enabled with the default budget.
    - ``reasoning.effort == "none"`` → thinking disabled.
    - Any other effort → enabled with a budget derived from
      ``max_output_tokens`` (falling back to 4096, the standard output limit).

    Args:
        request: The parsed Responses request.

    Returns:
        ThinkingConfig for the core converter layer.

    Examples:
        >>> extract_thinking_config_from_responses(
        ...     ResponsesRequest(model="claude-sonnet-4.5")
        ... )
        ThinkingConfig(enabled=True, budget_tokens=None)
    """
    reasoning = request.reasoning
    effort = getattr(reasoning, "effort", None) if reasoning is not None else None

    if not effort:
        return ThinkingConfig(enabled=True, budget_tokens=None)

    if effort == "none":
        logger.debug("Responses request disabled thinking via reasoning.effort='none'")
        return ThinkingConfig(enabled=False, budget_tokens=None)

    max_output_tokens = request.max_output_tokens or 4096
    budget = reasoning_effort_to_budget(max_output_tokens, effort)

    logger.debug(
        f"Extracted thinking config from Responses: reasoning.effort='{effort}', "
        f"max_output_tokens={max_output_tokens}, budget={budget}"
    )

    return ThinkingConfig(enabled=True, budget_tokens=budget)


# ==================================================================================================
# Unsupported field reporting
# ==================================================================================================

def validate_responses_request(request: ResponsesRequest) -> None:
    """
    Reject Responses requests the gateway cannot answer correctly.

    The gateway is stateless: it never persists a response, so it cannot
    reconstruct a conversation from ``previous_response_id``. Answering such a
    request with only the items present in ``input`` would silently drop the
    hidden history and produce a wrong answer, so it is refused instead.

    ``store: true`` is NOT refused: the client still sends the full ``input``, so
    the answer is correct — only the server-side copy is missing. That is
    reported as a WARNING and reflected as ``store: false`` on the response.

    Args:
        request: The parsed Responses request.

    Raises:
        UnsupportedResponsesFeatureError: If ``previous_response_id`` is set.
    """
    if request.previous_response_id:
        raise UnsupportedResponsesFeatureError(
            "This gateway does not store responses, so 'previous_response_id' cannot be "
            "resolved and the earlier turns of the conversation would be lost. "
            "Send the full conversation in the 'input' array instead (and set "
            "\"store\": false). If your client is OpenAI Codex CLI, this is already the "
            "default behaviour."
        )


def log_ignored_responses_fields(request: ResponsesRequest) -> None:
    """
    Name every accepted-but-ignored Responses request field in one WARNING.

    Kiro API exposes no knobs for sampling, structured output, service tiers or
    server-side truncation, and it produces no encrypted reasoning payloads.
    Those fields are accepted for compatibility, but the client deserves to know
    they had no effect.

    Args:
        request: The parsed Responses request.
    """
    ignored: List[str] = []

    if request.store:
        ignored.append(
            "store=true (the response is not persisted and cannot be fetched later)"
        )
    if request.tool_choice not in (None, "auto"):
        ignored.append(f"tool_choice={request.tool_choice!r} (Kiro API always decides)")
    if request.parallel_tool_calls is False:
        ignored.append("parallel_tool_calls=false (Kiro API may still return several calls)")
    if request.text is not None and request.text.format is not None:
        ignored.append("text.format (Kiro API has no structured-output mode)")
    if request.temperature is not None:
        ignored.append("temperature")
    if request.top_p is not None:
        ignored.append("top_p")
    if request.truncation is not None:
        ignored.append(f"truncation={request.truncation!r}")
    if request.service_tier is not None:
        ignored.append(f"service_tier={request.service_tier!r}")
    if request.max_output_tokens is not None:
        ignored.append(
            "max_output_tokens (used only to size the thinking budget; Kiro API enforces "
            "its own output limit)"
        )
    if request.include:
        unsupported_include = [
            entry for entry in request.include if entry != "reasoning.encrypted_content"
        ]
        if "reasoning.encrypted_content" in request.include:
            ignored.append(
                "include=reasoning.encrypted_content (reasoning is reconstructed from plain "
                "text, so no encrypted payload is produced)"
            )
        for entry in unsupported_include:
            ignored.append(f"include={entry!r}")

    if ignored:
        logger.warning(
            "POST /v1/responses accepted but ignored: " + "; ".join(ignored)
        )


# ==================================================================================================
# Main Entry Point
# ==================================================================================================

@dataclass
class ResponsesConversionResult:
    """
    Result of converting a Responses request to a Kiro API payload.

    Attributes:
        payload: The complete Kiro API payload.
        tool_registry: Routing table used by the streaming layer to report tool
            calls back under the client's own tool names and namespaces.
        messages_for_tokenizer: Unified messages as plain dicts, used for the
            tiktoken fallback when Kiro API returns no context usage.
        tools_for_tokenizer: Kiro tool specifications as plain dicts, used for
            the tiktoken fallback.
    """
    payload: Dict[str, Any]
    tool_registry: ToolRegistry
    messages_for_tokenizer: List[Dict[str, Any]]
    tools_for_tokenizer: Optional[List[Dict[str, Any]]]


def build_kiro_payload_responses(
    request_data: ResponsesRequest,
    conversation_id: str,
    profile_arn: str,
) -> ResponsesConversionResult:
    """
    Build a complete Kiro API payload from a Responses API request.

    This is the single entry point for Responses → Kiro conversion. It reuses the
    shared core converter, so tool sanitization, message merging, role
    alternation, thinking-tag injection and payload-size guards behave exactly as
    they do for Chat Completions and the Messages API.

    Args:
        request_data: Request in OpenAI Responses format.
        conversation_id: Unique conversation ID for Kiro API.
        profile_arn: AWS CodeWhisperer profile ARN.

    Returns:
        ResponsesConversionResult with the payload and the tool routing table.

    Raises:
        UnsupportedResponsesFeatureError: If the request requires server-side
            conversation state.
        EmptyResponsesInputError: If the request carries no content to send.
        ToolSpecValidationError: If a tool name cannot be sent to Kiro API.
        ValueError: If the core converter cannot build a payload.
    """
    validate_responses_request(request_data)
    log_ignored_responses_fields(request_data)

    additional_tools = extract_additional_tools(request_data.input)
    unified_tools, tool_registry = convert_responses_tools_to_unified(
        request_data.tools,
        additional_tools,
    )

    system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.input,
        request_data.instructions,
        tool_registry,
    )

    if not unified_messages:
        raise EmptyResponsesInputError(
            "The 'input' field contains no message, tool call or tool result to send. "
            "Add at least one input item, for example "
            "{\"type\": \"message\", \"role\": \"user\", "
            "\"content\": [{\"type\": \"input_text\", \"text\": \"...\"}]}, "
            "or pass 'input' as a non-empty string."
        )

    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)
    thinking_config = extract_thinking_config_from_responses(request_data)

    logger.debug(
        f"Converting Responses request: model={request_data.model} -> {model_id}, "
        f"input_items={len(unified_messages)}, tools={len(unified_tools) if unified_tools else 0} "
        f"({len(additional_tools)} from '{ADDITIONAL_TOOLS_ITEM_TYPE}'), "
        f"system_prompt_length={len(system_prompt)}, "
        f"thinking_enabled={thinking_config.enabled}, "
        f"thinking_budget={thinking_config.budget_tokens}"
    )

    result = core_build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=thinking_config,
    )

    messages_for_tokenizer = [
        {
            "role": message.role,
            "content": message.content,
            "tool_calls": message.tool_calls,
            "tool_results": message.tool_results,
        }
        for message in unified_messages
    ]
    if system_prompt:
        messages_for_tokenizer.insert(0, {"role": "system", "content": system_prompt})

    tools_for_tokenizer: Optional[List[Dict[str, Any]]] = None
    if unified_tools:
        tools_for_tokenizer = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in unified_tools
        ]

    return ResponsesConversionResult(
        payload=result.payload,
        tool_registry=tool_registry,
        messages_for_tokenizer=messages_for_tokenizer,
        tools_for_tokenizer=tools_for_tokenizer,
    )
