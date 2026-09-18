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
Pydantic models for the OpenAI Responses API (``POST /v1/responses``).

The Responses API is the wire protocol used by OpenAI Codex CLI when it is
configured with ``wire_api = "responses"``. It is a different surface from Chat
Completions (``kiro/models_openai.py``), so it lives in its own module:

- Requests carry a flat ``input`` item list instead of ``messages``.
- Tool specifications are flat (``{"type": "function", "name": ...}``) instead
  of nested under a ``"function"`` key, and may be grouped in a ``namespace``
  container tool.
- Responses carry an ``output`` item list instead of ``choices``.
- Streaming uses named SSE events (``response.output_text.delta`` and friends)
  instead of ``chat.completion.chunk`` deltas.

Every model uses ``extra="allow"`` so that new upstream fields never turn into
HTTP 422 for the client. The shapes below were verified against a real request
captured from Codex CLI 0.153.4 and against the Codex SSE decoder in
``codex-rs/codex-api/src/sse/responses.rs``.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ==================================================================================================
# Request: input items
# ==================================================================================================

class ResponsesContentPart(BaseModel):
    """
    Single content part inside a Responses ``message`` input item.

    Known part types:
    - ``input_text`` / ``output_text`` / ``summary_text``: carry ``text``
    - ``input_image``: carries ``image_url`` (data URL or remote URL)
    - ``input_audio``: carries ``audio_url`` (not supported by Kiro API)
    - ``refusal``: carries ``refusal``

    Attributes:
        type: Part type discriminator.
        text: Text payload for textual parts.
        image_url: Image URL (data URL or ``http(s)://``) for ``input_image``.
        detail: Optional image detail hint (``auto``/``low``/``high``).
        audio_url: Audio URL for ``input_audio``.
        refusal: Refusal text for ``refusal`` parts.
        annotations: Optional annotation list emitted by OpenAI.
    """
    type: Optional[str] = None
    text: Optional[str] = None
    image_url: Optional[str] = None
    detail: Optional[str] = None
    audio_url: Optional[str] = None
    refusal: Optional[str] = None
    annotations: Optional[List[Any]] = None

    model_config = {"extra": "allow"}


class ResponsesSummaryPart(BaseModel):
    """
    Single reasoning summary part inside a ``reasoning`` item.

    Attributes:
        type: Part type, always ``summary_text`` in practice.
        text: Summary text.
    """
    type: Optional[str] = None
    text: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponsesInputItem(BaseModel):
    """
    Single item of the Responses ``input`` array.

    One model covers every item type because the Responses API discriminates on
    ``type`` and the field sets overlap heavily. Unknown item types are accepted
    here and reported by the converter, which is the layer that decides what can
    be represented in Kiro API.

    Item types handled by the gateway:
    - ``message`` (or a bare ``{"role": ..., "content": ...}`` with no ``type``)
    - ``function_call``: assistant tool invocation, keyed by ``call_id``
    - ``function_call_output``: client tool result, keyed by ``call_id``
    - ``custom_tool_call``: assistant invocation of a freeform (``custom``)
      tool, carrying raw text in ``input`` instead of JSON ``arguments``
    - ``custom_tool_call_output``: client result of a freeform tool call
    - ``reasoning``: previous reasoning item replayed by the client
    - ``additional_tools``: tool declarations delivered inside ``input``
      instead of the top-level ``tools`` array (Codex CLI code mode)

    Attributes:
        type: Item type discriminator (``None`` is treated as ``message``).
        id: Item identifier assigned by the server that produced the item.
        role: Message role (``user``, ``assistant``, ``system``, ``developer``).
        content: String or list of content parts for ``message`` items.
        name: Tool name for ``function_call`` / ``custom_tool_call`` items.
        namespace: Tool namespace for ``function_call`` / ``custom_tool_call``
            items produced from a ``namespace`` container tool.
        arguments: JSON-encoded tool arguments string for ``function_call``.
        call_id: Tool call correlation ID shared by every ``*_call`` item and
            its matching ``*_call_output`` item.
        status: Item status reported by the server (``completed``, ...).
        output: Tool result payload for ``function_call_output`` and
            ``custom_tool_call_output``. On the wire this is either a plain
            string or a list of content items.
        input: Raw freeform text of a ``custom_tool_call`` item. Typed as
            ``Any`` so a malformed non-string payload is reported by the
            converter instead of turning into HTTP 422.
        tools: Tool declarations carried by an ``additional_tools`` item. Kept
            untyped for the same reason: a junk payload must be reported, not
            rejected outright.
        summary: Reasoning summary parts for ``reasoning`` items.
        encrypted_content: Opaque encrypted reasoning payload.
    """
    type: Optional[str] = None
    id: Optional[str] = None
    role: Optional[str] = None
    content: Optional[Union[str, List[ResponsesContentPart]]] = None

    # function_call / custom_tool_call
    name: Optional[str] = None
    namespace: Optional[str] = None
    arguments: Optional[str] = None
    call_id: Optional[str] = None
    status: Optional[str] = None

    # function_call_output / custom_tool_call_output
    output: Optional[Any] = None

    # custom_tool_call
    input: Optional[Any] = None

    # additional_tools
    tools: Optional[Any] = None

    # reasoning
    summary: Optional[List[ResponsesSummaryPart]] = None
    encrypted_content: Optional[str] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Request: tools
# ==================================================================================================

class ResponsesTool(BaseModel):
    """
    Single entry of the Responses ``tools`` array.

    Three shapes are observed in practice:

    1. Function tool (flat, NOT nested under ``"function"``)::

           {"type": "function", "name": "exec_command",
            "description": "...", "strict": false,
            "parameters": {"type": "object", ...}}

    2. Namespace container that groups function tools (Codex ``multi_agent_v1``)::

           {"type": "namespace", "name": "multi_agent_v1",
            "description": "...", "tools": [ {function tool}, ... ]}

    3. Built-in server-side tool with no schema, e.g.::

           {"type": "web_search", "external_web_access": true}

    4. Freeform ("custom") tool that takes raw text instead of JSON arguments
       (Codex CLI code mode ``exec``, and ``apply_patch`` when the model preset
       sets ``apply_patch_tool_type = "freeform"``)::

           {"type": "custom", "name": "exec",
            "description": "...",
            "format": {"type": "grammar", "syntax": "lark",
                       "definition": "start: ..."}}

    Attributes:
        type: Tool type discriminator.
        name: Tool (or namespace) name.
        description: Human-readable tool description.
        parameters: JSON Schema of the tool arguments.
        strict: OpenAI strict-schema flag (accepted, not forwarded to Kiro).
        tools: Nested function/custom tools for ``namespace`` containers.
        format: Input format descriptor of a ``custom`` tool. Codex sends
            ``{"type": "grammar", "syntax": "lark", "definition": ...}``; the
            gateway inlines it into the tool description so the model knows what
            text to produce.
        defer_loading: Codex hint that the tool body is loaded on demand
            (accepted, not forwarded to Kiro).
    """
    type: str = "function"
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None
    tools: Optional[List["ResponsesTool"]] = None
    format: Optional[Dict[str, Any]] = None
    defer_loading: Optional[bool] = None

    model_config = {"extra": "allow"}


ResponsesTool.model_rebuild()


# ==================================================================================================
# Request: reasoning / text configuration
# ==================================================================================================

class ResponsesReasoningConfig(BaseModel):
    """
    ``reasoning`` block of a Responses request.

    Attributes:
        effort: Requested reasoning effort. Mapped onto the gateway's
            ThinkingConfig budget.
        summary: Requested summary verbosity (``auto``, ``concise``,
            ``detailed``, ``none``). Codex sends ``"auto"``.
        generate_summary: Legacy alias of ``summary``.
    """
    effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None
    summary: Optional[str] = None
    generate_summary: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponsesTextConfig(BaseModel):
    """
    ``text`` block of a Responses request (structured-output configuration).

    Attributes:
        format: Output format descriptor, e.g. ``{"type": "json_object"}``.
        verbosity: Requested verbosity level.
    """
    format: Optional[Dict[str, Any]] = None
    verbosity: Optional[str] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Request
# ==================================================================================================

class ResponsesRequest(BaseModel):
    """
    Request body of ``POST /v1/responses``.

    Only ``model`` is strictly required: ``input`` may legitimately be omitted
    when ``instructions`` carries the whole prompt, and the converter raises an
    actionable error if nothing at all can be sent to Kiro.

    Attributes:
        model: Model ID to generate with.
        input: Prompt as a plain string or as an ordered item list.
        instructions: System-level instructions. Folded into the Kiro system
            prompt ahead of any ``system``/``developer`` input items.
        tools: Available tools (flat function specs and/or namespace groups).
        tool_choice: Tool selection strategy (accepted, Kiro has no equivalent).
        stream: Whether to stream the response as SSE.
        max_output_tokens: Output token ceiling, used to size the thinking
            budget when ``reasoning.effort`` is set.
        reasoning: Reasoning configuration.
        store: Whether the server should persist the response. The gateway is
            stateless, so ``True`` is rejected with an actionable error.
        previous_response_id: Server-side conversation reference. The gateway is
            stateless, so this is rejected with an actionable error.
        parallel_tool_calls: Whether several tool calls may be returned at once.
        text: Structured-output configuration (accepted, not forwarded).
        include: Extra payloads the client wants inlined, e.g.
            ``["reasoning.encrypted_content"]``.
        metadata: Free-form client metadata echoed back on the response.
        temperature: Sampling temperature (accepted, Kiro has no equivalent).
        top_p: Nucleus sampling parameter (accepted, no equivalent).
        truncation: Server-side truncation strategy (accepted, no equivalent).
        service_tier: Requested service tier (accepted, no equivalent).
        user: End-user identifier (accepted, no equivalent).
        prompt_cache_key: Cache partition key sent by Codex CLI.
        client_metadata: Client telemetry blob sent by Codex CLI.
    """
    model: str
    input: Optional[Union[str, List[ResponsesInputItem]]] = None
    instructions: Optional[str] = None

    tools: Optional[List[ResponsesTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    stream: bool = False
    max_output_tokens: Optional[int] = None
    reasoning: Optional[ResponsesReasoningConfig] = None

    # Server-side conversation state (unsupported by this stateless gateway)
    store: Optional[bool] = None
    previous_response_id: Optional[str] = None

    parallel_tool_calls: Optional[bool] = None
    text: Optional[ResponsesTextConfig] = None
    include: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Accepted for compatibility, no Kiro equivalent
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    truncation: Optional[str] = None
    service_tier: Optional[str] = None
    user: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    client_metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Response: output items
# ==================================================================================================

class ResponsesOutputTextPart(BaseModel):
    """
    ``output_text`` content part of an assistant ``message`` output item.

    Attributes:
        type: Always ``output_text``.
        text: Assistant text.
        annotations: Always an empty list — the gateway produces no citations.
    """
    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: List[Any] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class ResponsesMessageItem(BaseModel):
    """
    Assistant ``message`` output item.

    Attributes:
        id: Item ID, stable across the ``added``/``done`` events.
        type: Always ``message``.
        status: ``in_progress`` while streaming, ``completed`` when finished.
        role: Always ``assistant``.
        content: List of output content parts.
    """
    id: str
    type: Literal["message"] = "message"
    status: str = "completed"
    role: Literal["assistant"] = "assistant"
    content: List[ResponsesOutputTextPart] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class ResponsesReasoningItem(BaseModel):
    """
    ``reasoning`` output item carrying the model's thinking summary.

    ``encrypted_content`` is always ``None``: the gateway reconstructs reasoning
    from Kiro's plain-text thinking blocks and has nothing to encrypt.

    Attributes:
        id: Item ID, stable across the ``added``/``done`` events.
        type: Always ``reasoning``.
        summary: Reasoning summary parts (``{"type": "summary_text", ...}``).
        content: Always an empty list.
        encrypted_content: Always ``None``.
    """
    id: str
    type: Literal["reasoning"] = "reasoning"
    summary: List[ResponsesSummaryPart] = Field(default_factory=list)
    content: List[Any] = Field(default_factory=list)
    encrypted_content: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponsesFunctionCallItem(BaseModel):
    """
    ``function_call`` output item.

    Attributes:
        id: Item ID, stable across the ``added``/``done`` events.
        type: Always ``function_call``.
        status: ``in_progress`` while streaming, ``completed`` when finished.
        name: Tool name as declared by the client.
        namespace: Namespace of the tool when it came from a ``namespace``
            container tool, otherwise ``None``.
        arguments: JSON-encoded arguments string.
        call_id: Correlation ID the client echoes back in
            ``function_call_output``.
    """
    id: str
    type: Literal["function_call"] = "function_call"
    status: str = "completed"
    name: str = ""
    namespace: Optional[str] = None
    arguments: str = "{}"
    call_id: str = ""

    model_config = {"extra": "allow"}


class ResponsesCustomToolCallItem(BaseModel):
    """
    ``custom_tool_call`` output item — an invocation of a freeform tool.

    Freeform (``{"type": "custom"}``) tools take raw text rather than JSON
    arguments, so the payload lives in ``input`` instead of ``arguments``. Codex
    CLI dispatches these strictly on the item type: its ``exec`` runtime accepts
    only a ``custom_tool_call``, and answers a ``function_call`` for the same
    tool with "expects raw JavaScript source text". Verified against
    ``codex-rs/core/src/tools/router.rs`` and
    ``codex-rs/core/src/tools/code_mode/execute_handler.rs`` at tag
    ``rust-v0.153.4``.

    Attributes:
        id: Item ID, stable across the ``added``/``done`` events.
        type: Always ``custom_tool_call``.
        status: ``in_progress`` while streaming, ``completed`` when finished.
        name: Tool name as declared by the client.
        namespace: Namespace of the tool when it came from a ``namespace``
            container tool, otherwise ``None``.
        input: Raw freeform text the tool was called with.
        call_id: Correlation ID the client echoes back in
            ``custom_tool_call_output``.
    """
    id: str
    type: Literal["custom_tool_call"] = "custom_tool_call"
    status: str = "completed"
    name: str = ""
    namespace: Optional[str] = None
    input: str = ""
    call_id: str = ""

    model_config = {"extra": "allow"}


# ==================================================================================================
# Response: usage and envelope
# ==================================================================================================

class ResponsesInputTokensDetails(BaseModel):
    """
    Breakdown of input token usage.

    Attributes:
        cached_tokens: Prompt tokens served from cache. Kiro API reports no
            cache statistics, so this is always 0.
    """
    cached_tokens: int = 0

    model_config = {"extra": "allow"}


class ResponsesOutputTokensDetails(BaseModel):
    """
    Breakdown of output token usage.

    Attributes:
        reasoning_tokens: Tokens spent on thinking content.
    """
    reasoning_tokens: int = 0

    model_config = {"extra": "allow"}


class ResponsesUsage(BaseModel):
    """
    Token usage block of a Responses response.

    Attributes:
        input_tokens: Prompt tokens.
        input_tokens_details: Input token breakdown.
        output_tokens: Completion tokens (including reasoning tokens).
        output_tokens_details: Output token breakdown.
        total_tokens: Sum of input and output tokens.
        credits_used: Kiro-specific metering payload. Added by this gateway on
            top of the OpenAI schema; clients that do not know it ignore it.
    """
    input_tokens: int = 0
    input_tokens_details: ResponsesInputTokensDetails = Field(
        default_factory=ResponsesInputTokensDetails
    )
    output_tokens: int = 0
    output_tokens_details: ResponsesOutputTokensDetails = Field(
        default_factory=ResponsesOutputTokensDetails
    )
    total_tokens: int = 0
    credits_used: Optional[Any] = None

    model_config = {"extra": "allow"}


class ResponsesIncompleteDetails(BaseModel):
    """
    Reason a response stopped before the model was done.

    Attributes:
        reason: Machine-readable reason, e.g. ``max_output_tokens``.
    """
    reason: str

    model_config = {"extra": "allow"}


class ResponsesResponse(BaseModel):
    """
    Full Responses API response object.

    This is both the body of a non-streaming ``POST /v1/responses`` call and the
    payload embedded in the ``response.created`` / ``response.completed``
    streaming events.

    Attributes:
        id: Response ID (``resp_...``).
        object: Always ``response``.
        created_at: Unix timestamp of creation.
        status: ``in_progress``, ``completed`` or ``incomplete``.
        model: Model name echoed from the request.
        output: Ordered output items (reasoning, message, function calls).
        usage: Token usage, ``None`` until the response completes.
        incomplete_details: Set when ``status`` is ``incomplete``.
        instructions: Instructions echoed from the request.
        metadata: Metadata echoed from the request.
        error: Error payload, always ``None`` on success.
        parallel_tool_calls: Echoed from the request.
        tool_choice: Echoed from the request.
        tools: Echoed from the request.
        max_output_tokens: Echoed from the request.
        previous_response_id: Always ``None`` (stateless gateway).
        reasoning: Echoed reasoning configuration.
        store: Always ``False`` (stateless gateway).
        text: Echoed structured-output configuration.
        truncation: Echoed truncation strategy.
        temperature: Echoed sampling temperature.
        top_p: Echoed nucleus sampling parameter.
        user: Echoed end-user identifier.
    """
    id: str
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: str = "in_progress"
    model: str
    output: List[Any] = Field(default_factory=list)
    usage: Optional[ResponsesUsage] = None
    incomplete_details: Optional[ResponsesIncompleteDetails] = None

    instructions: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    parallel_tool_calls: bool = True
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    tools: List[Any] = Field(default_factory=list)
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None
    store: bool = False
    text: Optional[Dict[str, Any]] = None
    truncation: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}
