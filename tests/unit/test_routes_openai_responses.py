# -*- coding: utf-8 -*-
"""
Unit tests for the OpenAI Responses API endpoint (routes_openai_responses.py).

Tests POST /v1/responses:
- Authentication and request validation (422 paths)
- Request echo construction and error envelopes
- Unsupported server-side state fields (previous_response_id, store)
- Empty input and missing/malformed profileArn
- HTTP client selection (per-request for streaming, shared for non-streaming)
- Upstream non-200 handling and network-error passthrough
"""
import json

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from codex_responses_payload import (
    CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST,
    CODEX_RESPONSES_CODE_MODE_REQUEST,
    CODEX_RESPONSES_REQUEST,
)
from kiro.models_openai_responses import ResponsesRequest
from kiro.routes_openai_responses import (
    ENDPOINT_PATH,
    build_error_response,
    build_request_echo,
    resolve_payload,
    router,
)

VALID_PROFILE_ARN = "arn:aws:codewhisperer:us-east-1:123456789012:profile/TEST"


# =============================================================================
# Router registration
# =============================================================================

class TestRouterRegistration:
    """Tests that the endpoint is wired up correctly."""

    def test_responses_route_is_registered_for_post(self):
        """
        What it does: Finds POST /v1/responses on the router.
        Purpose: The endpoint is useless if the router is not registered.
        """
        print("Checking: /v1/responses route...")
        paths = {route.path: route.methods for route in router.routes}

        assert ENDPOINT_PATH in paths
        assert "POST" in paths[ENDPOINT_PATH]

    def test_route_is_reachable_through_the_application(self, test_client):
        """
        What it does: Confirms the app exposes the endpoint.
        Purpose: main.py must include the router.
        """
        print("Checking: app route table...")
        paths = {route.path for route in test_client.app.routes}

        assert ENDPOINT_PATH in paths

    def test_endpoint_is_debug_logged(self):
        """
        What it does: Confirms the endpoint is in LOGGED_ENDPOINTS.
        Purpose: Debug logging must capture raw bodies before validation, which
                 is how the Codex contract was captured in the first place.
        """
        from kiro.debug_middleware import LOGGED_ENDPOINTS

        assert ENDPOINT_PATH in LOGGED_ENDPOINTS


# =============================================================================
# Authentication
# =============================================================================

class TestResponsesAuthentication:
    """Tests for endpoint authentication."""

    def test_missing_api_key_is_rejected(self, test_client):
        """
        What it does: Rejects an unauthenticated request.
        Purpose: The endpoint proxies a paid upstream and must be protected.
        """
        print("Action: POST without Authorization...")
        response = test_client.post(
            ENDPOINT_PATH, json={"model": "claude-sonnet-4.5", "input": "hi"}
        )

        assert response.status_code == 401

    def test_invalid_api_key_is_rejected(self, test_client, invalid_proxy_api_key):
        """
        What it does: Rejects a wrong API key.
        Purpose: Authentication must actually compare the key.
        """
        print("Action: POST with a wrong key...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {invalid_proxy_api_key}"},
            json={"model": "claude-sonnet-4.5", "input": "hi"},
        )

        assert response.status_code == 401


# =============================================================================
# Request validation
# =============================================================================

class TestResponsesValidation:
    """Tests for request-level validation."""

    def test_missing_model_returns_422(self, test_client, valid_proxy_api_key):
        """
        What it does: Rejects a body without 'model'.
        Purpose: The gateway cannot guess the model.
        """
        print("Action: POST without model...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"input": "hi"},
        )

        assert response.status_code == 422

    def test_invalid_json_returns_422(self, test_client, valid_proxy_api_key):
        """
        What it does: Rejects a malformed body.
        Purpose: Broken JSON must not reach the converter.
        """
        print("Action: POST with broken JSON...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={
                "Authorization": f"Bearer {valid_proxy_api_key}",
                "Content-Type": "application/json",
            },
            content=b"{not json",
        )

        assert response.status_code == 422

    def test_invalid_reasoning_effort_returns_422(self, test_client, valid_proxy_api_key):
        """
        What it does: Rejects an unknown reasoning effort.
        Purpose: Only documented levels map to a thinking budget.
        """
        print("Action: POST with reasoning.effort='turbo'...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "m", "input": "hi", "reasoning": {"effort": "turbo"}},
        )

        assert response.status_code == 422

    def test_previous_response_id_returns_actionable_400(self, test_client, valid_proxy_api_key):
        """
        What it does: Rejects previous_response_id with HTTP 400.
        Purpose: Ignoring it would silently answer without the hidden history.
                 The error must tell the client what to do instead.
        """
        print("Action: POST with previous_response_id...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "m", "input": "hi", "previous_response_id": "resp_123"},
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "previous_response_id" in body["error"]["message"]
        assert "input" in body["error"]["message"]

    def test_empty_input_returns_actionable_400(self, test_client, valid_proxy_api_key):
        """
        What it does: Rejects an empty input list with HTTP 400.
        Purpose: There is nothing to ask the model; the message shows the shape
                 of a valid input item.
        """
        print("Action: POST with empty input...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "m", "input": []},
        )

        assert response.status_code == 400
        message = response.json()["error"]["message"]
        assert "input" in message
        assert "input_text" in message

    def test_store_true_is_not_rejected(self, test_client, valid_proxy_api_key):
        """
        What it does: Does not turn store=true into an error.
        Purpose: The OpenAI SDK defaults to store=true and still sends the full
                 input, so the answer is correct; only persistence is missing.
        """
        print("Action: POST with store=true...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "m", "input": "hi", "store": True},
        )

        assert response.status_code != 400


# =============================================================================
# Request echo
# =============================================================================

class TestBuildRequestEcho:
    """Tests for build_request_echo()."""

    def test_stateless_fields_are_pinned(self):
        """
        What it does: Forces store=False and previous_response_id=None.
        Purpose: The response must never claim a server-side conversation exists.
        """
        echo = build_request_echo(ResponsesRequest(model="m", input="hi", store=True))

        assert echo["store"] is False
        assert echo["previous_response_id"] is None

    def test_defaults_match_openai(self):
        """
        What it does: Defaults tool_choice to "auto" and parallel_tool_calls True.
        Purpose: Matches the values OpenAI reports when the client omits them.
        """
        echo = build_request_echo(ResponsesRequest(model="m", input="hi"))

        assert echo["tool_choice"] == "auto"
        assert echo["parallel_tool_calls"] is True
        assert echo["tools"] == []

    def test_client_values_are_echoed(self):
        """
        What it does: Echoes the values the client actually sent.
        Purpose: The response describes the effective configuration.
        """
        echo = build_request_echo(
            ResponsesRequest(
                model="m",
                input="hi",
                instructions="SYS",
                metadata={"a": "b"},
                tool_choice="required",
                parallel_tool_calls=False,
                max_output_tokens=256,
                reasoning={"effort": "high", "summary": "auto"},
                text={"format": {"type": "json_object"}},
                truncation="auto",
                temperature=0.3,
                top_p=0.8,
                user="u1",
            )
        )

        assert echo["instructions"] == "SYS"
        assert echo["metadata"] == {"a": "b"}
        assert echo["tool_choice"] == "required"
        assert echo["parallel_tool_calls"] is False
        assert echo["max_output_tokens"] == 256
        assert echo["reasoning"] == {"effort": "high", "summary": "auto"}
        assert echo["text"] == {"format": {"type": "json_object"}}
        assert echo["truncation"] == "auto"
        assert echo["temperature"] == 0.3
        assert echo["top_p"] == 0.8
        assert echo["user"] == "u1"

    def test_tools_are_echoed_without_none_fields(self):
        """
        What it does: Drops unset tool fields from the echo.
        Purpose: Keeps the echoed tool list close to what the client sent.
        """
        echo = build_request_echo(
            ResponsesRequest(
                model="m",
                input="hi",
                tools=[{"type": "function", "name": "run", "parameters": {}}],
            )
        )

        assert echo["tools"] == [{"type": "function", "name": "run", "parameters": {}}]

    def test_captured_codex_request_echo(self):
        """
        What it does: Builds the echo for the real Codex CLI request.
        Purpose: Codex sends store=false already; the echo must agree.
        """
        echo = build_request_echo(ResponsesRequest(**CODEX_RESPONSES_REQUEST))

        assert echo["store"] is False
        assert echo["parallel_tool_calls"] is True
        assert echo["reasoning"] == {"summary": "auto"}
        assert len(echo["tools"]) == len(CODEX_RESPONSES_REQUEST["tools"])


# =============================================================================
# Error envelope
# =============================================================================

class TestBuildErrorResponse:
    """Tests for build_error_response()."""

    def test_error_envelope_shape(self):
        """
        What it does: Builds an OpenAI-shaped error body.
        Purpose: Clients parse error.message / error.type / error.code.
        """
        response = build_error_response(400, "bad thing", "invalid_request_error")
        body = json.loads(response.body)

        assert response.status_code == 400
        assert body == {
            "error": {
                "message": "bad thing",
                "type": "invalid_request_error",
                "code": 400,
            }
        }


# =============================================================================
# Payload resolution
# =============================================================================

class TestResolvePayload:
    """Tests for resolve_payload()."""

    def test_missing_profile_arn_returns_actionable_400(self):
        """
        What it does: Refuses to call Kiro without a profileArn.
        Purpose: Kiro answers with an opaque 400; the gateway must explain what
                 the user has to configure.
        """
        auth_manager = MagicMock()
        auth_manager.profile_arn = None

        with patch("kiro.routes_openai_responses.resolve_profile_arn", return_value=""):
            payload, registry, messages, tools, error = resolve_payload(
                ResponsesRequest(model="m", input="hi"), auth_manager
            )

        assert payload is None
        assert registry is None
        assert messages is None
        assert tools is None
        assert error.status_code == 400
        assert "profileArn" in json.loads(error.body)["error"]["message"]

    def test_malformed_profile_arn_returns_actionable_400(self):
        """
        What it does: Rejects a placeholder profileArn.
        Purpose: "arn:aws:codewhisperer:us-east-1:..." is a documentation
                 placeholder users copy verbatim.
        """
        auth_manager = MagicMock()

        with patch(
            "kiro.routes_openai_responses.resolve_profile_arn",
            return_value="arn:aws:codewhisperer:us-east-1:...",
        ):
            _, _, _, _, error = resolve_payload(
                ResponsesRequest(model="m", input="hi"), auth_manager
            )

        assert error.status_code == 400

    def test_valid_request_returns_payload_and_registry(self):
        """
        What it does: Returns the payload, registry and tokenizer inputs.
        Purpose: The route needs all four to stream a response.
        """
        auth_manager = MagicMock()

        with patch(
            "kiro.routes_openai_responses.resolve_profile_arn", return_value=VALID_PROFILE_ARN
        ):
            payload, registry, messages, tools, error = resolve_payload(
                ResponsesRequest(
                    model="claude-sonnet-4.5",
                    input="hi",
                    tools=[{"type": "function", "name": "run", "parameters": {}}],
                ),
                auth_manager,
            )

        assert error is None
        assert payload["profileArn"] == VALID_PROFILE_ARN
        assert "run" in registry.routes
        assert messages[-1]["role"] == "user"
        assert tools[0]["name"] == "run"

    def test_illegal_tool_name_returns_400(self):
        """
        What it does: Turns a rejected tool name into HTTP 400.
        Purpose: ToolSpecValidationError is a ValueError, so the route must map
                 it to an actionable invalid_request_error.
        """
        auth_manager = MagicMock()

        with patch(
            "kiro.routes_openai_responses.resolve_profile_arn", return_value=VALID_PROFILE_ARN
        ):
            _, _, _, _, error = resolve_payload(
                ResponsesRequest(
                    model="m",
                    input="hi",
                    tools=[{"type": "function", "name": "bad name!", "parameters": {}}],
                ),
                auth_manager,
            )

        assert error.status_code == 400
        assert json.loads(error.body)["error"]["type"] == "invalid_request_error"


# =============================================================================
# HTTP client selection
# =============================================================================

class TestHTTPClientSelection:
    """Tests for streaming vs non-streaming HTTP client selection."""

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_streaming_uses_per_request_client(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Uses a per-request client for streaming.
        Purpose: A shared client leaks CLOSE_WAIT sockets when the network
                 interface changes mid-stream.
        """
        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(side_effect=RuntimeError("network blocked"))
        instance.close = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST with stream=true...")
        try:
            test_client.post(
                ENDPOINT_PATH,
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "claude-sonnet-4.5", "input": "hi", "stream": True},
            )
        except RuntimeError:
            pass

        assert mock_client_class.called
        assert mock_client_class.call_args[1]["shared_client"] is None

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_non_streaming_uses_shared_client(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Uses the shared pooled client for non-streaming.
        Purpose: Connection reuse matters for short request/response cycles.
        """
        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(side_effect=RuntimeError("network blocked"))
        instance.close = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST with stream=false...")
        try:
            test_client.post(
                ENDPOINT_PATH,
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "claude-sonnet-4.5", "input": "hi", "stream": False},
            )
        except RuntimeError:
            pass

        assert mock_client_class.called
        assert mock_client_class.call_args[1]["shared_client"] is not None


# =============================================================================
# Upstream errors
# =============================================================================

class TestUpstreamErrors:
    """Tests for how upstream failures surface to the client."""

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_upstream_non_200_is_returned_as_kiro_api_error(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Maps an upstream 4xx onto the OpenAI error envelope.
        Purpose: Clients need the upstream status and a readable message.
        """
        upstream = AsyncMock()
        upstream.status_code = 400
        upstream.aread = AsyncMock(
            return_value=json.dumps({"message": "Improperly formed request."}).encode()
        )
        upstream.aclose = AsyncMock()

        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(return_value=upstream)
        instance.close = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST with an upstream 400...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "claude-sonnet-4.5", "input": "hi"},
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error"]["type"] == "kiro_api_error"
        assert body["error"]["code"] == 400
        assert body["error"]["message"]

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_network_error_status_is_preserved(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Surfaces a 504 raised by the HTTP client as 504.
        Purpose: 502/504 mean a network failure; the account system relies on the
                 status to decide whether to fail over, so it must not become 500.
        """
        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(
            side_effect=HTTPException(status_code=504, detail="Server response timeout")
        )
        instance.close = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST with a network timeout...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "claude-sonnet-4.5", "input": "hi"},
        )

        assert response.status_code == 504

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_unreadable_error_body_does_not_crash(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Still returns the upstream status when the body is
                      unreadable.
        Purpose: A broken error body must not turn into an internal error.
        """
        import httpx

        upstream = AsyncMock()
        upstream.status_code = 503
        upstream.aread = AsyncMock(side_effect=httpx.ReadError("connection reset"))
        upstream.aclose = AsyncMock()

        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(return_value=upstream)
        instance.close = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST with an unreadable upstream error body...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "claude-sonnet-4.5", "input": "hi"},
        )

        assert response.status_code == 503


# =============================================================================
# Successful requests
# =============================================================================

class TestSuccessfulRequests:
    """Tests for the success paths with a mocked upstream stream."""

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_non_streaming_returns_a_response_object(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Returns a full Responses object for stream=false.
        Purpose: The non-streaming contract must be complete: id, object, status,
                 output items and usage.
        """
        upstream = AsyncMock()
        upstream.status_code = 200
        upstream.aclose = AsyncMock()

        async def aiter_bytes():
            yield b'{"content":"Hello"}'
            yield b'{"contextUsagePercentage":5.0}'

        upstream.aiter_bytes = aiter_bytes

        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(return_value=upstream)
        instance.close = AsyncMock()
        instance.client = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST with stream=false...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "claude-sonnet-4.5", "input": "hi", "stream": False},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "response"
        assert body["id"].startswith("resp_")
        assert body["status"] == "completed"
        assert body["output"][0]["content"][0]["text"] == "Hello"
        assert body["usage"]["total_tokens"] > 0
        assert body["store"] is False

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_streaming_returns_sse_with_terminal_completed(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Returns an SSE stream ending in response.completed.
        Purpose: Codex aborts with "stream closed before response.completed" if
                 the terminal event is missing, and needs the SSE content type
                 plus the anti-buffering headers.
        """
        upstream = AsyncMock()
        upstream.status_code = 200
        upstream.aclose = AsyncMock()

        async def aiter_bytes():
            yield b'{"content":"Hi"}'
            yield b'{"contextUsagePercentage":1.0}'

        upstream.aiter_bytes = aiter_bytes

        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(return_value=upstream)
        instance.close = AsyncMock()
        instance.client = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST with stream=true...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "claude-sonnet-4.5", "input": "hi", "stream": True},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        assert response.headers["cache-control"] == "no-cache"

        body = response.text
        assert body.startswith("event: response.created")
        assert "event: response.completed" in body

        events = [
            json.loads(line[len("data:"):].strip())
            for line in body.splitlines()
            if line.startswith("data:")
        ]
        numbers = [event["sequence_number"] for event in events]
        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers))
        assert events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hi"

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_captured_codex_request_is_served(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Serves the real captured Codex CLI request end to end.
        Purpose: The decisive integration check at the route level — the exact
                 body Codex sends must produce a valid Responses SSE stream.
        """
        upstream = AsyncMock()
        upstream.status_code = 200
        upstream.aclose = AsyncMock()

        async def aiter_bytes():
            yield b'{"content":"Hello there, friend."}'
            yield b'{"contextUsagePercentage":7.5}'

        upstream.aiter_bytes = aiter_bytes

        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(return_value=upstream)
        instance.close = AsyncMock()
        instance.client = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST the captured Codex CLI body...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json=CODEX_RESPONSES_REQUEST,
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = [
            json.loads(line[len("data:"):].strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        types = [event["type"] for event in events]

        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        assert "response.output_item.added" in types
        assert "response.output_text.delta" in types

        completed = events[-1]["response"]
        assert completed["status"] == "completed"
        assert completed["output"][0]["role"] == "assistant"
        assert completed["output"][0]["content"][0]["text"] == "Hello there, friend."
        assert completed["usage"]["input_tokens"] > 0

        # The Kiro payload must have been built from the captured tools
        kiro_payload = instance.request_with_retry.call_args[0][2]
        tool_specs = kiro_payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]["tools"]
        names = [spec["toolSpecification"]["name"] for spec in tool_specs]
        assert "exec_command" in names
        assert "multi_agent_v1__spawn_agent" in names


# =============================================================================
# Codex CLI code mode (tools declared inside the input array)
# =============================================================================

class TestCodeModeRequests:
    """Route-level tests for the Codex CLI code-mode wire shape."""

    def test_additional_tools_reach_the_kiro_payload(self):
        """
        What it does: Resolves a payload whose tools live only in the input.
        Purpose: This is the shape that silently produced a tool-less request;
                 the resolved payload must now carry the tools.
        """
        auth_manager = MagicMock()
        request = ResponsesRequest(
            model="gpt-5.6-terra",
            input=[
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "tools": [
                                {"type": "custom", "name": "exec", "description": "Run JS"},
                                {
                                    "type": "function",
                                    "name": "wait",
                                    "parameters": {"type": "object"},
                                },
                            ],
                        }
                    ],
                },
                {"type": "message", "role": "user", "content": "do it"},
            ],
        )

        with patch(
            "kiro.routes_openai_responses.resolve_profile_arn", return_value=VALID_PROFILE_ARN
        ):
            payload, registry, _messages, tools, error = resolve_payload(request, auth_manager)

        assert error is None
        assert [tool["name"] for tool in tools] == ["functions__exec", "functions__wait"]
        assert registry.routes["functions__exec"].is_custom is True
        specs = payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]["tools"]
        assert [spec["toolSpecification"]["name"] for spec in specs] == [
            "functions__exec",
            "functions__wait",
        ]

    def test_explicitly_empty_tools_array_plus_additional_tools(self):
        """
        What it does: Sends ``tools: []`` together with additional_tools.
        Purpose: An empty array must not suppress the merge.
        """
        auth_manager = MagicMock()
        request = ResponsesRequest(
            model="gpt-5.6-terra",
            tools=[],
            input=[
                {
                    "type": "additional_tools",
                    "tools": [{"type": "function", "name": "solo", "parameters": {}}],
                },
                {"type": "message", "role": "user", "content": "hi"},
            ],
        )

        with patch(
            "kiro.routes_openai_responses.resolve_profile_arn", return_value=VALID_PROFILE_ARN
        ):
            _payload, _registry, _messages, tools, error = resolve_payload(request, auth_manager)

        assert error is None
        assert [tool["name"] for tool in tools] == ["solo"]

    def test_malformed_additional_tools_does_not_break_the_request(self):
        """
        What it does: Resolves a payload with a junk additional_tools item.
        Purpose: A broken declaration must degrade to "no tools", not to a 500.
        """
        auth_manager = MagicMock()
        request = ResponsesRequest(
            model="gpt-5.6-terra",
            input=[
                {"type": "additional_tools", "tools": "not-a-list"},
                {"type": "message", "role": "user", "content": "hi"},
            ],
        )

        with patch(
            "kiro.routes_openai_responses.resolve_profile_arn", return_value=VALID_PROFILE_ARN
        ):
            payload, _registry, _messages, tools, error = resolve_payload(request, auth_manager)

        assert error is None
        assert tools is None
        assert payload is not None

    def test_additional_tools_only_request_is_not_treated_as_empty_input(self):
        """
        What it does: Sends an additional_tools item as the only input item.
        Purpose: It carries no conversation content, so the request must be
                 refused with the actionable empty-input error rather than
                 producing a Kiro payload with no message.
        """
        auth_manager = MagicMock()
        request = ResponsesRequest(
            model="gpt-5.6-terra",
            input=[
                {"type": "additional_tools", "tools": [{"type": "function", "name": "a"}]}
            ],
        )

        with patch(
            "kiro.routes_openai_responses.resolve_profile_arn", return_value=VALID_PROFILE_ARN
        ):
            _payload, _registry, _messages, _tools, error = resolve_payload(
                request, auth_manager
            )

        assert error is not None
        assert error.status_code == 400
        assert json.loads(error.body)["error"]["type"] == "invalid_request_error"

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_captured_code_mode_request_is_served(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Serves the real captured code-mode body end to end.
        Purpose: The decisive route-level check for the fix — the exact body
                 Codex sends for ``gpt-5.6-terra`` must produce a valid stream
                 AND a Kiro payload that offers all nine declared tools.
        """
        upstream = AsyncMock()
        upstream.status_code = 200
        upstream.aclose = AsyncMock()

        async def aiter_bytes():
            yield b'{"content":"Working on it."}'
            yield b'{"contextUsagePercentage":3.0}'

        upstream.aiter_bytes = aiter_bytes

        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(return_value=upstream)
        instance.close = AsyncMock()
        instance.client = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST the captured Codex CLI code-mode body...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json=CODEX_RESPONSES_CODE_MODE_REQUEST,
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = [
            json.loads(line[len("data:"):].strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"

        kiro_payload = instance.request_with_retry.call_args[0][2]
        tool_specs = kiro_payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]["tools"]
        names = [spec["toolSpecification"]["name"] for spec in tool_specs]
        assert len(names) == 9
        assert "functions__exec" in names
        assert "collaboration__spawn_agent" in names

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_captured_code_mode_follow_up_is_served(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Serves the real follow-up body after ``exec`` ran.
        Purpose: The custom_tool_call round trip must be accepted, otherwise the
                 second turn of every code-mode session fails.
        """
        upstream = AsyncMock()
        upstream.status_code = 200
        upstream.aclose = AsyncMock()

        async def aiter_bytes():
            yield b'{"content":"Done."}'
            yield b'{"contextUsagePercentage":4.0}'

        upstream.aiter_bytes = aiter_bytes

        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(return_value=upstream)
        instance.close = AsyncMock()
        instance.client = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST the captured Codex CLI code-mode follow-up body...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json=CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST,
        )

        assert response.status_code == 200

        kiro_payload = instance.request_with_retry.call_args[0][2]
        history = kiro_payload["conversationState"]["history"]
        tool_uses = [
            entry["assistantResponseMessage"]["toolUses"]
            for entry in history
            if "assistantResponseMessage" in entry
            and entry["assistantResponseMessage"].get("toolUses")
        ]
        assert tool_uses[0][0]["name"] == "functions__exec"

    @patch(
        "kiro.routes_openai_responses.resolve_profile_arn",
        return_value=VALID_PROFILE_ARN,
    )
    @patch("kiro.routes_openai_responses.KiroHttpClient")
    def test_freeform_tool_call_is_streamed_as_a_custom_tool_call(
        self, mock_client_class, _mock_arn, test_client, valid_proxy_api_key
    ):
        """
        What it does: Streams a model call to the declared freeform tool.
        Purpose: The whole loop hinges on Codex receiving a custom_tool_call item
                 with the raw text in ``input``.
        """
        upstream = AsyncMock()
        upstream.status_code = 200
        upstream.aclose = AsyncMock()

        # Kiro's wire form for a tool call: a ``{"name": ...}`` event carrying the
        # arguments object and a terminal ``stop`` flag.
        tool_use_frame = json.dumps(
            {
                "name": "functions__exec",
                "toolUseId": "call_1",
                "input": {"input": "text(1);"},
                "stop": True,
            }
        ).encode("utf-8")

        async def aiter_bytes():
            yield tool_use_frame
            yield b'{"contextUsagePercentage":2.0}'

        upstream.aiter_bytes = aiter_bytes

        instance = AsyncMock()
        instance.request_with_retry = AsyncMock(return_value=upstream)
        instance.close = AsyncMock()
        instance.client = AsyncMock()
        mock_client_class.return_value = instance

        print("Action: POST a code-mode request and stream a freeform tool call...")
        response = test_client.post(
            ENDPOINT_PATH,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "gpt-5.6-terra",
                "stream": True,
                "input": [
                    {
                        "type": "additional_tools",
                        "tools": [
                            {
                                "type": "namespace",
                                "name": "functions",
                                "tools": [{"type": "custom", "name": "exec"}],
                            }
                        ],
                    },
                    {"type": "message", "role": "user", "content": "print 1"},
                ],
            },
        )

        assert response.status_code == 200
        events = [
            json.loads(line[len("data:"):].strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        types = [event["type"] for event in events]
        assert "response.custom_tool_call_input.delta" in types
        assert "response.custom_tool_call_input.done" in types

        item = events[-1]["response"]["output"][0]
        assert item["type"] == "custom_tool_call"
        assert item["name"] == "exec"
        assert item["namespace"] == "functions"
        assert item["input"] == "text(1);"
        assert item["call_id"] == "call_1"
