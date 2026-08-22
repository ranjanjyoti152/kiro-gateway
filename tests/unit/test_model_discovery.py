# -*- coding: utf-8 -*-

"""
Tests for kiro/model_discovery.py - live model discovery via the management host.

The runtime host does not implement /ListAvailableModels (HTTP 404
UnknownOperationException); the management host does. These tests cover the whole
discovery concern:

- Request shape (method, path, query parameters, identity headers)
- Successful discovery of the real 18-model list
- Tolerant parsing (capitalized keys, missing fields, junk entries)
- Token limit mapping and defaults
- nextToken pagination, including loop protection
- Every failure mode falling back to the static list without raising

All HTTP traffic goes through httpx.MockTransport, so these tests never touch
the network.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from loguru import logger

from kiro.auth import KiroAuthManager
from kiro.cache import ModelInfoCache
from kiro.config import (
    DEFAULT_MAX_INPUT_TOKENS,
    FALLBACK_MODELS,
    MODEL_DISCOVERY_MAX_PAGES,
)
from kiro.model_discovery import (
    SOURCE_MANAGEMENT_ENDPOINT,
    SOURCE_STATIC_FALLBACK,
    ModelDiscoveryError,
    build_discovery_headers,
    build_discovery_params,
    build_discovery_url,
    discover_models_with_fallback,
    extract_models_page,
    fetch_available_models,
    normalize_model_entry,
)


# =============================================================================
# Real data observed on management.us-east-1.kiro.dev/ListAvailableModels
# =============================================================================

REAL_MODEL_IDS: List[str] = [
    "auto",
    "claude-sonnet-5",
    "claude-opus-4.8",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "claude-opus-4.7",
    "claude-opus-4.6",
    "claude-sonnet-4.6",
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "claude-sonnet-4",
    "claude-haiku-4.5",
    "deepseek-3.2",
    "minimax-m2.5",
    "minimax-m2.1",
    "glm-5",
    "qwen3-coder-next",
]

TEST_PROFILE_ARN = "arn:aws:codewhisperer:us-east-1:123456789012:profile/DISCOVERY"


def build_api_model_entry(model_id: str) -> Dict[str, Any]:
    """
    Build a model entry in the exact shape returned by the management endpoint.

    Note there are NO token limit fields - the real payload does not include
    them, which is why the gateway has to default them.

    Args:
        model_id: Model ID for the entry

    Returns:
        Model entry dictionary
    """
    return {
        "additionalModelRequestFieldsSchema": None,
        "availableOrigins": None,
        "description": f"Description of {model_id}",
        "modelId": model_id,
        "modelName": model_id.replace("-", " ").title(),
        "modelProvider": None,
        "promptCaching": {
            "maximumCacheCheckpointsPerRequest": 4,
            "minimumTokensPerCacheCheckpoint": 1024,
            "supportsPromptCaching": True,
        },
        "rateMultiplier": 1.0,
        "rateUnit": "Credit",
        "status": "ACTIVE",
        "supportedInputTypes": ["TEXT", "IMAGE"],
    }


def build_api_response(
    model_ids: List[str],
    next_token: Optional[str] = None,
    models_key: str = "models",
) -> Dict[str, Any]:
    """
    Build a full /ListAvailableModels response body.

    Args:
        model_ids: Model IDs to include
        next_token: Optional pagination token
        models_key: Key used for the models array ("models" or "Models")

    Returns:
        Response body dictionary
    """
    body: Dict[str, Any] = {models_key: [build_api_model_entry(m) for m in model_ids]}
    if next_token:
        body["nextToken"] = next_token
    return body


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def runtime_auth_manager() -> KiroAuthManager:
    """
    Auth manager for an account on the runtime endpoint (us-east-1).

    Has a valid, non-expiring token so discovery never triggers a refresh.
    """
    manager = KiroAuthManager(
        refresh_token="discovery_refresh_token",
        profile_arn=TEST_PROFILE_ARN,
        region="us-east-1",
        api_region="us-east-1",
    )
    manager._access_token = "discovery_access_token"
    manager._expires_at = datetime(2099, 1, 1, tzinfo=timezone.utc)
    return manager


@pytest.fixture
def captured_logs():
    """
    Capture loguru records emitted during a test.

    Yields:
        List of loguru records (appended as they are emitted)
    """
    records: List[Any] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
    yield records
    logger.remove(sink_id)


class RecordingClient:
    """
    Stand-in for httpx.AsyncClient that records requests and replays responses.

    Real ``httpx.Request`` objects are built from the arguments passed by the
    code under test, so URL, query string and header assembly are exercised by
    httpx itself while nothing ever leaves the process.

    Attributes:
        requests: Every request issued, in order
    """

    def __init__(self, responder):
        """
        Initialize the client.

        Args:
            responder: Callable receiving (request, call_index) and returning an
                httpx.Response, or raising an exception to simulate failure
        """
        self.requests: List[httpx.Request] = []
        self.closed = False
        self._responder = responder

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """
        Record a GET request and return the configured response.

        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers

        Returns:
            Response produced by the responder

        Raises:
            Exception: Whatever the responder raises (used to simulate failures)
        """
        request = httpx.Request("GET", url, params=params, headers=headers)
        index = len(self.requests)
        self.requests.append(request)

        response = self._responder(request, index)
        response.request = request
        return response

    async def aclose(self) -> None:
        """Mark the client as closed."""
        self.closed = True

    async def __aenter__(self) -> "RecordingClient":
        """Enter the async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the client on context exit."""
        await self.aclose()

    @property
    def call_count(self) -> int:
        """Number of requests handled."""
        return len(self.requests)


def make_client(recorder: RecordingClient) -> RecordingClient:
    """
    Return the recording client (kept as a helper for readability in tests).

    Args:
        recorder: Recording client instance

    Returns:
        The same recording client
    """
    return recorder


def static_responder(body: Any, status_code: int = 200, raw: Optional[str] = None):
    """
    Build a responder always returning the same response.

    Args:
        body: JSON-serializable body (ignored when raw is provided)
        status_code: HTTP status code to return
        raw: Optional raw text body (used to simulate malformed JSON)

    Returns:
        Responder callable for RequestRecorder
    """
    def _responder(request: httpx.Request, index: int) -> httpx.Response:
        if raw is not None:
            return httpx.Response(status_code, text=raw)
        return httpx.Response(status_code, json=body)

    return _responder


# =============================================================================
# URL / params / headers
# =============================================================================

class TestDiscoveryRequestBuilders:
    """Tests for URL, query parameter and header construction."""

    def test_url_targets_management_host_list_available_models(self, runtime_auth_manager):
        """
        What it does: Verifies the discovery URL uses the management host.
        Purpose: The runtime host answers 404 for this operation.
        """
        url = build_discovery_url(runtime_auth_manager)
        print(f"Discovery URL: {url}")

        assert url == "https://management.us-east-1.kiro.dev/ListAvailableModels"

    def test_url_has_no_double_slash_when_host_has_trailing_slash(self, runtime_auth_manager):
        """
        What it does: Verifies a trailing slash on the host does not break the path.
        Purpose: Defensive - hosts come from templates that could change.
        """
        runtime_auth_manager._management_host = "https://management.us-east-1.kiro.dev/"

        url = build_discovery_url(runtime_auth_manager)
        print(f"Discovery URL: {url}")

        assert url == "https://management.us-east-1.kiro.dev/ListAvailableModels"

    def test_params_include_origin_and_profile_arn(self, runtime_auth_manager):
        """
        What it does: Verifies origin and profileArn query parameters.
        Purpose: origin is mandatory upstream, profileArn scopes the model list.
        """
        params = build_discovery_params(runtime_auth_manager)
        print(f"Params: {params}")

        assert params["origin"] == "AI_EDITOR"
        assert params["profileArn"] == TEST_PROFILE_ARN
        assert "nextToken" not in params

    def test_params_include_next_token_when_provided(self, runtime_auth_manager):
        """
        What it does: Verifies nextToken is forwarded for subsequent pages.
        Purpose: Pagination requires echoing the token back.
        """
        params = build_discovery_params(runtime_auth_manager, next_token="page-2")
        print(f"Params: {params}")

        assert params["nextToken"] == "page-2"

    def test_params_omit_profile_arn_when_unresolvable(self, runtime_auth_manager):
        """
        What it does: Verifies profileArn is omitted when it cannot be resolved.
        Purpose: Sending an empty profileArn makes the API reject the request.
        """
        runtime_auth_manager._profile_arn = None

        with patch("kiro.profile_resolver.PROFILE_ARN", ""):
            params = build_discovery_params(runtime_auth_manager)

        print(f"Params: {params}")
        assert "profileArn" not in params
        assert params["origin"] == "AI_EDITOR"

    def test_headers_use_json_content_type_and_drop_amz_target(self, runtime_auth_manager):
        """
        What it does: Verifies restJson1 headers (JSON content type, no x-amz-target).
        Purpose: x-amz-target points at GenerateAssistantResponse and breaks this call.
        """
        headers = build_discovery_headers(runtime_auth_manager, "token-123")
        print(f"Headers: {sorted(headers)}")

        assert headers["Content-Type"] == "application/json"
        assert "x-amz-target" not in headers
        assert headers["Authorization"] == "Bearer token-123"

    def test_headers_keep_kiro_ide_identity(self, runtime_auth_manager):
        """
        What it does: Verifies KiroIDE identity headers are preserved.
        Purpose: Sibling endpoints answer 403 without the KiroIDE user agent.
        """
        headers = build_discovery_headers(runtime_auth_manager, "token-123")

        print(f"User-Agent: {headers['User-Agent']}")
        assert "KiroIDE" in headers["User-Agent"]
        assert "KiroIDE" in headers["x-amz-user-agent"]
        assert runtime_auth_manager.fingerprint in headers["User-Agent"]


# =============================================================================
# Successful discovery
# =============================================================================

class TestDiscoverySuccess:
    """Tests for successful model discovery."""

    @pytest.mark.asyncio
    async def test_discovers_all_real_models(self, runtime_auth_manager):
        """
        What it does: Discovers the full real 18-model list.
        Purpose: The whole point of the feature - expose the account's real models.
        """
        recorder = RecordingClient(static_responder(build_api_response(REAL_MODEL_IDS)))

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        discovered_ids = [m["modelId"] for m in models]
        print(f"Discovered {len(discovered_ids)} models: {discovered_ids}")

        assert len(models) == 18
        assert discovered_ids == REAL_MODEL_IDS

    @pytest.mark.asyncio
    async def test_discovered_models_land_in_cache_and_are_valid(self, runtime_auth_manager):
        """
        What it does: Feeds discovered models into ModelInfoCache.
        Purpose: Ensure the mapped schema is accepted by the cache used by both APIs.
        """
        recorder = RecordingClient(static_responder(build_api_response(REAL_MODEL_IDS)))

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        cache = ModelInfoCache()
        await cache.update(models)

        print(f"Cache size: {cache.size}")
        assert cache.size == 18
        assert sorted(cache.get_all_model_ids()) == sorted(REAL_MODEL_IDS)

        for new_model in ("gpt-5.6-sol", "claude-sonnet-5", "claude-opus-4.8"):
            print(f"Checking cache validity of {new_model}...")
            assert cache.is_valid_model(new_model) is True

    @pytest.mark.asyncio
    async def test_request_shape_is_get_with_expected_query_and_headers(
        self, runtime_auth_manager
    ):
        """
        What it does: Asserts the actual outgoing request shape.
        Purpose: Lock the empirically verified contract of the endpoint.
        """
        recorder = RecordingClient(static_responder(build_api_response(["auto"])))

        async with make_client(recorder) as client:
            await fetch_available_models(runtime_auth_manager, token="tok", client=client)

        assert recorder.call_count == 1
        request = recorder.requests[0]

        print(f"Method: {request.method}, URL: {request.url}")
        assert request.method == "GET"
        assert request.url.path == "/ListAvailableModels"
        assert request.url.host == "management.us-east-1.kiro.dev"
        assert request.url.params["origin"] == "AI_EDITOR"
        assert request.url.params["profileArn"] == TEST_PROFILE_ARN

        print(f"Request headers: {dict(request.headers)}")
        assert request.headers["content-type"] == "application/json"
        assert "x-amz-target" not in request.headers
        assert "KiroIDE" in request.headers["user-agent"]
        assert "KiroIDE" in request.headers["x-amz-user-agent"]
        assert request.headers["authorization"] == "Bearer tok"

    @pytest.mark.asyncio
    async def test_capitalized_models_key_is_accepted(self, runtime_auth_manager):
        """
        What it does: Parses a body using "Models" instead of "models".
        Purpose: AWS services are not always consistent about casing.
        """
        body = build_api_response(["auto", "glm-5"], models_key="Models")
        recorder = RecordingClient(static_responder(body))

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        print(f"Discovered: {[m['modelId'] for m in models]}")
        assert [m["modelId"] for m in models] == ["auto", "glm-5"]

    @pytest.mark.asyncio
    async def test_upstream_metadata_is_preserved(self, runtime_auth_manager):
        """
        What it does: Verifies upstream metadata survives normalization.
        Purpose: Response enrichment must not drop useful upstream data.
        """
        recorder = RecordingClient(static_responder(build_api_response(["auto"])))

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        entry = models[0]
        print(f"Normalized entry: {entry}")

        assert entry["rateUnit"] == "Credit"
        assert entry["promptCaching"]["supportsPromptCaching"] is True
        assert entry["supportedInputTypes"] == ["TEXT", "IMAGE"]
        assert entry["description"] == "Description of auto"

    @pytest.mark.asyncio
    async def test_token_is_fetched_from_auth_manager_when_not_supplied(
        self, runtime_auth_manager
    ):
        """
        What it does: Verifies the access token is requested when not passed in.
        Purpose: The function must be usable standalone (TTL refresh path).
        """
        recorder = RecordingClient(static_responder(build_api_response(["auto"])))

        async with make_client(recorder) as client:
            await fetch_available_models(runtime_auth_manager, client=client)

        auth_header = recorder.requests[0].headers["authorization"]
        print(f"Authorization: {auth_header}")
        assert auth_header == "Bearer discovery_access_token"

    @pytest.mark.asyncio
    async def test_default_client_is_created_and_closed(self, runtime_auth_manager):
        """
        What it does: Verifies a short-lived client is created and closed.
        Purpose: Prevent connection leaks when no client is supplied.
        """
        response = httpx.Response(200, json=build_api_response(["auto"]))
        fake_client = MagicMock()
        fake_client.get = AsyncMock(return_value=response)
        fake_client.aclose = AsyncMock()

        with patch(
            "kiro.model_discovery.httpx.AsyncClient", return_value=fake_client
        ) as client_factory:
            models = await fetch_available_models(runtime_auth_manager, token="tok")

        print(f"Client created: {client_factory.called}, closed: {fake_client.aclose.called}")
        assert client_factory.called is True
        assert fake_client.aclose.await_count == 1
        assert [m["modelId"] for m in models] == ["auto"]


# =============================================================================
# Entry normalization and token limits
# =============================================================================

class TestNormalizeModelEntry:
    """Tests for normalize_model_entry()."""

    def test_entry_without_token_limits_gets_default(self):
        """
        What it does: Verifies default token limits for entries without them.
        Purpose: The real payload has no token limit fields at all.
        """
        entry = normalize_model_entry(build_api_model_entry("auto"))
        print(f"Token limits: {entry['tokenLimits']}")

        assert entry["tokenLimits"]["maxInputTokens"] == DEFAULT_MAX_INPUT_TOKENS

    def test_entry_with_token_limits_is_preserved(self):
        """
        What it does: Verifies reported token limits are kept.
        Purpose: Do not overwrite real upstream data with defaults.
        """
        raw = build_api_model_entry("auto")
        raw["tokenLimits"] = {"maxInputTokens": 123456}

        entry = normalize_model_entry(raw)
        print(f"Token limits: {entry['tokenLimits']}")

        assert entry["tokenLimits"]["maxInputTokens"] == 123456

    def test_top_level_max_input_tokens_is_used(self):
        """
        What it does: Verifies a top-level maxInputTokens field is honored.
        Purpose: Tolerate a flatter payload shape.
        """
        raw = build_api_model_entry("auto")
        raw["maxInputTokens"] = 99000

        entry = normalize_model_entry(raw)
        print(f"Token limits: {entry['tokenLimits']}")

        assert entry["tokenLimits"]["maxInputTokens"] == 99000

    @pytest.mark.parametrize(
        "bad_value",
        [None, 0, -5, "200000", True, 1.5, {"nested": 1}, []],
        ids=["none", "zero", "negative", "string", "bool", "float", "dict", "list"],
    )
    def test_invalid_token_limits_fall_back_to_default(self, bad_value):
        """
        What it does: Verifies junk token limits are replaced with the default.
        Purpose: Never let None or a string reach token arithmetic.
        """
        raw = build_api_model_entry("auto")
        raw["tokenLimits"] = {"maxInputTokens": bad_value}

        entry = normalize_model_entry(raw)
        print(f"Input {bad_value!r} → {entry['tokenLimits']}")

        assert entry["tokenLimits"]["maxInputTokens"] == DEFAULT_MAX_INPUT_TOKENS

    def test_non_dict_token_limits_fall_back_to_default(self):
        """
        What it does: Verifies a non-object tokenLimits value is ignored.
        Purpose: Malformed upstream data must not raise.
        """
        raw = build_api_model_entry("auto")
        raw["tokenLimits"] = "not-an-object"

        entry = normalize_model_entry(raw)
        print(f"Token limits: {entry['tokenLimits']}")

        assert entry["tokenLimits"]["maxInputTokens"] == DEFAULT_MAX_INPUT_TOKENS

    def test_model_name_defaults_to_model_id(self):
        """
        What it does: Verifies modelName falls back to modelId.
        Purpose: Keep a display name available for every entry.
        """
        entry = normalize_model_entry({"modelId": "glm-5"})
        print(f"Entry: {entry}")

        assert entry["modelName"] == "glm-5"

    def test_model_id_is_stripped(self):
        """
        What it does: Verifies surrounding whitespace in modelId is removed.
        Purpose: A padded ID would never match a normalized request name.
        """
        entry = normalize_model_entry({"modelId": "  glm-5  "})
        print(f"Entry: {entry}")

        assert entry["modelId"] == "glm-5"

    @pytest.mark.parametrize(
        "bad_entry",
        [
            {"modelName": "No id"},
            {"modelId": ""},
            {"modelId": "   "},
            {"modelId": None},
            {"modelId": 42},
            "just-a-string",
            None,
            [],
            123,
        ],
        ids=[
            "missing_id",
            "empty_id",
            "whitespace_id",
            "null_id",
            "numeric_id",
            "string_entry",
            "none_entry",
            "list_entry",
            "int_entry",
        ],
    )
    def test_unusable_entries_are_rejected(self, bad_entry):
        """
        What it does: Verifies unusable entries are dropped, not crashed on.
        Purpose: The cache keys on modelId - a bad entry would raise KeyError.
        """
        result = normalize_model_entry(bad_entry)
        print(f"Entry {bad_entry!r} → {result}")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_max_input_tokens_usable_for_discovered_models(
        self, runtime_auth_manager
    ):
        """
        What it does: Reads get_max_input_tokens() for every discovered model.
        Purpose: Ensure no KeyError / None arithmetic downstream.
        """
        recorder = RecordingClient(static_responder(build_api_response(REAL_MODEL_IDS)))

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        cache = ModelInfoCache()
        await cache.update(models)

        for model_id in REAL_MODEL_IDS:
            max_tokens = cache.get_max_input_tokens(model_id)
            print(f"{model_id}: max_input_tokens={max_tokens}")
            assert isinstance(max_tokens, int)
            assert max_tokens > 0


# =============================================================================
# Page extraction
# =============================================================================

class TestExtractModelsPage:
    """Tests for extract_models_page()."""

    def test_extracts_models_and_next_token(self):
        """
        What it does: Extracts both models and the pagination token.
        Purpose: Baseline parsing behavior.
        """
        models, token = extract_models_page({"models": [{"modelId": "a"}], "nextToken": "t"})
        print(f"Models: {models}, token: {token}")

        assert models == [{"modelId": "a"}]
        assert token == "t"

    def test_accepts_capitalized_keys(self):
        """
        What it does: Handles "Models"/"NextToken" spellings.
        Purpose: Tolerate inconsistent AWS casing.
        """
        models, token = extract_models_page({"Models": [{"modelId": "a"}], "NextToken": "t"})
        print(f"Models: {models}, token: {token}")

        assert models == [{"modelId": "a"}]
        assert token == "t"

    @pytest.mark.parametrize(
        "payload",
        [{}, {"models": None}, {"models": "not-a-list"}, {"models": {}}, [], "text", None, 42],
        ids=["empty", "null", "string", "dict", "list_body", "text_body", "none", "int"],
    )
    def test_unusable_payloads_yield_empty_list(self, payload):
        """
        What it does: Verifies unusable payloads produce an empty list.
        Purpose: Parsing must never raise; the caller decides what empty means.
        """
        models, token = extract_models_page(payload)
        print(f"Payload {payload!r} → models={models}, token={token}")

        assert models == []
        assert token is None

    @pytest.mark.parametrize(
        "token_value",
        [None, "", "   ", 42, [], {}],
        ids=["none", "empty", "whitespace", "int", "list", "dict"],
    )
    def test_unusable_next_tokens_are_ignored(self, token_value):
        """
        What it does: Verifies junk pagination tokens are treated as absent.
        Purpose: An empty token must not trigger another request.
        """
        _, token = extract_models_page({"models": [], "nextToken": token_value})
        print(f"Token {token_value!r} → {token}")

        assert token is None


# =============================================================================
# Pagination
# =============================================================================

class TestPagination:
    """Tests for nextToken pagination handling."""

    @pytest.mark.asyncio
    async def test_two_pages_are_merged(self, runtime_auth_manager):
        """
        What it does: Merges two pages joined by nextToken.
        Purpose: The endpoint paginates; a partial list would hide models.
        """
        page_one = build_api_response(REAL_MODEL_IDS[:10], next_token="page-2")
        page_two = build_api_response(REAL_MODEL_IDS[10:])

        def responder(request: httpx.Request, index: int) -> httpx.Response:
            if index == 0:
                assert "nextToken" not in request.url.params
                return httpx.Response(200, json=page_one)
            assert request.url.params["nextToken"] == "page-2"
            return httpx.Response(200, json=page_two)

        recorder = RecordingClient(responder)

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        print(f"Requests: {recorder.call_count}, models: {len(models)}")
        assert recorder.call_count == 2
        assert [m["modelId"] for m in models] == REAL_MODEL_IDS

    @pytest.mark.asyncio
    async def test_repeated_next_token_stops_pagination(self, runtime_auth_manager, captured_logs):
        """
        What it does: Stops when the server repeats the same nextToken.
        Purpose: A stuck token must not cause an unbounded request loop.
        """
        body = build_api_response(["auto"], next_token="same-token")
        recorder = RecordingClient(static_responder(body))

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        print(f"Requests: {recorder.call_count}, models: {[m['modelId'] for m in models]}")
        assert recorder.call_count == 2
        assert [m["modelId"] for m in models] == ["auto"]

        warnings = [r["message"] for r in captured_logs if r["level"].name == "WARNING"]
        print(f"Warnings: {warnings}")
        assert any("repeated nextToken" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_page_limit_caps_request_count(self, runtime_auth_manager, captured_logs):
        """
        What it does: Stops after MODEL_DISCOVERY_MAX_PAGES with always-new tokens.
        Purpose: Hard cap against an endpoint that never stops paginating.
        """
        def responder(request: httpx.Request, index: int) -> httpx.Response:
            return httpx.Response(
                200,
                json=build_api_response([f"model-{index}"], next_token=f"token-{index}"),
            )

        recorder = RecordingClient(responder)

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        print(f"Requests: {recorder.call_count} (cap={MODEL_DISCOVERY_MAX_PAGES})")
        assert recorder.call_count == MODEL_DISCOVERY_MAX_PAGES
        assert len(models) == MODEL_DISCOVERY_MAX_PAGES

        warnings = [r["message"] for r in captured_logs if r["level"].name == "WARNING"]
        assert any("page limit reached" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_duplicate_models_across_pages_are_deduplicated(self, runtime_auth_manager):
        """
        What it does: Deduplicates a model repeated on two pages.
        Purpose: The upstream list can shift between paged requests.
        """
        page_one = build_api_response(["auto", "glm-5"], next_token="page-2")
        page_two = build_api_response(["glm-5", "deepseek-3.2"])

        def responder(request: httpx.Request, index: int) -> httpx.Response:
            return httpx.Response(200, json=page_one if index == 0 else page_two)

        recorder = RecordingClient(responder)

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        ids = [m["modelId"] for m in models]
        print(f"Merged ids: {ids}")
        assert ids == ["auto", "glm-5", "deepseek-3.2"]

    @pytest.mark.asyncio
    async def test_second_page_failure_propagates(self, runtime_auth_manager):
        """
        What it does: Fails discovery when a later page errors out.
        Purpose: A partially fetched list is not trustworthy - fall back instead.
        """
        def responder(request: httpx.Request, index: int) -> httpx.Response:
            if index == 0:
                return httpx.Response(
                    200, json=build_api_response(["auto"], next_token="page-2")
                )
            return httpx.Response(500, json={"message": "boom"})

        recorder = RecordingClient(responder)

        async with make_client(recorder) as client:
            with pytest.raises(ModelDiscoveryError) as exc_info:
                await fetch_available_models(
                    runtime_auth_manager, token="tok", client=client
                )

        print(f"Reason: {exc_info.value.reason}")
        assert "500" in exc_info.value.reason


# =============================================================================
# Failure modes
# =============================================================================

class TestDiscoveryFailures:
    """Tests for every failure mode of fetch_available_models()."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 403, 404, 429, 500, 503])
    async def test_non_200_status_raises(self, runtime_auth_manager, status_code):
        """
        What it does: Raises ModelDiscoveryError for non-200 responses.
        Purpose: Signal the caller to fall back.
        """
        recorder = RecordingClient(
            static_responder({"message": "error"}, status_code=status_code)
        )

        async with make_client(recorder) as client:
            with pytest.raises(ModelDiscoveryError) as exc_info:
                await fetch_available_models(
                    runtime_auth_manager, token="tok", client=client
                )

        print(f"Status {status_code} → reason: {exc_info.value.reason}")
        assert str(status_code) in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_timeout_raises_with_reason(self, runtime_auth_manager):
        """
        What it does: Converts httpx.TimeoutException into ModelDiscoveryError.
        Purpose: Startup must not hang or crash on a slow endpoint.
        """
        def responder(request: httpx.Request, index: int) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        recorder = RecordingClient(responder)

        async with make_client(recorder) as client:
            with pytest.raises(ModelDiscoveryError) as exc_info:
                await fetch_available_models(
                    runtime_auth_manager, token="tok", client=client
                )

        print(f"Reason: {exc_info.value.reason}")
        assert "timeout" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_network_error_raises_with_reason(self, runtime_auth_manager):
        """
        What it does: Converts httpx.RequestError into ModelDiscoveryError.
        Purpose: DNS or connection failures must be reported, not raised raw.
        """
        def responder(request: httpx.Request, index: int) -> httpx.Response:
            raise httpx.ConnectError("dns failure", request=request)

        recorder = RecordingClient(responder)

        async with make_client(recorder) as client:
            with pytest.raises(ModelDiscoveryError) as exc_info:
                await fetch_available_models(
                    runtime_auth_manager, token="tok", client=client
                )

        print(f"Reason: {exc_info.value.reason}")
        assert "network error" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self, runtime_auth_manager):
        """
        What it does: Handles a 200 response with a non-JSON body.
        Purpose: Proxies and error pages return HTML with a 200 status.
        """
        recorder = RecordingClient(static_responder(None, raw="<html>not json</html>"))

        async with make_client(recorder) as client:
            with pytest.raises(ModelDiscoveryError) as exc_info:
                await fetch_available_models(
                    runtime_auth_manager, token="tok", client=client
                )

        print(f"Reason: {exc_info.value.reason}")
        assert "malformed JSON" in exc_info.value.reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"models": []},
            {"models": [{"modelName": "no id"}]},
            {"models": ["string", 42, None]},
            {"other": "field"},
            [],
        ],
        ids=["no_key", "empty_list", "entry_without_id", "junk_entries", "wrong_key", "list_body"],
    )
    async def test_bodies_without_usable_models_raise(self, runtime_auth_manager, body):
        """
        What it does: Raises when no usable model entry is present.
        Purpose: An empty model list would leave the gateway with zero models.
        """
        recorder = RecordingClient(static_responder(body))

        async with make_client(recorder) as client:
            with pytest.raises(ModelDiscoveryError) as exc_info:
                await fetch_available_models(
                    runtime_auth_manager, token="tok", client=client
                )

        print(f"Body {body!r} → reason: {exc_info.value.reason}")
        assert "no usable models" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_partially_junk_page_keeps_valid_entries(self, runtime_auth_manager):
        """
        What it does: Keeps valid entries while dropping junk ones.
        Purpose: One bad entry must not discard the whole model list.
        """
        body = {
            "models": [
                build_api_model_entry("auto"),
                {"modelName": "broken"},
                "not-an-object",
                build_api_model_entry("glm-5"),
            ]
        }
        recorder = RecordingClient(static_responder(body))

        async with make_client(recorder) as client:
            models = await fetch_available_models(
                runtime_auth_manager, token="tok", client=client
            )

        print(f"Discovered: {[m['modelId'] for m in models]}")
        assert [m["modelId"] for m in models] == ["auto", "glm-5"]


# =============================================================================
# Fallback helper
# =============================================================================

class TestDiscoverWithFallback:
    """Tests for discover_models_with_fallback() - the never-raising entry point."""

    @pytest.mark.asyncio
    async def test_success_returns_management_source(self, runtime_auth_manager, captured_logs):
        """
        What it does: Returns discovered models labelled as the management source.
        Purpose: Users need to know where the exposed list came from.
        """
        recorder = RecordingClient(static_responder(build_api_response(REAL_MODEL_IDS)))

        async with make_client(recorder) as client:
            models, source = await discover_models_with_fallback(
                runtime_auth_manager, account_label="acct-1", token="tok", client=client
            )

        print(f"Source: {source}, models: {len(models)}")
        assert source == SOURCE_MANAGEMENT_ENDPOINT
        assert len(models) == 18

        infos = [r["message"] for r in captured_logs if r["level"].name == "INFO"]
        print(f"INFO logs: {infos}")
        assert any("18 models from management endpoint" in message for message in infos)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code",
        [403, 404, 500],
        ids=["forbidden", "not_found", "server_error"],
    )
    async def test_http_errors_fall_back_with_warning(
        self, runtime_auth_manager, captured_logs, status_code
    ):
        """
        What it does: Falls back to the static list on HTTP errors.
        Purpose: Startup must never break because of model discovery.
        """
        recorder = RecordingClient(
            static_responder({"message": "error"}, status_code=status_code)
        )

        async with make_client(recorder) as client:
            models, source = await discover_models_with_fallback(
                runtime_auth_manager, account_label="acct-1", token="tok", client=client
            )

        print(f"Status {status_code} → source: {source}, models: {len(models)}")
        assert source == SOURCE_STATIC_FALLBACK
        assert [m["modelId"] for m in models] == [m["modelId"] for m in FALLBACK_MODELS]

        warnings = [r["message"] for r in captured_logs if r["level"].name == "WARNING"]
        print(f"Warnings: {warnings}")
        assert any(str(status_code) in message for message in warnings)
        assert any("static model list" in message for message in warnings)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_factory,expected_fragment",
        [
            (lambda request: httpx.ReadTimeout("slow", request=request), "timeout"),
            (lambda request: httpx.ConnectTimeout("slow", request=request), "timeout"),
            (lambda request: httpx.ConnectError("dns", request=request), "network error"),
            (lambda request: httpx.RemoteProtocolError("broken", request=request), "network error"),
        ],
        ids=["read_timeout", "connect_timeout", "connect_error", "protocol_error"],
    )
    async def test_transport_errors_fall_back_with_warning(
        self, runtime_auth_manager, captured_logs, exception_factory, expected_fragment
    ):
        """
        What it does: Falls back on timeouts and network errors.
        Purpose: Offline or restricted networks must still start the gateway.
        """
        def responder(request: httpx.Request, index: int) -> httpx.Response:
            raise exception_factory(request)

        recorder = RecordingClient(responder)

        async with make_client(recorder) as client:
            models, source = await discover_models_with_fallback(
                runtime_auth_manager, account_label="acct-1", token="tok", client=client
            )

        print(f"Source: {source}, models: {len(models)}")
        assert source == SOURCE_STATIC_FALLBACK
        assert len(models) == len(FALLBACK_MODELS)

        warnings = [r["message"] for r in captured_logs if r["level"].name == "WARNING"]
        assert any(expected_fragment in message.lower() for message in warnings)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body,raw",
        [
            (None, "<html>error</html>"),
            ({"other": "field"}, None),
            ({"models": []}, None),
            ({"models": [{"modelName": "no id"}]}, None),
        ],
        ids=["malformed_json", "models_missing", "models_empty", "entries_without_id"],
    )
    async def test_bad_bodies_fall_back_with_warning(
        self, runtime_auth_manager, captured_logs, body, raw
    ):
        """
        What it does: Falls back when the body carries no usable models.
        Purpose: Never expose an empty model list to clients.
        """
        recorder = RecordingClient(static_responder(body, raw=raw))

        async with make_client(recorder) as client:
            models, source = await discover_models_with_fallback(
                runtime_auth_manager, account_label="acct-1", token="tok", client=client
            )

        print(f"Source: {source}, models: {len(models)}")
        assert source == SOURCE_STATIC_FALLBACK
        assert len(models) == len(FALLBACK_MODELS)

        warnings = [r["message"] for r in captured_logs if r["level"].name == "WARNING"]
        assert any("static model list" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_token_failure_falls_back(self, runtime_auth_manager, captured_logs):
        """
        What it does: Falls back when the access token cannot be obtained.
        Purpose: Token trouble is reported by the request path, not by discovery.
        """
        recorder = RecordingClient(static_responder(build_api_response(["auto"])))

        with patch.object(
            runtime_auth_manager,
            "get_access_token",
            AsyncMock(side_effect=ValueError("no token")),
        ):
            async with make_client(recorder) as client:
                models, source = await discover_models_with_fallback(
                    runtime_auth_manager, account_label="acct-1", client=client
                )

        print(f"Source: {source}, requests: {recorder.call_count}")
        assert source == SOURCE_STATIC_FALLBACK
        assert recorder.call_count == 0
        assert len(models) == len(FALLBACK_MODELS)

        warnings = [r["message"] for r in captured_logs if r["level"].name == "WARNING"]
        assert any("model discovery failed" in message.lower() for message in warnings)

    @pytest.mark.asyncio
    async def test_fallback_result_is_a_copy(self, runtime_auth_manager):
        """
        What it does: Verifies the returned fallback list is not the shared constant.
        Purpose: A caller mutating the result must not corrupt FALLBACK_MODELS.
        """
        recorder = RecordingClient(static_responder({"models": []}))

        async with make_client(recorder) as client:
            models, _ = await discover_models_with_fallback(
                runtime_auth_manager, account_label="acct-1", token="tok", client=client
            )

        models.append({"modelId": "injected"})

        print(f"FALLBACK_MODELS size after mutation: {len(FALLBACK_MODELS)}")
        assert all(m["modelId"] != "injected" for m in FALLBACK_MODELS)
