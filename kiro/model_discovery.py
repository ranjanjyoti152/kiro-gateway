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
Live model discovery for Kiro Gateway.

The runtime host (``runtime.{region}.kiro.dev``) answers ``/ListAvailableModels``
with HTTP 404 ``<UnknownOperationException>``: that operation is not part of the
runtime stack. It is served by the management host
(``management.{region}.kiro.dev``), which is exactly what Kiro IDE itself calls
(``GET /ListAvailableModels`` with a mandatory ``origin`` query parameter, AWS
restJson1 protocol).

This module owns the whole discovery concern so it can be reused by every code
path that needs a model list (initial account initialization and TTL refresh)
and tested in isolation:

- URL / query / header construction (identity headers matter: sibling endpoints
  reject requests without the KiroIDE User-Agent)
- ``nextToken`` pagination with a hard page cap
- Tolerant parsing of the response body
- Mapping of upstream entries into the schema expected by
  :class:`kiro.cache.ModelInfoCache`
- A never-raising helper that falls back to the static model list so startup can
  not break because of model discovery

Discovery is best-effort by design: the gateway must start with a usable model
list even when the management endpoint is unreachable.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import httpx
from loguru import logger

from kiro.config import (
    DEFAULT_MAX_INPUT_TOKENS,
    FALLBACK_MODELS,
    MODEL_DISCOVERY_MAX_PAGES,
    MODEL_DISCOVERY_ORIGIN,
    MODEL_DISCOVERY_PATH,
    MODEL_DISCOVERY_TIMEOUT,
)
from kiro.profile_resolver import resolve_profile_arn
from kiro.utils import get_kiro_headers

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager


# Human-readable discovery sources, used in logs and returned to callers so the
# user can always tell where the exposed model list came from.
SOURCE_MANAGEMENT_ENDPOINT: str = "management endpoint"
SOURCE_Q_ENDPOINT: str = "q endpoint"
SOURCE_STATIC_FALLBACK: str = "static fallback"

# Accepted spellings of the response fields. The upstream API is documented as
# restJson1 with lowerCamelCase keys, but AWS services are not always
# consistent, so both capitalizations are accepted.
_MODELS_KEYS: Tuple[str, ...] = ("models", "Models")
_NEXT_TOKEN_KEYS: Tuple[str, ...] = ("nextToken", "NextToken")


class ModelDiscoveryError(Exception):
    """
    Raised when the management endpoint cannot provide a usable model list.

    Carries a short, user-facing reason (HTTP status, timeout, malformed body)
    so callers can log something actionable before falling back.

    Attributes:
        reason: Short description of what went wrong
    """

    def __init__(self, reason: str):
        """
        Initialize the error.

        Args:
            reason: Short description of the failure (used in log messages)
        """
        super().__init__(reason)
        self.reason = reason


def build_discovery_url(auth_manager: "KiroAuthManager") -> str:
    """
    Build the /ListAvailableModels URL for the account's management host.

    Args:
        auth_manager: Account auth manager providing ``management_host``

    Returns:
        Absolute URL of the model discovery operation

    Examples:
        >>> build_discovery_url(auth_manager)  # doctest: +SKIP
        'https://management.us-east-1.kiro.dev/ListAvailableModels'
    """
    return f"{auth_manager.management_host.rstrip('/')}{MODEL_DISCOVERY_PATH}"


def build_discovery_params(
    auth_manager: "KiroAuthManager",
    next_token: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build query parameters for a model discovery request.

    ``origin`` is mandatory upstream. ``profileArn`` is included only when it can
    be resolved: sending an empty value makes the API reject the request.

    Args:
        auth_manager: Account auth manager (source of the profile ARN)
        next_token: Pagination token returned by a previous page, if any

    Returns:
        Dictionary of query parameters
    """
    params: Dict[str, str] = {"origin": MODEL_DISCOVERY_ORIGIN}

    profile_arn = resolve_profile_arn(auth_manager)
    if profile_arn:
        params["profileArn"] = profile_arn
    else:
        logger.debug("Model discovery: no profileArn resolved, omitting parameter")

    if next_token:
        params["nextToken"] = next_token

    return params


def build_discovery_headers(auth_manager: "KiroAuthManager", token: str) -> Dict[str, str]:
    """
    Build headers for a model discovery request.

    Starts from the shared Kiro identity headers (the KiroIDE ``User-Agent`` and
    ``x-amz-user-agent`` are required - sibling endpoints answer 403 without
    them), then adapts them for the restJson1 operation:

    - ``Content-Type`` becomes ``application/json`` instead of the
      ``application/x-amz-json-1.0`` used by the streaming RPC operation
    - ``x-amz-target`` is removed: it targets
      ``GenerateAssistantResponse`` and does not apply here

    Args:
        auth_manager: Account auth manager (source of the machine fingerprint)
        token: Valid access token

    Returns:
        Dictionary of HTTP headers
    """
    headers = get_kiro_headers(auth_manager, token)
    headers["Content-Type"] = "application/json"
    headers.pop("x-amz-target", None)
    return headers


def _extract_max_input_tokens(raw_model: Dict[str, Any]) -> int:
    """
    Resolve the maximum input token count for an upstream model entry.

    The management endpoint does not always report token limits (most entries
    only carry pricing and capability metadata), so a sensible default is used.
    The lookup order is:

    1. ``tokenLimits.maxInputTokens``
    2. top-level ``maxInputTokens``
    3. ``DEFAULT_MAX_INPUT_TOKENS``

    Values that are not positive integers are ignored, so a malformed upstream
    payload can never turn into ``None`` arithmetic downstream.

    Args:
        raw_model: Single model entry from the API response

    Returns:
        Positive maximum input token count
    """
    candidates: List[Any] = []

    token_limits = raw_model.get("tokenLimits")
    if isinstance(token_limits, dict):
        candidates.append(token_limits.get("maxInputTokens"))

    candidates.append(raw_model.get("maxInputTokens"))

    for candidate in candidates:
        # bool is an int subclass but is never a valid token count
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate

    return DEFAULT_MAX_INPUT_TOKENS


def normalize_model_entry(raw_model: Any) -> Optional[Dict[str, Any]]:
    """
    Map an upstream model entry into the schema expected by ModelInfoCache.

    The cache keys entries by ``modelId`` and reads token limits from
    ``tokenLimits.maxInputTokens``. Upstream metadata (description, rate
    multiplier, prompt caching, supported input types, ...) is preserved as-is so
    it stays available to callers, while the fields the gateway depends on are
    always present and well typed.

    Args:
        raw_model: Single entry from the ``models`` array (any type - the
            upstream payload is untrusted)

    Returns:
        Normalized entry, or None when the entry is unusable (not an object, or
        missing / empty ``modelId``)

    Examples:
        >>> normalize_model_entry({"modelId": "auto", "modelName": "Auto"})["tokenLimits"]
        {'maxInputTokens': 200000}
        >>> normalize_model_entry({"modelName": "No id"}) is None
        True
    """
    if not isinstance(raw_model, dict):
        logger.warning(
            f"Model discovery: skipping model entry of unexpected type "
            f"{type(raw_model).__name__}"
        )
        return None

    model_id = raw_model.get("modelId")
    if not isinstance(model_id, str) or not model_id.strip():
        logger.warning("Model discovery: skipping model entry without a usable 'modelId'")
        return None

    model_id = model_id.strip()

    normalized: Dict[str, Any] = dict(raw_model)
    normalized["modelId"] = model_id

    model_name = raw_model.get("modelName")
    normalized["modelName"] = model_name if isinstance(model_name, str) and model_name else model_id

    normalized["tokenLimits"] = {"maxInputTokens": _extract_max_input_tokens(raw_model)}

    return normalized


def extract_models_page(payload: Any) -> Tuple[List[Any], Optional[str]]:
    """
    Extract the raw model entries and the pagination token from a response body.

    Tolerates both ``models``/``Models`` and ``nextToken``/``NextToken``
    spellings. A missing or non-list models field yields an empty list rather
    than an error, so the caller decides what an empty page means.

    Args:
        payload: Parsed JSON body of a discovery response

    Returns:
        Tuple of (raw model entries, next pagination token or None)
    """
    if not isinstance(payload, dict):
        return [], None

    raw_models: List[Any] = []
    for key in _MODELS_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            raw_models = value
            break

    next_token: Optional[str] = None
    for key in _NEXT_TOKEN_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            next_token = value.strip()
            break

    return raw_models, next_token


async def fetch_available_models(
    auth_manager: "KiroAuthManager",
    token: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch the account's real model list from the management endpoint.

    Performs ``GET {management_host}/ListAvailableModels`` with ``origin`` and
    the resolved ``profileArn``, follows ``nextToken`` pagination up to
    ``MODEL_DISCOVERY_MAX_PAGES`` pages, and normalizes every entry for
    :class:`kiro.cache.ModelInfoCache`.

    Args:
        auth_manager: Account auth manager
        token: Valid access token. When omitted, a token is requested from the
            auth manager.
        client: Optional httpx client to use. When omitted, a short-lived client
            is created and closed by this function.

    Returns:
        Non-empty list of normalized model entries

    Raises:
        ModelDiscoveryError: On any non-200 status, network error, timeout,
            malformed body, or when no usable model entry was returned
    """
    url = build_discovery_url(auth_manager)

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(MODEL_DISCOVERY_TIMEOUT),
            follow_redirects=True,
        )

    models: List[Dict[str, Any]] = []
    skipped_entries = 0
    next_token: Optional[str] = None
    seen_tokens: Set[str] = set()
    pages_fetched = 0

    try:
        if token is None:
            token = await auth_manager.get_access_token()

        headers = build_discovery_headers(auth_manager, token)

        while pages_fetched < MODEL_DISCOVERY_MAX_PAGES:
            params = build_discovery_params(auth_manager, next_token=next_token)

            try:
                response = await client.get(url, params=params, headers=headers)
            except httpx.TimeoutException as e:
                raise ModelDiscoveryError(f"timeout after {MODEL_DISCOVERY_TIMEOUT}s ({e})") from e
            except httpx.RequestError as e:
                raise ModelDiscoveryError(f"network error ({type(e).__name__}: {e})") from e

            pages_fetched += 1

            if response.status_code != 200:
                raise ModelDiscoveryError(f"HTTP {response.status_code}")

            try:
                payload = response.json()
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
                raise ModelDiscoveryError(f"malformed JSON body ({e})") from e

            raw_models, next_token = extract_models_page(payload)

            for raw_model in raw_models:
                normalized = normalize_model_entry(raw_model)
                if normalized is None:
                    skipped_entries += 1
                    continue
                models.append(normalized)

            logger.debug(
                f"Model discovery page {pages_fetched}: {len(raw_models)} entries, "
                f"next_token={'yes' if next_token else 'no'}"
            )

            if not next_token:
                break

            if next_token in seen_tokens:
                logger.warning(
                    "Model discovery: repeated nextToken received, stopping pagination "
                    "to avoid an unbounded loop"
                )
                break

            seen_tokens.add(next_token)
        else:
            logger.warning(
                f"Model discovery: page limit reached "
                f"({MODEL_DISCOVERY_MAX_PAGES}), ignoring remaining pages"
            )

        if skipped_entries:
            logger.warning(
                f"Model discovery: skipped {skipped_entries} malformed model entry(ies)"
            )

        # Deduplicate while preserving order: the same model may legitimately
        # appear on two pages if the upstream list shifts between requests.
        unique_models: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for model in models:
            model_id = model["modelId"]
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            unique_models.append(model)

        if not unique_models:
            raise ModelDiscoveryError("response contained no usable models")

        return unique_models

    finally:
        if owns_client:
            try:
                await client.aclose()
            except (httpx.HTTPError, RuntimeError) as e:
                # Cleanup failure must not mask the original outcome
                logger.debug(f"Model discovery: error closing HTTP client: {e}")


async def discover_models_with_fallback(
    auth_manager: "KiroAuthManager",
    account_label: str,
    token: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Discover models from the management endpoint, falling back to the static list.

    This helper NEVER raises: model discovery is best-effort and must not break
    gateway startup. When the management endpoint cannot be used, a WARNING with
    the concrete reason is logged and ``FALLBACK_MODELS`` is returned.

    Args:
        auth_manager: Account auth manager
        account_label: Account identifier used in log messages
        token: Optional pre-fetched access token
        client: Optional httpx client to reuse

    Returns:
        Tuple of (model entries, source label). The source is either
        ``SOURCE_MANAGEMENT_ENDPOINT`` or ``SOURCE_STATIC_FALLBACK``.
    """
    try:
        models = await fetch_available_models(auth_manager, token=token, client=client)
    except ModelDiscoveryError as e:
        logger.warning(
            f"Account {account_label}: model discovery via {auth_manager.management_host} "
            f"failed ({e.reason}). Falling back to the static model list - "
            f"newer models may be missing until the endpoint is reachable again."
        )
        logger.info(
            f"Account {account_label}: {len(FALLBACK_MODELS)} models from "
            f"{SOURCE_STATIC_FALLBACK}"
        )
        return list(FALLBACK_MODELS), SOURCE_STATIC_FALLBACK
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        # Defensive: token retrieval or an unexpected payload shape must not
        # prevent the account from starting up.
        logger.warning(
            f"Account {account_label}: model discovery failed unexpectedly "
            f"({type(e).__name__}: {e}). Falling back to the static model list."
        )
        logger.info(
            f"Account {account_label}: {len(FALLBACK_MODELS)} models from "
            f"{SOURCE_STATIC_FALLBACK}"
        )
        return list(FALLBACK_MODELS), SOURCE_STATIC_FALLBACK

    logger.info(
        f"Account {account_label}: {len(models)} models from "
        f"{SOURCE_MANAGEMENT_ENDPOINT} ({auth_manager.management_host})"
    )
    return models, SOURCE_MANAGEMENT_ENDPOINT
