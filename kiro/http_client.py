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
HTTP client for Kiro API with retry logic support.

Handles:
- 403: automatic token refresh and retry
- 429: exponential backoff
- 5xx: exponential backoff
- Timeouts: exponential backoff

Supports both per-request clients and shared application-level client
with connection pooling for better resource management.
"""

import asyncio
import json
import ssl
import threading
import random
import time
from typing import Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import (
    MAX_RETRIES,
    BASE_RETRY_DELAY,
    FIRST_TOKEN_MAX_RETRIES,
    STREAMING_READ_TIMEOUT,
    SERVER_ERROR_MAX_RETRIES,
    MAX_RETRY_DELAY,
)
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers
from kiro.network_errors import (
    classify_network_error,
    get_short_error_message,
    build_http_error_detail,
    NetworkErrorInfo,
)


# ==================================================================================================
# Retry backoff helper
# ==================================================================================================

def compute_backoff_delay(
    attempt: int,
    base_delay: float = BASE_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
) -> float:
    """
    Compute an exponential backoff delay with full jitter, capped at ``max_delay``.

    Exponential backoff lets a transiently overloaded upstream (e.g. Kiro/Amazon
    returning HTTP 500 ``MODEL_TEMPORARILY_UNAVAILABLE``) recover between attempts,
    while jitter prevents many concurrent clients from retrying in lockstep
    ("thundering herd"). The delay is bounded so a long retry budget never
    produces an unbounded wait.

    Args:
        attempt: Zero-based attempt index (0 for the first retry wait).
        base_delay: Base delay in seconds before exponential growth.
        max_delay: Upper bound in seconds for the returned delay.

    Returns:
        Delay in seconds in the range [0, capped_exponential_delay], where
        capped_exponential_delay = min(base_delay * 2**attempt, max_delay).

    Examples:
        >>> # With base_delay=1.0, max_delay=30.0 the cap per attempt is:
        >>> # attempt 0 -> up to 1s, 1 -> up to 2s, 2 -> up to 4s, ... capped at 30s
        >>> 0.0 <= compute_backoff_delay(0, 1.0, 30.0) <= 1.0
        True
        >>> 0.0 <= compute_backoff_delay(10, 1.0, 30.0) <= 30.0
        True
    """
    capped = min(base_delay * (2 ** attempt), max_delay)
    # Full jitter: sample uniformly in [0, capped] to spread out retries.
    return random.uniform(0.0, capped)


# ==================================================================================================
# Shared TLS configuration
# ==================================================================================================

# Process-wide TLS context reused by every per-request client. Building an
# SSLContext loads and parses the full CA bundle, which costs ~20 ms of CPU on
# the event loop; streaming requests create one client each, so without reuse
# every streaming request (and every account failover) paid that cost again.
# An SSLContext holds configuration only, never connection state, so sharing it
# keeps the per-request client isolation that prevents CLOSE_WAIT leaks.
_shared_ssl_context: Optional[ssl.SSLContext] = None
_shared_ssl_context_lock = threading.Lock()


def get_shared_ssl_context() -> ssl.SSLContext:
    """
    Return the process-wide TLS context used by per-request HTTP clients.

    The context is created lazily on first use with httpx defaults
    (certificate verification enabled, ``SSL_CERT_FILE`` / ``SSL_CERT_DIR``
    honored), so it verifies exactly like a freshly constructed
    ``httpx.AsyncClient`` would.

    Returns:
        Shared ``ssl.SSLContext`` instance.

    Raises:
        ssl.SSLError: If the CA bundle cannot be loaded.
        OSError: If a configured CA file or directory cannot be read.
    """
    global _shared_ssl_context
    if _shared_ssl_context is None:
        with _shared_ssl_context_lock:
            if _shared_ssl_context is None:
                started = time.perf_counter()
                _shared_ssl_context = httpx.create_ssl_context(verify=True, trust_env=True)
                logger.debug(
                    f"Created shared TLS context for per-request clients "
                    f"in {(time.perf_counter() - started) * 1000:.1f} ms"
                )
    return _shared_ssl_context


def reset_shared_ssl_context() -> None:
    """
    Drop the cached TLS context so the next client builds a fresh one.

    Intended for tests and for picking up a changed CA bundle without a restart.
    """
    global _shared_ssl_context
    with _shared_ssl_context_lock:
        _shared_ssl_context = None


class KiroHttpClient:
    """
    HTTP client for Kiro API with retry logic support.
    
    Automatically handles errors and retries requests:
    - 403: refreshes token and retries
    - 429: waits with exponential backoff
    - 5xx: waits with exponential backoff
    - Timeouts: waits with exponential backoff
    
    Supports two modes of operation:
    1. Per-request client: Creates and owns its own httpx.AsyncClient
    2. Shared client: Uses an application-level shared client (recommended)
    
    Using a shared client reduces memory usage and enables connection pooling,
    which is especially important for handling concurrent requests.
    
    Attributes:
        auth_manager: Authentication manager for obtaining tokens
        client: httpx HTTP client (owned or shared)
    
    Example:
        >>> # Per-request client (legacy mode)
        >>> client = KiroHttpClient(auth_manager)
        >>> response = await client.request_with_retry(...)
        
        >>> # Shared client (recommended)
        >>> shared = httpx.AsyncClient(limits=httpx.Limits(...))
        >>> client = KiroHttpClient(auth_manager, shared_client=shared)
        >>> response = await client.request_with_retry(...)
    """
    
    def __init__(
        self,
        auth_manager: KiroAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None
    ):
        """
        Initializes the HTTP client.
        
        Args:
            auth_manager: Authentication manager
            shared_client: Optional shared httpx.AsyncClient for connection pooling.
                          If provided, this client will be used instead of creating
                          a new one. The shared client will NOT be closed by close().
        """
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client
    
    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        """
        Returns or creates an HTTP client with proper timeouts.
        
        If a shared client was provided at initialization, it is returned as-is.
        Otherwise, creates a new client with appropriate timeout configuration.
        
        httpx timeouts:
        - connect: TCP handshake (DNS + TCP SYN/ACK)
        - read: waiting for data from server between chunks
        - write: sending data to server
        - pool: waiting for free connection from pool
        
        IMPORTANT: FIRST_TOKEN_TIMEOUT is NOT used here!
        It is applied in streaming_openai.py via asyncio.wait_for() to control
        the wait time for the first token from the model (retry business logic).
        
        Args:
            stream: If True, uses STREAMING_READ_TIMEOUT for read (only for new clients)
        
        Returns:
            Active HTTP client
        """
        # If using shared client, return it directly
        # Shared client should be pre-configured with appropriate timeouts
        if self._shared_client is not None:
            return self._shared_client
        
        # Create new client if needed (per-request mode)
        if self.client is None or self.client.is_closed:
            if stream:
                # For streaming:
                # - connect: 30 sec (TCP connection, usually < 1 sec)
                # - read: STREAMING_READ_TIMEOUT (300 sec) - model may "think" between chunks
                # - write/pool: standard values
                timeout_config = httpx.Timeout(
                    connect=30.0,
                    read=STREAMING_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0
                )
                logger.debug(f"Creating streaming HTTP client (read_timeout={STREAMING_READ_TIMEOUT}s)")
            else:
                # For regular requests: single timeout of 300 sec
                timeout_config = httpx.Timeout(timeout=300.0)
                logger.debug("Creating non-streaming HTTP client (timeout=300s)")
            
            self.client = httpx.AsyncClient(
                timeout=timeout_config,
                follow_redirects=True,
                verify=get_shared_ssl_context(),
            )
        return self.client
    
    async def close(self) -> None:
        """
        Closes the HTTP client if this instance owns it.
        
        If using a shared client, this method does nothing - the shared client
        should be closed by the application lifecycle manager.
        
        Uses graceful exception handling to prevent errors during cleanup
        from masking the original exception in finally blocks.
        """
        # Don't close shared clients - they're managed by the application
        if not self._owns_client:
            return
        
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                # Log but don't propagate - we're in cleanup code
                # Propagating here could mask the original exception
                logger.warning(f"Error closing HTTP client: {e}")
    
    async def request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        stream: bool = False
    ) -> httpx.Response:
        """
        Executes an HTTP request with retry logic.
        
        Automatically handles various error types:
        - 403: refreshes token via auth_manager.force_refresh() and retries
        - 429: waits with exponential backoff (1s, 2s, 4s)
        - 5xx: waits with exponential backoff
        - Timeouts: waits with exponential backoff
        
        For streaming, STREAMING_READ_TIMEOUT is used for waiting between chunks.
        First token timeout is controlled separately in streaming_openai.py via asyncio.wait_for().
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            json_data: Optional JSON body (for POST/PUT/PATCH)
            params: Optional query parameters (for GET)
            stream: Use streaming (default False)
        
        Returns:
            httpx.Response with successful response
        
        Raises:
            HTTPException: On failure after all attempts (502/504)
        """
        # Determine the number of retry attempts.
        #
        # Streaming: the HTTP client only opens the initial connection here; the
        # first-token-timeout retry loop lives in the streaming layer, so we keep
        # the streaming budget at FIRST_TOKEN_MAX_RETRIES and do NOT inflate it.
        #
        # Non-streaming: 429/5xx retries (including transient Amazon overload
        # MODEL_TEMPORARILY_UNAVAILABLE) are handled right here, so we grant the
        # larger SERVER_ERROR_MAX_RETRIES budget to let the backend recover before
        # surfacing the error to the client.
        if stream:
            max_retries = FIRST_TOKEN_MAX_RETRIES
        else:
            max_retries = max(MAX_RETRIES, SERVER_ERROR_MAX_RETRIES)
        
        client = await self._get_client(stream=stream)
        last_error = None
        last_error_info: Optional[NetworkErrorInfo] = None
        last_response: Optional[httpx.Response] = None  # Для сохранения последнего 429/5xx
        
        for attempt in range(max_retries):
            try:
                # Get current token
                token = await self.auth_manager.get_access_token()
                headers = get_kiro_headers(self.auth_manager, token)
                
                # Build request kwargs based on parameters
                request_kwargs = {"headers": headers}
                
                if json_data is not None:
                    request_kwargs["content"] = json.dumps(json_data).encode()
                
                if params is not None:
                    request_kwargs["params"] = params
                
                send_started = time.monotonic()
                if stream:
                    # Prevent CLOSE_WAIT connection leak (issue #38)
                    headers["Connection"] = "close"
                    req = client.build_request(method, url, **request_kwargs)
                    logger.debug("Sending request to Kiro API...")
                    response = await client.send(req, stream=True)
                else:
                    logger.debug("Sending request to Kiro API...")
                    response = await client.request(method, url, **request_kwargs)
                
                # Kiro holds the response headers until the model starts
                # generating, so this is the upstream time-to-first-token
                # (plus connection setup), not gateway processing time.
                logger.debug(
                    f"Kiro API responded HTTP {response.status_code} after "
                    f"{time.monotonic() - send_started:.3f}s "
                    f"(attempt {attempt + 1}/{max_retries}, stream={stream})"
                )
                
                # Check status
                if response.status_code == 200:
                    return response
                
                # 403 - token expired, refresh and retry
                if response.status_code == 403:
                    logger.warning(f"Received 403, refreshing token (attempt {attempt + 1}/{MAX_RETRIES})")
                    await self.auth_manager.force_refresh()
                    continue
                
                # 429 - rate limit, wait and retry
                if response.status_code == 429:
                    last_response = response  # Keep for return after exhaustion
                    delay = compute_backoff_delay(attempt)
                    logger.warning(f"Received 429, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                
                # 5xx - server error, wait and retry
                # Transient upstream overload (e.g. MODEL_TEMPORARILY_UNAVAILABLE)
                # usually clears within a few seconds; jittered exponential backoff
                # with a larger budget gives it time to recover before failing.
                if 500 <= response.status_code < 600:
                    last_response = response  # Keep for return after exhaustion
                    delay = compute_backoff_delay(attempt)
                    logger.warning(f"Received {response.status_code}, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                
                # Other errors - return as is
                return response
                
            except httpx.TimeoutException as e:
                last_error = e
                
                # Classify timeout error for user-friendly messaging
                error_info = classify_network_error(e)
                last_error_info = error_info
                
                # Log with user-friendly message
                short_msg = get_short_error_message(error_info)
                
                if error_info.is_retryable and attempt < max_retries - 1:
                    delay = compute_backoff_delay(attempt)
                    logger.warning(f"{short_msg} - waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{short_msg} - no more retries (attempt {attempt + 1}/{max_retries})")
                    if not error_info.is_retryable:
                        break  # Don't retry non-retryable errors
                
            except httpx.RequestError as e:
                last_error = e
                
                # Classify the error for user-friendly messaging
                error_info = classify_network_error(e)
                last_error_info = error_info
                
                # Log with user-friendly message
                short_msg = get_short_error_message(error_info)
                
                if error_info.is_retryable and attempt < max_retries - 1:
                    delay = compute_backoff_delay(attempt)
                    logger.warning(f"{short_msg} - waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{short_msg} - no more retries (attempt {attempt + 1}/{max_retries})")
                    if not error_info.is_retryable:
                        break  # Don't retry non-retryable errors
        
        # If we have a last_response (429/5xx retry exhausted), return it
        # This allows the caller to see the real status code and error body
        if last_response is not None:
            logger.warning(
                f"Retries exhausted for HTTP {last_response.status_code}, "
                f"returning response to caller for classification"
            )
            return last_response
        
        # All attempts exhausted - provide detailed, user-friendly error message
        if last_error_info:
            # Use classified error information. The detail is built via the shared
            # formatter so connection-level failures here and mid-stream body drops
            # (classified in the streaming collectors) produce identical messages.
            raise HTTPException(
                status_code=last_error_info.suggested_http_code,
                detail=build_http_error_detail(last_error_info)
            )
        else:
            # Fallback if no error was captured (shouldn't happen)
            if stream:
                raise HTTPException(
                    status_code=504,
                    detail=f"Streaming failed after {max_retries} attempts. Unknown error."
                )
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Request failed after {max_retries} attempts. Unknown error."
                )
    
    async def __aenter__(self) -> "KiroHttpClient":
        """Async context manager support."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Closes the client when exiting context."""
        await self.close()