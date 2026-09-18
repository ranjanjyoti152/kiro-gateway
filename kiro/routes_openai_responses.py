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
FastAPI route for the OpenAI Responses API.

Exposes ``POST /v1/responses``, the wire protocol used by OpenAI Codex CLI when
configured with ``wire_api = "responses"``. Both streaming and non-streaming
modes are supported.

Error handling mirrors the Chat Completions and Messages routes:

- ``HTTPException`` is passed through unchanged.
- HTTP 502/504 (network-level failures) are treated as recoverable, so the
  account system fails over to the next account.
- ``debug_logger.log_kiro_request_payload`` records the translated payload, and
  buffers are flushed on error / discarded on success.
"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from kiro.account_errors import ErrorType, classify_error
from kiro.converters_openai_responses import (
    EmptyResponsesInputError,
    ToolRegistry,
    UnsupportedResponsesFeatureError,
    build_kiro_payload_responses,
    extract_additional_tools,
)
from kiro.exceptions import MalformedProfileArnError, MissingProfileArnError
from kiro.http_client import KiroHttpClient
from kiro.models_openai_responses import ResponsesRequest
from kiro.profile_resolver import is_valid_profile_arn, resolve_profile_arn
from kiro.routes_openai import verify_api_key
from kiro.streaming_openai_responses import (
    build_failed_event,
    collect_responses_response,
    generate_response_id,
    stream_responses_with_first_token_retry,
)
from kiro.utils import SSE_RESPONSE_HEADERS, generate_conversation_id

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


ENDPOINT_PATH: str = "/v1/responses"

router = APIRouter()


def build_request_echo(request_data: ResponsesRequest) -> Dict[str, Any]:
    """
    Build the request fields echoed back on the response object.

    The Responses API echoes the effective request configuration on every
    lifecycle event and on the final response body. ``store`` and
    ``previous_response_id`` are pinned to ``False``/``None`` because this
    gateway is stateless: reporting them any other way would promise a
    server-side conversation that does not exist.

    Args:
        request_data: The parsed Responses request.

    Returns:
        Dictionary of fields to merge into every response envelope.
    """
    return {
        "instructions": request_data.instructions,
        "metadata": request_data.metadata,
        "tools": (
            [tool.model_dump(exclude_none=True) for tool in request_data.tools]
            if request_data.tools
            else []
        ),
        "tool_choice": request_data.tool_choice if request_data.tool_choice is not None else "auto",
        "parallel_tool_calls": (
            True if request_data.parallel_tool_calls is None else request_data.parallel_tool_calls
        ),
        "max_output_tokens": request_data.max_output_tokens,
        "reasoning": (
            request_data.reasoning.model_dump(exclude_none=True)
            if request_data.reasoning is not None
            else None
        ),
        "store": False,
        "previous_response_id": None,
        "text": (
            request_data.text.model_dump(exclude_none=True)
            if request_data.text is not None
            else None
        ),
        "truncation": request_data.truncation,
        "temperature": request_data.temperature,
        "top_p": request_data.top_p,
        "user": request_data.user,
    }


def build_error_response(status_code: int, message: str, error_type: str) -> JSONResponse:
    """
    Build an OpenAI-shaped error response.

    Args:
        status_code: HTTP status code to return.
        message: Actionable, user-facing error message.
        error_type: OpenAI error type (``invalid_request_error``, ...).

    Returns:
        JSONResponse carrying the error envelope.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": status_code,
            }
        },
    )


def resolve_payload(
    request_data: ResponsesRequest,
    auth_manager: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[ToolRegistry], Optional[List[Dict[str, Any]]],
           Optional[List[Dict[str, Any]]], Optional[JSONResponse]]:
    """
    Resolve the profile ARN and translate the request into a Kiro payload.

    profileArn is required by ``runtime.kiro.dev`` for every auth type, so it is
    resolved through the shared profile resolver and rejected early with an
    actionable error instead of letting Kiro answer with an opaque HTTP 400.

    Args:
        request_data: The parsed Responses request.
        auth_manager: Auth manager of the account being used.

    Returns:
        Tuple of (payload, tool registry, tokenizer messages, tokenizer tools,
        error response). On success the error response is ``None``; on failure
        every other element is ``None``.
    """
    conversation_id = generate_conversation_id()
    profile_arn = resolve_profile_arn(auth_manager)

    try:
        if not profile_arn:
            raise MissingProfileArnError()
        if not is_valid_profile_arn(profile_arn):
            raise MalformedProfileArnError(profile_arn)

        conversion = build_kiro_payload_responses(
            request_data,
            conversation_id,
            profile_arn,
        )
    except (MissingProfileArnError, MalformedProfileArnError) as e:
        logger.warning(f"HTTP 400 - POST {ENDPOINT_PATH} - {e}")
        if debug_logger:
            debug_logger.flush_on_error(400, str(e))
        return None, None, None, None, build_error_response(400, str(e), "invalid_request_error")
    except (UnsupportedResponsesFeatureError, EmptyResponsesInputError) as e:
        logger.warning(f"HTTP 400 - POST {ENDPOINT_PATH} - {e}")
        if debug_logger:
            debug_logger.flush_on_error(400, str(e))
        return None, None, None, None, build_error_response(400, str(e), "invalid_request_error")
    except ValueError as e:
        logger.warning(f"HTTP 400 - POST {ENDPOINT_PATH} - {e}")
        if debug_logger:
            debug_logger.flush_on_error(400, str(e))
        return None, None, None, None, build_error_response(400, str(e), "invalid_request_error")

    return (
        conversion.payload,
        conversion.tool_registry,
        conversion.messages_for_tokenizer,
        conversion.tools_for_tokenizer,
        None,
    )


@router.post(ENDPOINT_PATH, dependencies=[Depends(verify_api_key)])
async def create_response(request: Request, request_data: ResponsesRequest):
    """
    Responses API endpoint — compatible with the OpenAI Responses API.

    Accepts requests in Responses format (flat ``input`` items, flat tool
    specifications) and translates them to Kiro API. Supports streaming (named
    SSE events) and non-streaming (single JSON response object) modes.

    Args:
        request: FastAPI request, used to reach ``app.state``.
        request_data: Request in OpenAI Responses format.

    Returns:
        StreamingResponse for streaming mode, JSONResponse otherwise.

    Raises:
        HTTPException: On authentication, validation or upstream API errors.
    """
    input_item_count = len(request_data.input) if isinstance(request_data.input, list) else (
        1 if request_data.input else 0
    )
    # Codex CLI code mode sends an empty top-level ``tools`` array and declares
    # everything inside an ``additional_tools`` input item, so both sources are
    # counted here; logging only the array made those requests look tool-less.
    additional_tool_count = len(extract_additional_tools(request_data.input, report=False))
    logger.info(
        f"Request to {ENDPOINT_PATH} (model={request_data.model}, "
        f"stream={request_data.stream}, input_items={input_item_count}, "
        f"tools={len(request_data.tools) if request_data.tools else 0}"
        f"+{additional_tool_count} additional)"
    )

    account_manager = request.app.state.account_manager
    account_system: bool = request.app.state.account_system
    request_echo = build_request_echo(request_data)

    if account_system:
        all_accounts = list(account_manager._accounts.keys())
        max_attempts = max(1, len(all_accounts) * 2)
    else:
        all_accounts = []
        max_attempts = 1

    tried_accounts: Set[str] = set()
    last_error_message: Optional[str] = None
    last_error_status: Optional[int] = None

    for _attempt in range(max_attempts):
        # --------------------------------------------------------------
        # Account selection
        # --------------------------------------------------------------
        if account_system:
            account = await account_manager.get_next_account(
                request_data.model, exclude_accounts=tried_accounts
            )
            if account is None:
                if len(all_accounts) == 1:
                    raise HTTPException(
                        status_code=last_error_status or 503,
                        detail=last_error_message or "Account unavailable",
                    )
                detail = "No available accounts for this model."
                if last_error_message:
                    detail += f" Error from last account: {last_error_message}"
                raise HTTPException(status_code=503, detail=detail)
            tried_accounts.add(account.id)
        else:
            account = account_manager.get_first_account()
            if not account.auth_manager:
                logger.error("No initialized accounts available (legacy mode)")
                raise HTTPException(status_code=503, detail="No initialized accounts available")

        auth_manager = account.auth_manager
        model_cache = account.model_cache

        # --------------------------------------------------------------
        # Request translation
        # --------------------------------------------------------------
        (
            kiro_payload,
            tool_registry,
            messages_for_tokenizer,
            tools_for_tokenizer,
            error_response,
        ) = resolve_payload(request_data, auth_manager)

        if error_response is not None:
            return error_response

        if debug_logger:
            debug_logger.log_kiro_request_payload(kiro_payload)

        url = f"{auth_manager.api_host}/generateAssistantResponse"
        logger.debug(f"Kiro API URL: {url} (account: {account.id})")

        # Streaming uses a per-request client to avoid CLOSE_WAIT leaks when the
        # network interface changes; non-streaming reuses the shared pool.
        if request_data.stream:
            http_client = KiroHttpClient(auth_manager, shared_client=None)
        else:
            http_client = KiroHttpClient(
                auth_manager, shared_client=request.app.state.http_client
            )

        response_id = generate_response_id()

        try:
            response = await http_client.request_with_retry(
                "POST", url, kiro_payload, stream=True
            )

            if response.status_code == 200:
                if account_system:
                    await account_manager.report_success(account.id, request_data.model)

                if request_data.stream:
                    async def stream_wrapper():
                        """Stream Responses SSE events and report the outcome."""
                        streaming_error = None
                        client_disconnected = False
                        try:
                            async def make_retry_request():
                                return await http_client.request_with_retry(
                                    "POST", url, kiro_payload, stream=True
                                )

                            async for frame in stream_responses_with_first_token_retry(
                                make_request=make_retry_request,
                                client=http_client.client,
                                model=request_data.model,
                                model_cache=model_cache,
                                auth_manager=auth_manager,
                                tool_registry=tool_registry,
                                response_id=response_id,
                                initial_response=response,
                                request_messages=messages_for_tokenizer,
                                request_tools=tools_for_tokenizer,
                                request_echo=request_echo,
                            ):
                                yield frame
                        except GeneratorExit:
                            client_disconnected = True
                            logger.debug(
                                "Client disconnected during streaming (GeneratorExit in routes)"
                            )
                        except Exception as e:
                            streaming_error = e
                            # Emit a typed terminal failure so the client stops
                            # waiting instead of seeing a truncated stream.
                            try:
                                yield build_failed_event(
                                    response_id=response_id,
                                    model=request_data.model,
                                    created_at=0,
                                    sequence_number=0,
                                    message=str(e) or "Upstream streaming failure",
                                )
                            except (GeneratorExit, RuntimeError):
                                logger.debug(
                                    "Could not deliver response.failed event "
                                    "(client already gone)"
                                )
                            raise
                        finally:
                            await http_client.close()
                            if streaming_error:
                                error_type = type(streaming_error).__name__
                                error_msg = str(streaming_error) or "(empty message)"
                                logger.error(
                                    f"HTTP 500 - POST {ENDPOINT_PATH} (streaming) - "
                                    f"[{error_type}] {error_msg[:100]}"
                                )
                            elif client_disconnected:
                                logger.info(
                                    f"HTTP 200 - POST {ENDPOINT_PATH} (streaming) - "
                                    f"client disconnected"
                                )
                            else:
                                logger.info(
                                    f"HTTP 200 - POST {ENDPOINT_PATH} (streaming) - completed"
                                )
                            if debug_logger:
                                if streaming_error:
                                    debug_logger.flush_on_error(500, str(streaming_error))
                                else:
                                    debug_logger.discard_buffers()

                    return StreamingResponse(
                        stream_wrapper(),
                        media_type="text/event-stream",
                        headers=SSE_RESPONSE_HEADERS,
                    )

                responses_object = await collect_responses_response(
                    client=http_client.client,
                    response=response,
                    model=request_data.model,
                    model_cache=model_cache,
                    auth_manager=auth_manager,
                    tool_registry=tool_registry,
                    response_id=response_id,
                    request_messages=messages_for_tokenizer,
                    request_tools=tools_for_tokenizer,
                    request_echo=request_echo,
                )

                await http_client.close()
                logger.info(f"HTTP 200 - POST {ENDPOINT_PATH} (non-streaming) - completed")

                if debug_logger:
                    debug_logger.discard_buffers()

                return JSONResponse(content=responses_object)

            # --------------------------------------------------------------
            # Upstream returned a non-200 status
            # --------------------------------------------------------------
            try:
                error_content = await response.aread()
            except (httpx.HTTPError, RuntimeError) as read_error:
                logger.debug(f"Could not read Kiro error body: {read_error}")
                error_content = b"Unknown error"

            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")

            error_reason = None
            error_message = error_text
            try:
                error_json = json.loads(error_text)
                from kiro.kiro_errors import enhance_kiro_error

                error_info = enhance_kiro_error(error_json)
                error_message = error_info.user_message
                error_reason = error_info.reason
                logger.debug(
                    f"Original Kiro error: {error_info.original_message} "
                    f"(reason: {error_info.reason})"
                )
            except (json.JSONDecodeError, KeyError):
                pass

            last_error_message = error_message
            last_error_status = response.status_code

            if account_system:
                error_type = classify_error(response.status_code, error_reason)
                await account_manager.report_failure(
                    account.id,
                    request_data.model,
                    error_type,
                    response.status_code,
                    error_reason,
                )
                if error_type == ErrorType.RECOVERABLE and len(all_accounts) > 1:
                    continue

            logger.warning(
                f"HTTP {response.status_code} - POST {ENDPOINT_PATH} - {error_message[:100]}"
            )
            if debug_logger:
                debug_logger.flush_on_error(response.status_code, error_message)

            return build_error_response(response.status_code, error_message, "kiro_api_error")

        except HTTPException as e:
            await http_client.close()

            # 502/504 come from request_with_retry for network-level failures
            # only, never for HTTP-level errors, so they are recoverable.
            if e.status_code in (502, 504):
                last_error_message = str(e.detail)
                last_error_status = e.status_code

                if account_system:
                    await account_manager.report_failure(
                        account.id, request_data.model, ErrorType.RECOVERABLE, e.status_code, None
                    )
                    if len(all_accounts) > 1:
                        logger.warning(
                            f"Network error on account {account.id}, trying next account"
                        )
                        continue
                else:
                    logger.warning("Network error (legacy mode, no failover available)")

            logger.error(f"HTTP {e.status_code} - POST {ENDPOINT_PATH} - {e.detail}")
            if debug_logger:
                debug_logger.flush_on_error(e.status_code, str(e.detail))
            raise
        except Exception as e:
            await http_client.close()
            logger.error(f"Internal error: {e}", exc_info=True)
            logger.error(f"HTTP 500 - POST {ENDPOINT_PATH} - {str(e)[:100]}")
            if debug_logger:
                debug_logger.flush_on_error(500, str(e))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    # All attempts exhausted (account system, multiple accounts)
    if len(all_accounts) <= 1 and last_error_status is not None:
        raise HTTPException(status_code=last_error_status, detail=last_error_message)

    detail = "All accounts failed after full circle."
    if last_error_message:
        detail += f" Error from last account: {last_error_message}"
    raise HTTPException(status_code=503, detail=detail)
