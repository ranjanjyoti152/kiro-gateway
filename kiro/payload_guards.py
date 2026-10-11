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
Payload size guard for Kiro API requests.

The Kiro API rejects payloads exceeding ~615KB with a misleading
"Improperly formed request." (reason: null) error. This module provides:
- Pre-flight size checking
- Auto-trimming of oldest history entries to fit under the limit

Ported from sametakofficial's payload_guards.py, simplified.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PayloadTrimStats:
    """Statistics from a payload trim operation."""
    original_bytes: int
    final_bytes: int
    original_entries: int
    final_entries: int
    trimmed: bool


def check_payload_size(payload: Dict[str, Any]) -> int:
    """Return the serialized byte size of the payload as UTF-8 JSON."""
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _strip_empty_tool_uses(history: list) -> None:
    """Remove empty toolUses arrays in-place (Kiro quirk)."""
    for entry in history:
        assistant = entry.get("assistantResponseMessage")
        if assistant and "toolUses" in assistant and assistant["toolUses"] == []:
            del assistant["toolUses"]


def _align_to_user_message(history: list) -> list:
    """Ensure history starts with a userInputMessage entry."""
    while history and "userInputMessage" not in history[0]:
        history.pop(0)
    return history


def _repair_orphaned_tool_results(history: list) -> None:
    """
    Remove orphaned toolResults that reference toolUseIds not present
    in the preceding assistant message. Preserve orphaned text content
    inline with a marker.
    """
    for i, entry in enumerate(history):
        user_msg = entry.get("userInputMessage")
        if not user_msg:
            continue

        ctx = user_msg.get("userInputMessageContext")
        if not ctx or "toolResults" not in ctx:
            continue

        # Collect toolUseIds from the preceding assistant message
        valid_ids = set()
        if i > 0:
            prev_assistant = history[i - 1].get("assistantResponseMessage")
            if prev_assistant:
                for tu in prev_assistant.get("toolUses", []):
                    tool_use_id = tu.get("toolUseId")
                    if tool_use_id:
                        valid_ids.add(tool_use_id)

        kept = []
        orphaned_text_parts = []
        for tr in ctx["toolResults"]:
            if tr.get("toolUseId") in valid_ids:
                kept.append(tr)
            else:
                # Preserve text content from orphaned results
                content = tr.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            orphaned_text_parts.append(part["text"])
                elif isinstance(content, str) and content:
                    orphaned_text_parts.append(content)

        if len(kept) != len(ctx["toolResults"]):
            if kept:
                ctx["toolResults"] = kept
            else:
                del ctx["toolResults"]
                if not ctx:
                    del user_msg["userInputMessageContext"]

            # Append orphaned text to user message content
            if orphaned_text_parts:
                marker = "\n[trimmed tool result] " + "; ".join(orphaned_text_parts)
                current_content = user_msg.get("content", "")
                user_msg["content"] = current_content + marker


def _first_user_message(history: list) -> Optional[Dict[str, Any]]:
    """Return the first ``userInputMessage`` dict in history, or None."""
    for entry in history:
        user_msg = entry.get("userInputMessage")
        if user_msg is not None:
            return user_msg
    return None


def _strip_system_prompt(history: list, system_prompt: str) -> None:
    """
    Remove a leading system-prompt block from the first user message in-place.

    ``build_kiro_payload`` prepends the system prompt to the first history user
    message as ``f"{system_prompt}\\n\\n{original}"``. Trimming removes oldest
    entries from the front, so without this the system prompt would be deleted.
    We strip it here so it can be re-attached to whatever user message survives
    trimming (see ``_reattach_system_prompt``), guaranteeing the model never
    loses its operating instructions.

    Args:
        history: Kiro history list (mutated in place).
        system_prompt: The exact system prompt string that was prepended.
    """
    first_user = _first_user_message(history)
    if first_user is None:
        return
    content = first_user.get("content", "")
    prefix = f"{system_prompt}\n\n"
    if isinstance(content, str) and content.startswith(prefix):
        first_user["content"] = content[len(prefix):]


def _reattach_system_prompt(history: list, system_prompt: str) -> None:
    """
    Prepend the system prompt to the first user message in-place (idempotent).

    Safe to call even if the prompt is already present: it checks first to
    avoid duplicating the block.

    Args:
        history: Kiro history list (mutated in place).
        system_prompt: The system prompt string to guarantee at the front.
    """
    first_user = _first_user_message(history)
    if first_user is None:
        return
    content = first_user.get("content", "")
    if not isinstance(content, str):
        content = ""
    prefix = f"{system_prompt}\n\n"
    if content.startswith(prefix):
        return  # already present
    first_user["content"] = f"{prefix}{content}"


def trim_payload_to_limit(
    payload: Dict[str, Any],
    max_bytes: int,
    system_prompt: Optional[str] = None,
) -> PayloadTrimStats:
    """
    Trim oldest history entries so the serialized payload fits under max_bytes.

    Trims in user/assistant pairs (2 entries at a time), aligns start to
    userInputMessage, and repairs orphaned toolResults after trimming.

    System prompt preservation (fixes context loss in Claude Code): the caller
    prepends the system prompt to the first history user message. Because
    trimming removes entries from the front, that message (and the system
    prompt with it) can be deleted, which strips the model's instructions and
    makes it "lose context completely". When ``system_prompt`` is provided we
    detach it before trimming and re-attach it to whichever user message
    survives, so the instructions are always present regardless of how much
    history is removed.

    Args:
        payload: Full Kiro payload (mutated in place).
        max_bytes: Target maximum serialized size in bytes.
        system_prompt: System prompt prepended to the first user message, if
            any. When provided, it is preserved across trimming.

    Returns:
        PayloadTrimStats describing the trim operation.
    """
    original_bytes = check_payload_size(payload)
    history = payload.get("conversationState", {}).get("history")

    if not history:
        return PayloadTrimStats(
            original_bytes=original_bytes,
            final_bytes=original_bytes,
            original_entries=0,
            final_entries=0,
            trimmed=False,
        )

    original_entries = len(history)

    # Detach the system prompt so it is not deleted when the first entries are
    # trimmed; it will be re-attached to the surviving first user message.
    if system_prompt:
        _strip_system_prompt(history, system_prompt)

    # Strip empty toolUses before measuring
    _strip_empty_tool_uses(history)

    # Trim pairs from the beginning until under limit (keep at least 2 entries).
    # When a system prompt will be re-attached, account for its size in the
    # measurement so re-attaching it afterwards does not push us back over.
    system_prompt_overhead = len(f"{system_prompt}\n\n".encode("utf-8")) if system_prompt else 0
    while len(history) > 2 and (check_payload_size(payload) + system_prompt_overhead) > max_bytes:
        # Remove 2 entries (a user/assistant pair)
        history.pop(0)
        history.pop(0)

    # Align to userInputMessage boundary
    _align_to_user_message(history)

    # Repair orphaned tool results after trimming
    _repair_orphaned_tool_results(history)

    # Re-attach the system prompt to the surviving first user message so the
    # model always keeps its operating instructions.
    if system_prompt:
        _reattach_system_prompt(history, system_prompt)

    final_bytes = check_payload_size(payload)
    return PayloadTrimStats(
        original_bytes=original_bytes,
        final_bytes=final_bytes,
        original_entries=original_entries,
        final_entries=len(history),
        trimmed=original_entries != len(history),
    )
