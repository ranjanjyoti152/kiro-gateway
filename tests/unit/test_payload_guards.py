# -*- coding: utf-8 -*-

"""
Unit tests for payload size guard logic.
Tests check_payload_size() and trim_payload_to_limit() functions.
"""

import json

import pytest

from kiro.payload_guards import (
    PayloadTrimStats,
    check_payload_size,
    trim_payload_to_limit,
)


def _make_payload(num_pairs=5, content_size=100):
    """Helper: build a minimal Kiro-shaped payload with N user/assistant pairs."""
    history = []
    for i in range(num_pairs):
        history.append({
            "userInputMessage": {"content": f"user message {i} " + "x" * content_size}
        })
        history.append({
            "assistantResponseMessage": {"content": f"assistant message {i} " + "y" * content_size}
        })
    return {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": "test-conv",
            "currentMessage": {
                "userInputMessage": {"content": "current message", "modelId": "test"}
            },
            "history": history,
        },
        "profileArn": "arn:aws:test",
    }


class TestCheckPayloadSize:

    def test_check_payload_size_returns_bytes(self):
        """Correct byte count for a simple payload."""
        payload = {"key": "value"}
        expected = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        assert check_payload_size(payload) == expected

    def test_check_payload_size_utf8(self):
        """Non-ASCII characters counted correctly."""
        payload = {"emoji": "\U0001f600", "chinese": "\u4f60\u597d"}
        size = check_payload_size(payload)
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        assert size == len(raw)
        # json.dumps escapes non-ASCII by default, so byte count matches
        # The important thing is consistency with what gets sent over the wire
        assert size > 0


class TestTrimPayloadToLimit:

    def test_trim_does_nothing_when_under_limit(self):
        """No-op when payload is small."""
        payload = _make_payload(num_pairs=2, content_size=10)
        original_size = check_payload_size(payload)
        stats = trim_payload_to_limit(payload, max_bytes=original_size + 1000)
        assert not stats.trimmed
        assert stats.original_entries == stats.final_entries
        assert stats.original_bytes == stats.final_bytes

    def test_trim_removes_oldest_history_pairs(self):
        """Removes pairs from the beginning of history."""
        payload = _make_payload(num_pairs=10, content_size=500)
        original_size = check_payload_size(payload)
        # Set limit to ~half the original size
        limit = original_size // 2
        stats = trim_payload_to_limit(payload, max_bytes=limit)

        assert stats.trimmed
        assert stats.final_entries < stats.original_entries
        assert stats.final_bytes <= limit
        # History should still exist and be shorter
        history = payload["conversationState"]["history"]
        assert len(history) == stats.final_entries
        assert len(history) < 20  # original was 20 entries (10 pairs)

    def test_trim_aligns_to_user_message(self):
        """Start index always lands on userInputMessage entry."""
        payload = _make_payload(num_pairs=10, content_size=500)
        limit = check_payload_size(payload) // 3
        trim_payload_to_limit(payload, max_bytes=limit)

        history = payload["conversationState"]["history"]
        assert len(history) > 0
        assert "userInputMessage" in history[0]

    def test_trim_preserves_minimum_history(self):
        """Never trims below 2 entries."""
        payload = _make_payload(num_pairs=5, content_size=1000)
        # Set an impossibly low limit
        stats = trim_payload_to_limit(payload, max_bytes=100)

        history = payload["conversationState"]["history"]
        assert len(history) >= 2
        assert stats.final_entries >= 2

    def test_trim_repairs_orphaned_tool_results(self):
        """Orphaned toolResults removed, text preserved inline."""
        history = [
            {"userInputMessage": {"content": "msg0"}},
            {
                "assistantResponseMessage": {
                    "content": "resp0",
                    "toolUses": [{"toolUseId": "tool-A", "name": "read", "input": "{}"}],
                }
            },
            {
                "userInputMessage": {
                    "content": "msg1",
                    "userInputMessageContext": {
                        "toolResults": [
                            {
                                "toolUseId": "tool-A",
                                "content": [{"text": "result from tool-A"}],
                            },
                            {
                                "toolUseId": "tool-ORPHAN",
                                "content": [{"text": "orphaned data"}],
                            },
                        ]
                    },
                }
            },
            {"assistantResponseMessage": {"content": "resp1"}},
        ]
        payload = {
            "conversationState": {
                "conversationId": "test",
                "chatTriggerType": "MANUAL",
                "currentMessage": {"userInputMessage": {"content": "now", "modelId": "m"}},
                "history": history,
            }
        }
        # Trim with a generous limit so only repair logic runs (no pair removal)
        big_limit = check_payload_size(payload) + 10000
        trim_payload_to_limit(payload, max_bytes=big_limit)

        # tool-A should remain, tool-ORPHAN should be removed with text preserved
        ctx = history[2]["userInputMessage"]["userInputMessageContext"]
        assert len(ctx["toolResults"]) == 1
        assert ctx["toolResults"][0]["toolUseId"] == "tool-A"
        # Orphaned text preserved in content
        assert "orphaned data" in history[2]["userInputMessage"]["content"]
        assert "[trimmed tool result]" in history[2]["userInputMessage"]["content"]

    def test_trim_strips_empty_tool_uses(self):
        """Empty toolUses: [] arrays cleaned before size measurement."""
        history = [
            {"userInputMessage": {"content": "msg"}},
            {"assistantResponseMessage": {"content": "resp", "toolUses": []}},
        ]
        payload = {
            "conversationState": {
                "conversationId": "test",
                "chatTriggerType": "MANUAL",
                "currentMessage": {"userInputMessage": {"content": "now", "modelId": "m"}},
                "history": history,
            }
        }
        big_limit = check_payload_size(payload) + 10000
        trim_payload_to_limit(payload, max_bytes=big_limit)

        # Empty toolUses should be stripped
        assert "toolUses" not in history[1]["assistantResponseMessage"]

    def test_trim_stats_accurate(self):
        """Stats reflect actual changes."""
        payload = _make_payload(num_pairs=8, content_size=500)
        original_size = check_payload_size(payload)
        limit = original_size // 2

        stats = trim_payload_to_limit(payload, max_bytes=limit)

        assert stats.original_bytes == original_size
        assert stats.original_entries == 16  # 8 pairs * 2
        assert stats.final_bytes == check_payload_size(payload)
        assert stats.final_entries == len(payload["conversationState"]["history"])
        assert stats.trimmed is True

    def test_trim_no_history(self):
        """Payload with no history returns no-op stats."""
        payload = {
            "conversationState": {
                "conversationId": "test",
                "chatTriggerType": "MANUAL",
                "currentMessage": {"userInputMessage": {"content": "hi", "modelId": "m"}},
            }
        }
        stats = trim_payload_to_limit(payload, max_bytes=100)
        assert not stats.trimmed
        assert stats.original_entries == 0
        assert stats.final_entries == 0


SYSTEM_PROMPT = "SYSTEM_PROMPT_MARKER important operating rules for the model"


def _make_payload_with_system_prompt(num_pairs=5, content_size=100, system_prompt=SYSTEM_PROMPT):
    """
    Build a payload where the system prompt is prepended to the first history
    user message, exactly as build_kiro_payload does.
    """
    payload = _make_payload(num_pairs=num_pairs, content_size=content_size)
    history = payload["conversationState"]["history"]
    first_user = history[0]["userInputMessage"]
    first_user["content"] = f"{system_prompt}\n\n{first_user['content']}"
    return payload


class TestTrimPreservesSystemPrompt:
    """
    Regression tests for the Claude Code context-loss bug: trimming must never
    delete the system prompt that was prepended to the first history message.
    """

    def test_system_prompt_survives_heavy_trim(self):
        """System prompt is retained even when most history is removed."""
        payload = _make_payload_with_system_prompt(num_pairs=40, content_size=2000)
        original_size = check_payload_size(payload)
        limit = original_size // 4

        stats = trim_payload_to_limit(payload, max_bytes=limit, system_prompt=SYSTEM_PROMPT)

        assert stats.trimmed
        assert stats.final_entries < stats.original_entries
        blob = json.dumps(payload)
        assert SYSTEM_PROMPT in blob, "System prompt must survive trimming"
        # Must be on the first surviving user message
        history = payload["conversationState"]["history"]
        assert history[0]["userInputMessage"]["content"].startswith(SYSTEM_PROMPT)

    def test_system_prompt_not_duplicated_when_no_trim(self):
        """When nothing is trimmed, the prompt is not duplicated."""
        payload = _make_payload_with_system_prompt(num_pairs=2, content_size=10)
        original_size = check_payload_size(payload)

        trim_payload_to_limit(payload, max_bytes=original_size + 1000, system_prompt=SYSTEM_PROMPT)

        content = payload["conversationState"]["history"][0]["userInputMessage"]["content"]
        assert content.count(SYSTEM_PROMPT) == 1

    def test_system_prompt_not_duplicated_after_trim(self):
        """After trimming, the prompt appears exactly once (strip + reattach)."""
        payload = _make_payload_with_system_prompt(num_pairs=30, content_size=1500)
        limit = check_payload_size(payload) // 3

        trim_payload_to_limit(payload, max_bytes=limit, system_prompt=SYSTEM_PROMPT)

        blob = json.dumps(payload)
        # The marker token appears once in the prompt; ensure it is not doubled.
        assert blob.count("SYSTEM_PROMPT_MARKER") == 1

    def test_system_prompt_present_even_at_minimum_history(self):
        """With an impossibly low limit (floor of 2 entries), prompt still survives."""
        payload = _make_payload_with_system_prompt(num_pairs=10, content_size=1000)

        trim_payload_to_limit(payload, max_bytes=100, system_prompt=SYSTEM_PROMPT)

        history = payload["conversationState"]["history"]
        assert len(history) >= 2
        assert history[0]["userInputMessage"]["content"].startswith(SYSTEM_PROMPT)

    def test_no_system_prompt_arg_is_backward_compatible(self):
        """Omitting system_prompt preserves the original (lossy) behavior."""
        payload = _make_payload_with_system_prompt(num_pairs=20, content_size=1500)
        limit = check_payload_size(payload) // 4

        # Without passing system_prompt, the old behavior applies: the first
        # entries (including the prompt) may be trimmed away.
        stats = trim_payload_to_limit(payload, max_bytes=limit)
        assert stats.trimmed
        # We don't assert the prompt is gone (depends on sizes), only that the
        # call succeeds with the default argument - backward compatibility.
        assert stats.final_entries >= 2

    def test_fits_under_limit_with_prompt_reattached(self):
        """Re-attaching the prompt must not push the payload back over the limit."""
        payload = _make_payload_with_system_prompt(num_pairs=50, content_size=2000)
        limit = check_payload_size(payload) // 5

        trim_payload_to_limit(payload, max_bytes=limit, system_prompt=SYSTEM_PROMPT)

        # Final size (prompt already re-attached) should respect the limit,
        # unless we hit the 2-entry floor. Verify the floor case explicitly.
        history = payload["conversationState"]["history"]
        final_size = check_payload_size(payload)
        assert final_size <= limit or len(history) <= 2
