# -*- coding: utf-8 -*-
"""
Real request payloads captured from OpenAI Codex CLI 0.153.4.

Captured by pointing Codex CLI at a local capture server with::

    [model_providers.kiro]
    base_url = "http://127.0.0.1:PORT/v1"
    wire_api = "responses"

Only long free-text values (``instructions``, prompt text, tool descriptions)
were truncated so the fixture stays readable. Every field name, item shape and
tool shape is byte-for-byte what Codex sent.

- ``CODEX_RESPONSES_REQUEST``: first turn of a session (plain prompt).
- ``CODEX_RESPONSES_FOLLOW_UP_REQUEST``: the turn Codex sends after running a
  tool, carrying the ``function_call`` / ``function_call_output`` round trip.

Both of the above were captured with ``model = "claude-sonnet-4.5"``, for which
Codex has no model metadata and therefore falls back to its default tool
delivery: nine entries in the top-level ``tools`` array, including a
``namespace`` container and a built-in ``web_search`` tool.

The two payloads below were captured with ``model = "gpt-5.6-terra"``, a slug
Codex ships metadata for. That metadata enables **code mode**, which changes
tool delivery completely:

- the top-level ``tools`` key is omitted entirely (Codex only serializes it when
  non-empty, so the gateway sees ``tools=None``);
- every tool is declared inside an ``additional_tools`` input item, grouped into
  ``namespace`` containers (``functions`` and ``collaboration``);
- the primary tool ``exec`` is a freeform ``{"type": "custom"}`` tool with a lark
  grammar instead of a JSON schema, and the model calls it with raw JavaScript;
- tool round-trips therefore use ``custom_tool_call`` / ``custom_tool_call_output``
  items rather than ``function_call`` / ``function_call_output``.

- ``CODEX_RESPONSES_CODE_MODE_REQUEST``: first turn of a code-mode session.
- ``CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST``: the turn Codex sends after
  running the freeform ``exec`` tool, carrying the ``custom_tool_call`` /
  ``custom_tool_call_output`` round trip.
"""

from typing import Any, Dict


CODEX_RESPONSES_REQUEST: Dict[str, Any] = {
    "model": "claude-sonnet-4.5",
    "instructions": "You are a coding agent running in the Codex CLI, a terminal-based coding assistant. Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.\n\nYour capabilities:\n\n- Receive user prompts and other context provided by the harness, such as files in the workspace.\n- Communicate with the user by streaming thinking & responses, and by making & updating plans.\n- Emit function calls to run terminal commands and apply patches. Depending on how this specific run is configured, you can request that these function calls be escalated to the user for approval befo\n[... truncated for the test fixture ...]",
    "input": [
        {
            "type": "message",
            "id": "msg_01a0752b-3b0d-78b0-8b20-87affc55e604",
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "<skills_instructions>\n## Skills\nA skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and a short path that can be expanded into an absolute path using the skill roots table.\n### Sk\n[... truncated for the test fixture ...]"
                },
                {
                    "type": "input_text",
                    "text": "<permissions instructions>\nFilesystem sandboxing defines which files can be read or written. `sandbox_mode` is `danger-full-access`: No filesystem sandboxing - all commands are permitted. Network access is enabled.\nApproval policy is currently never. Do not provide the `sandbox_permissions` for any \n[... truncated for the test fixture ...]"
                }
            ]
        },
        {
            "type": "message",
            "id": "msg_01a0752b-3b0d-78b0-8b20-87bde4a194a2",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<environment_context>\n  <cwd>/tmp/codex_capture/work</cwd>\n  <shell>bash</shell>\n  <current_date>2026-09-06</current_date>\n  <timezone>Asia/Kolkata</timezone>\n  <filesystem><workspace_roots><root>/tmp/codex_capture/work</root></workspace_roots><permission_profile type=\"disabled\"><file_system type=\"u\n[... truncated for the test fixture ...]"
                }
            ]
        },
        {
            "type": "message",
            "id": "msg_01a0752b-3b18-7671-956b-6f4b0f084527",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Say hello in exactly three words."
                }
            ]
        }
    ],
    "tools": [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute."
                    },
                    "justification": {
                        "type": "string",
                        "description": "User-facing approval question for `require_escalated`; omit otherwise."
                    },
                    "login": {
                        "type": "boolean",
                        "description": "True runs the shell with -l/-i semantics; false disables them. Defaults to true."
                    },
                    "max_output_tokens": {
                        "type": "number",
                        "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
                    },
                    "prefix_rule": {
                        "type": "array",
                        "description": "Reusable approval prefix for `cmd`, only with `sandbox_permissions: \"require_escalated\"`; for example [\"git\", \"pull\"].",
                        "items": {
                            "type": "string"
                        }
                    },
                    "sandbox_permissions": {
                        "type": "string",
                        "description": "Per-command sandbox override. Defaults to `use_default`; use `require_escalated` for unsandboxed execution.",
                        "enum": [
                            "use_default",
                            "require_escalated"
                        ]
                    },
                    "shell": {
                        "type": "string",
                        "description": "Shell binary to launch. Defaults to the user's default shell."
                    },
                    "tty": {
                        "type": "boolean",
                        "description": "True allocates a PTY for the command; false or omitted uses plain pipes."
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory for the command. Defaults to the turn cwd."
                    },
                    "yield_time_ms": {
                        "type": "number",
                        "description": "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms."
                    }
                },
                "required": [
                    "cmd"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "write_stdin",
            "description": "Writes characters to an existing unified exec session and returns recent output.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "chars": {
                        "type": "string",
                        "description": "Bytes to write to stdin. Defaults to empty, which polls without writing."
                    },
                    "max_output_tokens": {
                        "type": "number",
                        "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
                    },
                    "session_id": {
                        "type": "number",
                        "description": "Identifier of the running unified exec session."
                    },
                    "yield_time_ms": {
                        "type": "number",
                        "description": "Wait before yielding output. Non-empty writes default to 250 ms and cap at 30000 ms; empty polls wait 5000-300000 ms by default."
                    }
                },
                "required": [
                    "session_id"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "request_user_input",
            "description": "Request user input for one to three short questions and wait for the response. This tool is only available in Plan mode.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "Questions to show the user. Prefer 1 and do not exceed 3",
                        "items": {
                            "type": "object",
                            "properties": {
                                "header": {
                                    "type": "string",
                                    "description": "Short header label shown in the UI (12 or fewer chars)."
                                },
                                "id": {
                                    "type": "string",
                                    "description": "Stable identifier for mapping answers (snake_case)."
                                },
                                "options": {
                                    "type": "array",
                                    "description": "Provide 2-3 mutually exclusive choices. Put the recommended option first and suffix its label with \"(Recommended)\". Do not include an \"Other\" option in this list; the client will add a free-form \"Other\" option automatically.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "description": {
                                                "type": "string",
                                                "description": "One short sentence explaining impact/tradeoff if selected."
                                            },
                                            "label": {
                                                "type": "string",
                                                "description": "User-facing label (1-5 words)."
                                            }
                                        },
                                        "required": [
                                            "label",
                                            "description"
                                        ],
                                        "additionalProperties": False
                                    }
                                },
                                "question": {
                                    "type": "string",
                                    "description": "Single-sentence prompt shown to the user."
                                }
                            },
                            "required": [
                                "id",
                                "header",
                                "question",
                                "options"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                "required": [
                    "questions"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "view_image",
            "description": "View a local image file from the filesystem when visual inspection is needed. Use this for images already available on disk.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local filesystem path to an image file."
                    }
                },
                "required": [
                    "path"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Tools for spawning and managing sub-agents.",
            "tools": [
                {
                    "type": "function",
                    "name": "close_agent",
                    "description": "Close an agent and any open descendants when they are no longer needed, and return the target agent's previous status be\n[... truncated for the test fixture ...]",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Agent id to close (from spawn_agent)."
                            }
                        },
                        "required": [
                            "target"
                        ],
                        "additionalProperties": False
                    }
                },
                {
                    "type": "function",
                    "name": "resume_agent",
                    "description": "Resume a previously closed agent by id so it can receive send_input and wait_agent calls.",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Agent id to resume."
                            }
                        },
                        "required": [
                            "id"
                        ],
                        "additionalProperties": False
                    }
                },
                {
                    "type": "function",
                    "name": "send_input",
                    "description": "Send a message to an existing agent. Use interrupt=true to redirect work immediately. You should reuse the agent by send\n[... truncated for the test fixture ...]",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "interrupt": {
                                "type": "boolean",
                                "description": "True interrupts the current task and handles this message immediately; false or omitted queues it."
                            },
                            "items": {
                                "type": "array",
                                "description": "Structured input items. Use this to pass explicit mentions (for example app:// connector paths).",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "audio_url": {
                                            "type": "string",
                                            "description": "Audio data URL when type is audio."
                                        },
                                        "image_url": {
                                            "type": "string",
                                            "description": "Image URL when type is image."
                                        },
                                        "name": {
                                            "type": "string",
                                            "description": "Display name when type is skill or mention."
                                        },
                                        "path": {
                                            "type": "string",
                                            "description": "Path when type is local_image/local_audio/skill, or structured mention target such as app://<connector-id> or plugin://<plugin-name>@<marketplace-name> when type is mention."
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": "Text content when type is text."
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "Input item type: text, image, local_image, audio, local_audio, skill, or mention."
                                        }
                                    },
                                    "additionalProperties": False
                                }
                            },
                            "message": {
                                "type": "string",
                                "description": "Legacy plain-text message to send to the agent. Use either message or items."
                            },
                            "target": {
                                "type": "string",
                                "description": "Agent id to message (from spawn_agent)."
                            }
                        },
                        "required": [
                            "target"
                        ],
                        "additionalProperties": False
                    }
                },
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "\n        \n        Available model overrides (optional; inherited parent model is preferred):\n- `gpt-6-astra`: Our most c\n[... truncated for the test fixture ...]",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fork_context": {
                                "type": "boolean",
                                "description": "True forks the current thread history into the new agent; false or omitted starts with only the initial prompt."
                            },
                            "items": {
                                "type": "array",
                                "description": "Structured input items. Use this to pass explicit mentions (for example app:// connector paths).",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "audio_url": {
                                            "type": "string",
                                            "description": "Audio data URL when type is audio."
                                        },
                                        "image_url": {
                                            "type": "string",
                                            "description": "Image URL when type is image."
                                        },
                                        "name": {
                                            "type": "string",
                                            "description": "Display name when type is skill or mention."
                                        },
                                        "path": {
                                            "type": "string",
                                            "description": "Path when type is local_image/local_audio/skill, or structured mention target such as app://<connector-id> or plugin://<plugin-name>@<marketplace-name> when type is mention."
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": "Text content when type is text."
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "Input item type: text, image, local_image, audio, local_audio, skill, or mention."
                                        }
                                    },
                                    "additionalProperties": False
                                }
                            },
                            "message": {
                                "type": "string",
                                "description": "Initial plain-text task for the new agent. Use either message or items."
                            },
                            "model": {
                                "type": "string",
                                "description": "Model override for the new agent. Omit unless an explicit override is needed."
                            },
                            "reasoning_effort": {
                                "type": "string",
                                "description": "Reasoning effort override for the new agent. Omit to inherit the parent effort."
                            }
                        },
                        "additionalProperties": False
                    }
                },
                {
                    "type": "function",
                    "name": "wait_agent",
                    "description": "Wait for agents to reach a final status. Completed statuses may include the agent's final message. Returns empty status \n[... truncated for the test fixture ...]",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "targets": {
                                "type": "array",
                                "description": "Agent ids to wait on. Pass multiple ids to wait for whichever finishes first.",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "timeout_ms": {
                                "type": "number",
                                "description": "Timeout in milliseconds. Defaults to 30000, min 10000, max 3600000. Prefer longer waits (minutes) to avoid busy polling."
                            }
                        },
                        "required": [
                            "targets"
                        ],
                        "additionalProperties": False
                    }
                }
            ]
        },
        {
            "type": "function",
            "name": "get_goal",
            "description": "Get the current goal for this thread, including status, budgets, token and elapsed-time usage, and remaining token budget.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "create_goal",
            "description": "Create a goal only when explicitly requested by the user or system/developer instructions; do not infer goals from ordinary tasks.\nSet token_budget only when an\n[... truncated for the test fixture ...]",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Required. The concrete objective to start pursuing. This starts a new active goal when no goal exists or replaces the current goal when it is complete."
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Positive token budget for the new goal. Omit unless explicitly requested."
                    }
                },
                "required": [
                    "objective"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "update_goal",
            "description": "Update the existing goal.\nUse this tool only to mark the goal achieved or genuinely blocked.\nSet status to `complete` only when the objective has actually been \n[... truncated for the test fixture ...]",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Required. Set to `complete` only when the objective is achieved and no required work remains. Set to `blocked` only after the same blocking condition has recurred for at least three consecutive goal turns and the agent is at an impasse. After a previously blocked goal is resumed, the resumed run starts a fresh blocked audit.",
                        "enum": [
                            "complete",
                            "blocked"
                        ]
                    }
                },
                "required": [
                    "status"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "web_search",
            "external_web_access": True
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "reasoning": {
        "summary": "auto"
    },
    "store": False,
    "stream": True,
    "include": [
        "reasoning.encrypted_content"
    ],
    "prompt_cache_key": "01a0752b-3af5-7990-bded-96bee5a5db6d",
    "client_metadata": {
        "x-codex-installation-id": "c5a00c2d-d76b-40c4-a59f-b42c5834124c",
        "session_id": "01a0752b-3af5-7990-bded-96bee5a5db6d",
        "x-codex-turn-metadata": "{\"installation_id\":\"c5a00c2d-d76b-40c4-a59f-b42c5834124c\",\"session_id\":\"01a0752b-3af5-7990-bded-96bee5a5db6d\",\"thread_id\":\"01a0752b-3af5-7990-bded-96bee5a5db6d\",\"agent_name\":\"/root\",\"turn_id\":\"01a0752b-3afb-7551-9772-d1c497b3745e\",\"window_id\":\"01a0752b-3af5-7990-bded-96bee5a5db6d:0\",\"window_number\":0,\"context_window_id\":\"01a0752b-3af5-7990-bded-96c62f9dde70\",\"request_kind\":\"turn\",\"root_turn_id\":\"01a0752b-3afb-7551-9772-d1c497b3745e\",\"thread_source\":\"user\",\"sandbox\":\"none\",\"sandbox_mode\":\"danger-full-access\",\"auto_review_enabled\":false,\"node_repl_auto_review_required\":false,\"node_repl_disabled\":false,\"turn_started_at_unix_ms\":1788672162556}",
        "turn_id": "01a0752b-3afb-7551-9772-d1c497b3745e",
        "thread_id": "01a0752b-3af5-7990-bded-96bee5a5db6d",
        "x-codex-window-id": "01a0752b-3af5-7990-bded-96bee5a5db6d:0",
        "root_turn_id": "01a0752b-3afb-7551-9772-d1c497b3745e"
    }
}


CODEX_RESPONSES_FOLLOW_UP_REQUEST: Dict[str, Any] = {
    "model": "claude-sonnet-4.5",
    "instructions": "You are a coding agent running in the Codex CLI, a terminal-based coding assistant. Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.\n\nYour capabilities:\n\n- Receive user prompts and other context provided by the harness, such as files in the workspace.\n- Communicate with the user by streaming thinking & responses, and by making & updating plans.\n- Emit function calls to run terminal commands and apply patches. Depending on how this specific run is configured, you can request that these function calls be escalated to the user for approval befo\n[... truncated for the test fixture ...]",
    "input": [
        {
            "type": "message",
            "id": "msg_01a07546-3232-76f1-b43b-7931b5340c43",
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "<skills_instructions>\n## Skills\nA skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and a short path that can be expanded into an absolute path using the skill roots table.\n### Sk\n[... truncated for the test fixture ...]"
                },
                {
                    "type": "input_text",
                    "text": "<permissions instructions>\nFilesystem sandboxing defines which files can be read or written. `sandbox_mode` is `danger-full-access`: No filesystem sandboxing - all commands are permitted. Network access is enabled.\nApproval policy is currently never. Do not provide the `sandbox_permissions` for any \n[... truncated for the test fixture ...]"
                }
            ]
        },
        {
            "type": "message",
            "id": "msg_01a07546-3232-76f1-b43b-794017bf5526",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<environment_context>\n  <cwd>/tmp/codex_capture/work</cwd>\n  <shell>bash</shell>\n  <current_date>2026-09-06</current_date>\n  <timezone>Asia/Kolkata</timezone>\n  <filesystem><workspace_roots><root>/tmp/codex_capture/work</root></workspace_roots><permission_profile type=\"disabled\"><file_system type=\"u\n[... truncated for the test fixture ...]"
                }
            ]
        },
        {
            "type": "message",
            "id": "msg_01a07546-323c-70a1-9cf4-458400c4522b",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Count the lines in data.txt with a shell command and report the number."
                }
            ]
        },
        {
            "type": "message",
            "id": "msg_a834b51581cf4384a17b91500c517079",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "I'll count the lines in data.txt for you."
                }
            ]
        },
        {
            "type": "function_call",
            "id": "fc_2363cd47d35e4c2093643883021f1795",
            "name": "exec_command",
            "arguments": "{\"cmd\": \"wc -l data.txt\"}",
            "call_id": "tooluse_GZgUDAkyT8fiV28FmI12oy"
        },
        {
            "type": "function_call_output",
            "id": "fco_01a07546-41d3-72b2-aeb0-94e8006fc80c",
            "call_id": "tooluse_GZgUDAkyT8fiV28FmI12oy",
            "output": "Chunk ID: 8c5717\nWall time: 0.0000 seconds\nProcess exited with code 0\nOriginal token count: 3\nOutput:\n2 data.txt\n"
        }
    ],
    "tools": [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute."
                    },
                    "justification": {
                        "type": "string",
                        "description": "User-facing approval question for `require_escalated`; omit otherwise."
                    },
                    "login": {
                        "type": "boolean",
                        "description": "True runs the shell with -l/-i semantics; false disables them. Defaults to true."
                    },
                    "max_output_tokens": {
                        "type": "number",
                        "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
                    },
                    "prefix_rule": {
                        "type": "array",
                        "description": "Reusable approval prefix for `cmd`, only with `sandbox_permissions: \"require_escalated\"`; for example [\"git\", \"pull\"].",
                        "items": {
                            "type": "string"
                        }
                    },
                    "sandbox_permissions": {
                        "type": "string",
                        "description": "Per-command sandbox override. Defaults to `use_default`; use `require_escalated` for unsandboxed execution.",
                        "enum": [
                            "use_default",
                            "require_escalated"
                        ]
                    },
                    "shell": {
                        "type": "string",
                        "description": "Shell binary to launch. Defaults to the user's default shell."
                    },
                    "tty": {
                        "type": "boolean",
                        "description": "True allocates a PTY for the command; false or omitted uses plain pipes."
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory for the command. Defaults to the turn cwd."
                    },
                    "yield_time_ms": {
                        "type": "number",
                        "description": "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms."
                    }
                },
                "required": [
                    "cmd"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "write_stdin",
            "description": "Writes characters to an existing unified exec session and returns recent output.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "chars": {
                        "type": "string",
                        "description": "Bytes to write to stdin. Defaults to empty, which polls without writing."
                    },
                    "max_output_tokens": {
                        "type": "number",
                        "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
                    },
                    "session_id": {
                        "type": "number",
                        "description": "Identifier of the running unified exec session."
                    },
                    "yield_time_ms": {
                        "type": "number",
                        "description": "Wait before yielding output. Non-empty writes default to 250 ms and cap at 30000 ms; empty polls wait 5000-300000 ms by default."
                    }
                },
                "required": [
                    "session_id"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "request_user_input",
            "description": "Request user input for one to three short questions and wait for the response. This tool is only available in Plan mode.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "Questions to show the user. Prefer 1 and do not exceed 3",
                        "items": {
                            "type": "object",
                            "properties": {
                                "header": {
                                    "type": "string",
                                    "description": "Short header label shown in the UI (12 or fewer chars)."
                                },
                                "id": {
                                    "type": "string",
                                    "description": "Stable identifier for mapping answers (snake_case)."
                                },
                                "options": {
                                    "type": "array",
                                    "description": "Provide 2-3 mutually exclusive choices. Put the recommended option first and suffix its label with \"(Recommended)\". Do not include an \"Other\" option in this list; the client will add a free-form \"Other\" option automatically.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "description": {
                                                "type": "string",
                                                "description": "One short sentence explaining impact/tradeoff if selected."
                                            },
                                            "label": {
                                                "type": "string",
                                                "description": "User-facing label (1-5 words)."
                                            }
                                        },
                                        "required": [
                                            "label",
                                            "description"
                                        ],
                                        "additionalProperties": False
                                    }
                                },
                                "question": {
                                    "type": "string",
                                    "description": "Single-sentence prompt shown to the user."
                                }
                            },
                            "required": [
                                "id",
                                "header",
                                "question",
                                "options"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                "required": [
                    "questions"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "view_image",
            "description": "View a local image file from the filesystem when visual inspection is needed. Use this for images already available on disk.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local filesystem path to an image file."
                    }
                },
                "required": [
                    "path"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Tools for spawning and managing sub-agents.",
            "tools": [
                {
                    "type": "function",
                    "name": "close_agent",
                    "description": "Close an agent and any open descendants when they are no longer needed, and return the target agent's previous status be\n[... truncated for the test fixture ...]",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Agent id to close (from spawn_agent)."
                            }
                        },
                        "required": [
                            "target"
                        ],
                        "additionalProperties": False
                    }
                },
                {
                    "type": "function",
                    "name": "resume_agent",
                    "description": "Resume a previously closed agent by id so it can receive send_input and wait_agent calls.",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Agent id to resume."
                            }
                        },
                        "required": [
                            "id"
                        ],
                        "additionalProperties": False
                    }
                },
                {
                    "type": "function",
                    "name": "send_input",
                    "description": "Send a message to an existing agent. Use interrupt=true to redirect work immediately. You should reuse the agent by send\n[... truncated for the test fixture ...]",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "interrupt": {
                                "type": "boolean",
                                "description": "True interrupts the current task and handles this message immediately; false or omitted queues it."
                            },
                            "items": {
                                "type": "array",
                                "description": "Structured input items. Use this to pass explicit mentions (for example app:// connector paths).",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "audio_url": {
                                            "type": "string",
                                            "description": "Audio data URL when type is audio."
                                        },
                                        "image_url": {
                                            "type": "string",
                                            "description": "Image URL when type is image."
                                        },
                                        "name": {
                                            "type": "string",
                                            "description": "Display name when type is skill or mention."
                                        },
                                        "path": {
                                            "type": "string",
                                            "description": "Path when type is local_image/local_audio/skill, or structured mention target such as app://<connector-id> or plugin://<plugin-name>@<marketplace-name> when type is mention."
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": "Text content when type is text."
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "Input item type: text, image, local_image, audio, local_audio, skill, or mention."
                                        }
                                    },
                                    "additionalProperties": False
                                }
                            },
                            "message": {
                                "type": "string",
                                "description": "Legacy plain-text message to send to the agent. Use either message or items."
                            },
                            "target": {
                                "type": "string",
                                "description": "Agent id to message (from spawn_agent)."
                            }
                        },
                        "required": [
                            "target"
                        ],
                        "additionalProperties": False
                    }
                },
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "\n        \n        Available model overrides (optional; inherited parent model is preferred):\n- `gpt-6-astra`: Our most c\n[... truncated for the test fixture ...]",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fork_context": {
                                "type": "boolean",
                                "description": "True forks the current thread history into the new agent; false or omitted starts with only the initial prompt."
                            },
                            "items": {
                                "type": "array",
                                "description": "Structured input items. Use this to pass explicit mentions (for example app:// connector paths).",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "audio_url": {
                                            "type": "string",
                                            "description": "Audio data URL when type is audio."
                                        },
                                        "image_url": {
                                            "type": "string",
                                            "description": "Image URL when type is image."
                                        },
                                        "name": {
                                            "type": "string",
                                            "description": "Display name when type is skill or mention."
                                        },
                                        "path": {
                                            "type": "string",
                                            "description": "Path when type is local_image/local_audio/skill, or structured mention target such as app://<connector-id> or plugin://<plugin-name>@<marketplace-name> when type is mention."
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": "Text content when type is text."
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "Input item type: text, image, local_image, audio, local_audio, skill, or mention."
                                        }
                                    },
                                    "additionalProperties": False
                                }
                            },
                            "message": {
                                "type": "string",
                                "description": "Initial plain-text task for the new agent. Use either message or items."
                            },
                            "model": {
                                "type": "string",
                                "description": "Model override for the new agent. Omit unless an explicit override is needed."
                            },
                            "reasoning_effort": {
                                "type": "string",
                                "description": "Reasoning effort override for the new agent. Omit to inherit the parent effort."
                            }
                        },
                        "additionalProperties": False
                    }
                },
                {
                    "type": "function",
                    "name": "wait_agent",
                    "description": "Wait for agents to reach a final status. Completed statuses may include the agent's final message. Returns empty status \n[... truncated for the test fixture ...]",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "targets": {
                                "type": "array",
                                "description": "Agent ids to wait on. Pass multiple ids to wait for whichever finishes first.",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "timeout_ms": {
                                "type": "number",
                                "description": "Timeout in milliseconds. Defaults to 30000, min 10000, max 3600000. Prefer longer waits (minutes) to avoid busy polling."
                            }
                        },
                        "required": [
                            "targets"
                        ],
                        "additionalProperties": False
                    }
                }
            ]
        },
        {
            "type": "function",
            "name": "get_goal",
            "description": "Get the current goal for this thread, including status, budgets, token and elapsed-time usage, and remaining token budget.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "create_goal",
            "description": "Create a goal only when explicitly requested by the user or system/developer instructions; do not infer goals from ordinary tasks.\nSet token_budget only when an\n[... truncated for the test fixture ...]",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Required. The concrete objective to start pursuing. This starts a new active goal when no goal exists or replaces the current goal when it is complete."
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Positive token budget for the new goal. Omit unless explicitly requested."
                    }
                },
                "required": [
                    "objective"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": "update_goal",
            "description": "Update the existing goal.\nUse this tool only to mark the goal achieved or genuinely blocked.\nSet status to `complete` only when the objective has actually been \n[... truncated for the test fixture ...]",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Required. Set to `complete` only when the objective is achieved and no required work remains. Set to `blocked` only after the same blocking condition has recurred for at least three consecutive goal turns and the agent is at an impasse. After a previously blocked goal is resumed, the resumed run starts a fresh blocked audit.",
                        "enum": [
                            "complete",
                            "blocked"
                        ]
                    }
                },
                "required": [
                    "status"
                ],
                "additionalProperties": False
            }
        },
        {
            "type": "web_search",
            "external_web_access": True
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "reasoning": {
        "summary": "auto"
    },
    "store": False,
    "stream": True,
    "include": [
        "reasoning.encrypted_content"
    ],
    "prompt_cache_key": "01a07546-320d-7c52-93c7-9f9292b64054",
    "client_metadata": {
        "thread_id": "01a07546-320d-7c52-93c7-9f9292b64054",
        "turn_id": "01a07546-3229-7f13-812b-f256b9ffc2c7",
        "x-codex-turn-metadata": "{\"installation_id\":\"c5a00c2d-d76b-40c4-a59f-b42c5834124c\",\"session_id\":\"01a07546-320d-7c52-93c7-9f9292b64054\",\"thread_id\":\"01a07546-320d-7c52-93c7-9f9292b64054\",\"agent_name\":\"/root\",\"turn_id\":\"01a07546-3229-7f13-812b-f256b9ffc2c7\",\"window_id\":\"01a07546-320d-7c52-93c7-9f9292b64054:0\",\"window_number\":0,\"context_window_id\":\"01a07546-320d-7c52-93c7-9faea86682a1\",\"request_kind\":\"turn\",\"root_turn_id\":\"01a07546-3229-7f13-812b-f256b9ffc2c7\",\"thread_source\":\"user\",\"sandbox\":\"none\",\"sandbox_mode\":\"danger-full-access\",\"auto_review_enabled\":false,\"node_repl_auto_review_required\":false,\"node_repl_disabled\":false,\"turn_started_at_unix_ms\":1788673929770}",
        "x-codex-window-id": "01a07546-320d-7c52-93c7-9f9292b64054:0",
        "x-codex-installation-id": "c5a00c2d-d76b-40c4-a59f-b42c5834124c",
        "root_turn_id": "01a07546-3229-7f13-812b-f256b9ffc2c7",
        "session_id": "01a07546-320d-7c52-93c7-9f9292b64054"
    }
}


CODEX_RESPONSES_CODE_MODE_REQUEST: Dict[str, Any] = {'model': 'gpt-5.6-terra',
 'input': [{'type': 'additional_tools',
            'id': 'at_6a6f305a-a32b-516b-83f3-8a7ac672ca8b',
            'role': 'developer',
            'tools': [{'type': 'namespace',
                       'name': 'functions',
                       'description': '',
                       'tools': [{'type': 'custom',
                                  'name': 'exec',
                                  'description': 'Run JavaScript code to orchestrate/compose '
                                                 'tool calls\n'
                                                 '- Evaluates the provided JavaScript code in a '
                                                 'fresh V8 isolate as an async module.\n'
                                                 '- All nested tools are available on the global '
                                                 '`tools` object, for example `await '
                                                 'tools.exec_command(...)`. Tool names are '
                                                 'exposed as normalized JavaScript identifiers, '
                                                 'for example `await '
                                                 'tools.mcp__ologs__get_profile(...)`.\n'
                                                 '- Nested tool methods take either a st\n'
                                                 '[... truncated for the test fixture ...]',
                                  'format': {'type': 'grammar',
                                             'syntax': 'lark',
                                             'definition': '\n'
                                                           'start: pragma_source | plain_source\n'
                                                           'pragma_source: PRAGMA_LINE NEWLINE '
                                                           'SOURCE\n'
                                                           'plain_source: SOURCE\n'
                                                           '\n'
                                                           'PRAGMA_LINE: /[ \\t]*\\/\\/ '
                                                           '@exec:[^\\r\\n]*/\n'
                                                           'NEWLINE: /\\r?\\n/\n'
                                                           'SOURCE: /[\\s\\S]+/\n'}},
                                 {'type': 'function',
                                  'name': 'wait',
                                  'description': 'Waits on a yielded `exec` cell and returns new '
                                                 'output or completion.\n'
                                                 '- Use `wait` only after `exec` returns `Script '
                                                 'running with cell ID ...`.\n'
                                                 '- `cell_id` identifies the running `exec` cell '
                                                 'to resume.\n'
                                                 '- `yield_time_ms` controls how long to wait '
                                                 'for more output before yielding again. '
                                                 'Defaults to 10000 ms.\n'
                                                 '- `max_tokens` limits how much new output this '
                                                 'wait call returns. Defaults to 10000 tokens.\n'
                                                 '- \n'
                                                 '[... truncated for the test fixture ...]',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'cell_id': {'type': 'string',
                                                                            'description': 'Identifier '
                                                                                           'of '
                                                                                           'the '
                                                                                           'running '
                                                                                           'exec '
                                                                                           'cell.'},
                                                                'max_tokens': {'type': 'number',
                                                                               'description': 'Output '
                                                                                              'token '
                                                                                              'budget '
                                                                                              'for '
                                                                                              'this '
                                                                                              'wait '
                                                                                              'call. '
                                                                                              'Defaults '
                                                                                              'to '
                                                                                              '10000 '
                                                                                              'tokens.'},
                                                                'terminate': {'type': 'boolean',
                                                                              'description': 'True '
                                                                                             'stops '
                                                                                             'the '
                                                                                             'running '
                                                                                             'exec '
                                                                                             'cell; '
                                                                                             'false '
                                                                                             'or '
                                                                                             'omitted '
                                                                                             'waits '
                                                                                             'for '
                                                                                             'output.'},
                                                                'yield_time_ms': {'type': 'number',
                                                                                  'description': 'Wait '
                                                                                                 'before '
                                                                                                 'yielding '
                                                                                                 'more '
                                                                                                 'output. '
                                                                                                 'Defaults '
                                                                                                 'to '
                                                                                                 '10000 '
                                                                                                 'ms.'}},
                                                 'required': ['cell_id'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'request_user_input',
                                  'description': 'Request user input for one to three short '
                                                 'questions and wait for the response. This tool '
                                                 'is only available in Plan mode.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'questions': {'type': 'array',
                                                                              'description': 'Questions '
                                                                                             'to '
                                                                                             'show '
                                                                                             'the '
                                                                                             'user. '
                                                                                             'Prefer '
                                                                                             '1 '
                                                                                             'and '
                                                                                             'do '
                                                                                             'not '
                                                                                             'exceed '
                                                                                             '3',
                                                                              'items': {'type': 'object',
                                                                                        'properties': {'header': {'type': 'string',
                                                                                                                  'description': 'Short '
                                                                                                                                 'header '
                                                                                                                                 'label '
                                                                                                                                 'shown '
                                                                                                                                 'in '
                                                                                                                                 'the '
                                                                                                                                 'UI '
                                                                                                                                 '(12 '
                                                                                                                                 'or '
                                                                                                                                 'fewer '
                                                                                                                                 'chars).'},
                                                                                                       'id': {'type': 'string',
                                                                                                              'description': 'Stable '
                                                                                                                             'identifier '
                                                                                                                             'for '
                                                                                                                             'mapping '
                                                                                                                             'answers '
                                                                                                                             '(snake_case).'},
                                                                                                       'options': {'type': 'array',
                                                                                                                   'description': 'Provide '
                                                                                                                                  '2-3 '
                                                                                                                                  'mutually '
                                                                                                                                  'exclusive '
                                                                                                                                  'choices. '
                                                                                                                                  'Put '
                                                                                                                                  'the '
                                                                                                                                  'recommended '
                                                                                                                                  'option '
                                                                                                                                  'first '
                                                                                                                                  'and '
                                                                                                                                  'suffix '
                                                                                                                                  'its '
                                                                                                                                  'label '
                                                                                                                                  'with '
                                                                                                                                  '"(Recommended)". '
                                                                                                                                  'Do '
                                                                                                                                  'not '
                                                                                                                                  'include '
                                                                                                                                  'an '
                                                                                                                                  '"Other" '
                                                                                                                                  'option '
                                                                                                                                  'in '
                                                                                                                                  'this '
                                                                                                                                  'list; '
                                                                                                                                  'the '
                                                                                                                                  'client '
                                                                                                                                  'will '
                                                                                                                                  'add '
                                                                                                                                  'a '
                                                                                                                                  'free-form '
                                                                                                                                  '"Other" '
                                                                                                                                  'option '
                                                                                                                                  'automatically.',
                                                                                                                   'items': {'type': 'object',
                                                                                                                             'properties': {'description': {'type': 'string',
                                                                                                                                                            'description': 'One '
                                                                                                                                                                           'short '
                                                                                                                                                                           'sentence '
                                                                                                                                                                           'explaining '
                                                                                                                                                                           'impact/tradeoff '
                                                                                                                                                                           'if '
                                                                                                                                                                           'selected.'},
                                                                                                                                            'label': {'type': 'string',
                                                                                                                                                      'description': 'User-facing '
                                                                                                                                                                     'label '
                                                                                                                                                                     '(1-5 '
                                                                                                                                                                     'words).'}},
                                                                                                                             'required': ['label',
                                                                                                                                          'description'],
                                                                                                                             'additionalProperties': False}},
                                                                                                       'question': {'type': 'string',
                                                                                                                    'description': 'Single-sentence '
                                                                                                                                   'prompt '
                                                                                                                                   'shown '
                                                                                                                                   'to '
                                                                                                                                   'the '
                                                                                                                                   'user.'}},
                                                                                        'required': ['id',
                                                                                                     'header',
                                                                                                     'question',
                                                                                                     'options'],
                                                                                        'additionalProperties': False}}},
                                                 'required': ['questions'],
                                                 'additionalProperties': False}}]},
                      {'type': 'namespace',
                       'name': 'collaboration',
                       'description': 'Tools for spawning and managing sub-agents.',
                       'tools': [{'type': 'function',
                                  'name': 'followup_task',
                                  'description': 'Send a follow-up task to an existing non-root '
                                                 'target agent and trigger a turn if it is idle. '
                                                 'If the target is already running, deliver the '
                                                 'task promptly at message boundaries while '
                                                 'sampling, or after the pending tool call '
                                                 'completes.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'message': {'type': 'string',
                                                                            'description': 'Message '
                                                                                           'text '
                                                                                           'to '
                                                                                           'send '
                                                                                           'to '
                                                                                           'the '
                                                                                           'target '
                                                                                           'agent.',
                                                                            'encrypted': True},
                                                                'target': {'type': 'string',
                                                                           'description': 'Agent '
                                                                                          'id or '
                                                                                          'canonical '
                                                                                          'task '
                                                                                          'name '
                                                                                          'to '
                                                                                          'send '
                                                                                          'a '
                                                                                          'follow-up '
                                                                                          'task '
                                                                                          'to '
                                                                                          '(from '
                                                                                          'spawn_agent).'}},
                                                 'required': ['target', 'message'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'interrupt_agent',
                                  'description': "Interrupt an agent's current turn, if any, and "
                                                 'return its previous status. The agent remains '
                                                 'available for messages and follow-up tasks.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'target': {'type': 'string',
                                                                           'description': 'Agent '
                                                                                          'id or '
                                                                                          'canonical '
                                                                                          'task '
                                                                                          'name '
                                                                                          'to '
                                                                                          'interrupt '
                                                                                          '(from '
                                                                                          'spawn_agent).'}},
                                                 'required': ['target'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'list_agents',
                                  'description': 'List live agents in the current root thread '
                                                 'tree. Optionally filter by task-path prefix.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'path_prefix': {'type': 'string',
                                                                                'description': 'Task-path '
                                                                                               'prefix '
                                                                                               'filter '
                                                                                               'without '
                                                                                               'a '
                                                                                               'trailing '
                                                                                               'slash. '
                                                                                               'Omit '
                                                                                               'to '
                                                                                               'list '
                                                                                               'all '
                                                                                               'live '
                                                                                               'agents.'}},
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'send_message',
                                  'description': 'Send a message to an existing agent. The '
                                                 'message will be delivered promptly. Does not '
                                                 'trigger a new turn.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'message': {'type': 'string',
                                                                            'description': 'Message '
                                                                                           'text '
                                                                                           'to '
                                                                                           'queue '
                                                                                           'on '
                                                                                           'the '
                                                                                           'target '
                                                                                           'agent.',
                                                                            'encrypted': True},
                                                                'target': {'type': 'string',
                                                                           'description': 'Relative '
                                                                                          'or '
                                                                                          'canonical '
                                                                                          'task '
                                                                                          'name '
                                                                                          'to '
                                                                                          'message '
                                                                                          '(from '
                                                                                          'spawn_agent).'}},
                                                 'required': ['target', 'message'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'spawn_agent',
                                  'description': '\n'
                                                 '        Available model overrides (optional; '
                                                 'inherited parent model is preferred):\n'
                                                 '- `gpt-6-astra`: Our most capable model for '
                                                 'complex, demanding work. Reasoning efforts: '
                                                 'low (default), medium, high, xhigh, max, '
                                                 'ultra. Service tiers: priority.\n'
                                                 '- `gpt-5.6-sol`: Latest frontier agentic '
                                                 'coding model. Reasoning efforts: low '
                                                 '(default), medium, high, xhigh, max, ultra. '
                                                 'Service tiers: priority, ultrafas\n'
                                                 '[... truncated for the test fixture ...]',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'fork_turns': {'type': 'string',
                                                                               'description': 'Optional '
                                                                                              'number '
                                                                                              'of '
                                                                                              'turns '
                                                                                              'to '
                                                                                              'fork. '
                                                                                              'Defaults '
                                                                                              'to '
                                                                                              '`all`. '
                                                                                              'Use '
                                                                                              '`none`, '
                                                                                              '`all`, '
                                                                                              'or '
                                                                                              'a '
                                                                                              'positive '
                                                                                              'integer '
                                                                                              'string '
                                                                                              'such '
                                                                                              'as '
                                                                                              '`3` '
                                                                                              'to '
                                                                                              'fork '
                                                                                              'only '
                                                                                              'the '
                                                                                              'most '
                                                                                              'recent '
                                                                                              'turns.'},
                                                                'message': {'type': 'string',
                                                                            'description': 'Initial '
                                                                                           'plain-text '
                                                                                           'task '
                                                                                           'for '
                                                                                           'the '
                                                                                           'new '
                                                                                           'agent.',
                                                                            'encrypted': True},
                                                                'model': {'type': 'string',
                                                                          'description': 'Model '
                                                                                         'override '
                                                                                         'for '
                                                                                         'the '
                                                                                         'new '
                                                                                         'agent. '
                                                                                         'Omit '
                                                                                         'unless '
                                                                                         'an '
                                                                                         'explicit '
                                                                                         'override '
                                                                                         'is '
                                                                                         'needed.'},
                                                                'reasoning_effort': {'type': 'string',
                                                                                     'description': 'Reasoning '
                                                                                                    'effort '
                                                                                                    'override '
                                                                                                    'for '
                                                                                                    'the '
                                                                                                    'new '
                                                                                                    'agent. '
                                                                                                    'Omit '
                                                                                                    'to '
                                                                                                    'inherit '
                                                                                                    'the '
                                                                                                    'parent '
                                                                                                    'effort.'},
                                                                'task_name': {'type': 'string',
                                                                              'description': 'Task '
                                                                                             'name '
                                                                                             'for '
                                                                                             'the '
                                                                                             'new '
                                                                                             'agent. '
                                                                                             'Use '
                                                                                             'lowercase '
                                                                                             'letters, '
                                                                                             'digits, '
                                                                                             'and '
                                                                                             'underscores.'}},
                                                 'required': ['task_name', 'message'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'wait_agent',
                                  'description': 'Wait for a mailbox update from any live agent, '
                                                 'including queued messages and final-status '
                                                 'notifications. The wait also ends early when '
                                                 'new user input is steered into the active '
                                                 'turn. Does not return the content; returns '
                                                 'either a summary of which agents have updates '
                                                 '(if any), an interruption summary for steered '
                                                 'input, or a timeout summary if no activity '
                                                 'arrives before the deadline.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'timeout_ms': {'type': 'number',
                                                                               'description': 'Timeout '
                                                                                              'in '
                                                                                              'milliseconds. '
                                                                                              'Defaults '
                                                                                              'to '
                                                                                              '30000, '
                                                                                              'min '
                                                                                              '10000, '
                                                                                              'max '
                                                                                              '3600000.'}},
                                                 'additionalProperties': False}}]}]},
           {'type': 'message',
            'id': 'msg_705916b2-5982-5100-bedc-d5cd68faea5c',
            'role': 'developer',
            'content': [{'type': 'input_text',
                         'text': 'You are Codex, an agent based on GPT-5. You and the user share '
                                 'one workspace, and your job is to collaborate with them until '
                                 'their goal is genuinely handled.\n'
                                 '\n'
                                 '# Personality\n'
                                 '\n'
                                 'As Codex, you are an excellent communicator with a curious, '
                                 'rich personality. You match the tone and understanding of the '
                                 'user\n'
                                 '[... truncated for the test fixture ...]'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b5b-7800-9c96-0e38b8989d55',
            'role': 'developer',
            'content': [{'type': 'input_text',
                         'text': '<skills_instructions>\n'
                                 '## Skills\n'
                                 'A skill is a set of local instructions to follow that is '
                                 'stored in a `SKILL.md` file. Below is the list of skills that '
                                 'can be used. Each entry includes a name, description, and a '
                                 'short path that can be expanded into an absolute path using '
                                 'the skill roots table.\n'
                                 '### Sk\n'
                                 '[... truncated for the test fixture ...]'},
                        {'type': 'input_text',
                         'text': '<permissions instructions>\n'
                                 'Filesystem sandboxing defines which files can be read or '
                                 'written. `sandbox_mode` is `danger-full-access`: No filesystem '
                                 'sandboxing - all commands are permitted. Network access is '
                                 'enabled.\n'
                                 'Approval policy is currently never. Do not provide the '
                                 '`sandbox_permissions` for any \n'
                                 '[... truncated for the test fixture ...]'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b5b-7800-9c96-0e442ff56cee',
            'role': 'developer',
            'content': [{'type': 'input_text',
                         'text': 'You are `/root`, the primary agent in a team of agents '
                                 "collaborating to fulfill the user's goals.\n"
                                 '\n'
                                 'At the start of your turn, you are the active agent.\n'
                                 'You can spawn sub-agents to handle subtasks, and those '
                                 'sub-agents can spawn their own sub-agents.\n'
                                 'All agents in the team, including the agents that \n'
                                 '[... truncated for the test fixture ...]'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b5b-7800-9c96-0e5257e103a1',
            'role': 'developer',
            'content': [{'type': 'input_text',
                         'text': '<multi_agent_mode>Any earlier instruction enabling proactive '
                                 'multi-agent delegation no longer applies. Do not spawn '
                                 'sub-agents unless the user or applicable AGENTS.md/skill '
                                 'instructions explicitly ask for sub-agents, delegation, or '
                                 'parallel agent work.</multi_agent_mode>'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b5b-7800-9c96-0e673e920bd3',
            'role': 'user',
            'content': [{'type': 'input_text',
                         'text': '<environment_context>\n'
                                 '  <cwd>/tmp/codex_tools/work</cwd>\n'
                                 '  <shell>bash</shell>\n'
                                 '  <current_date>2026-09-06</current_date>\n'
                                 '  <timezone>Asia/Kolkata</timezone>\n'
                                 '  '
                                 '<filesystem><workspace_roots><root>/tmp/codex_tools/work</root></workspace_roots><permission_profile '
                                 'type="disabled"><file_system type="unres\n'
                                 '[... truncated for the test fixture ...]'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b67-74a3-82af-5a9b3f554339',
            'role': 'user',
            'content': [{'type': 'input_text',
                         'text': 'Read fileA.txt and write the sum of its two numbers into '
                                 'fileB.txt, then read fileB.txt back.'}]}],
 'tool_choice': 'auto',
 'parallel_tool_calls': False,
 'reasoning': {'effort': 'medium', 'context': 'all_turns'},
 'store': False,
 'stream': True,
 'include': ['reasoning.encrypted_content'],
 'prompt_cache_key': '01a07595-9b36-7ed0-b05b-e5c20046b0c6',
 'text': {'verbosity': 'low'},
 'client_metadata': {'turn_id': '01a07595-9b52-7213-9579-4b91a392d961',
                     'x-codex-window-id': '01a07595-9b36-7ed0-b05b-e5c20046b0c6:0',
                     'session_id': '01a07595-9b36-7ed0-b05b-e5c20046b0c6',
                     'thread_id': '01a07595-9b36-7ed0-b05b-e5c20046b0c6',
                     'root_turn_id': '01a07595-9b52-7213-9579-4b91a392d961',
                     'x-codex-turn-metadata': '{"installation_id":"bef5e7da-76eb-4b31-a853-da9f56c2e864","session_id":"01a07595-9b36-7ed0-b05b-e5c20046b0c6","thread_id":"01a07595-9b36-7ed0-b05b-e5c20046b0c6","agent_name":"/root","turn_id":"01a07595-9b52-7213-9579-4b91a392d961","window_id":"01a07595-9b36-7ed0-b05b-e5c20046b0c6:0","window_number":\n'
                                              '[... truncated for the test fixture ...]',
                     'x-codex-installation-id': 'bef5e7da-76eb-4b31-a853-da9f56c2e864'}}

CODEX_RESPONSES_CODE_MODE_FOLLOW_UP_REQUEST: Dict[str, Any] = {'model': 'gpt-5.6-terra',
 'input': [{'type': 'additional_tools',
            'id': 'at_6a6f305a-a32b-516b-83f3-8a7ac672ca8b',
            'role': 'developer',
            'tools': [{'type': 'namespace',
                       'name': 'functions',
                       'description': '',
                       'tools': [{'type': 'custom',
                                  'name': 'exec',
                                  'description': 'Run JavaScript code to orchestrate/compose '
                                                 'tool calls\n'
                                                 '- Evaluates the provided JavaScript code in a '
                                                 'fresh V8 isolate as an async module.\n'
                                                 '- All nested tools are available on the global '
                                                 '`tools` object, for example `await '
                                                 'tools.exec_command(...)`. Tool names are '
                                                 'exposed as normalized JavaScript identifiers, '
                                                 'for example `await '
                                                 'tools.mcp__ologs__get_profile(...)`.\n'
                                                 '- Nested tool methods take either a st\n'
                                                 '[... truncated for the test fixture ...]',
                                  'format': {'type': 'grammar',
                                             'syntax': 'lark',
                                             'definition': '\n'
                                                           'start: pragma_source | plain_source\n'
                                                           'pragma_source: PRAGMA_LINE NEWLINE '
                                                           'SOURCE\n'
                                                           'plain_source: SOURCE\n'
                                                           '\n'
                                                           'PRAGMA_LINE: /[ \\t]*\\/\\/ '
                                                           '@exec:[^\\r\\n]*/\n'
                                                           'NEWLINE: /\\r?\\n/\n'
                                                           'SOURCE: /[\\s\\S]+/\n'}},
                                 {'type': 'function',
                                  'name': 'wait',
                                  'description': 'Waits on a yielded `exec` cell and returns new '
                                                 'output or completion.\n'
                                                 '- Use `wait` only after `exec` returns `Script '
                                                 'running with cell ID ...`.\n'
                                                 '- `cell_id` identifies the running `exec` cell '
                                                 'to resume.\n'
                                                 '- `yield_time_ms` controls how long to wait '
                                                 'for more output before yielding again. '
                                                 'Defaults to 10000 ms.\n'
                                                 '- `max_tokens` limits how much new output this '
                                                 'wait call returns. Defaults to 10000 tokens.\n'
                                                 '- \n'
                                                 '[... truncated for the test fixture ...]',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'cell_id': {'type': 'string',
                                                                            'description': 'Identifier '
                                                                                           'of '
                                                                                           'the '
                                                                                           'running '
                                                                                           'exec '
                                                                                           'cell.'},
                                                                'max_tokens': {'type': 'number',
                                                                               'description': 'Output '
                                                                                              'token '
                                                                                              'budget '
                                                                                              'for '
                                                                                              'this '
                                                                                              'wait '
                                                                                              'call. '
                                                                                              'Defaults '
                                                                                              'to '
                                                                                              '10000 '
                                                                                              'tokens.'},
                                                                'terminate': {'type': 'boolean',
                                                                              'description': 'True '
                                                                                             'stops '
                                                                                             'the '
                                                                                             'running '
                                                                                             'exec '
                                                                                             'cell; '
                                                                                             'false '
                                                                                             'or '
                                                                                             'omitted '
                                                                                             'waits '
                                                                                             'for '
                                                                                             'output.'},
                                                                'yield_time_ms': {'type': 'number',
                                                                                  'description': 'Wait '
                                                                                                 'before '
                                                                                                 'yielding '
                                                                                                 'more '
                                                                                                 'output. '
                                                                                                 'Defaults '
                                                                                                 'to '
                                                                                                 '10000 '
                                                                                                 'ms.'}},
                                                 'required': ['cell_id'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'request_user_input',
                                  'description': 'Request user input for one to three short '
                                                 'questions and wait for the response. This tool '
                                                 'is only available in Plan mode.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'questions': {'type': 'array',
                                                                              'description': 'Questions '
                                                                                             'to '
                                                                                             'show '
                                                                                             'the '
                                                                                             'user. '
                                                                                             'Prefer '
                                                                                             '1 '
                                                                                             'and '
                                                                                             'do '
                                                                                             'not '
                                                                                             'exceed '
                                                                                             '3',
                                                                              'items': {'type': 'object',
                                                                                        'properties': {'header': {'type': 'string',
                                                                                                                  'description': 'Short '
                                                                                                                                 'header '
                                                                                                                                 'label '
                                                                                                                                 'shown '
                                                                                                                                 'in '
                                                                                                                                 'the '
                                                                                                                                 'UI '
                                                                                                                                 '(12 '
                                                                                                                                 'or '
                                                                                                                                 'fewer '
                                                                                                                                 'chars).'},
                                                                                                       'id': {'type': 'string',
                                                                                                              'description': 'Stable '
                                                                                                                             'identifier '
                                                                                                                             'for '
                                                                                                                             'mapping '
                                                                                                                             'answers '
                                                                                                                             '(snake_case).'},
                                                                                                       'options': {'type': 'array',
                                                                                                                   'description': 'Provide '
                                                                                                                                  '2-3 '
                                                                                                                                  'mutually '
                                                                                                                                  'exclusive '
                                                                                                                                  'choices. '
                                                                                                                                  'Put '
                                                                                                                                  'the '
                                                                                                                                  'recommended '
                                                                                                                                  'option '
                                                                                                                                  'first '
                                                                                                                                  'and '
                                                                                                                                  'suffix '
                                                                                                                                  'its '
                                                                                                                                  'label '
                                                                                                                                  'with '
                                                                                                                                  '"(Recommended)". '
                                                                                                                                  'Do '
                                                                                                                                  'not '
                                                                                                                                  'include '
                                                                                                                                  'an '
                                                                                                                                  '"Other" '
                                                                                                                                  'option '
                                                                                                                                  'in '
                                                                                                                                  'this '
                                                                                                                                  'list; '
                                                                                                                                  'the '
                                                                                                                                  'client '
                                                                                                                                  'will '
                                                                                                                                  'add '
                                                                                                                                  'a '
                                                                                                                                  'free-form '
                                                                                                                                  '"Other" '
                                                                                                                                  'option '
                                                                                                                                  'automatically.',
                                                                                                                   'items': {'type': 'object',
                                                                                                                             'properties': {'description': {'type': 'string',
                                                                                                                                                            'description': 'One '
                                                                                                                                                                           'short '
                                                                                                                                                                           'sentence '
                                                                                                                                                                           'explaining '
                                                                                                                                                                           'impact/tradeoff '
                                                                                                                                                                           'if '
                                                                                                                                                                           'selected.'},
                                                                                                                                            'label': {'type': 'string',
                                                                                                                                                      'description': 'User-facing '
                                                                                                                                                                     'label '
                                                                                                                                                                     '(1-5 '
                                                                                                                                                                     'words).'}},
                                                                                                                             'required': ['label',
                                                                                                                                          'description'],
                                                                                                                             'additionalProperties': False}},
                                                                                                       'question': {'type': 'string',
                                                                                                                    'description': 'Single-sentence '
                                                                                                                                   'prompt '
                                                                                                                                   'shown '
                                                                                                                                   'to '
                                                                                                                                   'the '
                                                                                                                                   'user.'}},
                                                                                        'required': ['id',
                                                                                                     'header',
                                                                                                     'question',
                                                                                                     'options'],
                                                                                        'additionalProperties': False}}},
                                                 'required': ['questions'],
                                                 'additionalProperties': False}}]},
                      {'type': 'namespace',
                       'name': 'collaboration',
                       'description': 'Tools for spawning and managing sub-agents.',
                       'tools': [{'type': 'function',
                                  'name': 'followup_task',
                                  'description': 'Send a follow-up task to an existing non-root '
                                                 'target agent and trigger a turn if it is idle. '
                                                 'If the target is already running, deliver the '
                                                 'task promptly at message boundaries while '
                                                 'sampling, or after the pending tool call '
                                                 'completes.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'message': {'type': 'string',
                                                                            'description': 'Message '
                                                                                           'text '
                                                                                           'to '
                                                                                           'send '
                                                                                           'to '
                                                                                           'the '
                                                                                           'target '
                                                                                           'agent.',
                                                                            'encrypted': True},
                                                                'target': {'type': 'string',
                                                                           'description': 'Agent '
                                                                                          'id or '
                                                                                          'canonical '
                                                                                          'task '
                                                                                          'name '
                                                                                          'to '
                                                                                          'send '
                                                                                          'a '
                                                                                          'follow-up '
                                                                                          'task '
                                                                                          'to '
                                                                                          '(from '
                                                                                          'spawn_agent).'}},
                                                 'required': ['target', 'message'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'interrupt_agent',
                                  'description': "Interrupt an agent's current turn, if any, and "
                                                 'return its previous status. The agent remains '
                                                 'available for messages and follow-up tasks.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'target': {'type': 'string',
                                                                           'description': 'Agent '
                                                                                          'id or '
                                                                                          'canonical '
                                                                                          'task '
                                                                                          'name '
                                                                                          'to '
                                                                                          'interrupt '
                                                                                          '(from '
                                                                                          'spawn_agent).'}},
                                                 'required': ['target'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'list_agents',
                                  'description': 'List live agents in the current root thread '
                                                 'tree. Optionally filter by task-path prefix.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'path_prefix': {'type': 'string',
                                                                                'description': 'Task-path '
                                                                                               'prefix '
                                                                                               'filter '
                                                                                               'without '
                                                                                               'a '
                                                                                               'trailing '
                                                                                               'slash. '
                                                                                               'Omit '
                                                                                               'to '
                                                                                               'list '
                                                                                               'all '
                                                                                               'live '
                                                                                               'agents.'}},
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'send_message',
                                  'description': 'Send a message to an existing agent. The '
                                                 'message will be delivered promptly. Does not '
                                                 'trigger a new turn.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'message': {'type': 'string',
                                                                            'description': 'Message '
                                                                                           'text '
                                                                                           'to '
                                                                                           'queue '
                                                                                           'on '
                                                                                           'the '
                                                                                           'target '
                                                                                           'agent.',
                                                                            'encrypted': True},
                                                                'target': {'type': 'string',
                                                                           'description': 'Relative '
                                                                                          'or '
                                                                                          'canonical '
                                                                                          'task '
                                                                                          'name '
                                                                                          'to '
                                                                                          'message '
                                                                                          '(from '
                                                                                          'spawn_agent).'}},
                                                 'required': ['target', 'message'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'spawn_agent',
                                  'description': '\n'
                                                 '        Available model overrides (optional; '
                                                 'inherited parent model is preferred):\n'
                                                 '- `gpt-6-astra`: Our most capable model for '
                                                 'complex, demanding work. Reasoning efforts: '
                                                 'low (default), medium, high, xhigh, max, '
                                                 'ultra. Service tiers: priority.\n'
                                                 '- `gpt-5.6-sol`: Latest frontier agentic '
                                                 'coding model. Reasoning efforts: low '
                                                 '(default), medium, high, xhigh, max, ultra. '
                                                 'Service tiers: priority, ultrafas\n'
                                                 '[... truncated for the test fixture ...]',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'fork_turns': {'type': 'string',
                                                                               'description': 'Optional '
                                                                                              'number '
                                                                                              'of '
                                                                                              'turns '
                                                                                              'to '
                                                                                              'fork. '
                                                                                              'Defaults '
                                                                                              'to '
                                                                                              '`all`. '
                                                                                              'Use '
                                                                                              '`none`, '
                                                                                              '`all`, '
                                                                                              'or '
                                                                                              'a '
                                                                                              'positive '
                                                                                              'integer '
                                                                                              'string '
                                                                                              'such '
                                                                                              'as '
                                                                                              '`3` '
                                                                                              'to '
                                                                                              'fork '
                                                                                              'only '
                                                                                              'the '
                                                                                              'most '
                                                                                              'recent '
                                                                                              'turns.'},
                                                                'message': {'type': 'string',
                                                                            'description': 'Initial '
                                                                                           'plain-text '
                                                                                           'task '
                                                                                           'for '
                                                                                           'the '
                                                                                           'new '
                                                                                           'agent.',
                                                                            'encrypted': True},
                                                                'model': {'type': 'string',
                                                                          'description': 'Model '
                                                                                         'override '
                                                                                         'for '
                                                                                         'the '
                                                                                         'new '
                                                                                         'agent. '
                                                                                         'Omit '
                                                                                         'unless '
                                                                                         'an '
                                                                                         'explicit '
                                                                                         'override '
                                                                                         'is '
                                                                                         'needed.'},
                                                                'reasoning_effort': {'type': 'string',
                                                                                     'description': 'Reasoning '
                                                                                                    'effort '
                                                                                                    'override '
                                                                                                    'for '
                                                                                                    'the '
                                                                                                    'new '
                                                                                                    'agent. '
                                                                                                    'Omit '
                                                                                                    'to '
                                                                                                    'inherit '
                                                                                                    'the '
                                                                                                    'parent '
                                                                                                    'effort.'},
                                                                'task_name': {'type': 'string',
                                                                              'description': 'Task '
                                                                                             'name '
                                                                                             'for '
                                                                                             'the '
                                                                                             'new '
                                                                                             'agent. '
                                                                                             'Use '
                                                                                             'lowercase '
                                                                                             'letters, '
                                                                                             'digits, '
                                                                                             'and '
                                                                                             'underscores.'}},
                                                 'required': ['task_name', 'message'],
                                                 'additionalProperties': False}},
                                 {'type': 'function',
                                  'name': 'wait_agent',
                                  'description': 'Wait for a mailbox update from any live agent, '
                                                 'including queued messages and final-status '
                                                 'notifications. The wait also ends early when '
                                                 'new user input is steered into the active '
                                                 'turn. Does not return the content; returns '
                                                 'either a summary of which agents have updates '
                                                 '(if any), an interruption summary for steered '
                                                 'input, or a timeout summary if no activity '
                                                 'arrives before the deadline.',
                                  'strict': False,
                                  'parameters': {'type': 'object',
                                                 'properties': {'timeout_ms': {'type': 'number',
                                                                               'description': 'Timeout '
                                                                                              'in '
                                                                                              'milliseconds. '
                                                                                              'Defaults '
                                                                                              'to '
                                                                                              '30000, '
                                                                                              'min '
                                                                                              '10000, '
                                                                                              'max '
                                                                                              '3600000.'}},
                                                 'additionalProperties': False}}]}]},
           {'type': 'message',
            'id': 'msg_705916b2-5982-5100-bedc-d5cd68faea5c',
            'role': 'developer',
            'content': [{'type': 'input_text',
                         'text': 'You are Codex, an agent based on GPT-5. You and the user share '
                                 'one workspace, and your job is to collaborate with them until '
                                 'their goal is genuinely handled.\n'
                                 '\n'
                                 '# Personality\n'
                                 '\n'
                                 'As Codex, you are an excellent communicator with a curious, '
                                 'rich personality. You match the tone and understanding of the '
                                 'user\n'
                                 '[... truncated for the test fixture ...]'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b5b-7800-9c96-0e38b8989d55',
            'role': 'developer',
            'content': [{'type': 'input_text',
                         'text': '<skills_instructions>\n'
                                 '## Skills\n'
                                 'A skill is a set of local instructions to follow that is '
                                 'stored in a `SKILL.md` file. Below is the list of skills that '
                                 'can be used. Each entry includes a name, description, and a '
                                 'short path that can be expanded into an absolute path using '
                                 'the skill roots table.\n'
                                 '### Sk\n'
                                 '[... truncated for the test fixture ...]'},
                        {'type': 'input_text',
                         'text': '<permissions instructions>\n'
                                 'Filesystem sandboxing defines which files can be read or '
                                 'written. `sandbox_mode` is `danger-full-access`: No filesystem '
                                 'sandboxing - all commands are permitted. Network access is '
                                 'enabled.\n'
                                 'Approval policy is currently never. Do not provide the '
                                 '`sandbox_permissions` for any \n'
                                 '[... truncated for the test fixture ...]'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b5b-7800-9c96-0e442ff56cee',
            'role': 'developer',
            'content': [{'type': 'input_text',
                         'text': 'You are `/root`, the primary agent in a team of agents '
                                 "collaborating to fulfill the user's goals.\n"
                                 '\n'
                                 'At the start of your turn, you are the active agent.\n'
                                 'You can spawn sub-agents to handle subtasks, and those '
                                 'sub-agents can spawn their own sub-agents.\n'
                                 'All agents in the team, including the agents that \n'
                                 '[... truncated for the test fixture ...]'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b5b-7800-9c96-0e5257e103a1',
            'role': 'developer',
            'content': [{'type': 'input_text',
                         'text': '<multi_agent_mode>Any earlier instruction enabling proactive '
                                 'multi-agent delegation no longer applies. Do not spawn '
                                 'sub-agents unless the user or applicable AGENTS.md/skill '
                                 'instructions explicitly ask for sub-agents, delegation, or '
                                 'parallel agent work.</multi_agent_mode>'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b5b-7800-9c96-0e673e920bd3',
            'role': 'user',
            'content': [{'type': 'input_text',
                         'text': '<environment_context>\n'
                                 '  <cwd>/tmp/codex_tools/work</cwd>\n'
                                 '  <shell>bash</shell>\n'
                                 '  <current_date>2026-09-06</current_date>\n'
                                 '  <timezone>Asia/Kolkata</timezone>\n'
                                 '  '
                                 '<filesystem><workspace_roots><root>/tmp/codex_tools/work</root></workspace_roots><permission_profile '
                                 'type="disabled"><file_system type="unres\n'
                                 '[... truncated for the test fixture ...]'}]},
           {'type': 'message',
            'id': 'msg_01a07595-9b67-74a3-82af-5a9b3f554339',
            'role': 'user',
            'content': [{'type': 'input_text',
                         'text': 'Read fileA.txt and write the sum of its two numbers into '
                                 'fileB.txt, then read fileB.txt back.'}]},
           {'type': 'message',
            'id': 'msg_ecaf4c68e95f4e29bec5e0a21539402f',
            'role': 'assistant',
            'content': [{'type': 'output_text',
                         'text': 'I’ll read the source values, write their sum to `fileB.txt`, '
                                 'and verify the written result.'}]},
           {'type': 'custom_tool_call',
            'id': 'ctc_6ac2c175ee1c4e2b98828b3db8039d3f',
            'status': 'completed',
            'call_id': 'call_2006e34c-19bc-4f88-9cf3-afef3e832de2',
            'name': 'exec',
            'namespace': 'functions',
            'input': 'const r = await tools.exec_command({cmd:"pwd && rg --files -g '
                     "'fileA.txt' -g "
                     '\'fileB.txt\'",workdir:"/tmp/codex_tools/work",yield_time_ms:10000,max_output_tokens:2000}); '
                     'text(r.output);'},
           {'type': 'custom_tool_call_output',
            'id': 'ctco_01a07595-a915-7172-9af3-bfd09dd15d87',
            'call_id': 'call_2006e34c-19bc-4f88-9cf3-afef3e832de2',
            'output': [{'type': 'input_text',
                        'text': 'Script completed\nWall time 0.0 seconds\nOutput:\n'},
                       {'type': 'input_text', 'text': '/tmp/codex_tools/work\nfileA.txt\n'}]}],
 'tool_choice': 'auto',
 'parallel_tool_calls': False,
 'reasoning': {'effort': 'medium', 'context': 'all_turns'},
 'store': False,
 'stream': True,
 'include': ['reasoning.encrypted_content'],
 'prompt_cache_key': '01a07595-9b36-7ed0-b05b-e5c20046b0c6',
 'text': {'verbosity': 'low'},
 'client_metadata': {'x-codex-window-id': '01a07595-9b36-7ed0-b05b-e5c20046b0c6:0',
                     'turn_id': '01a07595-9b52-7213-9579-4b91a392d961',
                     'x-codex-turn-metadata': '{"installation_id":"bef5e7da-76eb-4b31-a853-da9f56c2e864","session_id":"01a07595-9b36-7ed0-b05b-e5c20046b0c6","thread_id":"01a07595-9b36-7ed0-b05b-e5c20046b0c6","agent_name":"/root","turn_id":"01a07595-9b52-7213-9579-4b91a392d961","window_id":"01a07595-9b36-7ed0-b05b-e5c20046b0c6:0","window_number":\n'
                                              '[... truncated for the test fixture ...]',
                     'session_id': '01a07595-9b36-7ed0-b05b-e5c20046b0c6',
                     'thread_id': '01a07595-9b36-7ed0-b05b-e5c20046b0c6',
                     'x-codex-installation-id': 'bef5e7da-76eb-4b31-a853-da9f56c2e864',
                     'root_turn_id': '01a07595-9b52-7213-9579-4b91a392d961'}}
