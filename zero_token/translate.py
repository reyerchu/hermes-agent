"""Translate between OpenAI chat/completions and Anthropic Messages shapes.

The proxy speaks OpenAI on the client side (so hermes-agent's ``provider:
custom`` backend and any other OpenAI client can use it unchanged) and
Anthropic on the upstream side (so it can present the Claude Code identity and
OAuth headers the subscription API requires).

Covered:
* messages: system extraction, user/assistant/tool roles, string and
  multi-part content, images (data: URI and https).
* tools: OpenAI ``function`` <-> Anthropic ``tool``; ``tool_calls`` <->
  ``tool_use``; ``role:"tool"`` results <-> ``tool_result``.
* sampling params: ``max_tokens`` (required upstream), ``temperature``,
  ``top_p``, ``stop`` -> ``stop_sequences``.
* responses: content blocks -> ``message.content`` + ``tool_calls``;
  ``stop_reason`` -> ``finish_reason``; usage mapping.
* streaming: Anthropic SSE events -> OpenAI ``chat.completion.chunk`` deltas.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

# Default upstream cap when the client doesn't specify one. Anthropic *requires*
# max_tokens; OpenAI treats it as optional, so we supply a generous default.
DEFAULT_MAX_TOKENS = 8192

_STOP_REASON_TO_FINISH: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def _finish_reason(stop_reason: Any) -> str:
    """Map an Anthropic ``stop_reason`` to an OpenAI ``finish_reason``."""
    return _STOP_REASON_TO_FINISH.get(str(stop_reason), "stop")


# --------------------------------------------------------------------------
# content normalisation
# --------------------------------------------------------------------------

# Match data:<mediatype>[;param=value]*;base64,<data> — extra parameters
# (e.g. ;charset=utf-8) between the media type and ;base64 are tolerated.
_DATA_URI_RE = re.compile(
    r"^data:(?P<mt>[^;,]+)(?:;[^;,]+)*;base64,(?P<data>.+)$", re.DOTALL
)


def _openai_content_to_anthropic(content: Any) -> str | list[dict[str, Any]]:
    """Convert an OpenAI message ``content`` to Anthropic block(s)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            blocks.append({"type": "text", "text": str(part)})
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            m = _DATA_URI_RE.match(url)
            if m:
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": m.group("mt"),
                        "data": m.group("data"),
                    },
                })
            elif url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            # Unknown part — stringify so nothing is silently dropped.
            blocks.append({"type": "text", "text": json.dumps(part)})
    return blocks


# --------------------------------------------------------------------------
# request: OpenAI -> Anthropic
# --------------------------------------------------------------------------


def _tools_openai_to_anthropic(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters")
            or {"type": "object", "properties": {}},
        })
    return out or None


def _tool_choice_openai_to_anthropic(choice: Any) -> dict[str, Any] | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}
        if choice == "none":
            # Keep the tools defined (Anthropic requires that when history holds
            # tool_use/tool_result), but forbid new calls this turn.
            return {"type": "none"}
        return {"type": "auto"}
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def _messages_openai_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Any]:
    """Return ``(anthropic_messages, system)``.

    Consecutive tool results are merged into a single user message, as Anthropic
    expects ``tool_result`` blocks grouped in one user turn.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    def _append_user_block(block: dict[str, Any]) -> None:
        if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            content = msg.get("content")
            if isinstance(content, list):
                system_parts.append(
                    "".join(p.get("text", "") for p in content if isinstance(p, dict))
                )
            elif content:
                system_parts.append(str(content))
        elif role == "tool":
            # OpenAI tool result -> Anthropic tool_result block in a user turn.
            result_content = msg.get("content")
            if isinstance(result_content, list):
                result_content = "".join(
                    p.get("text", "") for p in result_content if isinstance(p, dict)
                )
            _append_user_block({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": result_content if result_content is not None else "",
            })
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = msg.get("content")
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            elif isinstance(text, list):
                norm = _openai_content_to_anthropic(text)
                if isinstance(norm, list):
                    blocks.extend(norm)
                elif norm:
                    blocks.append({"type": "text", "text": norm})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except json.JSONDecodeError:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args if isinstance(args, dict) else {},
                })
            # Anthropic rejects messages with empty content. An assistant turn
            # with neither text nor tool_use (e.g. a filtered/empty stored turn)
            # is dropped rather than sent as content:"".
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        else:  # user (or unknown -> treat as user)
            content = _openai_content_to_anthropic(msg.get("content"))
            if content == "" or content == []:
                continue  # skip empty user turns (Anthropic 400s on them)
            out.append({"role": "user", "content": content})

    system: Any = "\n\n".join(p for p in system_parts if p) if system_parts else None
    return out, system


def openai_to_anthropic_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Build an Anthropic Messages request body from an OpenAI request."""
    messages, system = _messages_openai_to_anthropic(payload.get("messages") or [])
    body: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "max_tokens": int(
            payload.get("max_tokens")
            or payload.get("max_completion_tokens")
            or DEFAULT_MAX_TOKENS
        ),
    }
    if system is not None:
        body["system"] = system

    if payload.get("temperature") is not None:
        body["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        body["top_p"] = payload["top_p"]

    stop = payload.get("stop")
    if isinstance(stop, str):
        body["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        body["stop_sequences"] = [s for s in stop if isinstance(s, str)]

    tools = _tools_openai_to_anthropic(payload.get("tools"))
    if tools:
        body["tools"] = tools
        tc = _tool_choice_openai_to_anthropic(payload.get("tool_choice"))
        if tc is not None:
            body["tool_choice"] = tc

    # Anthropic's metadata only permits `user_id`; forwarding arbitrary OpenAI
    # metadata keys 400s the request. Map user_id (or the top-level `user`).
    user_id = (payload.get("metadata") or {}).get("user_id") or payload.get("user")
    if user_id:
        body["metadata"] = {"user_id": str(user_id)}

    return body


def add_cache_control(body: dict[str, Any]) -> dict[str, Any]:
    """Insert ephemeral prompt-cache breakpoints, Claude-Code style.

    An agent replays the same large system prompt + tool schemas + conversation
    prefix on every turn. Without cache breakpoints Anthropic re-bills all of it
    as fresh input each call; with them, the repeated prefix is billed once and
    then served as (~10%-cost) cache reads — which also drains the subscription
    usage window far more slowly. Marks up to three breakpoints (well under
    Anthropic's limit of four): the last system block, the last tool, and the
    last message's final content block. Mutates and returns ``body``.
    """
    cc = {"type": "ephemeral"}

    system = body.get("system")
    if isinstance(system, list) and system and isinstance(system[-1], dict):
        system[-1].setdefault("cache_control", dict(cc))

    tools = body.get("tools")
    if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
        tools[-1].setdefault("cache_control", dict(cc))

    messages = body.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[-1], dict):
        content = messages[-1].get("content")
        if isinstance(content, str) and content:
            messages[-1]["content"] = [
                {"type": "text", "text": content, "cache_control": dict(cc)}
            ]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            content[-1].setdefault("cache_control", dict(cc))
    return body


# --------------------------------------------------------------------------
# response: Anthropic -> OpenAI (non-streaming)
# --------------------------------------------------------------------------


def anthropic_to_openai_response(
    data: dict[str, Any], *, response_id: str, created: int, model: str
) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish = _finish_reason(data.get("stop_reason"))
    usage = data.get("usage") or {}
    in_tok = usage.get("input_tokens", 0) or 0
    out_tok = usage.get("output_tokens", 0) or 0
    # Anthropic reports cached input separately; fold both into prompt_tokens so
    # the total is honest, and surface the read portion as OpenAI's standard
    # prompt_tokens_details.cached_tokens (else prompt caching is invisible).
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_create = usage.get("cache_creation_input_tokens", 0) or 0
    prompt_tokens = in_tok + cache_read + cache_create

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": out_tok,
            "total_tokens": prompt_tokens + out_tok,
            "prompt_tokens_details": {"cached_tokens": cache_read},
        },
    }


# --------------------------------------------------------------------------
# response: Anthropic SSE -> OpenAI chunks (streaming)
# --------------------------------------------------------------------------


def _chunk(
    response_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish: str | None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class AnthropicStreamTranslator:
    """Stateful translator from Anthropic SSE events to OpenAI chunk dicts.

    Feed each parsed Anthropic event dict to :meth:`handle`; it yields zero or
    more OpenAI chunk dicts. Tool-call arguments arrive as ``input_json_delta``
    fragments which map straight onto OpenAI's incremental
    ``tool_calls[].function.arguments`` deltas.
    """

    def __init__(self, response_id: str, created: int, model: str) -> None:
        self._id = response_id
        self._created = created
        self._model = model
        self._role_sent = False
        # Anthropic content-block index -> OpenAI tool_calls array index.
        self._tool_slot: dict[int, int] = {}
        self._next_tool_index = 0
        self._finish: str | None = None

    def _mk(self, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return _chunk(self._id, self._created, self._model, delta, finish)

    def handle(self, event: dict[str, Any]) -> Iterable[dict[str, Any]]:
        etype = event.get("type")
        if etype == "message_start":
            if not self._role_sent:
                self._role_sent = True
                yield self._mk({"role": "assistant"})
        elif etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                idx = event.get("index", 0)
                slot = self._next_tool_index
                self._next_tool_index += 1
                self._tool_slot[idx] = slot
                yield self._mk({
                    "tool_calls": [
                        {
                            "index": slot,
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": "",
                            },
                        }
                    ]
                })
        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text", "")
                if text:
                    yield self._mk({"content": text})
            elif dtype == "input_json_delta":
                idx = event.get("index", 0)
                slot = self._tool_slot.get(idx, 0)
                yield self._mk({
                    "tool_calls": [
                        {
                            "index": slot,
                            "function": {"arguments": delta.get("partial_json", "")},
                        }
                    ]
                })
        elif etype == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason")
            if stop_reason:
                self._finish = _finish_reason(stop_reason)
        elif etype == "message_stop":
            yield self._mk({}, finish=self._finish or "stop")

    @property
    def finish_reason(self) -> str | None:
        return self._finish
