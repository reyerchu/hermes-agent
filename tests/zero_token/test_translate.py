"""OpenAI <-> Anthropic translation unit tests (no network)."""

from __future__ import annotations

import json

from zero_token import translate as tr


def test_system_messages_extracted_and_joined():
    payload = {
        "model": "claude-opus-4-8",
        "messages": [
            {"role": "system", "content": "sys A"},
            {"role": "system", "content": "sys B"},
            {"role": "user", "content": "hi"},
        ],
    }
    body = tr.openai_to_anthropic_body(payload)
    assert body["system"] == "sys A\n\nsys B"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == tr.DEFAULT_MAX_TOKENS


def test_max_tokens_and_stop_and_sampling_passthrough():
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        "temperature": 0.4,
        "top_p": 0.9,
        "stop": ["END", "STOP"],
    }
    body = tr.openai_to_anthropic_body(payload)
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.4
    assert body["top_p"] == 0.9
    assert body["stop_sequences"] == ["END", "STOP"]


def test_tools_and_tool_choice_translation():
    payload = {
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    body = tr.openai_to_anthropic_body(payload)
    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "look up weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    assert body["tool_choice"] == {"type": "tool", "name": "get_weather"}


def test_tool_choice_none_maps_to_anthropic_none():
    body = tr.openai_to_anthropic_body({
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "none",
    })
    assert body["tool_choice"] == {"type": "none"}


def test_metadata_only_forwards_user_id():
    body = tr.openai_to_anthropic_body({
        "messages": [{"role": "user", "content": "x"}],
        "metadata": {"session": "abc", "trace_id": "t1", "user_id": "u9"},
    })
    assert body["metadata"] == {"user_id": "u9"}


def test_top_level_user_maps_to_metadata_user_id():
    body = tr.openai_to_anthropic_body({
        "messages": [{"role": "user", "content": "x"}],
        "user": "u42",
    })
    assert body["metadata"] == {"user_id": "u42"}


def test_arbitrary_metadata_without_user_id_is_dropped():
    body = tr.openai_to_anthropic_body({
        "messages": [{"role": "user", "content": "x"}],
        "metadata": {"session": "abc"},
    })
    assert "metadata" not in body


def test_empty_assistant_and_user_messages_are_dropped():
    body = tr.openai_to_anthropic_body({
        "messages": [
            {"role": "user", "content": "real"},
            {"role": "assistant", "content": None},  # empty, no tool_calls
            {"role": "user", "content": ""},  # empty user turn
        ],
    })
    # only the real user turn survives; no content:"" messages emitted
    assert body["messages"] == [{"role": "user", "content": "real"}]


def test_data_uri_with_extra_params_still_parsed():
    body = tr.openai_to_anthropic_body({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;charset=utf-8;base64,QUJD"
                        },
                    },
                ],
            }
        ],
    })
    blk = body["messages"][0]["content"][0]
    assert blk["source"]["media_type"] == "image/png"
    assert blk["source"]["data"] == "QUJD"


def test_stream_translator_error_event_is_not_swallowed_as_finish():
    # The translator itself yields nothing for an error event; the server layer
    # handles it. Verify no spurious finish chunk is produced.
    t = tr.AnthropicStreamTranslator("rid", 1, "m")
    out = list(t.handle({"type": "error", "error": {"type": "overloaded_error"}}))
    assert out == []
    assert t.finish_reason is None


def test_tool_choice_required_maps_to_any():
    body = tr.openai_to_anthropic_body({
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "required",
    })
    assert body["tool_choice"] == {"type": "any"}


def test_assistant_tool_calls_and_tool_results_roundtrip_shape():
    payload = {
        "messages": [
            {"role": "user", "content": "call it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Taipei"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
    }
    body = tr.openai_to_anthropic_body(payload)
    # assistant turn becomes tool_use block
    assistant = body["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {"city": "Taipei"},
    }
    # tool result becomes a user turn with a tool_result block
    tool_turn = body["messages"][2]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "sunny",
    }


def test_consecutive_tool_results_merge_into_one_user_turn():
    payload = {
        "messages": [
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "tool", "tool_call_id": "b", "content": "rb"},
        ],
    }
    body = tr.openai_to_anthropic_body(payload)
    assert len(body["messages"]) == 1
    assert [b["tool_use_id"] for b in body["messages"][0]["content"]] == ["a", "b"]


def test_image_data_uri_becomes_base64_block():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                ],
            }
        ],
    }
    body = tr.openai_to_anthropic_body(payload)
    blocks = body["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "what is this"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_response_text_translation():
    data = {
        "content": [{"type": "text", "text": "hello world"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }
    out = tr.anthropic_to_openai_response(
        data, response_id="id1", created=123, model="m"
    )
    assert out["choices"][0]["message"]["content"] == "hello world"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }


def test_response_tool_use_translation_sets_tool_calls_finish():
    data = {
        "content": [
            {"type": "text", "text": "let me check"},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "get_weather",
                "input": {"city": "Taipei"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    out = tr.anthropic_to_openai_response(data, response_id="id", created=1, model="m")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "let me check"
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    tc = msg["tool_calls"][0]
    assert tc["id"] == "tu_1"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Taipei"}


def test_stream_translator_text_flow():
    t = tr.AnthropicStreamTranslator("rid", 100, "m")
    chunks: list[dict] = []
    for event in [
        {"type": "message_start", "message": {}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hel"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "lo"},
        },
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]:
        chunks.extend(t.handle(event))
    # first chunk sets role
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_translator_tool_call_flow():
    t = tr.AnthropicStreamTranslator("rid", 100, "m")
    chunks: list[dict] = []
    for event in [
        {"type": "message_start", "message": {}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_9", "name": "f"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"a":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "1}"},
        },
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]:
        chunks.extend(t.handle(event))
    # collect the tool-call deltas
    args = ""
    name = None
    tid = None
    for c in chunks:
        for tc in c["choices"][0]["delta"].get("tool_calls", []):
            name = tc.get("function", {}).get("name") or name
            tid = tc.get("id") or tid
            args += tc.get("function", {}).get("arguments", "")
    assert name == "f"
    assert tid == "tu_9"
    assert json.loads(args) == {"a": 1}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
