"""Server-level tests for the OpenAI surface, with a mocked Anthropic upstream."""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from zero_token import server as srv


class _FakeStore:
    def access_token(self):
        return "sk-ant-oat01-TESTTOKEN"

    def describe(self):
        return {"source": "fake", "refreshable": True, "subscriptionType": "max"}


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    async def aread(self):
        return json.dumps(self._payload).encode()


class _CapturingClient:
    """Minimal stand-in for httpx.AsyncClient capturing the outgoing request."""

    def __init__(self, resp: _FakeResp):
        self._resp = resp
        self.last_json = None
        self.last_headers = None
        self.last_url = None

    async def post(self, url, json, headers, timeout):  # noqa: A002
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        return self._resp

    async def aclose(self):
        pass


class _StreamResp:
    """Fake httpx streaming response yielding preset SSE lines."""

    def __init__(self, status_code, sse_lines):
        self.status_code = status_code
        self._lines = sse_lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aiter_raw(self):
        for ln in self._lines:
            yield (ln + "\n").encode()

    async def aread(self):
        return "\n".join(self._lines).encode()


class _StreamingClient:
    def __init__(self, stream_resp):
        self._stream_resp = stream_resp

    def stream(self, method, url, json, headers, timeout):  # noqa: A002
        return self._stream_resp

    async def aclose(self):
        pass


async def _make_client(anthropic_payload, status=200, http=None):
    app = srv.build_app()
    app[srv.STORE_KEY] = _FakeStore()
    fake = http or _CapturingClient(_FakeResp(status, anthropic_payload))

    # Replace the startup hook so it installs our fake http client.
    app.on_startup.clear()

    async def _startup(a):
        a[srv.HTTP_KEY] = fake

    app.on_startup.append(_startup)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, fake


@pytest.mark.asyncio
async def test_chat_completions_non_stream_happy_path():
    anthropic_resp = {
        "content": [{"type": "text", "text": "HERMES_OAUTH_OK"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 4},
        "model": "claude-opus-4-8",
    }
    client, fake = await _make_client(anthropic_resp)
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/claude-opus-4-8",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "say ok"},
                ],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "HERMES_OAUTH_OK"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["total_tokens"] == 16

        # upstream got the mandatory identity block + Bearer + stripped vendor prefix
        assert fake.last_json["model"] == "claude-opus-4-8"
        assert fake.last_json["system"][0]["text"].startswith("You are Claude Code")
        assert fake.last_headers["Authorization"] == "Bearer sk-ant-oat01-TESTTOKEN"
        assert "oauth-2025-04-20" in fake.last_headers["anthropic-beta"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_completions_rejects_empty_messages():
    client, _ = await _make_client({})
    try:
        resp = await client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upstream_error_is_forwarded():
    client, _ = await _make_client(
        {"error": {"message": "boom", "type": "overloaded_error"}}, status=529
    )
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status == 529
        data = await resp.json()
        assert data["error"]["type"] == "overloaded_error"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_health_reports_credentials_when_no_auth_required():
    client, _ = await _make_client({})
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["ok"] is True
        assert data["credentials"]["subscriptionType"] == "max"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_health_hides_credentials_from_unauthenticated_caller(monkeypatch):
    monkeypatch.setattr(srv, "AUTH_TOKEN", "s3cret")
    client, _ = await _make_client({})
    try:
        resp = await client.get("/health")  # no Authorization header
        data = await resp.json()
        assert data == {"ok": True}  # no path, no subscription, no expiry leaked
        # with the right token, details come back
        resp2 = await client.get("/health", headers={"Authorization": "Bearer s3cret"})
        data2 = await resp2.json()
        assert data2["credentials"]["subscriptionType"] == "max"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stream_midstream_error_event_surfaces_error_not_clean_done():
    sse_lines = [
        'data: {"type":"message_start","message":{}}',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}',
        'data: {"type":"error","error":{"type":"overloaded_error","message":"boom"}}',
    ]
    http = _StreamingClient(_StreamResp(200, sse_lines))
    client, _ = await _make_client({}, http=http)
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        body = await resp.text()
        assert "overloaded_error" in body  # the error is surfaced
        assert '"error"' in body
        assert "[DONE]" in body  # stream still terminates cleanly for the client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stream_truncated_without_message_stop_gets_finish_chunk():
    sse_lines = [
        'data: {"type":"message_start","message":{}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
        # connection ends here — no message_delta, no message_stop
    ]
    http = _StreamingClient(_StreamResp(200, sse_lines))
    client, _ = await _make_client({}, http=http)
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        body = await resp.text()
        assert '"finish_reason": "stop"' in body  # terminal chunk emitted
        assert "[DONE]" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_messages_passthrough_non_200_is_forwarded_not_streamed():
    http = _StreamingClient(
        _StreamResp(401, ['{"type":"error","error":{"type":"authentication_error"}}'])
    )
    client, _ = await _make_client({}, http=http)
    try:
        resp = await client.post(
            "/v1/messages",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        # status is honest (stream never started), body carries the error
        assert resp.status == 401
        data = await resp.json()
        assert data["error"]["type"] == "authentication_error"
    finally:
        await client.close()
