"""aiohttp proxy exposing the Claude subscription via OpenAI/Anthropic APIs.

Endpoints (all bind to 127.0.0.1 by default):

* ``POST /v1/chat/completions`` — OpenAI Chat Completions. Streaming and
  non-streaming. This is what hermes-agent's ``provider: custom`` backend uses.
* ``POST /v1/messages``          — Anthropic Messages passthrough (OAuth headers
  and the Claude Code identity block are injected for you). Streaming and not.
* ``GET  /v1/models``            — model list (proxied from Anthropic, with a
  static fallback).
* ``GET  /health``               — liveness plus non-secret token metadata.

Auth: when ``CLAUDE_PROXY_TOKEN`` (a.k.a. ``ZT_AUTH_TOKEN``) is set, every
request must carry ``Authorization: Bearer <token>``. This proxy can spend the
user's Claude subscription, so treat that token as the security boundary; the
127.0.0.1 bind is only defence in depth.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import signal
import time
import uuid
from typing import Any

import httpx
from aiohttp import web

from . import anthropic_oauth as ao
from .credentials import (
    CredentialPool,
    CredentialsError,
    is_auth_error,
    is_usage_limited,
)
from . import translate as tr

LOG = logging.getLogger("zero-token.server")

# --- configuration --------------------------------------------------------

LISTEN_HOST = os.environ.get("ZT_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("ZT_PORT", "3031"))
DEFAULT_MODEL = os.environ.get("ZT_DEFAULT_MODEL", "claude-opus-4-8").strip()
REQUEST_TIMEOUT_S = float(os.environ.get("ZT_REQUEST_TIMEOUT", "300"))
# Optional hard cap on max_tokens forwarded upstream (0/empty = no cap).
try:
    MAX_TOKENS_CAP = int(os.environ.get("ZT_MAX_TOKENS_CAP", "0"))
except ValueError:
    MAX_TOKENS_CAP = 0
# Optional JSON object remapping client model names to Anthropic model ids.
try:
    MODEL_MAP: dict[str, str] = json.loads(os.environ.get("ZT_MODEL_MAP", "") or "{}")
except json.JSONDecodeError:
    MODEL_MAP = {}

AUTH_TOKEN = (
    os.environ.get("ZT_AUTH_TOKEN") or os.environ.get("CLAUDE_PROXY_TOKEN") or ""
).strip()

# Typed application keys (aiohttp best practice — avoids NotAppKeyWarning).
POOL_KEY: web.AppKey[CredentialPool] = web.AppKey("pool", CredentialPool)
HTTP_KEY: web.AppKey[httpx.AsyncClient] = web.AppKey("http", httpx.AsyncClient)

# Cooldowns for failover (seconds).
_CREDS_COOLDOWN_S = 300.0  # account whose credentials can't be resolved
_NET_COOLDOWN_S = 30.0  # transient network error reaching an account
_RATELIMIT_COOLDOWN_S = 60.0  # 429 without a Retry-After
_AUTH_COOLDOWN_S = 300.0  # upstream 401/403 — token bad, rest it and rotate

_STATIC_MODELS = (
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
)


def _normalise_model(model: str | None) -> str:
    if not model:
        return DEFAULT_MODEL
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    # Strip a leading vendor prefix like "anthropic/" or "claude/".
    if "/" in model:
        model = model.split("/", 1)[1]
    return model


# --- auth -----------------------------------------------------------------


def _auth_ok(request: web.Request) -> bool:
    """True when the request presents the correct bearer token (or none is set)."""
    if not AUTH_TOKEN:
        return True
    presented = request.headers.get("Authorization", "")
    try:
        return hmac.compare_digest(presented, f"Bearer {AUTH_TOKEN}")
    except TypeError:
        # Non-ASCII header — compare_digest rejects it; treat as a failed auth.
        return False


def _auth_failed(request: web.Request) -> web.Response | None:
    if _auth_ok(request):
        return None
    return web.json_response(
        {"error": {"message": "unauthorized", "type": "authentication_error"}},
        status=401,
    )


def _pool(request: web.Request) -> CredentialPool:
    return request.app[POOL_KEY]


def _http(request: web.Request) -> httpx.AsyncClient:
    return request.app[HTTP_KEY]


def _retry_after(resp: httpx.Response) -> float | None:
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    return None


# --- Anthropic call helpers ----------------------------------------------


def _prepare_for_account(
    body: dict[str, Any], token: str, account: Any
) -> tuple[dict[str, Any], dict[str, str], str]:
    """Finalise the request body, headers, and messages URL for one account.

    Applies the account's model override, endpoint, identity-spoof policy, and
    beta flags.
    """
    body = dict(body)
    body["model"] = account.model or _normalise_model(body.get("model"))
    if account.send_identity:
        body["system"] = ao.ensure_claude_code_system(body.get("system"))
    if MAX_TOKENS_CAP and int(body.get("max_tokens") or 0) > MAX_TOKENS_CAP:
        body["max_tokens"] = MAX_TOKENS_CAP
    headers = ao.build_headers(
        token, base_betas=account.betas, extra_betas=ao.body_required_betas(body)
    )
    url = f"{account.base_url}{ao.ANTHROPIC_MESSAGES_PATH}"
    return body, headers, url


async def _read_upstream_error(resp: httpx.Response) -> tuple[int, dict[str, Any]]:
    """Read an upstream error body and return (status, OpenAI-ish error dict)."""
    raw = await resp.aread()
    detail = raw.decode(errors="replace")[:1000]
    LOG.error("anthropic upstream %d: %s", resp.status_code, detail)
    try:
        payload = json.loads(detail)
        if not isinstance(payload, dict) or "error" not in payload:
            payload = {"error": {"message": detail, "type": "upstream_error"}}
    except json.JSONDecodeError:
        payload = {"error": {"message": detail, "type": "upstream_error"}}
    return resp.status_code, payload


async def _iter_sse_events(resp: httpx.Response):
    """Yield parsed JSON data objects from an Anthropic SSE stream."""
    async for line in resp.aiter_lines():
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


# --- /v1/chat/completions -------------------------------------------------


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    denied = _auth_failed(request)
    if denied is not None:
        return denied
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001 - malformed client input
        return web.json_response(
            {
                "error": {
                    "message": f"invalid JSON: {exc}",
                    "type": "invalid_request_error",
                }
            },
            status=400,
        )

    messages = payload.get("messages")
    if not messages:
        return web.json_response(
            {
                "error": {
                    "message": "messages required",
                    "type": "invalid_request_error",
                }
            },
            status=400,
        )

    stream = bool(payload.get("stream"))
    model = _normalise_model(payload.get("model"))
    anthropic_body = tr.openai_to_anthropic_body(payload)
    rid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not stream:
        acct, data, status, err = await _nonstream_with_failover(
            request, anthropic_body
        )
        if data is None:
            return web.json_response(err, status=status)
        out = tr.anthropic_to_openai_response(
            data, response_id=rid, created=created, model=model
        )
        LOG.info(
            "chat.completions ok via %s: model=%s finish=%s out_tok=%s",
            acct.name if acct else "?",
            model,
            out["choices"][0]["finish_reason"],
            out["usage"]["completion_tokens"],
        )
        return web.json_response(out)

    # Streaming: pick the first non-limited account whose stream opens 200,
    # failing over on usage limits, THEN prepare the SSE response.
    stream_cm, resp, chosen, status, err = await _open_stream_with_failover(
        request, anthropic_body
    )
    if resp is None:
        return web.json_response(err, status=status)

    sse = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await sse.prepare(request)
    translator = tr.AnthropicStreamTranslator(rid, created, model)
    try:
        sent_finish = False
        async for event in _iter_sse_events(resp):
            etype = event.get("type")
            if etype == "error":
                # Anthropic can emit an in-band error event (e.g.
                # overloaded_error) after a 200 and then close. Surface it
                # instead of silently ending with a clean [DONE].
                err = event.get("error") or {"message": "upstream stream error"}
                LOG.error("anthropic mid-stream error: %s", err)
                await sse.write(f"data: {json.dumps({'error': err})}\n\n".encode())
                await sse.write(
                    f"data: {json.dumps(tr._chunk(rid, created, model, {}, 'stop'))}\n\n".encode()
                )
                await sse.write(b"data: [DONE]\n\n")
                await sse.write_eof()
                return sse
            if etype == "ping":
                # Keepalive: reset the client's idle timer while we wait for
                # the first real token.
                await sse.write(b": ping\n\n")
                continue
            for chunk in translator.handle(event):
                if chunk["choices"][0]["finish_reason"] is not None:
                    sent_finish = True
                await sse.write(f"data: {json.dumps(chunk)}\n\n".encode())
        # If the upstream stream ended without a terminal chunk (no message_stop,
        # e.g. a truncated connection), emit one so a truncated response is not
        # presented to the client as a clean completion.
        if not sent_finish:
            await sse.write(
                f"data: {json.dumps(tr._chunk(rid, created, model, {}, 'stop'))}\n\n".encode()
            )
        await sse.write(b"data: [DONE]\n\n")
        await sse.write_eof()
    except (httpx.HTTPError, ConnectionResetError) as exc:
        LOG.error("streaming upstream failed: %s", exc)
        try:
            errchunk = tr._chunk(
                rid, created, model, {"content": f"\n\n[proxy error: {exc}]"}, "stop"
            )
            await sse.write(f"data: {json.dumps(errchunk)}\n\n".encode())
            await sse.write(b"data: [DONE]\n\n")
            await sse.write_eof()
        except Exception:  # noqa: BLE001 - client already gone
            pass
    finally:
        await stream_cm.__aexit__(None, None, None)
    return sse


# --- failover call helpers ------------------------------------------------


async def _nonstream_with_failover(
    request: web.Request, anthropic_body: dict[str, Any]
) -> tuple[Any, dict[str, Any] | None, int, dict[str, Any] | None]:
    """Try accounts in order for a non-streaming call.

    Returns ``(account, data, status, error)``. On success ``data`` is the
    Anthropic response dict; otherwise it is None and ``(status, error)`` is the
    OpenAI-style error to return.
    """
    pool = _pool(request)
    http = _http(request)
    tried: set[str] = set()
    last_status = 503
    last_err: dict[str, Any] = {
        "error": {
            "message": "all accounts are cooling down; try again shortly",
            "type": "upstream_error",
        }
    }
    for _ in range(pool.size):
        acct = pool.active()
        if acct.name in tried:
            break
        tried.add(acct.name)
        try:
            token = await asyncio.to_thread(acct.store.access_token)
        except CredentialsError as exc:
            pool.mark_limited(
                acct, cooldown_s=_CREDS_COOLDOWN_S, reason=f"creds: {exc}"
            )
            last_status, last_err = (
                401,
                {"error": {"message": str(exc), "type": "authentication_error"}},
            )
            continue
        body, headers, url = _prepare_for_account(anthropic_body, token, acct)
        body["stream"] = False
        try:
            resp = await http.post(
                url, json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
            )
        except httpx.HTTPError as exc:
            pool.mark_limited(acct, cooldown_s=_NET_COOLDOWN_S, reason="network")
            last_status, last_err = (
                502,
                {
                    "error": {
                        "message": f"upstream request failed: {exc}",
                        "type": "upstream_error",
                    }
                },
            )
            continue
        if resp.status_code == 200:
            return acct, resp.json(), 200, None
        status, payload = await _read_upstream_error(resp)
        if is_usage_limited(status, payload):
            cd = (
                (_retry_after(resp) or _RATELIMIT_COOLDOWN_S) if status == 429 else None
            )
            pool.mark_limited(
                acct,
                cooldown_s=cd,
                reason=str((payload.get("error") or {}).get("message", ""))[:100],
            )
            last_status, last_err = status, payload
            continue
        if is_auth_error(status):
            # Account-specific bad token — rest it and try the next account.
            pool.mark_limited(acct, cooldown_s=_AUTH_COOLDOWN_S, reason=f"auth {status}")
            last_status, last_err = status, payload
            continue
        # Any other error is account-independent — return it, don't burn others.
        return acct, None, status, payload
    return None, None, last_status, last_err


async def _open_stream_with_failover(
    request: web.Request, anthropic_body: dict[str, Any]
) -> tuple[Any, httpx.Response | None, Any, int, dict[str, Any] | None]:
    """Open the upstream stream on the first non-limited account.

    Returns ``(stream_cm, resp, account, status, error)``. On success ``resp``
    is the entered httpx streaming response (caller must ``__aexit__`` the
    returned ``stream_cm``); otherwise ``resp`` is None and ``(status, error)``
    is the error to return.
    """
    pool = _pool(request)
    http = _http(request)
    tried: set[str] = set()
    last_status = 503
    last_err: dict[str, Any] = {
        "error": {
            "message": "all accounts are cooling down; try again shortly",
            "type": "upstream_error",
        }
    }
    for _ in range(pool.size):
        acct = pool.active()
        if acct.name in tried:
            break
        tried.add(acct.name)
        try:
            token = await asyncio.to_thread(acct.store.access_token)
        except CredentialsError as exc:
            pool.mark_limited(
                acct, cooldown_s=_CREDS_COOLDOWN_S, reason=f"creds: {exc}"
            )
            last_status, last_err = (
                401,
                {"error": {"message": str(exc), "type": "authentication_error"}},
            )
            continue
        body, headers, url = _prepare_for_account(anthropic_body, token, acct)
        body["stream"] = True
        cm = http.stream(
            "POST", url, json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
        )
        try:
            resp = await cm.__aenter__()
        except httpx.HTTPError as exc:
            pool.mark_limited(acct, cooldown_s=_NET_COOLDOWN_S, reason="network")
            last_status, last_err = (
                502,
                {
                    "error": {
                        "message": f"upstream request failed: {exc}",
                        "type": "upstream_error",
                    }
                },
            )
            continue
        if resp.status_code == 200:
            return cm, resp, acct, 200, None
        status, payload = await _read_upstream_error(resp)
        await cm.__aexit__(None, None, None)
        if is_usage_limited(status, payload):
            cd = (
                (_retry_after(resp) or _RATELIMIT_COOLDOWN_S) if status == 429 else None
            )
            pool.mark_limited(
                acct,
                cooldown_s=cd,
                reason=str((payload.get("error") or {}).get("message", ""))[:100],
            )
            last_status, last_err = status, payload
            continue
        if is_auth_error(status):
            pool.mark_limited(acct, cooldown_s=_AUTH_COOLDOWN_S, reason=f"auth {status}")
            last_status, last_err = status, payload
            continue
        return cm, None, acct, status, payload
    return None, None, None, last_status, last_err


# --- /v1/messages (Anthropic passthrough) --------------------------------


async def handle_messages(request: web.Request) -> web.StreamResponse:
    denied = _auth_failed(request)
    if denied is not None:
        return denied
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response(
            {
                "error": {
                    "message": f"invalid JSON: {exc}",
                    "type": "invalid_request_error",
                }
            },
            status=400,
        )
    stream = bool(payload.get("stream"))

    if not stream:
        _acct, data, status, err = await _nonstream_with_failover(request, payload)
        if data is None:
            return web.json_response(err, status=status)
        return web.json_response(data)

    stream_cm, resp, _acct, status, err = await _open_stream_with_failover(
        request, payload
    )
    if resp is None:
        return web.json_response(err, status=status)
    sse = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await sse.prepare(request)
    try:
        async for raw in resp.aiter_raw():
            await sse.write(raw)
        await sse.write_eof()
    except (httpx.HTTPError, ConnectionResetError) as exc:
        LOG.error("messages stream failed: %s", exc)
        try:
            await sse.write_eof()
        except Exception:  # noqa: BLE001 - client already gone
            pass
    finally:
        await stream_cm.__aexit__(None, None, None)
    return sse


# --- /v1/models -----------------------------------------------------------


async def handle_models(request: web.Request) -> web.Response:
    denied = _auth_failed(request)
    if denied is not None:
        return denied
    try:
        acct = _pool(request).active()
        token = await asyncio.to_thread(acct.store.access_token)
        headers = ao.build_headers(token, base_betas=acct.betas)
        resp = await _http(request).get(
            f"{acct.base_url}{ao.ANTHROPIC_MODELS_PATH}",
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            upstream = resp.json()
            models = [
                {
                    "id": m.get("id"),
                    "object": "model",
                    "created": 0,
                    "owned_by": "anthropic",
                }
                for m in upstream.get("data", [])
                if m.get("id")
            ]
            if models:
                return web.json_response({"object": "list", "data": models})
    except (CredentialsError, httpx.HTTPError) as exc:
        LOG.warning("model list fetch failed, using static list: %s", exc)
    return web.json_response({
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "anthropic"}
            for m in _STATIC_MODELS
        ],
    })


# --- /health --------------------------------------------------------------


async def handle_health(request: web.Request) -> web.Response:
    # Liveness is public; credential metadata (path with OS username,
    # subscription type, expiry) is only exposed to an authenticated caller.
    info: dict[str, Any] = {"ok": True}
    if _auth_ok(request):
        info["auth_required"] = bool(AUTH_TOKEN)
        info["default_model"] = DEFAULT_MODEL
        try:
            info["accounts"] = _pool(request).describe()
        except Exception as exc:  # noqa: BLE001
            info["ok"] = False
            info["accounts_error"] = str(exc)
    return web.json_response(info)


# --- app wiring -----------------------------------------------------------


# Allow large request bodies (base64 images, tool-heavy histories). aiohttp's
# default client_max_size is 1 MiB, which 413s such requests.
MAX_BODY_SIZE = int(os.environ.get("ZT_MAX_BODY_BYTES", str(32 * 1024 * 1024)))


async def _on_startup(app: web.Application) -> None:
    app[HTTP_KEY] = httpx.AsyncClient()
    # Warm the cached `claude --version` lookup off the request path so the
    # first request's header build doesn't block on a subprocess.
    await asyncio.to_thread(ao.detect_claude_code_version)


async def _on_cleanup(app: web.Application) -> None:
    await app[HTTP_KEY].aclose()


def build_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BODY_SIZE)
    app[POOL_KEY] = CredentialPool.from_env()
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    return app


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # Ignore SIGHUP so a controlling-terminal hangup can't take the proxy down
    # (it typically runs headless under systemd).
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except (ValueError, AttributeError):
        pass

    LOG.info("zero-token proxy starting on %s:%d", LISTEN_HOST, LISTEN_PORT)
    _loopback = LISTEN_HOST in ("127.0.0.1", "::1", "localhost")
    if AUTH_TOKEN:
        LOG.info("bearer auth ENABLED")
    elif not _loopback:
        # Refuse to expose an unauthenticated, subscription-spending endpoint to
        # a non-loopback interface. Set CLAUDE_PROXY_TOKEN or bind to 127.0.0.1.
        raise SystemExit(
            f"refusing to bind {LISTEN_HOST}:{LISTEN_PORT} without auth — set "
            "CLAUDE_PROXY_TOKEN (recommended) or ZT_HOST=127.0.0.1. An "
            "unauthenticated non-loopback bind lets anyone on the network spend "
            "this machine's Claude subscription."
        )
    else:
        LOG.warning(
            "bearer auth DISABLED (ZT_AUTH_TOKEN/CLAUDE_PROXY_TOKEN unset) — any local "
            "process reaching %s:%d can spend this machine's Claude subscription. "
            "Set a token.",
            LISTEN_HOST,
            LISTEN_PORT,
        )
    app = build_app()
    try:
        accounts = app[POOL_KEY].describe()
        LOG.info("credential pool: %d account(s)", len(accounts))
        for a in accounts:
            LOG.info(
                "  - %s [%s] %s (refreshable=%s, sub=%s)",
                a.get("name"),
                a.get("provider"),
                a.get("source"),
                a.get("refreshable"),
                a.get("subscriptionType"),
            )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("credential preflight: %s", exc)
    web.run_app(app, host=LISTEN_HOST, port=LISTEN_PORT, print=None)


if __name__ == "__main__":
    main()
