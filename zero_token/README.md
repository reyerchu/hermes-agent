# zero_token — Claude account-token proxy

A small, self-contained **OpenAI-/Anthropic-compatible HTTP proxy** that
authenticates upstream to `api.anthropic.com` with the OAuth **account** token
minted by the Claude CLI (`claude /login` or `claude setup-token`) — so no paid
`ANTHROPIC_API_KEY` is required. Requests bill against your Claude Pro/Max
subscription.

It lets any OpenAI-compatible client — hermes-agent's `provider: custom`
backend, or any other tool — consume the Claude subscription through one local
endpoint, without embedding OAuth/refresh logic of its own.

## Why (vs. the old `claude -p` proxy)

The legacy `claude_code_proxy.py` spawned `claude -p` as a subprocess for every
message and pinned a single resumed Claude Code session. As that session's
context grew, the per-call `--max-budget-usd` ceiling was consumed just loading
history, wedging the bot with `error_max_budget_usd`.

This proxy instead makes a **direct HTTPS call** to the Anthropic Messages API
per request:

- No subprocess, no `--max-budget-usd` ceiling to trip, no cold-spawn latency.
- Real OAuth **token refresh** with atomic write-back to
  `~/.claude/.credentials.json` (kept in sync with the `claude` CLI).
- **Stateless** — the calling agent owns the conversation, so nothing snowballs
  on the proxy side. Each request is one model turn.

## Run

```bash
uv pip install -e '.[zero-token]'        # aiohttp; httpx is already core
CLAUDE_PROXY_TOKEN=$(openssl rand -hex 32) python -m zero_token
```

Then point hermes at it (see `config.example.yaml`) or curl it directly:

```bash
curl -s http://127.0.0.1:3031/v1/chat/completions \
  -H "Authorization: Bearer $CLAUDE_PROXY_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}'
```

For a persistent deployment use `packaging/systemd/hermes-zero-token.service`.

## Endpoints

| Method | Path                    | Purpose                                        |
|--------|-------------------------|------------------------------------------------|
| POST   | `/v1/chat/completions`  | OpenAI Chat Completions (stream + non-stream). |
| POST   | `/v1/messages`          | Anthropic Messages passthrough (headers added).|
| GET    | `/v1/models`            | Model list (from Anthropic, static fallback).  |
| GET    | `/health`               | Liveness + non-secret token metadata.          |

## Configuration (environment)

| Var                        | Default            | Meaning                                              |
|----------------------------|--------------------|------------------------------------------------------|
| `ZT_HOST`                  | `127.0.0.1`        | Bind address.                                        |
| `ZT_PORT`                  | `3031`             | Bind port (matches the legacy proxy default).        |
| `ZT_DEFAULT_MODEL`         | `claude-opus-4-8`  | Model when the client omits one.                     |
| `CLAUDE_PROXY_TOKEN` / `ZT_AUTH_TOKEN` | *(unset)* | Bearer secret required on every request. **Set it.** |
| `ZT_REQUEST_TIMEOUT`       | `300`              | Per-request upstream timeout (seconds).              |
| `ZT_MAX_TOKENS_CAP`        | `0` (off)          | Optional hard cap on forwarded `max_tokens`.         |
| `ZT_MODEL_MAP`             | `{}`               | JSON remap of client model names → Anthropic ids.    |
| `CLAUDE_CREDENTIALS_PATH`  | `~/.claude/.credentials.json` | Credentials file location.                |
| `ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` / `ANTHROPIC_OAUTH_TOKEN` | *(unset)* | Static setup-token; disables refresh. |

## How the OAuth request is shaped

Subscription tokens require a specific request shape (all three reference
implementations agree — see below):

- `Authorization: Bearer <accessToken>` (never `x-api-key`).
- `anthropic-beta: claude-code-20250219,oauth-2025-04-20` (never `context-1m`).
- `anthropic-version: 2023-06-01`, `user-agent: claude-cli/<version> (external, cli)`, `x-app: cli`.
- The **first** `system` block must be exactly
  `"You are Claude Code, Anthropic's official CLI for Claude."` — the caller's
  real system prompt is appended after it.

Token refresh POSTs to `https://platform.claude.com/v1/oauth/token` with the
public Claude Code client id and a `claude-cli/...` User-Agent (the endpoint is
behind Cloudflare and 403s the default UA); rotated tokens are written back
atomically with `0600` perms.

## Modules

- `credentials.py` — `ClaudeOAuthStore`: read / refresh / persist the token.
- `anthropic_oauth.py` — headers, the mandatory identity block, version detect.
- `translate.py` — OpenAI ↔ Anthropic messages, tools, responses, streaming.
- `server.py` — the aiohttp app; `__main__.py` — `python -m zero_token`.

## Provenance

Ported for hermes-agent from three references that implement the same
account-token mechanics:

- `reyerchu/any-llm-in-claude` — file read + refresh-token grant + atomic
  write-back; mandatory identity block; `platform.claude.com` refresh endpoint.
- `linuxhsj/openclaw-zero-token` — Bearer + OAuth beta headers, Claude Code
  identity, credential file/keychain reading.
- `tashfeenahmed/freellmapi` — provider/auth abstraction and OpenAI↔upstream
  translation patterns.

## Caveat

Anthropic permits subscription tokens for Claude Code; using them from another
client is a **technical** compatibility, not a policy guarantee. Expect the
occasional `401` ("OAuth token refresh failed") after a forced re-auth — re-run
`claude /login` or `claude setup-token`.
