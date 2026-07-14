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
| `ZT_ACCOUNTS_JSON`         | *(unset)*          | Multi-account failover config, inline JSON (see below). |
| `ZT_ACCOUNTS_FILE`         | *(unset)*          | Path to a `0600` file holding the same JSON array (preferred — keeps tokens out of the systemd env). |
| `ZT_BACKUP_CREDENTIALS`    | *(unset)*          | Simple: `os.pathsep`-separated backup credential files. |
| `ZT_BACKUP_TOKENS`         | *(unset)*          | Simple: comma-separated backup static tokens.        |
| `ZT_USAGE_COOLDOWN_S`      | `900`              | Seconds to rest an account after "out of extra usage". |

## Multi-account failover

The proxy holds an ordered **pool** of accounts. Each request uses the first
account not in cooldown; when an account returns *"out of extra usage"* (HTTP
400) or a 429 rate limit, it is cooled down and the request transparently
retries on the next account. Each account refreshes its own token independently,
and accounts may be **different providers** (e.g. Claude subscriptions plus a
Kimi Code subscription — all Anthropic-Messages-compatible).

Configure with `ZT_ACCOUNTS_JSON` (a JSON array, tried in order):

```json
[
  {"name": "claude1", "provider": "anthropic"},
  {"name": "claude2", "provider": "anthropic",
   "credentials_path": "~/.claude/.credentials.account2.json"},
  {"name": "kimi", "provider": "kimi", "token": "sk-...",
   "model": "kimi-for-coding"}
]
```

Per-account fields: `name`; `provider` (`anthropic` | `kimi` | `generic`); one
of `credentials_path` (an OAuth file that auto-refreshes) or `token` (a static
token); optional `base_url`, `model` (override — required for a non-Claude
provider whose model ids differ), `send_identity`, `betas`. Omit both
`credentials_path` and `token` on the primary to use the default Claude
credentials file plus the env token.

Provider presets: `anthropic` → `api.anthropic.com` with the Claude Code
identity block + OAuth betas; `kimi` → `api.kimi.com/coding` (a plain Kimi
Code API key, no OAuth beta), no identity
block; `generic` → `api.anthropic.com`, no identity, no betas.

`GET /health` (authenticated) lists every account with its provider, endpoint,
and cooldown state.

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
