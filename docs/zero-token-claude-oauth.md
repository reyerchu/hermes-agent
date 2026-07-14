# Zero-token: run hermes on a Claude account token (no API key)

`zero_token` is a standalone OpenAI-/Anthropic-compatible proxy that
authenticates to Anthropic with the **account** OAuth token from the Claude CLI
(`claude /login` / `claude setup-token`) instead of a paid `ANTHROPIC_API_KEY`.
Point hermes-agent's `provider: custom` backend at it and every model call is
billed against your Claude Pro/Max subscription.

Full reference: [`zero_token/README.md`](../zero_token/README.md).

## When to use this

- You have a Claude Pro/Max subscription and want hermes to use it directly
  rather than paying per-token for API access.
- You are migrating off the older `claude_code_proxy.py` (the `claude -p`
  subprocess proxy) that wedged with `error_max_budget_usd` as its single
  resumed session grew. This proxy is stateless and calls the API directly, so
  that failure mode is gone.

hermes also has a **native** anthropic OAuth path (set `model.provider:
anthropic` and let it read `~/.claude/.credentials.json`). Use `zero_token`
instead when you specifically want an OpenAI-compatible endpoint — e.g. to keep
hermes on `provider: custom`, to share one subscription endpoint with other
OpenAI clients, or to keep the proxy on its own service boundary.

## Setup

1. Authenticate the Claude CLI once (writes `~/.claude/.credentials.json`):

   ```bash
   claude /login        # or: claude setup-token
   ```

2. Install the extra and start the proxy (or use the systemd unit):

   ```bash
   cd ~/hermes-agent
   uv pip install -e '.[zero-token]'
   export CLAUDE_PROXY_TOKEN=$(openssl rand -hex 32)
   python -m zero_token
   # persistent: packaging/systemd/hermes-zero-token.service
   ```

3. Point hermes at the proxy — merge into `~/.hermes/config.yaml`
   (see [`zero_token/config.example.yaml`](../zero_token/config.example.yaml)):

   ```yaml
   model:
     provider: custom
     base_url: http://127.0.0.1:3031/v1
     api_key: <same value as CLAUDE_PROXY_TOKEN>
     default: claude-opus-4-8
   ```

4. Restart the gateway:

   ```bash
   systemctl --user restart hermes-gateway    # or `hermes gateway run --replace`
   ```

## Verify

```bash
curl -s http://127.0.0.1:3031/health -H "Authorization: Bearer $CLAUDE_PROXY_TOKEN"
# {"ok": true, ..., "credentials": {"subscriptionType": "max", "expiresInSeconds": ...}}
```

## Architecture

```
Telegram ─▶ hermes-gateway ──OpenAI /v1/chat/completions──▶ zero_token proxy
                                                              │  (translate +
                                                              │   OAuth headers +
                                                              │   Claude Code identity)
                                                              ▼
                                                     api.anthropic.com/v1/messages
                                                     (Authorization: Bearer sk-ant-oat…)
```

hermes owns the conversation, memory, and tool loop; the proxy is a stateless
translator + authenticator. Token refresh (with atomic write-back to the
credentials file) happens just-in-time inside the proxy.

## Troubleshooting

- **`401 unauthorized`** — the request's bearer token doesn't match
  `CLAUDE_PROXY_TOKEN`. Make `model.api_key` equal to it.
- **`401` / "OAuth token refresh failed"** — the refresh token expired or was
  invalidated; re-run `claude /login` (or `claude setup-token`).
- **`credentials not found`** — the CLI hasn't been logged in on this machine,
  or `CLAUDE_CREDENTIALS_PATH` points elsewhere. Run `claude /login`, or set
  `ANTHROPIC_TOKEN` to a `setup-token` value (static, no refresh).
- **Policy note** — subscription tokens are supported for Claude Code; use from
  other clients is technical compatibility, not a policy guarantee.
