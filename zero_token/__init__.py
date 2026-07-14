"""zero_token — use a Claude CLI *account* token instead of an Anthropic API key.

This package is a small, self-contained OpenAI-/Anthropic-compatible HTTP proxy
that authenticates upstream to ``api.anthropic.com`` with the OAuth *account*
token minted by the Claude CLI (``claude /login`` or ``claude setup-token``),
so no paid ``ANTHROPIC_API_KEY`` is required — the request is billed against the
user's Claude Pro/Max subscription instead.

It exists so any OpenAI-compatible client (hermes-agent's ``provider: custom``
backend, or any other tool) can consume the Claude subscription through one
local endpoint, without embedding OAuth/refresh logic of its own.

Design goals that distinguish this from the older ``claude_code_proxy.py``:

* No ``claude -p`` subprocess. Each request is a direct HTTPS call to the
  Anthropic Messages API, so there is no per-call ``--max-budget-usd`` ceiling
  to trip (the ``error_max_budget_usd`` wedge) and no cold-spawn latency.
* Real OAuth token refresh with atomic write-back to
  ``~/.claude/.credentials.json`` (kept in sync with the ``claude`` CLI itself).
* Stateless with respect to conversation history — the calling agent owns the
  context, so nothing snowballs on the proxy side.

Modules:
    credentials      — ClaudeOAuthStore: read/refresh/persist the account token.
    anthropic_oauth  — headers, mandatory Claude Code identity, version detect.
    translate        — OpenAI chat/completions <-> Anthropic Messages mapping.
    server           — aiohttp app exposing /v1/chat/completions, /v1/messages,
                       /v1/models, /health.
    __main__         — ``python -m zero_token`` entrypoint.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
