"""Anthropic Messages API request construction for OAuth *account* tokens.

Authenticating to ``api.anthropic.com`` with a Claude subscription OAuth token
(rather than a paid API key) requires a specific request shape that all three
reference implementations agree on:

1. ``Authorization: Bearer <accessToken>`` — NOT ``x-api-key``.
2. ``anthropic-beta: claude-code-20250219,oauth-2025-04-20`` — the OAuth beta
   flags. ``x-api-key`` must be absent.
3. ``anthropic-version: 2023-06-01``.
4. A Claude-Code-shaped ``User-Agent`` (``claude-cli/<version> (external, cli)``)
   and ``x-app: cli``.
5. The FIRST system block MUST be exactly the Claude Code identity string
   ``"You are Claude Code, Anthropic's official CLI for Claude."`` — the API
   rejects subscription tokens otherwise. The caller's real system prompt is
   appended as a second block.

This module builds the headers and enforces the identity block; it does not do
any OpenAI translation (see ``translate``) or transport (see ``server``).
"""

from __future__ import annotations

import functools
import logging
import re
import subprocess
from typing import Any

LOG = logging.getLogger("zero-token.anthropic")

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"
ANTHROPIC_MODELS_PATH = "/v1/models"
ANTHROPIC_VERSION = "2023-06-01"

# OAuth beta flags. claude-code-20250219 marks the request as Claude Code;
# oauth-2025-04-20 enables subscription-token auth. context-1m is deliberately
# NOT included here — Anthropic rejects it under OAuth.
OAUTH_BETAS = ("claude-code-20250219", "oauth-2025-04-20")

# Mandatory identity block. Must be the first system block verbatim.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

# Fallback CLI version if `claude --version` can't be run.
_FALLBACK_CLAUDE_VERSION = "2.1.208"


@functools.lru_cache(maxsize=1)
def detect_claude_code_version() -> str:
    """Return the installed Claude CLI version string, e.g. ``2.1.208``.

    Cached: the binary doesn't change under a running proxy. Falls back to a
    recent known-good version if the CLI is absent or unparseable.
    """
    try:
        out = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return _FALLBACK_CLAUDE_VERSION
    if out.returncode != 0:
        return _FALLBACK_CLAUDE_VERSION
    m = re.search(r"(\d+\.\d+\.\d+)", out.stdout or "")
    return m.group(1) if m else _FALLBACK_CLAUDE_VERSION


def user_agent() -> str:
    return f"claude-cli/{detect_claude_code_version()} (external, cli)"


def build_headers(
    access_token: str, extra_betas: tuple[str, ...] = ()
) -> dict[str, str]:
    """Build the request headers for an OAuth Messages call.

    ``extra_betas`` are merged (de-duplicated, order-preserving) into the
    ``anthropic-beta`` header — e.g. ``context-management-2025-06-27`` when the
    body carries a ``context_management`` field.
    """
    betas: list[str] = []
    for b in (*OAUTH_BETAS, *extra_betas):
        if b and b not in betas:
            betas.append(b)
    return {
        "Authorization": f"Bearer {access_token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": ",".join(betas),
        "content-type": "application/json",
        "accept": "application/json",
        "user-agent": user_agent(),
        "x-app": "cli",
        "anthropic-dangerous-direct-browser-access": "true",
    }


def _identity_block() -> dict[str, Any]:
    return {"type": "text", "text": CLAUDE_CODE_IDENTITY}


def ensure_claude_code_system(system: Any) -> list[dict[str, Any]]:
    """Return a system value whose first block is the Claude Code identity.

    Idempotent: re-processing an already-identity'd system does not double
    insert. Accepts ``None``, a plain string, or a list of Anthropic system
    blocks, and always returns a list of blocks.
    """
    # Normalise to a list of blocks.
    if system is None or system == "":
        blocks: list[dict[str, Any]] = []
    elif isinstance(system, str):
        blocks = [{"type": "text", "text": system}]
    elif isinstance(system, list):
        blocks = [
            dict(b) if isinstance(b, dict) else {"type": "text", "text": str(b)}
            for b in system
        ]
    else:
        blocks = [{"type": "text", "text": str(system)}]

    if (
        blocks
        and blocks[0].get("type") == "text"
        and blocks[0].get("text") == CLAUDE_CODE_IDENTITY
    ):
        return blocks
    return [_identity_block(), *blocks]


def body_required_betas(body: dict[str, Any]) -> tuple[str, ...]:
    """Betas that must be re-added because the body carries a beta-gated field."""
    extra: list[str] = []
    if body.get("context_management") is not None:
        extra.append("context-management-2025-06-27")
    return tuple(extra)
