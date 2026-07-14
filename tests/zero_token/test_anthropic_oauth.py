"""Tests for header construction and the mandatory Claude Code identity block."""

from __future__ import annotations

from zero_token import anthropic_oauth as ao


def test_build_headers_uses_bearer_not_api_key():
    h = ao.build_headers("sk-ant-oat01-TEST")
    assert h["Authorization"] == "Bearer sk-ant-oat01-TEST"
    assert "x-api-key" not in {k.lower() for k in h}
    assert h["anthropic-version"] == ao.ANTHROPIC_VERSION
    assert h["x-app"] == "cli"
    assert h["user-agent"].startswith("claude-cli/")


def test_build_headers_includes_oauth_betas():
    betas = ao.build_headers("t")["anthropic-beta"].split(",")
    assert "oauth-2025-04-20" in betas
    assert "claude-code-20250219" in betas
    # context-1m must never be sent under OAuth
    assert "context-1m-2025-08-07" not in betas


def test_build_headers_merges_extra_betas_dedup():
    betas = ao.build_headers(
        "t", extra_betas=("oauth-2025-04-20", "context-management-2025-06-27")
    )
    parts = betas["anthropic-beta"].split(",")
    assert parts.count("oauth-2025-04-20") == 1
    assert "context-management-2025-06-27" in parts


def test_identity_prepended_for_empty_system():
    blocks = ao.ensure_claude_code_system(None)
    assert blocks[0]["text"] == ao.CLAUDE_CODE_IDENTITY
    assert len(blocks) == 1


def test_identity_prepended_before_string_system():
    blocks = ao.ensure_claude_code_system("You are a helpful bot.")
    assert blocks[0]["text"] == ao.CLAUDE_CODE_IDENTITY
    assert blocks[1] == {"type": "text", "text": "You are a helpful bot."}


def test_identity_idempotent_when_already_first():
    once = ao.ensure_claude_code_system("real prompt")
    twice = ao.ensure_claude_code_system(once)
    assert twice == once
    assert sum(1 for b in twice if b["text"] == ao.CLAUDE_CODE_IDENTITY) == 1


def test_identity_prepended_before_list_system():
    blocks = ao.ensure_claude_code_system([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ])
    assert blocks[0]["text"] == ao.CLAUDE_CODE_IDENTITY
    assert [b["text"] for b in blocks[1:]] == ["a", "b"]


def test_body_required_betas_for_context_management():
    assert ao.body_required_betas({"context_management": {"x": 1}}) == (
        "context-management-2025-06-27",
    )
    assert ao.body_required_betas({}) == ()
