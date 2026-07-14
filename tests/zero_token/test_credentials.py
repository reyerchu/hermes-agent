"""Tests for ClaudeOAuthStore: read, expiry, refresh, atomic write-back."""

from __future__ import annotations

import json
import os
import time
import stat

import pytest

from zero_token import credentials as cred
from zero_token.credentials import ClaudeOAuthStore, CredentialsError


def _write_creds(
    path, *, access="sk-ant-oat01-A", refresh="sk-ant-ort01-R", expires_ms=None
):
    if expires_ms is None:
        expires_ms = int((time.time() + 3600) * 1000)
    path.write_text(
        json.dumps({
            "claudeAiOauth": {
                "accessToken": access,
                "refreshToken": refresh,
                "expiresAt": expires_ms,
                "subscriptionType": "max",
            }
        })
    )


def test_returns_fresh_token_without_refresh(tmp_path):
    p = tmp_path / ".credentials.json"
    _write_creds(p, access="sk-ant-oat01-FRESH")
    store = ClaudeOAuthStore(credentials_path=p)
    assert store.access_token() == "sk-ant-oat01-FRESH"


def test_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cred.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    store = ClaudeOAuthStore(credentials_path=tmp_path / "nope.json")
    with pytest.raises(CredentialsError):
        store.access_token()


def test_static_env_token_short_circuits(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-STATIC")
    store = ClaudeOAuthStore(credentials_path=tmp_path / "unused.json")
    assert store.uses_static_token is True
    assert store.access_token() == "sk-ant-oat01-STATIC"


def test_expired_token_triggers_refresh_and_writeback(tmp_path, monkeypatch):
    p = tmp_path / ".credentials.json"
    _write_creds(
        p,
        access="sk-ant-oat01-OLD",
        refresh="sk-ant-ort01-OLD",
        expires_ms=int((time.time() - 10) * 1000),
    )  # already expired

    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "access_token": "sk-ant-oat01-NEW",
                "refresh_token": "sk-ant-ort01-NEW",
                "expires_in": 28800,
            }

        @property
        def text(self):
            return ""

    def fake_post(url, json, headers, timeout):  # noqa: A002 - match httpx signature
        captured["url"] = url
        captured["payload"] = json
        captured["ua"] = headers.get("User-Agent")
        return FakeResp()

    monkeypatch.setattr(cred.httpx, "post", fake_post)

    store = ClaudeOAuthStore(credentials_path=p)
    tok = store.access_token()

    assert tok == "sk-ant-oat01-NEW"
    # correct refresh request shape
    assert captured["url"] == cred.OAUTH_TOKEN_URL
    assert captured["payload"]["grant_type"] == "refresh_token"
    assert captured["payload"]["refresh_token"] == "sk-ant-ort01-OLD"
    assert captured["payload"]["client_id"] == cred.OAUTH_CLIENT_ID
    # Cloudflare-friendly UA required
    assert captured["ua"].startswith("claude-cli/")

    # written back atomically with rotated refresh token + ms expiry
    doc = json.loads(p.read_text())["claudeAiOauth"]
    assert doc["accessToken"] == "sk-ant-oat01-NEW"
    assert doc["refreshToken"] == "sk-ant-ort01-NEW"
    assert doc["expiresAt"] > int(time.time() * 1000)
    # 0600 perms
    mode = stat.S_IMODE(os.stat(p).st_mode)
    assert mode == 0o600


def test_refresh_prefers_fresh_file_written_by_cli(tmp_path, monkeypatch):
    """If the file already holds a fresh token (CLI refreshed it), don't burn our refresh token."""
    p = tmp_path / ".credentials.json"
    _write_creds(
        p, access="sk-ant-oat01-CLIFRESH", expires_ms=int((time.time() + 3600) * 1000)
    )

    def boom(*a, **k):
        raise AssertionError("refresh must not be called when file is already fresh")

    monkeypatch.setattr(cred.httpx, "post", boom)
    store = ClaudeOAuthStore(credentials_path=p)
    assert store.access_token() == "sk-ant-oat01-CLIFRESH"


def test_refresh_http_4xx_raises_reauth_hint(tmp_path, monkeypatch):
    p = tmp_path / ".credentials.json"
    _write_creds(p, expires_ms=int((time.time() - 10) * 1000))

    class FakeResp:
        status_code = 400
        text = "invalid_grant"

        def json(self):
            return {}

    monkeypatch.setattr(cred.httpx, "post", lambda *a, **k: FakeResp())
    store = ClaudeOAuthStore(credentials_path=p)
    with pytest.raises(CredentialsError) as ei:
        store.access_token()
    assert (
        "re-run" in str(ei.value).lower() or "re-authenticate" in str(ei.value).lower()
    )


def test_persist_failure_falls_back_to_cached_token(tmp_path, monkeypatch):
    """If write-back fails, the in-memory refreshed token is reused (not re-refreshed)."""
    p = tmp_path / ".credentials.json"
    _write_creds(
        p, access="sk-ant-oat01-OLD", expires_ms=int((time.time() - 10) * 1000)
    )

    calls = {"n": 0}

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {"access_token": "sk-ant-oat01-NEW", "expires_in": 28800}

    def fake_post(*a, **k):
        calls["n"] += 1
        return FakeResp()

    monkeypatch.setattr(cred.httpx, "post", fake_post)
    # Simulate persist failure — the file keeps the OLD expired token.
    monkeypatch.setattr(ClaudeOAuthStore, "_persist", lambda self, doc, **k: None)

    store = ClaudeOAuthStore(credentials_path=p)
    assert store.access_token() == "sk-ant-oat01-NEW"  # refreshed (1 http call)
    assert store.access_token() == "sk-ant-oat01-NEW"  # served from cache, no 2nd call
    assert calls["n"] == 1


def test_refresh_rejected_but_cli_rotated_file_is_used(tmp_path, monkeypatch):
    """On a 4xx refresh (race with the CLI), a fresh token on disk is picked up."""
    p = tmp_path / ".credentials.json"
    _write_creds(
        p,
        access="sk-ant-oat01-OLD",
        refresh="sk-ant-ort01-OLD",
        expires_ms=int((time.time() - 10) * 1000),
    )

    class Rejected:
        status_code = 400
        text = "invalid_grant"

        def json(self):
            return {}

    def fake_post(*a, **k):
        # Simulate the Claude CLI having refreshed concurrently: a fresh token
        # is now on disk, and our (now-stale) refresh token is rejected.
        _write_creds(
            p,
            access="sk-ant-oat01-CLINEW",
            refresh="sk-ant-ort01-CLINEW",
            expires_ms=int((time.time() + 3600) * 1000),
        )
        return Rejected()

    monkeypatch.setattr(cred.httpx, "post", fake_post)
    store = ClaudeOAuthStore(credentials_path=p)
    assert store.access_token() == "sk-ant-oat01-CLINEW"


def test_persist_skips_when_file_changed_under_us(tmp_path):
    """The lost-update guard: don't clobber a newer token written by the CLI."""
    p = tmp_path / ".credentials.json"
    # The CLI already wrote a newer token to disk.
    _write_creds(p, access="sk-ant-oat01-CLINEWER")
    store = ClaudeOAuthStore(credentials_path=p)
    # We try to persist a doc we started building from an OLDER token.
    ours = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-OURS"}}
    store._persist(ours, started_from="sk-ant-oat01-OLDER")
    # File must be untouched (CLI's newer token preserved).
    assert (
        json.loads(p.read_text())["claudeAiOauth"]["accessToken"]
        == "sk-ant-oat01-CLINEWER"
    )


def test_describe_reports_expiry_and_subscription(tmp_path):
    p = tmp_path / ".credentials.json"
    _write_creds(p, expires_ms=int((time.time() + 1800) * 1000))
    info = ClaudeOAuthStore(credentials_path=p).describe()
    assert info["refreshable"] is True
    assert info["subscriptionType"] == "max"
    assert 0 < info["expiresInSeconds"] <= 1800


# --- CredentialPool -------------------------------------------------------

from zero_token.credentials import CredentialPool, _Account  # noqa: E402


def _acct(name, token, **kw):
    return _Account(
        name, ClaudeOAuthStore(static_token=token, read_env_token=False), **kw
    )


def test_pool_active_is_first_account_then_skips_cooled_down():
    a, b = _acct("primary", "t1"), _acct("backup1", "t2")
    pool = CredentialPool([a, b])
    assert pool.active().name == "primary"
    pool.mark_limited(a, cooldown_s=60, reason="usage")
    assert pool.active().name == "backup1"


def test_pool_all_cooled_returns_soonest_to_recover():
    a, b = _acct("primary", "t1"), _acct("backup1", "t2")
    pool = CredentialPool([a, b])
    pool.mark_limited(a, cooldown_s=1000, reason="usage")
    pool.mark_limited(b, cooldown_s=10, reason="usage")
    assert pool.all_cooled_down() is True
    assert pool.active().name == "backup1"  # smaller cooldown → soonest


def test_pool_cooldown_expires(monkeypatch):
    a, b = _acct("primary", "t1"), _acct("backup1", "t2")
    pool = CredentialPool([a, b])
    pool.mark_limited(a, cooldown_s=-1, reason="usage")  # already elapsed
    assert pool.active().name == "primary"


def test_provider_presets_kimi_endpoint_and_no_identity():
    k = _acct("kimi", "tok", provider="kimi", model="kimi-k2-0711-preview")
    assert k.base_url == "https://api.kimi.com/coding/v1"
    assert k.send_identity is False
    assert k.model == "kimi-k2-0711-preview"
    assert k.provider == "kimi"


def test_pool_from_env_json_builds_mixed_providers(monkeypatch, tmp_path):
    cfg = [
        {"name": "claude1", "provider": "anthropic", "token": "tok-claude1"},
        {
            "name": "kimi",
            "provider": "kimi",
            "token": "tok-kimi",
            "model": "kimi-k2-0711-preview",
        },
    ]
    monkeypatch.setenv("ZT_ACCOUNTS_JSON", json.dumps(cfg))
    pool = CredentialPool.from_env()
    accts = pool.accounts()
    assert [a.name for a in accts] == ["claude1", "kimi"]
    assert accts[0].provider == "anthropic"
    assert accts[1].provider == "kimi"
    assert accts[1].base_url == "https://api.kimi.com/coding/v1"
    assert accts[1].model == "kimi-k2-0711-preview"


def test_pool_from_env_backup_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-PRIMARY")
    monkeypatch.setenv("ZT_BACKUP_TOKENS", "sk-ant-oat01-B1,sk-ant-oat01-B2")
    pool = CredentialPool.from_env()
    assert pool.size == 3
    assert pool.active().store.access_token() == "sk-ant-oat01-PRIMARY"
