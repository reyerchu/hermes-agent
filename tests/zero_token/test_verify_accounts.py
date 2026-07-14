"""Tests for the shared-subscription (same-org) detector."""

from __future__ import annotations

from zero_token import credentials as cred
from zero_token import verify_accounts as va
from zero_token.credentials import ClaudeOAuthStore, CredentialPool, _Account


def _pool(*names):
    return CredentialPool([
        _Account(n, ClaudeOAuthStore(static_token=f"tok-{n}", read_env_token=False))
        for n in names
    ])


def test_duplicate_orgs_flags_shared_org():
    rows = [
        {"name": "claude1", "provider": "anthropic", "org": "ORG-A"},
        {"name": "claude2", "provider": "anthropic", "org": "ORG-A"},
        {"name": "claude3", "provider": "anthropic", "org": "ORG-B"},
    ]
    assert va.duplicate_orgs(rows) == {"ORG-A": ["claude1", "claude2"]}


def test_duplicate_orgs_ignores_missing_and_distinct():
    rows = [
        {"name": "a", "provider": "anthropic", "org": "ORG-A"},
        {"name": "b", "provider": "anthropic", "org": "ORG-B"},
        {"name": "kimi", "provider": "kimi", "org": None},
        {"name": "c", "provider": "anthropic", "org": None},  # probe failed
    ]
    assert va.duplicate_orgs(rows) == {}


def test_probe_pool_only_probes_anthropic(monkeypatch):
    seen = []

    def fake_org(token, *, base_url, timeout=15.0):
        seen.append(token)
        return "ORG-A"

    monkeypatch.setattr(cred, "anthropic_org_id", fake_org)
    monkeypatch.setattr(va, "anthropic_org_id", fake_org)

    pool = CredentialPool([
        _Account("claude1", ClaudeOAuthStore(static_token="t1", read_env_token=False)),
        _Account(
            "kimi",
            ClaudeOAuthStore(static_token="t2", read_env_token=False),
            provider="kimi",
            model="kimi-code",
        ),
    ])
    rows = va.probe_pool(pool)
    assert [r["provider"] for r in rows] == ["anthropic", "kimi"]
    assert rows[0]["org"] == "ORG-A"
    assert rows[1]["org"] is None  # kimi is not probed
    assert seen == ["t1"]  # only the anthropic account was probed
