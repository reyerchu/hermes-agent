"""Verify a failover pool actually has *independent* accounts.

Two credential slots whose tokens differ can still bill against the **same**
Anthropic subscription (one ``claude /login`` token + one ``claude setup-token``
from the same org). Such a "backup" adds no capacity: when the primary's usage
window fills, the duplicate's does too. This tool probes each anthropic-provider
account's ``anthropic-organization-id`` and reports any slots that share an org.

Run::

    python -m zero_token.verify_accounts     # reads ZT_ACCOUNTS_FILE / ZT_ACCOUNTS_JSON / env

Exit code is always 0 (a duplicate is a warning, not a hard failure) so it can be
dropped into deploy scripts without aborting them.
"""

from __future__ import annotations

import sys

from .credentials import CredentialPool, CredentialsError, anthropic_org_id


def probe_pool(pool: CredentialPool) -> list[dict]:
    """Return one row per account: ``{name, provider, org, error}`` (best-effort)."""
    rows: list[dict] = []
    for a in pool.accounts():
        org: str | None = None
        err: str | None = None
        if a.provider == "anthropic":
            try:
                org = anthropic_org_id(a.store.access_token(), base_url=a.base_url)
            except CredentialsError as exc:
                err = str(exc)
        rows.append(
            {"name": a.name, "provider": a.provider, "org": org, "error": err}
        )
    return rows


def duplicate_orgs(rows: list[dict]) -> dict[str, list[str]]:
    """Map org id -> [account names] for orgs claimed by more than one account."""
    groups: dict[str, list[str]] = {}
    for r in rows:
        org = r.get("org")
        if org:
            groups.setdefault(org, []).append(r["name"])
    return {org: names for org, names in groups.items() if len(names) > 1}


def main() -> int:
    pool = CredentialPool.from_env()
    rows = probe_pool(pool)
    print(f"failover pool: {len(rows)} account(s)")
    for r in rows:
        if r["error"]:
            detail = f"credential error: {r['error']}"
        elif r["org"]:
            detail = f"org {r['org']}"
        elif r["provider"] != "anthropic":
            detail = "(non-anthropic; org check n/a)"
        else:
            detail = "org unknown (probe failed)"
        print(f"  {r['name']:12} {r['provider']:9} {detail}")

    dups = duplicate_orgs(rows)
    if dups:
        print()
        for org, names in dups.items():
            print(
                f"⚠️  {', '.join(names)} share Anthropic org {org} — SAME "
                "subscription, so this is NOT real failover."
            )
        print(
            "   A real backup needs a *different* Anthropic org (a separate "
            "account/subscription) or a different provider (e.g. Kimi)."
        )
    else:
        anthropic = [r for r in rows if r["provider"] == "anthropic" and r["org"]]
        if len(anthropic) > 1:
            print("\n✓ anthropic accounts are on distinct orgs (independent failover).")
        else:
            print("\n✓ no shared-org duplicates detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
