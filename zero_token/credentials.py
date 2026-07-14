"""Claude CLI OAuth account-token store: read, refresh, and persist.

The Claude CLI keeps its subscription OAuth credentials in
``~/.claude/.credentials.json`` (or, on macOS, the login keychain item
``Claude Code-credentials``) under the top-level key ``claudeAiOauth``:

    {
      "claudeAiOauth": {
        "accessToken":  "sk-ant-oat01-...",
        "refreshToken": "sk-ant-ort01-...",
        "expiresAt":    1784032080076,      # milliseconds since epoch
        "subscriptionType": "max",
        ...
      }
    }

``ClaudeOAuthStore`` returns a valid access token on demand, refreshing it
just-in-time via the public Claude Code OAuth client when it is within
``_REFRESH_SKEW_SECONDS`` of expiry, and writing the rotated credentials back
to the same file atomically (0600) so the ``claude`` CLI keeps working.

A static token may instead be supplied via the environment (``ANTHROPIC_TOKEN``
/ ``CLAUDE_CODE_OAUTH_TOKEN`` / ``ANTHROPIC_OAUTH_TOKEN``); such a token is used
verbatim with no refresh.

The refresh mechanics (endpoint, client id, the Cloudflare-friendly
``User-Agent`` requirement, millisecond expiry, atomic write-back) match the
behaviour of the reference implementations in ``any-llm-in-claude`` and
``openclaw-zero-token`` and of hermes-agent's own ``anthropic_adapter``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx

LOG = logging.getLogger("zero-token.credentials")

# Public Claude Code OAuth client id (same value used by the Claude CLI and by
# every reference implementation). Not a secret — it identifies the app, not
# the user.
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# Refresh endpoint. platform.claude.com is the current host; the older
# console.anthropic.com endpoint is kept as a fallback because some tokens were
# minted against it.
OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
OAUTH_TOKEN_URL_FALLBACK = "https://console.anthropic.com/v1/oauth/token"

# The refresh host sits behind Cloudflare and 403s the default httpx UA, so a
# Claude-CLI-shaped User-Agent is REQUIRED on the refresh call. This is a fixed
# string (unlike the Messages-call UA, which tracks the installed CLI version)
# because it only has to look like *a* CLI to clear the WAF.
_REFRESH_USER_AGENT = "claude-cli/2.1.208 (external, cli)"

# Refresh this many seconds before the stated expiry so a token can't lapse
# mid-request.
_REFRESH_SKEW_SECONDS = 120

# Top-level key in the credentials file.
_OAUTH_KEY = "claudeAiOauth"

# macOS keychain generic-password service name used by the Claude CLI.
_MACOS_KEYCHAIN_SERVICE = "Claude Code-credentials"

# Env vars that supply a static account token (checked in this order). A value
# here disables the file store entirely (no refresh).
_STATIC_TOKEN_ENV_VARS = (
    "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_OAUTH_TOKEN",
)


def default_credentials_path() -> Path:
    """Return the credentials file path, honouring CLAUDE_CREDENTIALS_PATH."""
    override = os.environ.get("CLAUDE_CREDENTIALS_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".claude" / ".credentials.json"


class CredentialsError(RuntimeError):
    """Raised when no usable account token can be obtained."""


class _RefreshRejected(Exception):
    """Internal: the OAuth refresh endpoint returned a 4xx (token rejected)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"refresh rejected ({status}): {detail}")
        self.status = status
        self.detail = detail


class ClaudeOAuthStore:
    """Resolve and refresh the Claude CLI subscription OAuth access token.

    Thread-safe: ``access_token()`` may be called concurrently from aiohttp
    handlers; a lock serialises the read-refresh-persist critical section.
    """

    def __init__(
        self,
        credentials_path: Path | None = None,
        *,
        static_token: str | None = None,
        read_env_token: bool = True,
    ) -> None:
        self._path = credentials_path or default_credentials_path()
        self._lock = threading.Lock()
        self._cache: dict[str, Any] | None = None  # full file document

        # A static token short-circuits everything (no refresh, no file). An
        # explicit static_token wins; otherwise the env vars are consulted (only
        # for the default/primary store — pool backup accounts pass their own).
        self._static_token: str | None = static_token.strip() if static_token else None
        if self._static_token:
            LOG.info("account using an explicit static token (refresh disabled)")
        elif read_env_token:
            for var in _STATIC_TOKEN_ENV_VARS:
                val = os.environ.get(var, "").strip()
                if val:
                    self._static_token = val
                    LOG.info(
                        "using static account token from %s (refresh disabled)", var
                    )
                    break

    # -- public API --------------------------------------------------------

    @property
    def uses_static_token(self) -> bool:
        return self._static_token is not None

    def access_token(self) -> str:
        """Return a currently-valid access token, refreshing if needed.

        This is synchronous (blocking file I/O and, on refresh, a blocking HTTP
        call). Async callers MUST offload it, e.g. ``await
        asyncio.to_thread(store.access_token)``, so the event loop is not frozen
        during a refresh.
        """
        if self._static_token is not None:
            return self._static_token
        with self._lock:
            doc = self._read_file()
            oauth = doc.get(_OAUTH_KEY) or {}
            token = oauth.get("accessToken")
            if token and not self._needs_refresh(oauth):
                self._cache = doc
                return token
            # File token is stale/absent. If a previous in-memory refresh
            # produced a still-valid token that we failed to persist, serve
            # that rather than burning the refresh token again (the on-disk
            # refresh token is already consumed and would 400).
            cached = self._cache_token_if_valid()
            if cached is not None:
                return cached
            # Otherwise refresh (which re-reads the file first, in case the CLI
            # already rotated the token for us).
            return self._refresh_locked()

    def _cache_token_if_valid(self) -> str | None:
        if not self._cache:
            return None
        oauth = self._cache.get(_OAUTH_KEY) or {}
        tok = oauth.get("accessToken")
        if tok and not self._needs_refresh(oauth):
            return tok
        return None

    def describe(self) -> dict[str, Any]:
        """Return non-secret metadata for /health and diagnostics."""
        if self._static_token is not None:
            return {"source": "static-env", "refreshable": False}
        try:
            oauth = (self._read_file().get(_OAUTH_KEY)) or {}
        except CredentialsError:
            return {"source": "missing", "refreshable": False}
        exp_ms = oauth.get("expiresAt")
        info: dict[str, Any] = {
            "source": str(self._path),
            "refreshable": bool(oauth.get("refreshToken")),
            "subscriptionType": oauth.get("subscriptionType"),
            "hasAccessToken": bool(oauth.get("accessToken")),
        }
        if isinstance(exp_ms, (int, float)):
            info["expiresInSeconds"] = int(exp_ms / 1000 - time.time())
        return info

    # -- internals ---------------------------------------------------------

    def _needs_refresh(self, oauth: dict[str, Any]) -> bool:
        exp_ms = oauth.get("expiresAt")
        if not isinstance(exp_ms, (int, float)):
            # Unknown expiry — refresh if we have the means, else use as-is.
            return bool(oauth.get("refreshToken"))
        return time.time() >= (exp_ms / 1000.0) - _REFRESH_SKEW_SECONDS

    def _read_file(self) -> dict[str, Any]:
        """Load the credentials document from disk (or macOS keychain)."""
        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            keychain = self._read_macos_keychain()
            if keychain is not None:
                return keychain
            raise CredentialsError(
                f"Claude CLI credentials not found at {self._path}. "
                "Run `claude /login` (or `claude setup-token`), or set "
                "ANTHROPIC_TOKEN to a setup-token value."
            )
        except OSError as exc:
            raise CredentialsError(f"cannot read {self._path}: {exc}") from exc
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CredentialsError(f"{self._path} is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise CredentialsError(f"{self._path} does not contain a JSON object")
        return doc

    def _read_macos_keychain(self) -> dict[str, Any] | None:
        """Best-effort read of the macOS keychain credentials item."""
        try:
            out = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    _MACOS_KEYCHAIN_SERVICE,
                    "-w",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        if out.returncode != 0 or not out.stdout.strip():
            return None
        try:
            doc = json.loads(out.stdout.strip())
        except json.JSONDecodeError:
            return None
        return doc if isinstance(doc, dict) else None

    def _refresh_locked(self) -> str:
        """Refresh the access token. Caller must hold ``self._lock``."""
        # Re-read first: the Claude CLI may have already refreshed and written a
        # new token. Anthropic rotates single-use refresh tokens, so racing the
        # CLI would burn our refresh token; prefer whatever is freshest on disk.
        doc = self._read_file()
        oauth = doc.get(_OAUTH_KEY) or {}
        if oauth.get("accessToken") and not self._needs_refresh(oauth):
            self._cache = doc
            return oauth["accessToken"]

        refresh_token = oauth.get("refreshToken")
        if not refresh_token:
            token = oauth.get("accessToken")
            if token:
                # No way to refresh, but we still have *a* token — use it and
                # let the upstream 401 surface if it is actually dead.
                self._cache = doc
                return token
            raise CredentialsError(
                f"{self._path} has no refreshToken and no accessToken; "
                "re-authenticate with `claude /login`."
            )

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OAUTH_CLIENT_ID,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _REFRESH_USER_AGENT,
        }
        try:
            data = self._post_refresh(payload, headers)
        except _RefreshRejected as rej:
            # The refresh token was rejected. The most common benign cause is a
            # race: the Claude CLI (or another instance) refreshed concurrently
            # and rotated our single-use refresh token out from under us. Re-read
            # the file once — if a fresh token is now on disk, use it.
            fresh = self._read_file()
            fresh_oauth = fresh.get(_OAUTH_KEY) or {}
            if fresh_oauth.get("accessToken") and not self._needs_refresh(fresh_oauth):
                self._cache = fresh
                return fresh_oauth["accessToken"]
            raise CredentialsError(
                f"OAuth refresh rejected ({rej.status}): the refresh token is "
                "expired or was invalidated — re-run `claude /login` "
                "(or `claude setup-token`)."
            ) from rej

        new_access = data.get("access_token")
        if not new_access:
            raise CredentialsError("OAuth refresh response missing access_token")
        new_refresh = data.get("refresh_token") or refresh_token
        expires_in = data.get("expires_in")
        new_oauth = dict(oauth)
        new_oauth["accessToken"] = new_access
        new_oauth["refreshToken"] = new_refresh
        if isinstance(expires_in, (int, float)):
            new_oauth["expiresAt"] = int(time.time() * 1000) + int(expires_in) * 1000
        doc[_OAUTH_KEY] = new_oauth
        self._cache = doc
        self._persist(doc, started_from=oauth.get("accessToken"))
        LOG.info(
            "refreshed account token (expires in %ss)",
            int(expires_in) if isinstance(expires_in, (int, float)) else "?",
        )
        return new_access

    def _post_refresh(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for url in (OAUTH_TOKEN_URL, OAUTH_TOKEN_URL_FALLBACK):
            try:
                resp = httpx.post(url, json=payload, headers=headers, timeout=30)
            except httpx.HTTPError as exc:
                last_exc = exc
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    continue
            # 4xx from the primary endpoint is authoritative (bad/expired/rotated
            # refresh token) — signal the caller to re-read and retry rather than
            # falling through to the fallback endpoint.
            if 400 <= resp.status_code < 500 and url == OAUTH_TOKEN_URL:
                raise _RefreshRejected(resp.status_code, resp.text[:300])
            last_exc = CredentialsError(
                f"OAuth refresh HTTP {resp.status_code} from {url}: {resp.text[:200]}"
            )
        raise CredentialsError(f"OAuth token refresh failed: {last_exc}")

    def _persist(self, doc: dict[str, Any], *, started_from: str | None = None) -> None:
        """Atomically write the credentials document back with 0600 perms.

        Guards against a lost update: if the file's access token changed since
        the refresh began (the CLI wrote a newer credential in the meantime),
        skip the write so a concurrently-rotated token is not clobbered.
        """
        try:
            if started_from is not None:
                try:
                    on_disk = (self._read_file().get(_OAUTH_KEY) or {}).get(
                        "accessToken"
                    )
                except CredentialsError:
                    on_disk = started_from
                if on_disk not in (None, started_from):
                    LOG.info("credentials file changed under us; not overwriting")
                    return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".credentials.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(doc, fh, indent=2)
                os.chmod(tmp, 0o600)
                os.replace(tmp, self._path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError as exc:
            # Non-fatal: the in-memory token (self._cache) still works for this
            # process and access_token() will serve it. We just couldn't share
            # the rotation with the CLI.
            LOG.warning(
                "could not persist refreshed credentials to %s: %s", self._path, exc
            )


# --------------------------------------------------------------------------
# multi-account failover
# --------------------------------------------------------------------------

# How long to cool down an account after it reports "out of extra usage" before
# trying it again. Subscription usage frees on a rolling window, so this is a
# retry interval, not a hard reset. Override with ZT_USAGE_COOLDOWN_S.
_DEFAULT_USAGE_COOLDOWN_S = int(os.environ.get("ZT_USAGE_COOLDOWN_S", "900"))
# Fallback cooldown for a 429 with no Retry-After.
_DEFAULT_RATELIMIT_COOLDOWN_S = 60


def is_usage_limited(status: int, body: dict[str, Any] | None) -> bool:
    """True if an upstream response means "this account is out of quota".

    Covers the subscription-exhaustion 400 ("You're out of extra usage") and
    429 rate limits — both are reasons to fail over to another account.
    """
    if status == 429:
        return True
    if status == 400 and isinstance(body, dict):
        msg = str((body.get("error") or {}).get("message", "")).lower()
        return (
            "out of extra usage" in msg or "usage limit" in msg or "rate limit" in msg
        )
    return False


def is_auth_error(status: int) -> bool:
    """True if an upstream response means "this account's token is bad".

    A 401/403 is account-specific (expired/revoked/unauthorized token), so it
    should fail over to the next account rather than be returned to the caller —
    unlike a 400 validation error, which would fail identically on every account.
    """
    return status in (401, 403)


# Per-provider defaults. Accounts may override any field. All providers here
# speak the Anthropic Messages API shape (Bearer auth), which is what the proxy
# emits; only the endpoint, model, identity-spoof, and beta flags differ.
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    # Claude Pro/Max subscription via the account OAuth token.
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "send_identity": True,
        "betas": ("claude-code-20250219", "oauth-2025-04-20"),
    },
    # Kimi Code subscription (Moonshot). Anthropic-compatible: the coding
    # endpoint lives at https://api.kimi.com/coding and serves /v1/messages
    # (base_url must NOT already include /v1 — the proxy appends it). Auth is a
    # plain Kimi Code *API key* (Bearer), not an OAuth token, so no oauth beta.
    "kimi": {
        "base_url": "https://api.kimi.com/coding",
        "send_identity": False,
        "betas": (),
    },
    # Any other Anthropic-compatible endpoint reached with a Bearer token.
    "generic": {
        "base_url": "https://api.anthropic.com",
        "send_identity": False,
        "betas": (),
    },
}


class _Account:
    """One failover slot: a credential store plus its provider endpoint config."""

    __slots__ = (
        "name",
        "store",
        "provider",
        "base_url",
        "model",
        "send_identity",
        "betas",
        "cooldown_until",
        "last_error",
    )

    def __init__(
        self,
        name: str,
        store: ClaudeOAuthStore,
        *,
        provider: str = "anthropic",
        base_url: str | None = None,
        model: str | None = None,
        send_identity: bool | None = None,
        betas: tuple[str, ...] | None = None,
    ) -> None:
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["generic"])
        self.name = name
        self.store = store
        self.provider = provider
        self.base_url = (base_url or preset["base_url"]).rstrip("/")
        self.model = model  # None => use the request's model
        self.send_identity = (
            preset["send_identity"] if send_identity is None else send_identity
        )
        self.betas = preset["betas"] if betas is None else tuple(betas)
        self.cooldown_until = 0.0
        self.last_error: str | None = None


class CredentialPool:
    """An ordered set of accounts with automatic failover on usage limits.

    The first account not in cooldown is the active one. When an account hits a
    usage/rate limit, the server calls :meth:`mark_limited`, which cools it down
    so the next request rotates to the following account. Each account has its
    own independent token refresh (its own credentials file).
    """

    def __init__(self, accounts: list[_Account]) -> None:
        if not accounts:
            raise ValueError("CredentialPool needs at least one account")
        self._accounts = accounts
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._accounts)

    def accounts(self) -> list[_Account]:
        return list(self._accounts)

    def active(self) -> _Account:
        """Return the first account not in cooldown, else the soonest-to-recover."""
        now = time.time()
        with self._lock:
            for a in self._accounts:
                if a.cooldown_until <= now:
                    return a
            # All cooled down — pick the one whose cooldown expires soonest so a
            # request still has a chance rather than failing outright.
            return min(self._accounts, key=lambda a: a.cooldown_until)

    def all_cooled_down(self) -> bool:
        now = time.time()
        with self._lock:
            return all(a.cooldown_until > now for a in self._accounts)

    def mark_limited(
        self, account: _Account, *, cooldown_s: float | None = None, reason: str = ""
    ) -> None:
        cd = _DEFAULT_USAGE_COOLDOWN_S if cooldown_s is None else cooldown_s
        with self._lock:
            account.cooldown_until = time.time() + cd
            account.last_error = reason
        LOG.warning(
            "account %r limited (%s); cooling down %.0fs, rotating to next",
            account.name,
            reason or "usage",
            cd,
        )

    def describe(self) -> list[dict[str, Any]]:
        now = time.time()
        out: list[dict[str, Any]] = []
        for a in self._accounts:
            info: dict[str, Any] = {
                "name": a.name,
                "provider": a.provider,
                "base_url": a.base_url,
            }
            if a.model:
                info["model"] = a.model
            info.update(a.store.describe())
            cd = a.cooldown_until - now
            info["cooling_down"] = cd > 0
            if cd > 0:
                info["cooldown_remaining_s"] = int(cd)
                info["last_error"] = a.last_error
            out.append(info)
        return out

    @classmethod
    def _account_from_config(cls, i: int, cfg: dict[str, Any]) -> _Account:
        name = cfg.get("name") or f"account{i}"
        provider = cfg.get("provider", "anthropic")
        token = cfg.get("token")
        cpath = cfg.get("credentials_path")
        if token:
            store = ClaudeOAuthStore(static_token=str(token), read_env_token=False)
        elif cpath:
            store = ClaudeOAuthStore(
                credentials_path=Path(str(cpath)).expanduser(), read_env_token=False
            )
        else:
            # Default file + env token (only sensible for the primary anthropic).
            store = ClaudeOAuthStore()
        betas = cfg.get("betas")
        return _Account(
            name,
            store,
            provider=provider,
            base_url=cfg.get("base_url"),
            model=cfg.get("model"),
            send_identity=cfg.get("send_identity"),
            betas=tuple(betas) if isinstance(betas, list) else None,
        )

    @classmethod
    def from_env(cls) -> CredentialPool:
        """Build the pool from env config.

        Full control (recommended for mixed providers) via ``ZT_ACCOUNTS_JSON``,
        a JSON array of account objects tried in order::

            [
              {"name": "claude1", "provider": "anthropic"},
              {"name": "claude2", "provider": "anthropic",
               "credentials_path": "~/.claude/.credentials.account2.json"},
              {"name": "kimi", "provider": "kimi", "token": "sk-...",
               "model": "kimi-for-coding"}
            ]

        Each object: ``name``, ``provider`` (anthropic|kimi|generic), one of
        ``credentials_path`` / ``token`` (omit both to use the default Claude
        credentials file + env token), and optional ``base_url`` / ``model`` /
        ``send_identity`` / ``betas``.

        The same JSON array may instead live in a file referenced by
        ``ZT_ACCOUNTS_FILE`` (preferred for real deployments: keep tokens in one
        ``0600`` file rather than a systemd ``Environment=`` line, which both
        mangles JSON quoting and exposes secrets via ``systemctl show``).
        ``ZT_ACCOUNTS_JSON`` takes precedence if both are set.

        Simple Claude-only alternative (no JSON): the primary is the default
        Claude credentials file, and ``ZT_BACKUP_CREDENTIALS`` (os.pathsep-
        separated credential files) / ``ZT_BACKUP_TOKENS`` (comma-separated
        static tokens) add anthropic backups.
        """
        raw_json = os.environ.get("ZT_ACCOUNTS_JSON", "").strip()
        source = "ZT_ACCOUNTS_JSON"
        if not raw_json:
            file_path = os.environ.get("ZT_ACCOUNTS_FILE", "").strip()
            if file_path:
                source = f"ZT_ACCOUNTS_FILE ({file_path})"
                try:
                    raw_json = Path(file_path).expanduser().read_text().strip()
                except OSError as exc:
                    raise CredentialsError(
                        f"ZT_ACCOUNTS_FILE could not be read: {exc}"
                    ) from exc
        if raw_json:
            try:
                cfgs = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise CredentialsError(
                    f"{source} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(cfgs, list) or not cfgs:
                raise CredentialsError(
                    f"{source} must be a non-empty JSON array"
                )
            return cls([cls._account_from_config(i, c) for i, c in enumerate(cfgs)])

        # Simple Claude-only path.
        accounts: list[_Account] = [_Account("primary", ClaudeOAuthStore())]
        raw_paths = os.environ.get("ZT_BACKUP_CREDENTIALS", "").strip()
        if raw_paths:
            for i, p in enumerate(x for x in raw_paths.split(os.pathsep) if x.strip()):
                accounts.append(
                    _Account(
                        f"backup{i + 1}",
                        ClaudeOAuthStore(
                            credentials_path=Path(p.strip()).expanduser(),
                            read_env_token=False,
                        ),
                    )
                )
        raw_tokens = os.environ.get("ZT_BACKUP_TOKENS", "").strip()
        if raw_tokens:
            offset = len(accounts)
            for i, tok in enumerate(t for t in raw_tokens.split(",") if t.strip()):
                accounts.append(
                    _Account(
                        f"backup{offset + i}",
                        ClaudeOAuthStore(
                            static_token=tok.strip(), read_env_token=False
                        ),
                    )
                )
        return cls(accounts)
