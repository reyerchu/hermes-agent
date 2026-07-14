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

    def __init__(self, credentials_path: Path | None = None) -> None:
        self._path = credentials_path or default_credentials_path()
        self._lock = threading.Lock()
        self._cache: dict[str, Any] | None = None  # full file document

        # A static env token short-circuits everything (no refresh, no file).
        self._static_token: str | None = None
        for var in _STATIC_TOKEN_ENV_VARS:
            val = os.environ.get(var, "").strip()
            if val:
                self._static_token = val
                LOG.info("using static account token from %s (refresh disabled)", var)
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
