"""Enterprise features for Synth v2.0 (spec: Enterprise tier).

Covers the capabilities that distinguish an on-prem / air-gapped / regulated
deployment from the base open-source tool:

  * SkillMarketplace — publish, search, install, and rate skills in a local
    SQLite catalog (no remote registry required, works air-gapped).
  * SkillSigner       — HMAC-SHA256 skill signing/verification so installed
    skills can be integrity-checked before execution.
  * ComplianceChecker — static checks against OWASP / GDPR / SOC2 / ISO27001
    control sets by scanning a repository tree.
  * AirGappedMode     — detect whether the host has outbound network and flip
    a process-wide flag that disables all network-dependent tools.
  * OnPremConfig      — read/write/validate the on-prem deployment descriptor.
  * SSOConfig         — persist SSO/SAML provider settings and build OAuth2
    authorization URLs.
  * FineTuningExport  — dump sessions (>=N messages) from sessions.db into a
    JSONL file suitable for LLM fine-tuning.

All persistence lives under ~/.synth/. Stdlib-only; no third-party deps.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import socket
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR, SESSION_DB_FILENAME
from synth.skills import Skill
from synth import security as _security

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Marketplace catalog file (lives next to skills.db / sessions.db).
MARKETPLACE_DB_FILENAME: str = "marketplace.db"

# On-prem deployment descriptor.
ONPREM_FILENAME: str = "onprem.json"

# SSO provider descriptor.
SSO_FILENAME: str = "sso.json"

# Default lower bound on session length for fine-tuning export.
MIN_MESSAGES_DEFAULT: int = 5

# Root for all enterprise persistence (~/.synth).
_CONFIG_ROOT: Path = Path.home() / CONFIG_DIR

# Environment variable toggled by AirGappedMode.enable().
_AIRGAPPED_ENV: str = "SYNTH_AIRGAPPED"


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _config_dir() -> Path:
    """Return (creating if needed) the ~/.synth directory."""
    root = _CONFIG_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# SkillMarketplace
# ---------------------------------------------------------------------------


class MarketplaceError(Exception):
    """Raised on marketplace publish/search/install/rate failures."""


class SkillMarketplace:
    """Local SQLite-backed skill catalog.

    The marketplace is a catalog of *published* skills (separate from the
    user's personal ``SkillStore``). Each published skill gets a stable
    marketplace ID (UUID) and carries an author, rating, and install count.

    Args:
        db_path: Override the SQLite path (tests use temp dirs).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = _config_dir() / MARKETPLACE_DB_FILENAME
        self.db_path: Path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.db_path))
        except sqlite3.Error as exc:
            raise MarketplaceError(
                f"cannot open marketplace db: {exc}"
            ) from exc
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # -- context manager --
    def __enter__(self) -> "SkillMarketplace":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # -- schema --
    def _ensure_schema(self) -> None:
        try:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS marketplace (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    code TEXT NOT NULL,
                    author TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    rating_sum INTEGER NOT NULL DEFAULT 0,
                    rating_count INTEGER NOT NULL DEFAULT 0,
                    install_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MarketplaceError(
                f"marketplace schema init failed: {exc}"
            ) from exc

    # -- publish --
    def publish(self, skill: Skill, author: str) -> str:
        """Publish a skill to the marketplace.

        Returns the generated marketplace ID (UUID hex).
        """
        if not isinstance(author, str) or not author.strip():
            raise MarketplaceError("author must be a non-empty string")
        if not isinstance(skill, Skill):
            raise MarketplaceError("skill must be a Skill instance")
        if not skill.name.strip() or not skill.code.strip():
            raise MarketplaceError("skill name and code must be non-empty")

        marketplace_id = uuid.uuid4().hex
        try:
            self._conn.execute(
                """
                INSERT INTO marketplace
                    (id, name, description, code, author, created_at,
                     rating_sum, rating_count, install_count)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0)
                """,
                (
                    marketplace_id,
                    skill.name,
                    skill.description,
                    skill.code,
                    author.strip(),
                    _now_iso(),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MarketplaceError(
                f"failed to publish skill: {exc}"
            ) from exc
        return marketplace_id

    # -- search --
    def search(self, query: str) -> list[dict[str, Any]]:
        """Search published skills by name/description/author.

        Returns a list of dicts with keys: id, name, description, author,
        rating, install_count.
        """
        if not isinstance(query, str) or not query.strip():
            raise MarketplaceError("query must be a non-empty string")
        like = f"%{query.strip()}%"
        try:
            rows = self._conn.execute(
                """
                SELECT id, name, description, author, rating_sum,
                       rating_count, install_count
                FROM marketplace
                WHERE name LIKE ? OR description LIKE ? OR author LIKE ?
                ORDER BY install_count DESC, name ASC
                """,
                (like, like, like),
            ).fetchall()
        except sqlite3.Error as exc:
            raise MarketplaceError(
                f"marketplace search failed: {exc}"
            ) from exc

        out: list[dict[str, Any]] = []
        for r in rows:
            count = r["rating_count"] or 0
            total = r["rating_sum"] or 0
            rating = (total / count) if count else 0.0
            out.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "description": r["description"],
                    "author": r["author"],
                    "rating": round(rating, 2),
                    "install_count": r["install_count"],
                }
            )
        return out

    # -- install --
    def install(self, marketplace_id: str) -> Skill:
        """Install a published skill: bump install_count and return a Skill."""
        if not isinstance(marketplace_id, str) or not marketplace_id.strip():
            raise MarketplaceError("marketplace_id must be a non-empty string")
        try:
            row = self._conn.execute(
                """
                SELECT name, description, code FROM marketplace
                WHERE id = ?
                """,
                (marketplace_id.strip(),),
            ).fetchone()
        except sqlite3.Error as exc:
            raise MarketplaceError(
                f"marketplace install query failed: {exc}"
            ) from exc
        if row is None:
            raise MarketplaceError(
                f"no marketplace skill with id {marketplace_id!r}"
            )
        try:
            self._conn.execute(
                "UPDATE marketplace SET install_count = install_count + 1 "
                "WHERE id = ?",
                (marketplace_id.strip(),),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MarketplaceError(
                f"failed to bump install_count: {exc}"
            ) from exc
        return Skill(
            name=row["name"],
            description=row["description"],
            code=row["code"],
        )

    # -- rate --
    def rate(self, marketplace_id: str, stars: int) -> None:
        """Rate a published skill (1-5 stars)."""
        if not isinstance(stars, int) or not (1 <= stars <= 5):
            raise MarketplaceError("stars must be an int in 1..5")
        if not isinstance(marketplace_id, str) or not marketplace_id.strip():
            raise MarketplaceError("marketplace_id must be a non-empty string")
        try:
            cur = self._conn.execute(
                "UPDATE marketplace SET rating_sum = rating_sum + ?, "
                "rating_count = rating_count + 1 WHERE id = ?",
                (stars, marketplace_id.strip()),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MarketplaceError(
                f"failed to rate skill: {exc}"
            ) from exc
        if cur.rowcount == 0:
            raise MarketplaceError(
                f"no marketplace skill with id {marketplace_id!r}"
            )


# ---------------------------------------------------------------------------
# SkillSigner
# ---------------------------------------------------------------------------


class SkillSignerError(Exception):
    """Raised on signing/verification failures."""


class SkillSigner:
    """HMAC-SHA256 skill signing.

    A simplified signing scheme: the signature is HMAC-SHA256 over the
    concatenation ``skill.name + "\0" + skill.code`` keyed by the provided
    PEM/key material. Verification recomputes the MAC and compares with
    ``hmac.compare_digest``.

    The "private_key_pem" / "public_key_pem" parameters are used directly as
    the HMAC key bytes — this keeps the scheme stdlib-only (no RSA deps) while
    still providing integrity + authenticity as long as the key is shared
    securely.
    """

    @staticmethod
    def _material(skill: Skill) -> bytes:
        """Return the canonical bytes that get signed."""
        if not isinstance(skill, Skill):
            raise SkillSignerError("skill must be a Skill instance")
        return (skill.name + "\0" + skill.code).encode("utf-8")

    @staticmethod
    def _key_bytes(key_pem: str) -> bytes:
        if not isinstance(key_pem, str) or not key_pem:
            raise SkillSignerError("key material must be a non-empty string")
        return key_pem.encode("utf-8")

    def sign(self, skill: Skill, private_key_pem: str) -> str:
        """Return the hex HMAC-SHA256 signature of the skill."""
        key = self._key_bytes(private_key_pem)
        msg = self._material(skill)
        return hmac.new(key, msg, hashlib.sha256).hexdigest()

    def verify(
        self, skill: Skill, signature: str, public_key_pem: str
    ) -> bool:
        """Return True iff ``signature`` matches the skill under the key.

        Constant-time comparison via ``hmac.compare_digest``.
        """
        if not isinstance(signature, str) or not signature:
            return False
        expected = self.sign(skill, public_key_pem)
        return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# ComplianceChecker
# ---------------------------------------------------------------------------


class ComplianceError(Exception):
    """Raised on compliance-check failures."""


# GDPR-style hardcoded-PII patterns.
_PII_PATTERNS: list[tuple[str, str, str]] = [
    # Email — simple but catches the common hardcoded case.
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "Hardcoded Email", "HIGH"),
    # US SSN (xxx-xx-xxxx)
    (r"\b\d{3}-\d{2}-\d{4}\b", "Hardcoded SSN", "CRITICAL"),
    # E.164 / common phone patterns
    (r"\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}", "Hardcoded Phone", "MEDIUM"),
]

# SOC2: look for audit-logging signals (or the *absence* of them). We flag
# risky code paths that bypass logging.
_SOC2_PATTERNS: list[tuple[str, str, str]] = [
    (r"\beval\s*\(", "Unlogged eval() — no audit trail", "HIGH"),
    (r"\bexec\s*\(", "Unlogged exec() — no audit trail", "HIGH"),
    (r"os\.system\s*\(", "Shell call without audit log", "MEDIUM"),
    (r"subprocess\.(?:run|call|Popen)\(.*shell\s*=\s*True", "Shell=True without audit log", "MEDIUM"),
]

# ISO27001: access-control pattern gaps — hardcoded credentials, wildcard
# permissions, disabled auth checks.
_ISO27001_PATTERNS: list[tuple[str, str, str]] = [
    (r"(?:password|passwd|secret|api_key|token)\s*=\s*['\"][^'\"]{4,}['\"]", "Hardcoded credential", "CRITICAL"),
    (r"permission\s*=\s*['\"]\*['\"]", "Wildcard permission grant", "HIGH"),
    (r"@login_required\s*=\s*False", "Auth check disabled", "HIGH"),
    (r"allow_anonymous\s*=\s*True", "Anonymous access allowed", "HIGH"),
]

_SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
_SCAN_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".yaml", ".yml", ".json",
             ".toml", ".cfg", ".ini", ".env", ".txt", ".md"}


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _iter_repo_files(root: Path, *, max_files: int = 1000) -> list[Path]:
    """Yield scannable files under root (bounded)."""
    out: list[Path] = []
    try:
        for p in root.rglob("*"):
            if len(out) >= max_files:
                break
            if not p.is_file() or _should_skip(p):
                continue
            if p.suffix.lower() in _SCAN_EXTS or p.name == ".env":
                out.append(p)
    except OSError:
        pass
    return out


def _scan_patterns(
    text: str, patterns: list[tuple[str, str, str]], file_path: Path
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for line_num, line in enumerate(text.splitlines(), 1):
        for regex, category, severity in patterns:
            if re.search(regex, line):
                findings.append(
                    {
                        "category": category,
                        "severity": severity,
                        "description": f"{category} in {file_path.name}:{line_num}",
                        "file": str(file_path),
                    }
                )
    return findings


class ComplianceChecker:
    """Static compliance scanning against named control frameworks.

    Args:
        repo_root: Repository root to scan.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root: Path = Path(repo_root) if repo_root else Path.cwd()

    def check_standard(
        self, standard: str, repo_root: Path | None = None
    ) -> list[dict[str, Any]]:
        """Scan ``repo_root`` against the named standard.

        Returns a list of finding dicts:
        ``{category, severity, description, file}``.

        Supported standards: ``owasp``, ``gdpr``, ``soc2``, ``iso27001``.
        """
        root = Path(repo_root) if repo_root else self.repo_root
        if not isinstance(standard, str) or not standard.strip():
            raise ComplianceError("standard must be a non-empty string")
        std = standard.strip().lower()

        if not root.exists() or not root.is_dir():
            raise ComplianceError(f"repo_root does not exist: {root}")

        if std == "owasp":
            return self._check_owasp(root)
        if std == "gdpr":
            return self._check_gdpr(root)
        if std == "soc2":
            return self._check_soc2(root)
        if std == "iso27001":
            return self._check_iso27001(root)
        raise ComplianceError(f"unknown standard: {standard!r}")

    # -- OWASP: reuse security.py SAST + secret scan --
    def _check_owasp(self, root: Path) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        files = _iter_repo_files(root)
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Reuse SAST patterns from security.py (eval/exec/shell=True/...)
            for line_num, line in enumerate(text.splitlines(), 1):
                for regex, category, severity in _security.SAST_PATTERNS:
                    if re.search(regex, line):
                        findings.append(
                            {
                                "category": category,
                                "severity": severity,
                                "description": f"{category} in {fp.name}:{line_num}",
                                "file": str(fp),
                            }
                        )
            # Also reuse secret patterns.
            for line_num, line in enumerate(text.splitlines(), 1):
                for regex, category, severity in _security.SECRET_PATTERNS:
                    if re.search(regex, line, re.IGNORECASE):
                        findings.append(
                            {
                                "category": category,
                                "severity": severity,
                                "description": f"{category} in {fp.name}:{line_num}",
                                "file": str(fp),
                            }
                        )
        return findings

    # -- GDPR: hardcoded PII --
    def _check_gdpr(self, root: Path) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for fp in _iter_repo_files(root):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            findings.extend(_scan_patterns(text, _PII_PATTERNS, fp))
        return findings

    # -- SOC2: audit-log gaps --
    def _check_soc2(self, root: Path) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for fp in _iter_repo_files(root):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            findings.extend(_scan_patterns(text, _SOC2_PATTERNS, fp))
        return findings

    # -- ISO27001: access-control gaps --
    def _check_iso27001(self, root: Path) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for fp in _iter_repo_files(root):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            findings.extend(_scan_patterns(text, _ISO27001_PATTERNS, fp))
        return findings


# ---------------------------------------------------------------------------
# AirGappedMode
# ---------------------------------------------------------------------------


class AirGappedError(Exception):
    """Raised on air-gapped-mode failures."""


class AirGappedMode:
    """Detect and toggle air-gapped operation.

    When enabled, a process-wide env var (``SYNTH_AIRGAPPED=1``) is set so
    that all network-dependent code paths can short-circuit.
    """

    _PROBE_HOST: str = "8.8.8.8"
    _PROBE_PORT: int = 53
    _PROBE_TIMEOUT: float = 1.0

    def detect(self) -> bool:
        """Return True if the host currently has outbound network.

        Tries a 1s TCP connect to 8.8.8.8:53. Any failure (timeout, refused,
        DNS) → no network. If ``SYNTH_AIRGAPPED=1`` is already set, returns
        False immediately (env override wins).
        """
        if os.environ.get(_AIRGAPPED_ENV) == "1":
            return False
        try:
            with socket.create_connection(
                (self._PROBE_HOST, self._PROBE_PORT),
                timeout=self._PROBE_TIMEOUT,
            ):
                return True
        except OSError:
            return False

    def enable(self) -> None:
        """Enable air-gapped mode: set SYNTH_AIRGAPPED=1 process-wide."""
        os.environ[_AIRGAPPED_ENV] = "1"

    def disable(self) -> None:
        """Disable air-gapped mode (clears the env var)."""
        os.environ.pop(_AIRGAPPED_ENV, None)

    def is_enabled(self) -> bool:
        """Return True iff air-gapped mode is currently enabled."""
        return os.environ.get(_AIRGAPPED_ENV) == "1"


# ---------------------------------------------------------------------------
# OnPremConfig
# ---------------------------------------------------------------------------


class OnPremError(Exception):
    """Raised on on-prem config failures."""


class OnPremConfig:
    """Read/write/validate the on-prem deployment descriptor.

    The descriptor lives at ``~/.synth/onprem.json`` and has the shape::

        {
          "server_url": "https://synth.internal",
          "port": 8443,
          "auth_token_env": "SYNTH_AUTH_TOKEN",
          "allowed_models": ["llama-3", "qwen-2.5"]
        }
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path: Path = (
            Path(config_path)
            if config_path
            else _config_dir() / ONPREM_FILENAME
        )

    def load(self) -> dict[str, Any]:
        """Load and return the on-prem config dict.

        Returns an empty dict if the file does not exist (not yet configured).
        """
        if not self.config_path.exists():
            return {}
        try:
            text = self.config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OnPremError(
                f"cannot read {self.config_path}: {exc}"
            ) from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OnPremError(
                f"invalid JSON in {self.config_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise OnPremError(
                f"{self.config_path} must contain a JSON object"
            )
        return data

    def save(self, config: dict[str, Any]) -> None:
        """Persist the on-prem config dict to disk (pretty-printed)."""
        if not isinstance(config, dict):
            raise OnPremError("config must be a dict")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise OnPremError(
                f"cannot write {self.config_path}: {exc}"
            ) from exc

    def validate(self) -> bool:
        """Return True iff the loaded config has all required fields.

        Required: server_url (str), port (int 1..65535), auth_token_env
        (non-empty str), allowed_models (non-empty list).
        """
        data = self.load()
        if not data:
            return False
        url = data.get("server_url")
        port = data.get("port")
        env_var = data.get("auth_token_env")
        models = data.get("allowed_models")
        if not isinstance(url, str) or not url.strip():
            return False
        if not isinstance(port, int) or not (1 <= port <= 65535):
            return False
        if not isinstance(env_var, str) or not env_var.strip():
            return False
        if not isinstance(models, list) or not models:
            return False
        return True


# ---------------------------------------------------------------------------
# SSOConfig
# ---------------------------------------------------------------------------


class SSOError(Exception):
    """Raised on SSO configuration failures."""


class SSOConfig:
    """Persist SSO/SAML provider settings and build OAuth2 auth URLs.

    The descriptor lives at ``~/.synth/sso.json`` and has the shape::

        {
          "provider": "google",
          "client_id": "abc.apps.googleusercontent.com",
          "client_secret_env": "SYNTH_SSO_SECRET",
          "redirect_uri": "http://localhost:8080/callback"
        }
    """

    # Known provider authorize endpoints.
    _AUTH_ENDPOINTS: dict[str, str] = {
        "google": "https://accounts.google.com/o/oauth2/v2/auth",
        "github": "https://github.com/login/oauth/authorize",
        "microsoft": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "okta": "https://okta.com/oauth2/v1/authorize",
        "keycloak": "",  # server_url-dependent; built at runtime
        "generic": "",  # caller-supplied via provider == endpoint URL
    }

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path: Path = (
            Path(config_path)
            if config_path
            else _config_dir() / SSO_FILENAME
        )

    def configure(
        self,
        provider: str,
        client_id: str,
        client_secret_env: str,
        redirect_uri: str,
    ) -> None:
        """Persist SSO provider settings."""
        if not isinstance(provider, str) or not provider.strip():
            raise SSOError("provider must be a non-empty string")
        if not isinstance(client_id, str) or not client_id.strip():
            raise SSOError("client_id must be a non-empty string")
        if not isinstance(client_secret_env, str) or not client_secret_env.strip():
            raise SSOError("client_secret_env must be a non-empty string")
        if not isinstance(redirect_uri, str) or not redirect_uri.strip():
            raise SSOError("redirect_uri must be a non-empty string")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": provider.strip(),
            "client_id": client_id.strip(),
            "client_secret_env": client_secret_env.strip(),
            "redirect_uri": redirect_uri.strip(),
        }
        try:
            self.config_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise SSOError(
                f"cannot write {self.config_path}: {exc}"
            ) from exc

    def _load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            text = self.config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SSOError(
                f"cannot read {self.config_path}: {exc}"
            ) from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SSOError(
                f"invalid JSON in {self.config_path}: {exc}"
            ) from exc
        return data if isinstance(data, dict) else {}

    def is_configured(self) -> bool:
        """Return True iff SSO has been configured (all four fields present)."""
        data = self._load()
        if not data:
            return False
        required = ("provider", "client_id", "client_secret_env", "redirect_uri")
        return all(
            isinstance(data.get(k), str) and data.get(k, "").strip()
            for k in required
        )

    def get_auth_url(self) -> str:
        """Build the OAuth2 authorization URL for the configured provider.

        Uses ``response_type=code`` and ``access_type=offline`` (for refresh
        tokens on providers that support it). Scope is provider-specific.

        Raises:
            SSOError: if SSO is not configured or the provider is unknown.
        """
        data = self._load()
        if not data:
            raise SSOError("SSO not configured; call configure() first")
        provider = (data.get("provider") or "").strip()
        client_id = (data.get("client_id") or "").strip()
        redirect = (data.get("redirect_uri") or "").strip()
        if not provider or not client_id or not redirect:
            raise SSOError(
                "SSO config incomplete: provider/client_id/redirect_uri required"
            )

        endpoint = self._AUTH_ENDPOINTS.get(provider)
        if endpoint is None:
            # Treat unknown provider strings that look like URLs as endpoints.
            if provider.startswith("http://") or provider.startswith("https://"):
                endpoint = provider
            else:
                raise SSOError(f"unknown SSO provider: {provider!r}")
        if not endpoint:
            raise SSOError(
                f"provider {provider!r} requires a server_url; not supported "
                f"in get_auth_url()"
            )

        scope = "openid email profile"
        if provider == "github":
            scope = "read:user user:email"
        elif provider == "google":
            scope = "openid email profile"

        # URL-encode via urllib (stdlib).
        from urllib.parse import urlencode

        params = urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect,
                "scope": scope,
                "access_type": "offline",
                "prompt": "consent",
            }
        )
        sep = "&" if ("?" in endpoint) else "?"
        return f"{endpoint}{sep}{params}"


# ---------------------------------------------------------------------------
# FineTuningExport
# ---------------------------------------------------------------------------


class FineTuningError(Exception):
    """Raised on fine-tuning export failures."""


class FineTuningExport:
    """Export sessions from sessions.db to a JSONL fine-tuning file.

    Each output line is a JSON object of the form::

        {"messages": [{"role": "user", "content": "..."}, ...]}

    Sessions with fewer than ``min_messages`` messages are skipped.

    Args:
        sessions_db: Path to sessions.db. Defaults to ~/.synth/sessions.db.
        output_dir: Directory for the exported file. Defaults to ~/.synth/.
    """

    def __init__(
        self,
        sessions_db: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        if sessions_db is None:
            sessions_db = _config_dir() / SESSION_DB_FILENAME
        self.sessions_db: Path = Path(sessions_db)
        self.output_dir: Path = Path(output_dir) if output_dir else _config_dir()

    def export_sessions(
        self,
        format: str = "jsonl",
        min_messages: int = MIN_MESSAGES_DEFAULT,
    ) -> Path:
        """Export qualifying sessions to a file and return its path.

        Args:
            format: Output format. Currently only ``"jsonl"`` is supported.
            min_messages: Minimum number of messages a session must have to
                be included (default 5).

        Returns:
            Path to the exported file.

        Raises:
            FineTuningError: if sessions.db is missing, the format is
                unsupported, or the export write fails.
        """
        if format != "jsonl":
            raise FineTuningError(f"unsupported format: {format!r}")
        if not isinstance(min_messages, int) or min_messages < 1:
            raise FineTuningError("min_messages must be a positive int")
        if not self.sessions_db.exists():
            raise FineTuningError(
                f"sessions.db not found: {self.sessions_db}"
            )

        # Open read-only to avoid mutating the live DB.
        uri = f"file:{self.sessions_db}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise FineTuningError(
                f"cannot open sessions.db: {exc}"
            ) from exc

        rows: list[sqlite3.Row]
        try:
            # Group messages by session, count them, and only keep sessions
            # with >= min_messages. The messages table schema (role, content)
            # is stable per session.py.
            rows = conn.execute(
                """
                SELECT session_id, role, content
                FROM messages
                WHERE session_id IN (
                    SELECT session_id
                    FROM messages
                    GROUP BY session_id
                    HAVING COUNT(*) >= ?
                )
                ORDER BY session_id, id ASC
                """,
                (min_messages,),
            ).fetchall()
        except sqlite3.Error as exc:
            conn.close()
            raise FineTuningError(
                f"failed to query sessions: {exc}"
            ) from exc
        finally:
            pass
        conn.close()

        # Bucket messages per session (preserve order).
        buckets: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for r in rows:
            sid = r["session_id"]
            if sid not in buckets:
                buckets[sid] = []
                order.append(sid)
            buckets[sid].append(
                {
                    "role": r["role"],
                    "content": r["content"] or "",
                }
            )

        # Write JSONL.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = self.output_dir / f"finetune_{stamp}.jsonl"
        try:
            with out_path.open("w", encoding="utf-8") as fh:
                for sid in order:
                    payload = {"messages": buckets[sid]}
                    fh.write(json.dumps(payload, ensure_ascii=False))
                    fh.write("\n")
        except OSError as exc:
            raise FineTuningError(
                f"failed to write export: {exc}"
            ) from exc

        return out_path


__all__ = [
    "MARKETPLACE_DB_FILENAME",
    "ONPREM_FILENAME",
    "SSO_FILENAME",
    "MIN_MESSAGES_DEFAULT",
    "MarketplaceError",
    "SkillMarketplace",
    "SkillSignerError",
    "SkillSigner",
    "ComplianceError",
    "ComplianceChecker",
    "AirGappedError",
    "AirGappedMode",
    "OnPremError",
    "OnPremConfig",
    "SSOError",
    "SSOConfig",
    "FineTuningError",
    "FineTuningExport",
]
