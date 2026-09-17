"""Cloud & collaboration for Synth (spec v2.0 — Cloud & Collab).

Local-only mock of a cloud collaboration layer. No real cloud dependency:
every network call is stubbed so the module works offline. The design mirrors
OllamaClient (urllib scaffolding, graceful degrade) and SkillStore (SQLite +
parameterized SQL + path validation).

Components:
* CloudConfig      — dataclass holding api_url / api_key_env / enabled flag.
* CloudClient      — mock HTTP client; returns empty/False when disabled.
* Workspace        — shared workspace file manager with traversal protection.
* TeamSkillLibrary — extends SkillStore with team publish/install flows.
* RBAC             — role-based access control with add/remove roles.
* AuditLog         — JSON-lines audit log with filtered query.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR
from synth.skills import Skill, SkillStore, _validate_name

logger = logging.getLogger(__name__)

# --- constants -----------------------------------------------------------

# Subdir of ~/.synth/ holding workspace directories.
WORKSPACE_DIR_NAME = "workspaces"

# Audit log file lives directly under ~/.synth/.
AUDIT_LOG_FILENAME = "audit.log"

# Hard cap on files per workspace — prevents unbounded growth.
MAX_WORKSPACE_FILES = 1000

# SQLite DB for the team skill library (mirrors skills.db layout).
TEAM_SKILLS_DB_FILENAME = "team_skills.db"

# RBAC default role definitions. '*' means all actions.
_DEFAULT_ROLES: dict[str, list[str]] = {
    "admin": ["*"],
    "editor": ["run", "read", "write", "git"],
    "viewer": ["run", "read"],
}

# Required keys for an audit-log entry.
_AUDIT_ENTRY_KEYS = ("timestamp", "user", "action", "resource", "result")


# --- errors --------------------------------------------------------------

class CloudError(Exception):
    """Raised when a cloud operation fails (config, workspace, or team skill)."""


class WorkspaceError(Exception):
    """Raised when a workspace file operation fails or is rejected."""


class RBACError(Exception):
    """Raised when an RBAC operation is invalid."""


class AuditLogError(Exception):
    """Raised when the audit log cannot be read or written."""


# --- helpers -------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time in ISO-8601 (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_epoch() -> int:
    """Return current Unix epoch seconds."""
    return int(time.time())


def _validate_rel_path(path: str) -> Path:
    """Return a Path for *path* after rejecting traversal escapes.

    Accepts relative paths like ``"src/main.py"`` or ``"data/file.txt"``.
    Rejects any component that is ``..`` or any absolute path.
    """
    if not isinstance(path, str) or not path.strip():
        raise WorkspaceError("path must be a non-empty string")
    stripped = path.strip()
    if stripped.startswith("/"):
        raise WorkspaceError(f"absolute paths are not allowed: {path!r}")
    parts = stripped.split("/")
    if ".." in parts:
        raise WorkspaceError(f"path traversal rejected: {path!r}")
    return Path(*parts)


# ======================================================================
# CloudConfig
# ======================================================================

@dataclass
class CloudConfig:
    """Configuration for the Synth Cloud client.

    Attributes:
        api_url: Base URL of the (mock) cloud API.
        api_key_env: Name of the environment variable holding the API key.
        enabled: Whether cloud features are active. Defaults to False so a
            fresh install works fully offline.
    """

    api_url: str = "https://cloud.synth.dev/api"
    api_key_env: str = "SYNTH_CLOUD_API_KEY"
    enabled: bool = False


# ======================================================================
# CloudClient
# ======================================================================

class CloudClient:
    """Mock HTTP client for Synth Cloud.

    Uses urllib scaffolding (like OllamaClient) but never makes a real
    request when disabled. When enabled, returns deterministic mock data
    so tests are reproducible without a live server.

    Args:
        config: CloudConfig controlling behaviour. Defaults to disabled.
        timeout: Per-request timeout in seconds (unused when disabled).
    """

    def __init__(self, config: CloudConfig | None = None, timeout: int = 10) -> None:
        self.config = config or CloudConfig()
        self.timeout = timeout

    # --- liveness ---

    def ping(self) -> bool:
        """Return True when the cloud is reachable (mock).

        Returns False when disabled. When enabled, returns True to simulate
        a healthy cloud connection without a real network call.
        """
        if not self.config.enabled:
            return False
        logger.debug("CloudClient.ping: mock cloud reachable (enabled)")
        return True

    # --- skill sync ---

    def upload_skill(self, name: str, content: str) -> str:
        """Upload a skill to the cloud; return its cloud ID.

        Returns "" when disabled. When enabled, returns a deterministic
        mock cloud ID derived from the skill name.
        """
        if not self.config.enabled:
            return ""
        if not isinstance(name, str) or not name.strip():
            raise CloudError("skill name must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise CloudError("skill content must be non-empty")
        cloud_id = f"cloud-{name.strip().lower()}-{uuid.uuid4().hex[:8]}"
        logger.debug("CloudClient.upload_skill: mock upload %s -> %s", name, cloud_id)
        return cloud_id

    def download_skill(self, cloud_id: str) -> str:
        """Download a skill's content from the cloud by its ID.

        Returns "" when disabled or when the ID is unknown. When enabled,
        returns mock content for any well-formed cloud ID.
        """
        if not self.config.enabled:
            return ""
        if not isinstance(cloud_id, str) or not cloud_id.strip():
            raise CloudError("cloud_id must be a non-empty string")
        # Mock content: a trivial skill body referencing the cloud ID.
        return f"# mock skill from cloud {cloud_id}\nresult = 'downloaded'\n"

    def list_cloud_skills(self) -> list[dict]:
        """List skills available on the cloud.

        Returns [] when disabled. When enabled, returns a small mock list.
        """
        if not self.config.enabled:
            return []
        return [
            {"cloud_id": "cloud-demo-aaaaaaaa", "name": "demo", "description": "mock skill"},
            {"cloud_id": "cloud-greet-bbbbbbbb", "name": "greet", "description": "mock greet"},
        ]


# ======================================================================
# Workspace
# ======================================================================

class Workspace:
    """Shared workspace file manager.

    Each workspace is a directory under ``base_dir/<id>/``. File paths are
    validated to reject ``..`` traversal and absolute paths.

    Args:
        base_dir: Root directory for workspaces. Defaults to
            ``~/.synth/workspaces/``.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            self.base_dir = Path.home() / CONFIG_DIR / WORKSPACE_DIR_NAME
        else:
            self.base_dir = Path(base_dir)
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorkspaceError(f"cannot create workspace root {self.base_dir}: {exc}") from exc

    def _workspace_dir(self, workspace_id: str) -> Path:
        """Return the directory for *workspace_id*, raising if it doesn't exist."""
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise WorkspaceError("workspace_id must be a non-empty string")
        ws_dir = self.base_dir / workspace_id
        if not ws_dir.is_dir():
            raise WorkspaceError(f"workspace not found: {workspace_id}")
        return ws_dir

    def create_workspace(self, name: str) -> str:
        """Create a new workspace; return its generated ID.

        Args:
            name: Human-friendly workspace name (stored in metadata.json).
        """
        if not isinstance(name, str) or not name.strip():
            raise WorkspaceError("workspace name must be a non-empty string")
        workspace_id = uuid.uuid4().hex[:12]
        ws_dir = self.base_dir / workspace_id
        try:
            ws_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise WorkspaceError(f"workspace id collision: {workspace_id}") from exc
        except OSError as exc:
            raise WorkspaceError(f"cannot create workspace {ws_dir}: {exc}") from exc
        meta = {"name": name.strip(), "created_at": _now_iso(), "id": workspace_id}
        try:
            (ws_dir / ".meta.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            raise WorkspaceError(f"cannot write workspace metadata: {exc}") from exc
        return workspace_id

    def add_file(self, workspace_id: str, path: str, content: str) -> None:
        """Write *content* to *path* inside the workspace.

        Rejects traversal (``..``) and absolute paths. Enforces
        MAX_WORKSPACE_FILES on the workspace file count.
        """
        ws_dir = self._workspace_dir(workspace_id)
        rel = _validate_rel_path(path)
        target = ws_dir / rel
        # Reject if target resolves outside the workspace dir.
        try:
            target.resolve().relative_to(ws_dir.resolve())
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace: {path!r}") from exc
        # Enforce file count cap.
        existing = self.list_files(workspace_id)
        if len(existing) >= MAX_WORKSPACE_FILES and str(rel) not in existing:
            raise WorkspaceError(
                f"workspace file limit reached ({MAX_WORKSPACE_FILES})"
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise WorkspaceError(f"cannot write {path!r}: {exc}") from exc

    def get_file(self, workspace_id: str, path: str) -> str:
        """Read the contents of *path* inside the workspace.

        Raises WorkspaceError if the file does not exist or the path escapes.
        """
        ws_dir = self._workspace_dir(workspace_id)
        rel = _validate_rel_path(path)
        target = ws_dir / rel
        try:
            target.resolve().relative_to(ws_dir.resolve())
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace: {path!r}") from exc
        if not target.is_file():
            raise WorkspaceError(f"file not found: {path!r}")
        try:
            return target.read_text(encoding="utf-8")
        except OSError as exc:
            raise WorkspaceError(f"cannot read {path!r}: {exc}") from exc

    def list_files(self, workspace_id: str) -> list[str]:
        """Return all file paths (relative) in the workspace, sorted."""
        ws_dir = self._workspace_dir(workspace_id)
        files: list[str] = []
        for entry in ws_dir.rglob("*"):
            if entry.is_file() and entry.name != ".meta.json":
                files.append(str(entry.relative_to(ws_dir)))
        return sorted(files)

    def share_workspace(self, workspace_id: str, user_email: str) -> bool:
        """Mock: record a share action and return True.

        No real email is sent; the action is logged for audit purposes.
        Returns False if the workspace does not exist.
        """
        if not isinstance(user_email, str) or not user_email.strip():
            raise WorkspaceError("user_email must be a non-empty string")
        try:
            ws_dir = self._workspace_dir(workspace_id)
        except WorkspaceError:
            return False
        # Mock: append share record to metadata.
        meta_path = ws_dir / ".meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            meta = {}
        shared_with: list[str] = meta.get("shared_with", [])
        email = user_email.strip()
        if email not in shared_with:
            shared_with.append(email)
        meta["shared_with"] = shared_with
        try:
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            raise WorkspaceError(f"cannot persist share: {exc}") from exc
        logger.info("workspace %s shared with %s", workspace_id, email)
        return True


# ======================================================================
# TeamSkillLibrary
# ======================================================================

_TEAM_SCHEMA = """
CREATE TABLE IF NOT EXISTS team_skills (
    team_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    code TEXT NOT NULL,
    published_at TEXT NOT NULL,
    published_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_team_skills_name ON team_skills(name);
"""


class TeamSkillLibrary(SkillStore):
    """Extends SkillStore with team publish/install flows.

    Team skills live in a separate SQLite table (team_skills) so they don't
    collide with the local skills table. The inherited SkillStore table is
    used for installed team skills.

    Args:
        db_path: Optional override for the SQLite database path.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(db_path)
        try:
            self._conn.executescript(_TEAM_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise CloudError(f"cannot init team_skills schema: {exc}") from exc

    def publish_to_team(self, skill: Skill, published_by: str | None = None) -> str:
        """Publish a skill to the team library; return its team ID."""
        try:
            name = _validate_name(skill.name)
        except Exception as exc:
            raise CloudError(f"invalid skill name: {exc}") from exc
        if not skill.description.strip():
            raise CloudError("skill description must be non-empty")
        if not skill.code.strip():
            raise CloudError("skill code must be non-empty")
        team_id = f"team-{name}-{uuid.uuid4().hex[:8]}"
        try:
            self._conn.execute(
                "INSERT INTO team_skills (team_id, name, description, code, published_at, published_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (team_id, name, skill.description.strip(), skill.code, _now_iso(), published_by),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise CloudError(f"failed to publish skill {name!r}: {exc}") from exc
        return team_id

    def install_from_team(self, team_id: str) -> Skill:
        """Install a team skill into the local SkillStore; return the Skill."""
        if not isinstance(team_id, str) or not team_id.strip():
            raise CloudError("team_id must be a non-empty string")
        try:
            row = self._conn.execute(
                "SELECT team_id, name, description, code, published_at, published_by "
                "FROM team_skills WHERE team_id = ?",
                (team_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise CloudError(f"failed to query team skill {team_id!r}: {exc}") from exc
        if row is None:
            raise CloudError(f"team skill not found: {team_id}")
        skill = Skill(
            name=row["name"],
            description=row["description"],
            code=row["code"],
            created_at=row["published_at"],
        )
        # Save into the local skills table (inherited from SkillStore).
        self.save_skill(skill)
        return skill

    def list_team_skills(self) -> list[dict]:
        """List all published team skills as a list of dicts."""
        try:
            rows = self._conn.execute(
                "SELECT team_id, name, description, published_at, published_by "
                "FROM team_skills ORDER BY published_at DESC"
            ).fetchall()
        except sqlite3.Error as exc:
            raise CloudError(f"failed to list team skills: {exc}") from exc
        return [
            {
                "team_id": r["team_id"],
                "name": r["name"],
                "description": r["description"],
                "published_at": r["published_at"],
                "published_by": r["published_by"],
            }
            for r in rows
        ]


# ======================================================================
# RBAC
# ======================================================================

class RBAC:
    """Role-based access control.

    Roles map a role name to a list of permitted action strings. The
    wildcard ``"*"`` grants all actions. Roles can be added or removed
    at runtime.
    """

    def __init__(self, roles: dict[str, list[str]] | None = None) -> None:
        # Copy defaults so mutations don't leak into the module-level dict.
        self.roles: dict[str, list[str]] = {
            name: list(perms) for name, perms in (roles or _DEFAULT_ROLES).items()
        }

    def check_permission(self, role: str, action: str) -> bool:
        """Return True if *role* is permitted to perform *action*."""
        if not isinstance(role, str) or not role.strip():
            return False
        if not isinstance(action, str) or not action.strip():
            return False
        perms = self.roles.get(role)
        if perms is None:
            return False
        return "*" in perms or action.strip() in perms

    def add_role(self, name: str, permissions: list[str]) -> None:
        """Add or overwrite a role with the given permissions list."""
        if not isinstance(name, str) or not name.strip():
            raise RBACError("role name must be a non-empty string")
        if not isinstance(permissions, list):
            raise RBACError("permissions must be a list")
        for perm in permissions:
            if not isinstance(perm, str) or not perm.strip():
                raise RBACError(f"permission must be a non-empty string, got {perm!r}")
        self.roles[name.strip()] = [p.strip() for p in permissions]

    def remove_role(self, name: str) -> bool:
        """Remove a role; return True iff it existed."""
        if not isinstance(name, str) or not name.strip():
            raise RBACError("role name must be a non-empty string")
        if name.strip() not in self.roles:
            return False
        del self.roles[name.strip()]
        return True

    def list_roles(self) -> dict[str, list[str]]:
        """Return a copy of all roles."""
        return {name: list(perms) for name, perms in self.roles.items()}


# ======================================================================
# AuditLog
# ======================================================================

class AuditLog:
    """JSON-lines audit log written to ``~/.synth/audit.log``.

    Each entry is a dict with keys: timestamp, user, action, resource,
    result. Entries are appended as one JSON object per line.

    Args:
        log_path: Path to the audit log file. Defaults to
            ``~/.synth/audit.log``.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        if log_path is None:
            self.log_path = Path.home() / CONFIG_DIR / AUDIT_LOG_FILENAME
        else:
            self.log_path = Path(log_path)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.touch()
        except OSError as exc:
            raise AuditLogError(f"cannot create audit log {self.log_path}: {exc}") from exc

    def append(self, entry: dict) -> None:
        """Append one entry to the audit log as a JSON line.

        Ensures the entry has all required keys; fills missing optional
        fields with defaults. Always adds a timestamp if absent.
        """
        if not isinstance(entry, dict):
            raise AuditLogError("entry must be a dict")
        record = dict(entry)
        if "timestamp" not in record:
            record["timestamp"] = _now_iso()
        # Ensure all expected keys are present (empty string default).
        for key in _AUDIT_ENTRY_KEYS:
            record.setdefault(key, "")
        try:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except (OSError, TypeError) as exc:
            raise AuditLogError(f"cannot append to audit log: {exc}") from exc

    def query(self, filters: dict | None = None) -> list[dict]:
        """Query audit entries, optionally filtered.

        Filters is a dict that can contain:
            user: exact match on the user field.
            action: exact match on the action field.
            timestamp_start: ISO-8601 lower bound (inclusive).
            timestamp_end: ISO-8601 upper bound (inclusive).
            resource: exact match on the resource field.
            result: exact match on the result field.

        Returns a list of matching entry dicts, newest first.
        """
        filters = filters or {}
        results: list[dict] = []
        if not self.log_path.exists():
            return results
        try:
            text = self.log_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AuditLogError(f"cannot read audit log: {exc}") from exc
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed audit line: %s", line[:80])
                continue
            if not isinstance(entry, dict):
                continue
            if self._matches(entry, filters):
                results.append(entry)
        # Newest first: reverse chronological by timestamp.
        results.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return results

    @staticmethod
    def _matches(entry: dict, filters: dict) -> bool:
        """Return True if *entry* matches all non-empty filters."""
        for key in ("user", "action", "resource", "result"):
            val = filters.get(key)
            if val is not None and str(val) and entry.get(key) != val:
                return False
        ts = entry.get("timestamp", "")
        start = filters.get("timestamp_start")
        if start and str(start) and ts < str(start):
            return False
        end = filters.get("timestamp_end")
        if end and str(end) and ts > str(end):
            return False
        return True
