"""Tests for src/synth/cloud.py — Cloud & Collab (spec v2.0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth.cloud import (
    AUDIT_LOG_FILENAME,
    MAX_WORKSPACE_FILES,
    WORKSPACE_DIR_NAME,
    AuditLog,
    AuditLogError,
    CloudClient,
    CloudConfig,
    CloudError,
    RBAC,
    RBACError,
    TeamSkillLibrary,
    Workspace,
    WorkspaceError,
)
from synth.skills import Skill


# ======================================================================
# fixtures
# ======================================================================

@pytest.fixture
def disabled_client() -> CloudClient:
    return CloudClient(CloudConfig(enabled=False))


@pytest.fixture
def enabled_client() -> CloudClient:
    return CloudClient(CloudConfig(enabled=True))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws_root")


@pytest.fixture
def team_lib(tmp_path: Path) -> TeamSkillLibrary:
    return TeamSkillLibrary(tmp_path / "team.db")


@pytest.fixture
def audit_log(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "audit.log")


def make_skill(name: str = "hello", code: str = 'result = "hi"') -> Skill:
    return Skill(name=name, description="a test skill", code=code)


# ======================================================================
# CloudConfig
# ======================================================================

def test_cloud_config_defaults_disabled():
    cfg = CloudConfig()
    assert cfg.enabled is False
    assert cfg.api_url.startswith("http")
    assert cfg.api_key_env == "SYNTH_CLOUD_API_KEY"


def test_cloud_config_can_enable():
    cfg = CloudConfig(enabled=True, api_url="https://example.com", api_key_env="K")
    assert cfg.enabled is True
    assert cfg.api_url == "https://example.com"
    assert cfg.api_key_env == "K"


# ======================================================================
# CloudClient — disabled
# ======================================================================

def test_ping_disabled_returns_false(disabled_client):
    assert disabled_client.ping() is False


def test_upload_skill_disabled_returns_empty(disabled_client):
    assert disabled_client.upload_skill("hello", "code") == ""


def test_download_skill_disabled_returns_empty(disabled_client):
    assert disabled_client.download_skill("cloud-abc") == ""


def test_list_cloud_skills_disabled_returns_empty(disabled_client):
    assert disabled_client.list_cloud_skills() == []


# ======================================================================
# CloudClient — enabled (mock)
# ======================================================================

def test_ping_enabled_returns_true(enabled_client):
    assert enabled_client.ping() is True


def test_upload_skill_enabled_returns_cloud_id(enabled_client):
    cid = enabled_client.upload_skill("hello", "result = 1")
    assert cid.startswith("cloud-hello-")


def test_upload_skill_rejects_empty_name(enabled_client):
    with pytest.raises(CloudError):
        enabled_client.upload_skill("", "code")


def test_upload_skill_rejects_empty_content(enabled_client):
    with pytest.raises(CloudError):
        enabled_client.upload_skill("name", "   ")


def test_download_skill_enabled_returns_content(enabled_client):
    content = enabled_client.download_skill("cloud-demo-aaa")
    assert "mock skill" in content


def test_download_skill_rejects_empty_id(enabled_client):
    with pytest.raises(CloudError):
        enabled_client.download_skill("")


def test_list_cloud_skills_enabled_returns_list(enabled_client):
    skills = enabled_client.list_cloud_skills()
    assert len(skills) == 2
    assert all("cloud_id" in s for s in skills)


# ======================================================================
# Workspace
# ======================================================================

def test_workspace_create_returns_id(workspace):
    wid = workspace.create_workspace("project-x")
    assert isinstance(wid, str) and len(wid) > 0


def test_workspace_create_makes_dir(workspace):
    wid = workspace.create_workspace("project-x")
    assert (workspace.base_dir / wid).is_dir()


def test_workspace_create_rejects_empty_name(workspace):
    with pytest.raises(WorkspaceError):
        workspace.create_workspace("")


def test_workspace_add_and_get_file(workspace):
    wid = workspace.create_workspace("proj")
    workspace.add_file(wid, "main.py", "print('hi')")
    assert workspace.get_file(wid, "main.py") == "print('hi')"


def test_workspace_add_file_nested_path(workspace):
    wid = workspace.create_workspace("proj")
    workspace.add_file(wid, "src/deep/mod.py", "x = 1")
    assert workspace.get_file(wid, "src/deep/mod.py") == "x = 1"


def test_workspace_add_file_rejects_traversal(workspace):
    wid = workspace.create_workspace("proj")
    with pytest.raises(WorkspaceError):
        workspace.add_file(wid, "../../etc/passwd", "bad")


def test_workspace_add_file_rejects_absolute(workspace):
    wid = workspace.create_workspace("proj")
    with pytest.raises(WorkspaceError):
        workspace.add_file(wid, "/etc/passwd", "bad")


def test_workspace_get_file_missing_raises(workspace):
    wid = workspace.create_workspace("proj")
    with pytest.raises(WorkspaceError):
        workspace.get_file(wid, "nope.py")


def test_workspace_get_file_missing_workspace_raises(workspace):
    with pytest.raises(WorkspaceError):
        workspace.get_file("nonexistent", "main.py")


def test_workspace_list_files(workspace):
    wid = workspace.create_workspace("proj")
    workspace.add_file(wid, "a.py", "1")
    workspace.add_file(wid, "b.py", "2")
    workspace.add_file(wid, "sub/c.py", "3")
    files = workspace.list_files(wid)
    assert "a.py" in files
    assert "b.py" in files
    assert "sub/c.py" in files
    # meta file should not appear
    assert ".meta.json" not in files


def test_workspace_list_files_empty(workspace):
    wid = workspace.create_workspace("proj")
    assert workspace.list_files(wid) == []


def test_workspace_share_returns_true(workspace):
    wid = workspace.create_workspace("proj")
    assert workspace.share_workspace(wid, "user@example.com") is True


def test_workspace_share_idempotent(workspace):
    wid = workspace.create_workspace("proj")
    workspace.share_workspace(wid, "user@example.com")
    workspace.share_workspace(wid, "user@example.com")
    # Read meta to verify only one entry.
    meta_path = workspace.base_dir / wid / ".meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["shared_with"] == ["user@example.com"]


def test_workspace_share_rejects_empty_email(workspace):
    wid = workspace.create_workspace("proj")
    with pytest.raises(WorkspaceError):
        workspace.share_workspace(wid, "")


def test_workspace_share_nonexistent_returns_false(workspace):
    assert workspace.share_workspace("nope", "user@example.com") is False


# ======================================================================
# TeamSkillLibrary
# ======================================================================

def test_team_publish_returns_team_id(team_lib):
    skill = make_skill("greet", 'result = "hi"')
    tid = team_lib.publish_to_team(skill)
    assert tid.startswith("team-greet-")


def test_team_publish_rejects_empty_name(team_lib):
    skill = Skill(name="", description="d", code="result = 1")
    with pytest.raises(CloudError):
        team_lib.publish_to_team(skill)


def test_team_publish_rejects_empty_code(team_lib):
    skill = Skill(name="x", description="d", code="   ")
    with pytest.raises(CloudError):
        team_lib.publish_to_team(skill)


def test_team_install_round_trip(team_lib):
    skill = make_skill("greet", 'result = "hi"')
    tid = team_lib.publish_to_team(skill, published_by="alice")
    installed = team_lib.install_from_team(tid)
    assert installed.name == "greet"
    assert 'result = "hi"' in installed.code
    # Should also be in the local SkillStore table now.
    loaded = team_lib.load_skill("greet")
    assert loaded is not None
    assert loaded.code == 'result = "hi"'


def test_team_install_unknown_id_raises(team_lib):
    with pytest.raises(CloudError):
        team_lib.install_from_team("team-unknown-1234")


def test_team_install_rejects_empty_id(team_lib):
    with pytest.raises(CloudError):
        team_lib.install_from_team("")


def test_team_list_skills(team_lib):
    s1 = make_skill("a", "result = 1")
    s2 = make_skill("b", "result = 2")
    team_lib.publish_to_team(s1)
    team_lib.publish_to_team(s2)
    listed = team_lib.list_team_skills()
    assert len(listed) == 2
    names = {e["name"] for e in listed}
    assert names == {"a", "b"}


def test_team_list_skills_empty(team_lib):
    assert team_lib.list_team_skills() == []


def test_team_publish_records_publisher(team_lib):
    skill = make_skill("greet", 'result = "hi"')
    tid = team_lib.publish_to_team(skill, published_by="bob")
    listed = team_lib.list_team_skills()
    match = [e for e in listed if e["team_id"] == tid]
    assert match and match[0]["published_by"] == "bob"


# ======================================================================
# RBAC
# ======================================================================

def test_rbac_default_roles_exist():
    rbac = RBAC()
    roles = rbac.list_roles()
    assert "admin" in roles
    assert "editor" in roles
    assert "viewer" in roles


def test_rbac_admin_all_permissions():
    rbac = RBAC()
    assert rbac.check_permission("admin", "run") is True
    assert rbac.check_permission("admin", "delete") is True
    assert rbac.check_permission("admin", "anything") is True


def test_rbac_editor_permissions():
    rbac = RBAC()
    assert rbac.check_permission("editor", "run") is True
    assert rbac.check_permission("editor", "write") is True
    assert rbac.check_permission("editor", "git") is True
    assert rbac.check_permission("editor", "delete") is False


def test_rbac_viewer_permissions():
    rbac = RBAC()
    assert rbac.check_permission("viewer", "run") is True
    assert rbac.check_permission("viewer", "read") is True
    assert rbac.check_permission("viewer", "write") is False
    assert rbac.check_permission("viewer", "git") is False


def test_rbac_unknown_role_denied():
    rbac = RBAC()
    assert rbac.check_permission("ghost", "run") is False


def test_rbac_empty_role_denied():
    rbac = RBAC()
    assert rbac.check_permission("", "run") is False


def test_rbac_empty_action_denied():
    rbac = RBAC()
    assert rbac.check_permission("admin", "") is False


def test_rbac_add_custom_role():
    rbac = RBAC()
    rbac.add_role("tester", ["run", "test"])
    assert rbac.check_permission("tester", "run") is True
    assert rbac.check_permission("tester", "test") is True
    assert rbac.check_permission("tester", "write") is False


def test_rbac_add_role_with_wildcard():
    rbac = RBAC()
    rbac.add_role("super", ["*"])
    assert rbac.check_permission("super", "anything") is True


def test_rbac_add_role_rejects_empty_name():
    rbac = RBAC()
    with pytest.raises(RBACError):
        rbac.add_role("", ["run"])


def test_rbac_add_role_rejects_bad_permissions():
    rbac = RBAC()
    with pytest.raises(RBACError):
        rbac.add_role("bad", "run")  # type: ignore[arg-type]
    with pytest.raises(RBACError):
        rbac.add_role("bad2", ["", "run"])


def test_rbac_remove_role():
    rbac = RBAC()
    assert rbac.remove_role("viewer") is True
    assert "viewer" not in rbac.list_roles()
    # Second removal returns False.
    assert rbac.remove_role("viewer") is False


def test_rbac_remove_role_rejects_empty():
    rbac = RBAC()
    with pytest.raises(RBACError):
        rbac.remove_role("")


def test_rbac_list_roles_is_copy():
    rbac = RBAC()
    roles = rbac.list_roles()
    roles["injected"] = ["x"]
    assert "injected" not in rbac.list_roles()


# ======================================================================
# AuditLog
# ======================================================================

def test_audit_append_writes_line(audit_log, tmp_path):
    audit_log.append({"user": "alice", "action": "run", "resource": "skill:hi", "result": "ok"})
    text = (tmp_path / "audit.log").read_text()
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["user"] == "alice"
    assert entry["action"] == "run"
    assert "timestamp" in entry


def test_audit_append_adds_timestamp(audit_log):
    audit_log.append({"user": "bob", "action": "read"})
    entries = audit_log.query()
    assert len(entries) == 1
    assert "timestamp" in entries[0]


def test_audit_query_returns_all(audit_log):
    audit_log.append({"user": "a", "action": "run"})
    audit_log.append({"user": "b", "action": "read"})
    entries = audit_log.query()
    assert len(entries) == 2


def test_audit_query_filter_by_user(audit_log):
    audit_log.append({"user": "alice", "action": "run"})
    audit_log.append({"user": "bob", "action": "run"})
    entries = audit_log.query({"user": "alice"})
    assert len(entries) == 1
    assert entries[0]["user"] == "alice"


def test_audit_query_filter_by_action(audit_log):
    audit_log.append({"user": "a", "action": "run"})
    audit_log.append({"user": "b", "action": "delete"})
    entries = audit_log.query({"action": "delete"})
    assert len(entries) == 1
    assert entries[0]["action"] == "delete"


def test_audit_query_filter_by_resource(audit_log):
    audit_log.append({"user": "a", "action": "run", "resource": "r1"})
    audit_log.append({"user": "b", "action": "run", "resource": "r2"})
    entries = audit_log.query({"resource": "r2"})
    assert len(entries) == 1


def test_audit_query_filter_by_result(audit_log):
    audit_log.append({"user": "a", "action": "run", "result": "ok"})
    audit_log.append({"user": "b", "action": "run", "result": "error"})
    entries = audit_log.query({"result": "error"})
    assert len(entries) == 1


def test_audit_query_filter_by_timestamp_range(audit_log):
    audit_log.append({"user": "a", "action": "run", "timestamp": "2026-01-01T00:00:00+00:00"})
    audit_log.append({"user": "b", "action": "run", "timestamp": "2026-06-01T00:00:00+00:00"})
    audit_log.append({"user": "c", "action": "run", "timestamp": "2026-12-01T00:00:00+00:00"})
    entries = audit_log.query({"timestamp_start": "2026-03-01T00:00:00+00:00", "timestamp_end": "2026-09-01T00:00:00+00:00"})
    assert len(entries) == 1
    assert entries[0]["user"] == "b"


def test_audit_query_empty_when_no_match(audit_log):
    audit_log.append({"user": "a", "action": "run"})
    assert audit_log.query({"user": "nobody"}) == []


def test_audit_query_empty_when_log_empty(audit_log, tmp_path):
    # Touch but don't write entries.
    assert audit_log.query() == []


def test_audit_query_newest_first(audit_log):
    audit_log.append({"user": "a", "action": "run", "timestamp": "2026-01-01T00:00:00+00:00"})
    audit_log.append({"user": "b", "action": "run", "timestamp": "2026-06-01T00:00:00+00:00"})
    entries = audit_log.query()
    assert entries[0]["user"] == "b"
    assert entries[1]["user"] == "a"


def test_audit_append_rejects_non_dict(audit_log):
    with pytest.raises(AuditLogError):
        audit_log.append("not a dict")  # type: ignore[arg-type]


def test_audit_append_skips_malformed_lines_on_query(audit_log, tmp_path):
    audit_log.append({"user": "a", "action": "run"})
    # Corrupt the log file with a malformed line.
    with (tmp_path / "audit.log").open("a") as fh:
        fh.write("{not valid json\n")
    entries = audit_log.query()
    assert len(entries) == 1  # malformed line skipped


def test_audit_constants():
    assert AUDIT_LOG_FILENAME == "audit.log"
    assert WORKSPACE_DIR_NAME == "workspaces"
    assert MAX_WORKSPACE_FILES == 1000
