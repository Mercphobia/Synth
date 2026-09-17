"""Tests for src/synth/enterprise.py (v2.0 Enterprise tier)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from synth.enterprise import (
    MIN_MESSAGES_DEFAULT,
    AirGappedError,
    AirGappedMode,
    ComplianceChecker,
    FineTuningError,
    FineTuningExport,
    MarketplaceError,
    OnPremConfig,
    OnPremError,
    SkillMarketplace,
    SkillSigner,
    SSOConfig,
    SSOError,
)
from synth.skills import Skill


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def marketplace(tmp_path: Path) -> SkillMarketplace:
    mp = SkillMarketplace(tmp_path / "marketplace.db")
    yield mp
    mp.close()


@pytest.fixture
def signer() -> SkillSigner:
    return SkillSigner()


@pytest.fixture
def sample_skill() -> Skill:
    return Skill(
        name="hello-world",
        description="prints hello",
        code='result = "hello"',
    )


# --- SkillMarketplace -------------------------------------------------------


def test_marketplace_publish_returns_id(marketplace: SkillMarketplace, sample_skill: Skill):
    mid = marketplace.publish(sample_skill, author="alice")
    assert isinstance(mid, str) and len(mid) > 0


def test_marketplace_publish_and_search(marketplace: SkillMarketplace, sample_skill: Skill):
    mid = marketplace.publish(sample_skill, author="alice")
    hits = marketplace.search("hello")
    assert len(hits) == 1
    assert hits[0]["id"] == mid
    assert hits[0]["author"] == "alice"
    assert hits[0]["name"] == "hello-world"


def test_marketplace_search_empty_query_raises(marketplace: SkillMarketplace):
    with pytest.raises(MarketplaceError):
        marketplace.search("")


def test_marketplace_search_no_match(marketplace: SkillMarketplace, sample_skill: Skill):
    marketplace.publish(sample_skill, author="alice")
    assert marketplace.search("nonexistent") == []


def test_marketplace_install_returns_skill(marketplace: SkillMarketplace, sample_skill: Skill):
    mid = marketplace.publish(sample_skill, author="alice")
    installed = marketplace.install(mid)
    assert isinstance(installed, Skill)
    assert installed.name == "hello-world"
    assert installed.code == 'result = "hello"'


def test_marketplace_install_unknown_id_raises(marketplace: SkillMarketplace):
    with pytest.raises(MarketplaceError):
        marketplace.install("deadbeef")


def test_marketplace_install_bumps_count(marketplace: SkillMarketplace, sample_skill: Skill):
    mid = marketplace.publish(sample_skill, author="alice")
    marketplace.install(mid)
    marketplace.install(mid)
    hits = marketplace.search("hello")
    assert hits[0]["install_count"] == 2


def test_marketplace_rate_updates_rating(marketplace: SkillMarketplace, sample_skill: Skill):
    mid = marketplace.publish(sample_skill, author="alice")
    marketplace.rate(mid, 5)
    marketplace.rate(mid, 3)
    hits = marketplace.search("hello")
    assert hits[0]["rating"] == 4.0


def test_marketplace_rate_invalid_stars_raises(marketplace: SkillMarketplace, sample_skill: Skill):
    mid = marketplace.publish(sample_skill, author="alice")
    with pytest.raises(MarketplaceError):
        marketplace.rate(mid, 6)
    with pytest.raises(MarketplaceError):
        marketplace.rate(mid, 0)


def test_marketplace_rate_unknown_id_raises(marketplace: SkillMarketplace):
    with pytest.raises(MarketplaceError):
        marketplace.rate("deadbeef", 4)


def test_marketplace_publish_empty_author_raises(marketplace: SkillMarketplace, sample_skill: Skill):
    with pytest.raises(MarketplaceError):
        marketplace.publish(sample_skill, author="")


# --- SkillSigner -----------------------------------------------------------


def test_signer_sign_returns_hex(signer: SkillSigner, sample_skill: Skill):
    sig = signer.sign(sample_skill, private_key_pem="secret-key")
    # hex string of 64 chars (sha256)
    assert isinstance(sig, str)
    assert len(sig) == 64
    int(sig, 16)  # valid hex


def test_signer_verify_valid(signer: SkillSigner, sample_skill: Skill):
    sig = signer.sign(sample_skill, private_key_pem="secret-key")
    assert signer.verify(sample_skill, sig, public_key_pem="secret-key") is True


def test_signer_verify_wrong_key(signer: SkillSigner, sample_skill: Skill):
    sig = signer.sign(sample_skill, private_key_pem="secret-key")
    assert signer.verify(sample_skill, sig, public_key_pem="other-key") is False


def test_signer_verify_tampered_skill(signer: SkillSigner, sample_skill: Skill):
    sig = signer.sign(sample_skill, private_key_pem="secret-key")
    tampered = Skill(name=sample_skill.name, description="x", code="result = 'evil'")
    assert signer.verify(tampered, sig, public_key_pem="secret-key") is False


def test_signer_verify_empty_signature(signer: SkillSigner, sample_skill: Skill):
    assert signer.verify(sample_skill, "", public_key_pem="secret-key") is False


def test_signer_sign_empty_key_raises(signer: SkillSigner, sample_skill: Skill):
    with pytest.raises(Exception):
        signer.sign(sample_skill, private_key_pem="")


# --- ComplianceChecker ------------------------------------------------------


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Plant files under a tmp repo root."""
    root = tmp_path / "repo"
    root.mkdir()
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


def test_compliance_owasp_detects_eval(tmp_path: Path):
    root = _make_repo(tmp_path, {"app.py": "def f():\n    eval('1')\n"})
    checker = ComplianceChecker()
    findings = checker.check_standard("owasp", root)
    cats = [f["category"] for f in findings]
    assert any("eval" in c.lower() for c in cats)
    assert all(f["file"] and f["severity"] and f["description"] for f in findings)


def test_compliance_owasp_detects_hardcoded_secret(tmp_path: Path):
    root = _make_repo(
        tmp_path,
        {"config.py": 'api_key = "ghp_" + "x" * 36\ntoken = "sk-abc123def456"\n'},
    )
    checker = ComplianceChecker()
    findings = checker.check_standard("owasp", root)
    assert len(findings) >= 1


def test_compliance_owasp_clean_repo(tmp_path: Path):
    root = _make_repo(tmp_path, {"clean.py": "x = 1 + 2\nprint(x)\n"})
    checker = ComplianceChecker()
    assert checker.check_standard("owasp", root) == []


def test_compliance_gdpr_detects_email(tmp_path: Path):
    root = _make_repo(
        tmp_path,
        {"users.py": 'admin = "admin@company.com"\n'},
    )
    checker = ComplianceChecker()
    findings = checker.check_standard("gdpr", root)
    cats = [f["category"] for f in findings]
    assert any("email" in c.lower() for c in cats)


def test_compliance_gdpr_detects_ssn(tmp_path: Path):
    root = _make_repo(tmp_path, {"data.txt": "SSN: 123-45-6789\n"})
    checker = ComplianceChecker()
    findings = checker.check_standard("gdpr", root)
    cats = [f["category"] for f in findings]
    assert any("ssn" in c.lower() for c in cats)


def test_compliance_gdpr_clean_repo(tmp_path: Path):
    root = _make_repo(tmp_path, {"ok.py": "x = 42\n"})
    checker = ComplianceChecker()
    assert checker.check_standard("gdpr", root) == []


def test_compliance_soc2_detects_shell_true(tmp_path: Path):
    root = _make_repo(
        tmp_path,
        {"run.py": "import subprocess\nsubprocess.run('ls', shell=True)\n"},
    )
    checker = ComplianceChecker()
    findings = checker.check_standard("soc2", root)
    assert len(findings) >= 1


def test_compliance_soc2_detects_os_system(tmp_path: Path):
    root = _make_repo(tmp_path, {"os_call.py": "import os\nos.system('rm -rf /')\n"})
    checker = ComplianceChecker()
    findings = checker.check_standard("soc2", root)
    assert len(findings) >= 1


def test_compliance_iso27001_detects_hardcoded_credential(tmp_path: Path):
    root = _make_repo(
        tmp_path,
        {"auth.py": 'password = "supersecret123"\n'},
    )
    checker = ComplianceChecker()
    findings = checker.check_standard("iso27001", root)
    cats = [f["category"] for f in findings]
    assert any("credential" in c.lower() for c in cats)


def test_compliance_iso27001_detects_wildcard_permission(tmp_path: Path):
    root = _make_repo(tmp_path, {"perm.py": "permission = '*'\n"})
    checker = ComplianceChecker()
    findings = checker.check_standard("iso27001", root)
    assert len(findings) >= 1


def test_compliance_unknown_standard_raises(tmp_path: Path):
    root = _make_repo(tmp_path, {"x.py": "x = 1\n"})
    checker = ComplianceChecker()
    with pytest.raises(Exception):
        checker.check_standard("pci-dss", root)


def test_compliance_missing_root_raises(tmp_path: Path):
    checker = ComplianceChecker()
    with pytest.raises(Exception):
        checker.check_standard("owasp", tmp_path / "nope")


# --- AirGappedMode --------------------------------------------------------


def test_airgapped_detect_returns_bool():
    ag = AirGappedMode()
    result = ag.detect()
    assert isinstance(result, bool)


def test_airgapped_enable_sets_env(monkeypatch):
    monkeypatch.delenv("SYNTH_AIRGAPPED", raising=False)
    ag = AirGappedMode()
    assert ag.is_enabled() is False
    ag.enable()
    assert ag.is_enabled() is True
    assert __import__("os").environ.get("SYNTH_AIRGAPPED") == "1"


def test_airgapped_enable_disable_cycle(monkeypatch):
    monkeypatch.delenv("SYNTH_AIRGAPPED", raising=False)
    ag = AirGappedMode()
    ag.enable()
    assert ag.is_enabled() is True
    ag.disable()
    assert ag.is_enabled() is False


def test_airgapped_detect_returns_false_when_enabled(monkeypatch):
    monkeypatch.setenv("SYNTH_AIRGAPPED", "1")
    ag = AirGappedMode()
    assert ag.detect() is False


# --- OnPremConfig ----------------------------------------------------------


def test_onprem_save_and_load(tmp_path: Path):
    cfg_path = tmp_path / "onprem.json"
    op = OnPremConfig(cfg_path)
    data = {
        "server_url": "https://synth.internal",
        "port": 8443,
        "auth_token_env": "SYNTH_AUTH_TOKEN",
        "allowed_models": ["llama-3", "qwen-2.5"],
    }
    op.save(data)
    loaded = op.load()
    assert loaded == data


def test_onprem_load_missing_returns_empty(tmp_path: Path):
    op = OnPremConfig(tmp_path / "nonexistent.json")
    assert op.load() == {}


def test_onprem_validate_valid_config(tmp_path: Path):
    op = OnPremConfig(tmp_path / "onprem.json")
    op.save({
        "server_url": "https://synth.internal",
        "port": 8443,
        "auth_token_env": "SYNTH_AUTH_TOKEN",
        "allowed_models": ["llama-3"],
    })
    assert op.validate() is True


def test_onprem_validate_missing_fields(tmp_path: Path):
    op = OnPremConfig(tmp_path / "onprem.json")
    op.save({"server_url": "https://synth.internal"})
    assert op.validate() is False


def test_onprem_validate_empty_file(tmp_path: Path):
    op = OnPremConfig(tmp_path / "onprem.json")
    assert op.validate() is False


def test_onprem_validate_bad_port(tmp_path: Path):
    op = OnPremConfig(tmp_path / "onprem.json")
    op.save({
        "server_url": "https://synth.internal",
        "port": 99999,
        "auth_token_env": "SYNTH_AUTH_TOKEN",
        "allowed_models": ["llama-3"],
    })
    assert op.validate() is False


def test_onprem_validate_empty_models(tmp_path: Path):
    op = OnPremConfig(tmp_path / "onprem.json")
    op.save({
        "server_url": "https://synth.internal",
        "port": 8443,
        "auth_token_env": "SYNTH_AUTH_TOKEN",
        "allowed_models": [],
    })
    assert op.validate() is False


# --- SSOConfig -------------------------------------------------------------


def test_sso_configure_and_is_configured(tmp_path: Path):
    sso = SSOConfig(tmp_path / "sso.json")
    assert sso.is_configured() is False
    sso.configure(
        provider="google",
        client_id="abc.apps.googleusercontent.com",
        client_secret_env="SYNTH_SSO_SECRET",
        redirect_uri="http://localhost:8080/callback",
    )
    assert sso.is_configured() is True


def test_sso_get_auth_url_google(tmp_path: Path):
    sso = SSOConfig(tmp_path / "sso.json")
    sso.configure(
        provider="google",
        client_id="abc.apps.googleusercontent.com",
        client_secret_env="SYNTH_SSO_SECRET",
        redirect_uri="http://localhost:8080/callback",
    )
    url = sso.get_auth_url()
    assert "accounts.google.com" in url
    assert "client_id=abc.apps.googleusercontent.com" in url
    assert "response_type=code" in url


def test_sso_get_auth_url_github(tmp_path: Path):
    sso = SSOConfig(tmp_path / "sso.json")
    sso.configure(
        provider="github",
        client_id="gh-client",
        client_secret_env="SYNTH_SSO_SECRET",
        redirect_uri="http://localhost:8080/callback",
    )
    url = sso.get_auth_url()
    assert "github.com" in url
    assert "client_id=gh-client" in url


def test_sso_get_auth_url_not_configured_raises(tmp_path: Path):
    sso = SSOConfig(tmp_path / "sso.json")
    with pytest.raises(SSOError):
        sso.get_auth_url()


def test_sso_get_auth_url_unknown_provider_raises(tmp_path: Path):
    sso = SSOConfig(tmp_path / "sso.json")
    sso.configure(
        provider="unknown-sso",
        client_id="x",
        client_secret_env="Y",
        redirect_uri="http://localhost:8080/callback",
    )
    with pytest.raises(SSOError):
        sso.get_auth_url()


def test_sso_configure_empty_provider_raises(tmp_path: Path):
    sso = SSOConfig(tmp_path / "sso.json")
    with pytest.raises(SSOError):
        sso.configure(
            provider="",
            client_id="x",
            client_secret_env="Y",
            redirect_uri="http://localhost:8080/callback",
        )


# --- FineTuningExport ------------------------------------------------------


def _seed_sessions_db(db_path: Path) -> None:
    """Create a sessions.db with the same schema as session.py and seed it."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at INTEGER,
            updated_at INTEGER
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            tool_call_id TEXT,
            ts INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        """
    )
    now = int(time.time())
    # Session A: 6 messages (qualifies for default min_messages=5)
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("sess-a", "alpha", now, now),
    )
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, ts) "
            "VALUES (?, ?, ?, NULL, NULL, ?)",
            ("sess-a", role, f"msg-{i}", now + i),
        )
    # Session B: 2 messages (below threshold)
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("sess-b", "beta", now, now),
    )
    for i in range(2):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, ts) "
            "VALUES (?, ?, ?, NULL, NULL, ?)",
            ("sess-b", "user", f"b-{i}", now + i),
        )
    conn.commit()
    conn.close()


def test_finetune_export_produces_jsonl(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    _seed_sessions_db(db_path)
    out_dir = tmp_path / "export"
    exporter = FineTuningExport(sessions_db=db_path, output_dir=out_dir)
    path = exporter.export_sessions()
    assert path.exists()
    assert path.suffix == ".jsonl"


def test_finetune_export_filters_by_min_messages(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    _seed_sessions_db(db_path)
    exporter = FineTuningExport(sessions_db=db_path, output_dir=tmp_path / "out")
    path = exporter.export_sessions(min_messages=5)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    # Only session A qualifies (6 messages >= 5)
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert "messages" in obj
    assert len(obj["messages"]) == 6


def test_finetune_export_format_correct(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    _seed_sessions_db(db_path)
    exporter = FineTuningExport(sessions_db=db_path, output_dir=tmp_path / "out")
    path = exporter.export_sessions()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        obj = json.loads(line)
        assert "messages" in obj
        assert isinstance(obj["messages"], list)
        for m in obj["messages"]:
            assert "role" in m
            assert "content" in m


def test_finetune_export_min_messages_filter(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    _seed_sessions_db(db_path)
    exporter = FineTuningExport(sessions_db=db_path, output_dir=tmp_path / "out")
    # With min_messages=2, both sessions qualify
    path = exporter.export_sessions(min_messages=2)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_finetune_export_missing_db_raises(tmp_path: Path):
    exporter = FineTuningExport(
        sessions_db=tmp_path / "nope.db",
        output_dir=tmp_path / "out",
    )
    with pytest.raises(FineTuningError):
        exporter.export_sessions()


def test_finetune_export_unsupported_format_raises(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    _seed_sessions_db(db_path)
    exporter = FineTuningExport(sessions_db=db_path, output_dir=tmp_path / "out")
    with pytest.raises(FineTuningError):
        exporter.export_sessions(format="csv")


def test_finetune_export_default_min_messages_constant():
    assert MIN_MESSAGES_DEFAULT == 5
