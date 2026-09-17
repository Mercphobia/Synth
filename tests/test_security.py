"""Tests for security scanning functionality."""

import tempfile
from pathlib import Path

import pytest

from synth.security import (
    Finding,
    check_dependencies,
    full_scan,
    report_to_markdown,
    sast_scan,
    scan_directory,
    scan_file,
)


def test_scan_file_finds_aws_key():
    """Test that AWS access keys are detected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("AWS_KEY = 'AKIAIOSFODNN7EXAMPLE'\n")
        path = Path(f.name)
    
    try:
        findings = scan_file(path)
        assert len(findings) >= 1
        aws_finding = next(f for f in findings if f.category == "AWS Access Key")
        assert aws_finding.severity == "HIGH"
        # Verify masking (first 8 chars + ...)
        assert "AKIAIOSF" in aws_finding.description
        assert "..." in aws_finding.description
    finally:
        path.unlink()


def test_scan_file_finds_github_token():
    """Test that GitHub tokens are detected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("GITHUB_TOKEN = 'ghp_1234567890abcdefghijklmnopqrstuvwxyz'\n")
        path = Path(f.name)
    
    try:
        findings = scan_file(path)
        assert len(findings) >= 1
        token_finding = next(f for f in findings if f.category == "GitHub Token")
        assert token_finding.severity == "CRITICAL"  # GitHub tokens are CRITICAL
        assert "ghp_1234" in token_finding.description
    finally:
        path.unlink()


def test_scan_file_masks_secret():
    """Test that secret values are masked in findings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("api_key = 'super_secret_key_12345'\n")
        path = Path(f.name)
    
    try:
        findings = scan_file(path)
        assert len(findings) == 1
        assert "super_secret_key_12345" not in findings[0].description
        assert "..." in findings[0].description
    finally:
        path.unlink()


def test_scan_file_skips_binary():
    """Test that binary files are skipped."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"Binary\x00content\n")
        path = Path(f.name)
    
    try:
        findings = scan_file(path)
        assert len(findings) == 0
    finally:
        path.unlink()


def test_scan_file_skips_oversized():
    """Test that files exceeding max_size are skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("small content")
        path = Path(f.name)
    
    try:
        findings = scan_file(path, max_size=5)
        assert len(findings) == 0
    finally:
        path.unlink()


def test_scan_directory_finds_multiple():
    """Test scanning a directory tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create files with secrets
        (root / "file1.txt").write_text("aws = 'AKIAIOSFODNN7EXAMPLE'\n")
        (root / "file2.txt").write_text("token = 'ghp_1234567890abcdefghijklmnopqrstuvwxyz'\n")
        
        findings = scan_directory(root)
        assert len(findings) >= 2
        assert any(f.category == "AWS Access Key" for f in findings)
        assert any(f.category == "GitHub Token" for f in findings)


def test_scan_directory_skips_git():
    """Test that .git directory is skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        git_dir = root / ".git"
        git_dir.mkdir()
        (git_dir / "secret.txt").write_text("password = 'hidden'\n")
        
        findings = scan_directory(root)
        assert len(findings) == 0


def test_sast_scan_finds_eval():
    """Test that dangerous eval() is detected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("result = eval(user_input)\n")
        path = Path(f.name)
    
    try:
        findings = sast_scan(path)
        assert len(findings) == 1
        assert findings[0].category == "Dangerous eval()"
        assert findings[0].severity == "HIGH"
    finally:
        path.unlink()


def test_sast_scan_finds_exec():
    """Test that dangerous exec() is detected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("exec(code_string)\n")
        path = Path(f.name)
    
    try:
        findings = sast_scan(path)
        assert len(findings) == 1
        assert findings[0].category == "Dangerous exec()"
    finally:
        path.unlink()


def test_sast_scan_finds_shell_injection():
    """Test that subprocess shell=True is detected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import subprocess\nsubprocess.run(cmd, shell=True)\n")
        path = Path(f.name)
    
    try:
        findings = sast_scan(path)
        assert len(findings) == 1
        assert "Shell injection" in findings[0].category
    finally:
        path.unlink()


def test_sast_scan_skips_non_python():
    """Test that non-Python files are skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("eval(something)\n")
        path = Path(f.name)
    
    try:
        findings = sast_scan(path)
        assert len(findings) == 0
    finally:
        path.unlink()


def test_check_dependencies_finds_vulnerable():
    """Test that vulnerable dependencies are detected."""
    requirements = """
jinja2==3.0.0
requests==2.25.0
flask==2.3.2
"""
    findings = check_dependencies(requirements)
    assert len(findings) >= 2
    assert any("jinja2" in f.description for f in findings)
    assert any("requests" in f.description for f in findings)


def test_check_dependencies_safe_versions():
    """Test that safe versions pass."""
    requirements = """
jinja2==3.1.2
requests==2.31.0
flask==2.3.2
"""
    findings = check_dependencies(requirements)
    assert len(findings) == 0


def test_check_dependencies_ignores_unpinned():
    """Test that unpinned dependencies are ignored."""
    requirements = """
jinja2
requests>=2.25.0
"""
    findings = check_dependencies(requirements)
    assert len(findings) == 0


def test_full_scan_comprehensive():
    """Test full scan on a repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create vulnerable files
        (root / "config.py").write_text("api_key = 'AKIAIOSFODNN7EXAMPLE'\n")
        (root / "app.py").write_text("import subprocess\nsubprocess.run(cmd, shell=True)\n")
        (root / "requirements.txt").write_text("jinja2==3.0.0\n")
        
        report = full_scan(root)
        
        assert report["total"] >= 3
        assert report["summary"]["CRITICAL"] >= 0
        assert report["summary"]["HIGH"] >= 1
        assert report["summary"]["MEDIUM"] >= 1
        assert len(report["findings"]) >= 3


def test_report_to_markdown_format():
    """Test markdown report generation."""
    report = {
        "total": 3,
        "summary": {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 1, "LOW": 0},
        "findings": [
            Finding("CRITICAL", "GitHub Token", "config.py", 10, "Found token"),
            Finding("HIGH", "AWS Key", "env.py", 5, "Found key"),
            Finding("MEDIUM", "eval()", "app.py", 20, "Dangerous eval"),
        ],
    }
    
    markdown = report_to_markdown(report)
    
    assert "# Security Scan Report" in markdown
    assert "## Summary" in markdown
    assert "CRITICAL" in markdown
    assert "HIGH" in markdown
    assert "MEDIUM" in markdown
    assert "## Findings" in markdown
    assert "config.py:10" in markdown


def test_report_to_markdown_no_findings():
    """Test markdown report when no issues found."""
    report = {
        "total": 0,
        "summary": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "findings": [],
    }
    
    markdown = report_to_markdown(report)
    assert "No Issues Found" in markdown
