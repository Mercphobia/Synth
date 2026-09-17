"""Security scanner for Synth (defensive-only, spec 6.14).

Scans repositories for common security issues: secrets, SAST patterns,
CVE vulnerabilities, and OWASP misconfigurations. All scanning is static
analysis; never executes scanned code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth.constants import MAX_FILE_SIZE
from synth.tools import safe_path


@dataclass(frozen=True)
class Finding:
    """One security finding from a scan."""
    
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str
    path: str
    line: int
    description: str


class SecurityError(Exception):
    """Raised when security scanning fails."""


# --- Secret scanning ---

SECRET_PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key", "HIGH"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub Token", "CRITICAL"),
    (r"-----BEGIN.*PRIVATE KEY-----", "Private Key", "CRITICAL"),
    (r"(?:api_key|password|secret|token)\s*=\s*['\"][^'\"]{8,}['\"]", "Hardcoded Secret", "HIGH"),
]


def scan_file(path: Path, max_size: int = MAX_FILE_SIZE) -> list[Finding]:
    """Scan a single file for secrets."""
    if not path.exists() or not path.is_file():
        return []
    
    if path.stat().st_size > max_size:
        return []
    
    # Skip binary
    try:
        with path.open("rb") as f:
            if b"\x00" in f.read(1024):
                return []
    except OSError:
        return []
    
    findings = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_num, line in enumerate(text.splitlines(), 1):
            for pattern, category, severity in SECRET_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Mask secret value
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        secret = match.group(0)
                        masked = secret[:8] + "..." if len(secret) > 8 else secret
                        findings.append(Finding(
                            severity=severity,
                            category=category,
                            path=str(path),
                            line=line_num,
                            description=f"Found {category}: {masked}",
                        ))
    except OSError:
        pass
    
    return findings


def scan_directory(root: Path, max_files: int = 500) -> list[Finding]:
    """Scan a directory tree for secrets."""
    if not root.exists() or not root.is_dir():
        return []
    
    findings = []
    count = 0
    for path in root.rglob("*"):
        if count >= max_files:
            break
        if path.is_file() and not _should_skip(path):
            findings.extend(scan_file(path))
            count += 1
    
    return findings


def _should_skip(path: Path) -> bool:
    """Check if path should be skipped (git, venv, node_modules, etc)."""
    parts = path.parts
    skip_dirs = {".git", "venv", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
    return any(part in skip_dirs for part in parts)


# --- SAST scanning ---

SAST_PATTERNS = [
    (r"\beval\s*\(", "Dangerous eval()", "HIGH"),
    (r"\bexec\s*\(", "Dangerous exec()", "HIGH"),
    (r"subprocess\.run\(.*shell=True", "Shell injection risk", "MEDIUM"),
    (r"os\.system\s*\(", "Unsafe os.system", "MEDIUM"),
    (r"pickle\.loads?\s*\(", "Deserialization risk", "MEDIUM"),
    (r"yaml\.load\s*\([^)]*(?!SafeLoader)", "Unsafe YAML load", "MEDIUM"),
    (r"http://(?!localhost)", "Insecure HTTP URL", "LOW"),
]


def sast_scan(path: Path) -> list[Finding]:
    """Scan Python file for SAST patterns."""
    if not path.exists() or not path.is_file():
        return []
    
    if path.suffix != ".py":
        return []
    
    findings = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_num, line in enumerate(text.splitlines(), 1):
            for pattern, category, severity in SAST_PATTERNS:
                if re.search(pattern, line):
                    findings.append(Finding(
                        severity=severity,
                        category=category,
                        path=str(path),
                        line=line_num,
                        description=f"SAST: {category}",
                    ))
    except OSError:
        pass
    
    return findings


# --- CVE checking ---

VULN_DATABASE = [
    ("jinja2", "<3.1.2", "CVE-2024-22195: sandbox escape"),
    ("requests", "<2.31.0", "CVE-2023-32681: proxy credential leak"),
    ("pyyaml", "<6.0", "CVE-2020-14343: arbitrary code execution"),
    ("django", "<4.2.7", "CVE-2023-46695: potential ReDoS"),
    ("flask", "<2.3.2", "CVE-2023-30861: cookie session fixation"),
]


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse version string into comparable tuple."""
    try:
        parts = version.strip().split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0,)


def _version_less_than(v1: str, v2: str) -> bool:
    """Check if v1 < v2."""
    try:
        return _parse_version(v1) < _parse_version(v2)
    except Exception:
        return False


def check_dependencies(requirements_text: str) -> list[Finding]:
    """Check pinned dependencies against known vulnerabilities."""
    findings = []
    
    for line in requirements_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Parse package==version
        if "==" not in line:
            continue
        
        parts = line.split("==")
        if len(parts) != 2:
            continue
        
        package, version = parts[0].strip().lower(), parts[1].strip()
        
        for vuln_pkg, vuln_range, description in VULN_DATABASE:
            if package == vuln_pkg:
                if vuln_range.startswith("<") and _version_less_than(version, vuln_range[1:]):
                    findings.append(Finding(
                        severity="HIGH",
                        category="Vulnerable Dependency",
                        path="requirements.txt",
                        line=0,
                        description=f"{package}=={version} is {description}",
                    ))
    
    return findings


# --- Report generation ---

def full_scan(root: Path) -> dict[str, Any]:
    """Run all scanners on a repository."""
    if not root.exists() or not root.is_dir():
        return {"error": "root directory does not exist", "findings": []}
    
    findings = []
    
    # Secret scan
    findings.extend(scan_directory(root))
    
    # SAST scan
    for path in root.rglob("*.py"):
        if not _should_skip(path):
            findings.extend(sast_scan(path))
    
    # CVE check
    req_file = root / "requirements.txt"
    if req_file.exists():
        try:
            findings.extend(check_dependencies(req_file.read_text(encoding="utf-8")))
        except OSError:
            pass
    
    # Summary
    summary = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for f in findings:
        if f.severity in summary:
            summary[f.severity] += 1
    
    return {
        "findings": findings,
        "summary": summary,
        "total": len(findings),
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    """Convert scan report to markdown."""
    lines = ["# Security Scan Report\n"]
    
    summary = report.get("summary", {})
    lines.append("## Summary\n")
    lines.append(f"- **CRITICAL**: {summary.get('CRITICAL', 0)}")
    lines.append(f"- **HIGH**: {summary.get('HIGH', 0)}")
    lines.append(f"- **MEDIUM**: {summary.get('MEDIUM', 0)}")
    lines.append(f"- **LOW**: {summary.get('LOW', 0)}")
    lines.append(f"- **Total**: {report.get('total', 0)}\n")
    
    findings = report.get("findings", [])
    if findings:
        lines.append("## Findings\n")
        for f in findings[:20]:  # Limit to 20 findings
            lines.append(f"### [{f.severity}] {f.category}")
            lines.append(f"- **File**: `{f.path}:{f.line}`")
            lines.append(f"- **Description**: {f.description}\n")
        
        if len(findings) > 20:
            lines.append(f"... and {len(findings) - 20} more findings\n")
    else:
        lines.append("## No Issues Found ✓\n")
    
    return "\n".join(lines)
