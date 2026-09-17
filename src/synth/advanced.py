"""Synth v2.0 advanced features.

Implemented (stdlib-only):
  MemoryGraph, VotingSystem, HandoffProtocol, CrossSessionMessaging,
  SessionBranch, CodeGraph, CommandPalette, TokenSaviour, SemanticDedup,
  PromptCompressor.

Stubs (raise NotImplementedError — require external dependencies):
  AgentCanvas, BrowserAutomation, ImageInput, PDFExcelParser,
  VoiceInput, VoiceOutput, WebDashboard, MultiBackend, PRCreation,
  Provenance, Notebook, GPUAcceleration.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sqlite3
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from synth.best_of_n import Candidate

# --- Constants --------------------------------------------------------------

#: Maximum BFS depth for MemoryGraph.traverse.
GRAPH_MAX_DEPTH = 3

#: Jaccard word-overlap threshold above which two messages are considered
#: duplicates by SemanticDedup.
DEDUP_THRESHOLD = 0.8

#: Default compression ratio for PromptCompressor.compress — keep ~50% of
#: the original text measured by character count.
COMPRESS_DEFAULT_RATIO = 0.5

#: Directory names to skip when CodeGraph walks a repository tree.
_SKIP_DIRS = frozenset({
    "__pycache__", ".venv", "venv", ".git", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", ".eggs", "htmlcov", ".hypothesis",
})


# === MemoryGraph ============================================================

class MemoryGraph:
    """In-memory directed graph of nodes with labeled links.

    Each node is a dict ``{id, type, content, links}`` where *links* is a
    list of ``{target, label}`` dicts.  The graph can be serialised to and
    restored from a SQLite database file.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add_node(self, id: str, type: str, content: str) -> None:
        """Create or overwrite a node identified by *id*."""
        self.nodes[id] = {
            "id": id,
            "type": type,
            "content": content,
            "links": [],
        }

    def link(self, from_id: str, to_id: str, label: str = "") -> None:
        """Add a labeled directed edge *from_id* → *to_id*."""
        if from_id not in self.nodes:
            raise KeyError(f"Unknown source node: {from_id}")
        if to_id not in self.nodes:
            raise KeyError(f"Unknown target node: {to_id}")
        self.nodes[from_id]["links"].append(
            {"target": to_id, "label": label}
        )

    def traverse(self, start_id: str,
                 max_depth: int = GRAPH_MAX_DEPTH) -> list[str]:
        """Breadth-first traversal from *start_id*, up to *max_depth* hops.

        Returns the list of visited node IDs in BFS order.  An unknown
        *start_id* yields an empty list.
        """
        if start_id not in self.nodes:
            return []
        visited: list[str] = []
        seen: set[str] = set()
        queue: list[tuple[str, int]] = [(start_id, 0)]
        while queue:
            node_id, depth = queue.pop(0)
            if node_id in seen:
                continue
            seen.add(node_id)
            visited.append(node_id)
            if depth + 1 < max_depth:
                node = self.nodes.get(node_id)
                if node:
                    for link in node["links"]:
                        target = link["target"]
                        if target not in seen:
                            queue.append((target, depth + 1))
        return visited

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the graph."""
        lines = ["digraph memory {"]
        for nid, node in self.nodes.items():
            lines.append(
                f'    "{nid}" [label="{nid}\\n({node["type"]})"];'
            )
        for nid, node in self.nodes.items():
            for link in node["links"]:
                lbl = link.get("label", "")
                if lbl:
                    lines.append(
                        f'    "{nid}" -> "{link["target"]}" '
                        f'[label="{lbl}"];'
                    )
                else:
                    lines.append(
                        f'    "{nid}" -> "{link["target"]}";'
                    )
        lines.append("}")
        return "\n".join(lines)

    def save(self, db_path: str | Path) -> None:
        """Persist all nodes and links to a SQLite database file."""
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(
                "CREATE TABLE IF NOT EXISTS graph_nodes ("
                "  id TEXT PRIMARY KEY,"
                "  type TEXT,"
                "  content TEXT"
                ");"
                "CREATE TABLE IF NOT EXISTS graph_links ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  from_id TEXT,"
                "  to_id TEXT,"
                "  label TEXT"
                ");"
            )
            conn.execute("DELETE FROM graph_nodes")
            conn.execute("DELETE FROM graph_links")
            for nid, node in self.nodes.items():
                conn.execute(
                    "INSERT INTO graph_nodes (id, type, content) "
                    "VALUES (?, ?, ?)",
                    (nid, node["type"], node["content"]),
                )
                for link in node["links"]:
                    conn.execute(
                        "INSERT INTO graph_links (from_id, to_id, label) "
                        "VALUES (?, ?, ?)",
                        (nid, link["target"], link.get("label", "")),
                    )
            conn.commit()
        finally:
            conn.close()

    def load(self, db_path: str | Path) -> None:
        """Load nodes and links from a SQLite database file, replacing all
        current content."""
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            self.nodes = {}
            for row in conn.execute(
                "SELECT id, type, content FROM graph_nodes"
            ):
                self.nodes[row["id"]] = {
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "links": [],
                }
            for row in conn.execute(
                "SELECT from_id, to_id, label FROM graph_links"
            ):
                nid = row["from_id"]
                if nid in self.nodes:
                    self.nodes[nid]["links"].append(
                        {"target": row["to_id"], "label": row["label"]}
                    )
        finally:
            conn.close()


# === VotingSystem ===========================================================

class VotingSystem:
    """Majority-vote selection across 3+ best-of-N candidates.

    If two or more candidates share the same first-100-character prefix,
    the highest-scoring member of that majority group wins.  Otherwise the
    single highest-scoring candidate wins.
    """

    def vote(self, candidates: list[Candidate]) -> Candidate:
        if len(candidates) < 3:
            raise ValueError(
                "VotingSystem requires at least 3 candidates"
            )
        groups: dict[str, list[Candidate]] = defaultdict(list)
        for cand in candidates:
            prefix = (cand.text or "")[:100]
            groups[prefix].append(cand)

        best_group: list[Candidate] | None = None
        for prefix, group in groups.items():
            if prefix and len(group) >= 2:
                if best_group is None or len(group) > len(best_group):
                    best_group = group

        if best_group is not None:
            return max(best_group, key=lambda c: c.score)
        return max(candidates, key=lambda c: c.score)


# === HandoffProtocol ========================================================

class HandoffProtocol:
    """In-memory handoff registry: create a context handoff, retrieve it by
    ID."""

    def __init__(self) -> None:
        self._handoffs: dict[str, dict] = {}

    def create_handoff(self, agent_name: str, context: Any) -> str:
        """Store a handoff for *agent_name* with the given *context*.

        Returns a UUID handoff ID.
        """
        handoff_id = str(uuid.uuid4())
        self._handoffs[handoff_id] = {
            "agent_name": agent_name,
            "context": context,
        }
        return handoff_id

    def receive_handoff(self, handoff_id: str) -> dict:
        """Retrieve a stored handoff by ID.

        Raises:
            KeyError: If *handoff_id* is not registered.
        """
        if handoff_id not in self._handoffs:
            raise KeyError(f"Unknown handoff: {handoff_id}")
        return self._handoffs[handoff_id]


# === CrossSessionMessaging ==================================================

class CrossSessionMessaging:
    """In-memory per-session message queue.

    Messages sent to a session accumulate until they are consumed by
    ``receive``, which returns and clears the queue.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[dict]] = defaultdict(list)

    def send(self, from_session: str, to_session: str,
             message: str) -> None:
        """Enqueue *message* for *to_session*, tagged with *from_session*."""
        self._queues[to_session].append({
            "from": from_session,
            "to": to_session,
            "message": message,
        })

    def receive(self, session_id: str) -> list[dict]:
        """Pop and return all pending messages for *session_id*."""
        msgs = list(self._queues.get(session_id, []))
        self._queues[session_id] = []
        return msgs


# === SessionBranch ==========================================================

class SessionBranch:
    """Branch, merge, and compare sessions via a SessionStore."""

    def __init__(self, session_store: Any) -> None:
        self.store = session_store

    def branch(self, session_id: str, label: str) -> str:
        """Create a new session that is a copy of *session_id*.

        Returns the new session ID.
        """
        messages = self.store.load_session(session_id)
        new_id = self.store.create_session(title=label)
        for msg in messages:
            self.store.append_message(new_id, msg)
        return new_id

    def merge(self, target_session: str, source_session: str) -> None:
        """Append all messages from *source_session* to *target_session*."""
        messages = self.store.load_session(source_session)
        for msg in messages:
            self.store.append_message(target_session, msg)

    def compare(self, session_a: str, session_b: str) -> dict:
        """Compare two sessions by message-content hash.

        Returns ``{'only_a': [...], 'only_b': [...], 'common': [...]}``
        where each list contains the message content strings.
        """
        msgs_a = self.store.load_session(session_a)
        msgs_b = self.store.load_session(session_b)

        def _hash(msg: Any) -> str:
            content = msg.content or ""
            return hashlib.md5(content.encode("utf-8")).hexdigest()  # noqa: S324

        map_a: dict[str, str] = {}
        for m in msgs_a:
            if m.content:
                map_a[_hash(m)] = m.content
        map_b: dict[str, str] = {}
        for m in msgs_b:
            if m.content:
                map_b[_hash(m)] = m.content

        set_a = set(map_a)
        set_b = set(map_b)
        return {
            "only_a": [map_a[h] for h in (set_a - set_b)],
            "only_b": [map_b[h] for h in (set_b - set_a)],
            "common": [map_a[h] for h in (set_a & set_b)],
        }


# === CodeGraph =============================================================

class CodeGraph:
    """Build a dependency map of a Python repository using ``ast``.

    The result maps each ``.py`` file (relative path) to its imported
    modules, top-level class names, and function names.
    """

    def build_graph(self, repo_root: Path) -> dict:
        """Walk *repo_root* and return a graph dict.

        Returns:
            ``{'files': [...], 'imports': {file: [...]},
            'classes': {file: [...]}, 'functions': {file: [...]}}``
        """
        repo = Path(repo_root)
        files: list[str] = []
        imports: dict[str, list[str]] = {}
        classes: dict[str, list[str]] = {}
        functions: dict[str, list[str]] = {}

        for py_file in repo.rglob("*.py"):
            if any(part in _SKIP_DIRS for part in py_file.parts):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            rel = str(py_file.relative_to(repo))
            files.append(rel)
            imports[rel] = []
            classes[rel] = []
            functions[rel] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[rel].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports[rel].append(node.module)
                elif isinstance(node, ast.ClassDef):
                    classes[rel].append(node.name)
                elif isinstance(node, (ast.FunctionDef,
                                       ast.AsyncFunctionDef)):
                    functions[rel].append(node.name)

        return {
            "files": sorted(files),
            "imports": imports,
            "classes": classes,
            "functions": functions,
        }


# === CommandPalette ========================================================

class CommandPalette:
    """Register commands with descriptions and handlers, fuzzy-search, and
    execute."""

    def __init__(self) -> None:
        self._commands: dict[str, dict] = {}

    def register(self, command: str, description: str,
                 handler: Callable[..., Any]) -> None:
        """Register or overwrite a command."""
        self._commands[command] = {
            "description": description,
            "handler": handler,
        }

    def search(self, query: str) -> list[tuple[str, str]]:
        """Fuzzy-match *query* against command+description.

        A match requires every whitespace-separated query word to appear as
        a substring (case-insensitive) in ``command + description``.
        """
        q = query.lower().strip()
        if not q:
            return [
                (cmd, info["description"])
                for cmd, info in self._commands.items()
            ]
        q_words = q.split()
        results: list[tuple[str, str]] = []
        for cmd, info in self._commands.items():
            text = f"{cmd} {info['description']}".lower()
            if all(w in text for w in q_words):
                results.append((cmd, info["description"]))
        return results

    def execute(self, command: str, *args: Any) -> Any:
        """Invoke the handler for *command* with *args*.

        Raises:
            KeyError: If *command* is not registered.
        """
        if command not in self._commands:
            raise KeyError(f"Unknown command: {command}")
        return self._commands[command]["handler"](*args)


# === TokenSaviour ===========================================================

class TokenSaviour:
    """Pick the cheapest tool for a task using keyword heuristics."""

    def pick_cheapest_tool(self, task: str,
                           available_tools: list[dict]) -> str:
        """Return the name of the most economical tool for *task*.

        Heuristic:
          * 'read'/'find'/'search' → prefer ``grep``/``glob`` over
            ``read_file``.
          * 'run'/'execute'/'shell' → prefer ``bash``.
          * Otherwise the first available tool.
        """
        task_lower = (task or "").lower()
        names = [
            t.get("name", str(t.get("id", "")))
            for t in available_tools
        ]
        names_lower = [n.lower() for n in names]

        if any(kw in task_lower for kw in
               ("read", "find", "search", "grep", "locate")):
            for pref in ("grep", "glob", "rg", "fd"):
                if pref in names_lower:
                    return names[names_lower.index(pref)]

        if any(kw in task_lower for kw in
               ("run", "execute", "shell", "command")):
            for pref in ("bash", "shell", "exec", "terminal"):
                if pref in names_lower:
                    return names[names_lower.index(pref)]

        return names[0] if names else ""


# === SemanticDedup =========================================================

class SemanticDedup:
    """Remove near-duplicate messages using word-set Jaccard similarity."""

    def dedup(self, messages: list[dict]) -> list[dict]:
        """Return *messages* with near-duplicates removed.

        When two messages exceed ``DEDUP_THRESHOLD`` word overlap, the
        longer content is kept.
        """
        result: list[dict] = []
        for msg in messages:
            content = (
                msg.get("content", "")
                if isinstance(msg, dict)
                else str(msg)
            )
            words = set(content.lower().split())
            is_dup = False
            for i, kept in enumerate(result):
                kept_content = (
                    kept.get("content", "")
                    if isinstance(kept, dict)
                    else str(kept)
                )
                kept_words = set(kept_content.lower().split())
                if words and kept_words:
                    intersection = len(words & kept_words)
                    union = len(words | kept_words)
                    similarity = (
                        intersection / union if union > 0 else 0.0
                    )
                    if similarity >= DEDUP_THRESHOLD:
                        if len(content) > len(kept_content):
                            result[i] = msg
                        is_dup = True
                        break
            if not is_dup:
                result.append(msg)
        return result


# === PromptCompressor ======================================================

class PromptCompressor:
    """Extractive text compression by word-frequency sentence scoring."""

    def compress(self, text: str,
                 ratio: float = COMPRESS_DEFAULT_RATIO) -> str:
        """Return an extractive summary of *text* at ~*ratio* of its length.

        Sentences are scored by the aggregate frequency of their words;
        the highest-scoring sentences are kept until the target character
        budget (``len(text) * ratio``) is met, then reassembled in
        original order.
        """
        if not text or not text.strip():
            return text or ""

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return text

        words = re.findall(r"\w+", text.lower())
        freq = Counter(words)

        scored: list[tuple[int, str, float]] = []
        for idx, sent in enumerate(sentences):
            sent_words = re.findall(r"\w+", sent.lower())
            if sent_words:
                score = sum(freq[w] for w in sent_words) / len(sent_words)
            else:
                score = 0.0
            scored.append((idx, sent, score))

        scored.sort(key=lambda x: x[2], reverse=True)

        target_chars = max(1, int(len(text) * ratio))
        kept_indices: set[int] = set()
        total = 0
        for idx, sent, _score in scored:
            kept_indices.add(idx)
            total += len(sent)
            if total >= target_chars:
                break

        result = [sentences[i] for i in sorted(kept_indices)]
        return " ".join(result)


# === Stubs (v2.0 future — require external dependencies) ==================

class _Stub:
    """Base class for v2.0 stubs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "v2.0 future — requires external dependency"
        )


class AgentCanvas(_Stub):
    """Interactive agent canvas — v2.0 future — requires external dependency."""


class BrowserAutomation(_Stub):
    """Headless browser automation — v2.0 future — requires external dependency."""


class ImageInput(_Stub):
    """Image input processing — v2.0 future — requires external dependency."""


class PDFExcelParser(_Stub):
    """PDF and Excel parsing — v2.0 future — requires external dependency."""


class VoiceInput(_Stub):
    """Voice/speech-to-text input — v2.0 future — requires external dependency."""


class VoiceOutput(_Stub):
    """Voice/text-to-speech output — v2.0 future — requires external dependency."""


class WebDashboard(_Stub):
    """Web-based dashboard UI — v2.0 future — requires external dependency."""


class MultiBackend(_Stub):
    """Multi-backend LLM orchestration — v2.0 future — requires external dependency."""


class PRCreation(_Stub):
    """Automated pull-request creation — v2.0 future — requires external dependency."""


class Provenance(_Stub):
    """Provenance tracking — v2.0 future — requires external dependency."""


class Notebook(_Stub):
    """Notebook integration — v2.0 future — requires external dependency."""


class GPUAcceleration(_Stub):
    """GPU-accelerated inference — v2.0 future — requires external dependency."""
