"""Best-of-N candidate selection (spec 6.1 #7 / 6.3 #47).

Runs the same prompt against N models (or the same model N times), scores
the candidates, and picks the best. Scoring is heuristic (no LLM judge) so
the module stays dependency-free and deterministic for tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Scoring weights (spec: pick best answer). Commented per rules 2.5.
# Completeness: longer answers usually cover more, but length alone is a
# gaming vector — weight it low.
WEIGHT_LENGTH = 0.1
# Code presence: for coding tasks, a fenced code block is a strong signal.
WEIGHT_HAS_CODE = 0.35
# Hedging/uncertainty phrases are a quality smell.
WEIGHT_HEDGE_PENALTY = 0.15
# Structure (bullets/numbered steps) reads well in terminal output.
WEIGHT_STRUCTURE = 0.2

HEDGE_PATTERNS = (
    re.compile(r"\bI'?m not sure\b", re.IGNORECASE),
    re.compile(r"\bI cannot\b", re.IGNORECASE),
    re.compile(r"\bI can'?t\b", re.IGNORECASE),
    re.compile(r"\bAs an AI\b", re.IGNORECASE),
    re.compile(r"\bperhaps\b|maybe\b|might be\b", re.IGNORECASE),
)

CODE_FENCE = re.compile(r"```")
BULLET = re.compile(r"^\s*([-*]|\d+\.)\s+", re.MULTILINE)

# Length normalization cap: 4000 chars counts as "full marks"; beyond that no
# extra credit (prevents rambling from winning).
LENGTH_FULL_MARKS = 4000


@dataclass
class Candidate:
    """One answer with its model and score."""

    model: str
    text: str
    score: float = 0.0


@dataclass
class BestOfNResult:
    """Outcome of a best-of-N run."""

    best: Candidate
    candidates: list[Candidate] = field(default_factory=list)


def score_answer(text: str) -> float:
    """Heuristically score one answer in [0, 1].

    Args:
        text: The candidate answer.

    Returns:
        A score where higher is better. Empty answers score 0.
    """
    if not text or not text.strip():
        return 0.0

    score = 0.0

    # Length component (capped).
    score += WEIGHT_LENGTH * min(len(text), LENGTH_FULL_MARKS) / LENGTH_FULL_MARKS

    # Code component.
    if CODE_FENCE.search(text):
        score += WEIGHT_HAS_CODE

    # Structure component: up to 5 bullets/numbered items.
    bullets = len(BULLET.findall(text))
    score += WEIGHT_STRUCTURE * min(bullets, 5) / 5

    # Hedge penalty.
    hedges = sum(1 for pat in HEDGE_PATTERNS if pat.search(text))
    score -= WEIGHT_HEDGE_PENALTY * min(hedges, 3) / 3

    return max(0.0, score)


class BestOfN:
    """Run one prompt across N models and keep the best-scoring answer."""

    def __init__(self, clients: dict[str, object]) -> None:
        """Map model name -> client.

        Args:
            clients: {'model-name': client} where each client exposes
                .chat(messages, tools=None, stream=False) -> response
                with a .text attribute (LLMClient-compatible duck type).
        """
        if not clients:
            raise ValueError("BestOfN requires at least one model client")
        self.clients = clients

    def run(self, prompt: str) -> BestOfNResult:
        """Query every client once and return scored candidates.

        A client that raises contributes nothing (the failure is recorded as
        an empty candidate with score 0 so the report shows the dropout).

        Args:
            prompt: The user task.

        Returns:
            BestOfNResult with the best candidate and all scored candidates,
            sorted best-first.

        Raises:
            RuntimeError: If every client failed (no usable answer).
        """
        candidates: list[Candidate] = []
        messages = [{"role": "user", "content": prompt}]

        for model, client in self.clients.items():
            try:
                response = client.chat(messages, tools=None, stream=False)
                text = getattr(response, "text", None) or ""
            except Exception:  # noqa: BLE001 — dropout is a normal path
                text = ""
            candidates.append(Candidate(model=model, text=text,
                                        score=score_answer(text)))

        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]
        if not best.text:
            raise RuntimeError("all models failed to produce an answer")
        return BestOfNResult(best=best, candidates=candidates)
