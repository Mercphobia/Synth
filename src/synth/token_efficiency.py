"""Token efficiency layer: RTK shell filter, Ponytail prompt, Caveman mode,
AutoCompressor, TokenDashboard, and effort levels.

Implements spec section 8 — the flagship feature for keeping LLM context
windows lean by compressing tool outputs, prompts, and chat history before
they reach the model.
"""

from __future__ import annotations

import re
import sys
from typing import Any


# --- RTK shell output filter (spec 8.3) ---


class RtkFilter:
    """Compress known command outputs BEFORE they enter LLM context.

    Built-in filters recognize common verbose commands (git, pytest, ls, docker)
    and strip their chatty boilerplate while preserving all error lines and
    essential information. Unknown commands pass through unchanged after
    generic cleanup (ANSI codes, repeated lines, blank runs).

    Lossless for errors: any line containing 'error', 'Error', 'ERROR',
    'failed', or 'FAILED' is always kept verbatim.
    """

    def __init__(
        self,
        max_output_tokens: int = 500,  # chars proxy
        enabled_filters: list[str] | None = None,
    ) -> None:
        self.max_output_tokens = max_output_tokens
        self.enabled_filters = set(enabled_filters) if enabled_filters is not None else None

        # Precompile regexes for performance
        self._ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self._error_keywords = re.compile(
            r"(?i)\b(error|failed)\b"
        )

    def filter_output(self, cmd: str, output: str) -> str:
        """Apply compression to a shell command's stdout/stderr.

        Args:
            cmd: The original shell command (e.g., "git status")
            output: The raw command output to compress

        Returns:
            Compressed output string, never longer than max_output_tokens chars
        """
        if not output.strip():
            return output

        # Always apply generic cleanup first
        cleaned = self._generic_cleanup(output)

        # Apply command-specific filters if enabled
        cmd_name = cmd.split()[0] if cmd.strip() else ""
        if self._is_filter_enabled("git") and cmd_name == "git":
            cleaned = self._filter_git(cmd, cleaned)
        elif self._is_filter_enabled("pytest") and cmd_name == "pytest":
            cleaned = self._filter_pytest(cleaned)
        elif self._is_filter_enabled("ls") and cmd_name in ("ls", "dir"):
            cleaned = self._filter_ls(cleaned)
        elif self._is_filter_enabled("docker") and cmd_name == "docker":
            cleaned = self._filter_docker(cmd, cleaned)

        # Enforce length limit
        if len(cleaned) > self.max_output_tokens:
            cleaned = cleaned[: self.max_output_tokens - 3] + "..."

        return cleaned

    def _is_filter_enabled(self, filter_name: str) -> bool:
        """Check if a specific filter is enabled."""
        if self.enabled_filters is None:
            return True
        return filter_name in self.enabled_filters

    def _generic_cleanup(self, text: str) -> str:
        """Apply universal cleanup: ANSI codes, repeated lines, blank runs."""
        # Strip ANSI escape codes
        text = self._ansi_escape.sub("", text)

        lines = text.splitlines(keepends=True)
        filtered_lines: list[str] = []
        prev_line: str | None = None
        repeat_count = 0

        def flush_run() -> None:
            """Emit the pending run: once, or as a '[N repeated]' marker."""
            if prev_line is None:
                return
            if repeat_count > 1:
                filtered_lines.append(f"[{repeat_count} repeated]\n")
            else:
                filtered_lines.append(prev_line)

        for line in lines:
            # Preserve error lines exactly (they break any repeat run).
            if self._error_keywords.search(line):
                flush_run()
                prev_line, repeat_count = None, 0
                filtered_lines.append(line)
                continue

            if line == prev_line:
                repeat_count += 1
                continue

            flush_run()
            prev_line, repeat_count = line, 1

        flush_run()

        # Collapse runs of blank lines into a single blank line.
        result_lines = []
        blank_run = 0
        for line in filtered_lines:
            if not line.strip():
                blank_run += 1
                if blank_run == 1:
                    result_lines.append(line if line.endswith("\n") else line + "\n")
            else:
                blank_run = 0
                result_lines.append(line)

        return "".join(result_lines)

    def _filter_git(self, cmd: str, output: str) -> str:
        """Compress git command output."""
        args = cmd.split()[1:] if len(cmd.split()) > 1 else []
        
        if "status" in args:
            # Keep only changed-file lines, drop verbose hints
            lines = []
            for line in output.splitlines():
                if line.startswith((" M ", "?? ", "A  ", "D  ", "R  ", "C  ", "U  ")):
                    lines.append(line)
                elif self._error_keywords.search(line):
                    lines.append(line)
            return "\n".join(lines)
        
        elif "log" in args:
            # Keep commit header + subject (first indented line after it);
            # drop Author/Date metadata and body paragraphs.
            lines = []
            seen_subject = False
            for line in output.splitlines():
                if self._error_keywords.search(line):
                    lines.append(line)
                    continue
                if line.startswith("commit "):
                    lines.append(line)
                    seen_subject = False
                elif line.startswith(("Author:", "Date:")):
                    continue
                elif line.strip() and line.startswith("    ") and not seen_subject:
                    lines.append(line.strip())
                    seen_subject = True
            return "\n".join(lines)
        
        return output

    def _filter_pytest(self, output: str) -> str:
        """Keep failed lines + summary, drop passing dots."""
        lines = []
        in_failures = False
        
        for line in output.splitlines():
            if line.startswith("====") and "FAILURES" in line:
                in_failures = True
                lines.append(line)
            elif line.startswith("====") and "PASSES" in line:
                in_failures = False
                continue
            elif in_failures or self._error_keywords.search(line):
                lines.append(line)
            elif line.startswith(("=", "F", "E")) and not line.startswith("===="):
                # Keep summary line markers
                lines.append(line)
            elif line.strip() and not line.startswith("."):
                # Keep non-dot lines that aren't just whitespace
                lines.append(line)
        
        return "\n".join(lines)

    def _filter_ls(self, output: str) -> str:
        """Collapse to names only if >10 entries."""
        lines = output.splitlines()
        if len(lines) > 10:
            # Just return filenames without details
            return "\n".join(line.split()[-1] if line.strip() else "" for line in lines if line.strip())
        return output

    def _filter_docker(self, cmd: str, output: str) -> str:
        """Extract id|name|status columns from docker ps."""
        args = cmd.split()[1:] if len(cmd.split()) > 1 else []
        
        if "ps" in args:
            lines = output.splitlines()
            if len(lines) <= 1:
                return output
            
            # Keep header and extract relevant columns
            header = lines[0]
            result_lines = [header]
            
            # Find column positions (rough heuristic)
            parts = header.split()
            if len(parts) >= 3:
                for line in lines[1:]:
                    if self._error_keywords.search(line):
                        result_lines.append(line)
                    else:
                        cols = line.split()
                        if len(cols) >= 3:
                            # Take first (ID), last but one (NAME), last (STATUS)
                            simplified = f"{cols[0]} {cols[-2]} {cols[-1]}"
                            result_lines.append(simplified)
                        else:
                            result_lines.append(line)
            
            return "\n".join(result_lines)
        
        return output


# --- Ponytail decision-ladder prompt (spec 8.2) ---


def ponytail_prompt(
    goal: str,
    existing_symbols: list[str] | None = None,
    aggressiveness: str = "balanced",
    preserve: list[str] | None = None,
) -> str:
    """Return the 7-step decision ladder instruction text for prepending to agent prompts.

    Args:
        goal: The primary task goal
        existing_symbols: List of symbols already defined in scope
        aggressiveness: 'minimal'|'balanced'|'aggressive' - controls wording strength
        preserve: List of phrases that must appear in output when preserved

    Returns:
        Formatted decision ladder prompt

    Raises:
        ValueError: If aggressiveness level is unknown
    """
    if aggressiveness not in ("minimal", "balanced", "aggressive"):
        raise ValueError(f"Unknown aggressiveness level: {aggressiveness}")
    
    preserve_set = set(preserve or [])
    
    # Build preservation requirements
    preservations = []
    if "validation" in preserve_set:
        preservations.append("Include input validation")
    if "security" in preserve_set:
        preservations.append("Apply security best practices")
    if "error_handling" in preserve_set:
        preservations.append("Implement proper error handling")
    if "accessibility" in preserve_set:
        preservations.append("Ensure accessibility compliance")
    
    preservation_text = ""
    if preservations:
        preservation_text = "\n".join(f"- {p}" for p in preservations) + "\n\n"
    
    # Aggressiveness wordings
    if aggressiveness == "minimal":
        step_prefix = "Consider"
        action_word = "implement"
    elif aggressiveness == "balanced":
        step_prefix = "Evaluate"
        action_word = "build"
    else:  # aggressive
        step_prefix = "Demand"
        action_word = "create"
    
    ladder = f"""{preservation_text}Follow this 7-step decision ladder to {action_word} the solution:

1. {step_prefix} whether the goal "{goal}" can be achieved with existing tools
2. {step_prefix} if new code is needed or existing code can be modified
3. {step_prefix} the simplest possible implementation that meets requirements  
4. {step_prefix} edge cases and failure modes
5. {step_prefix} testing strategy and verification approach
6. {step_prefix} performance and resource implications
7. {step_prefix} documentation and maintainability

Always prefer tool use over code generation when possible.
"""
    
    return ladder


# --- Caveman mode (spec 8.4) ---


class Caveman:
    """Rule-based output compression for terse responses."""
    
    def __init__(self) -> None:
        # Compile filler phrases regex once
        self._filler_phrases = re.compile(
            r"\b(I have|I've|I will|Let me|Certainly|Of course|Sure,|just|really|actually|basically|Note that|Keep in mind)\b",
            re.IGNORECASE
        )
        
        self._code_fence_start = re.compile(r"```")
        self._code_fence_end = re.compile(r"```")
        
        # Meta-commentary patterns
        self._meta_patterns = [
            re.compile(r"^Now,\s*"),
            re.compile(r"^Next,\s*"),
        ]
        
        # Phrase replacements
        self._replacements = [
            (re.compile(r"\bthe following\b"), ""),
            (re.compile(r"\bin order to\b"), "to"),
            (re.compile(r"\butilize\b"), "use"),
            (re.compile(r"\ba large number of\b"), "many"),
        ]
    
    def terse(self, text: str) -> str:
        """Compress text by removing filler and applying rule-based transformations.
        
        Rules:
        - Strip filler phrases
        - Remove pleasantries 
        - Apply phrase replacements
        - Drop meta-commentary sentences
        - Preserve code blocks exactly
        - Keep bullet/number structure
        - Must be idempotent
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed text
        """
        if not text.strip():
            return text
            
        lines = text.splitlines(keepends=True)
        result_lines = []
        in_code_block = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for code fence boundaries
            if self._code_fence_start.search(line):
                if not in_code_block:
                    # Entering code block
                    result_lines.append(line)
                    in_code_block = True
                    i += 1
                    # Copy everything until closing fence
                    while i < len(lines):
                        code_line = lines[i]
                        result_lines.append(code_line)
                        if self._code_fence_end.search(code_line):
                            in_code_block = False
                            break
                        i += 1
                    i += 1
                    continue
                else:
                    # Already in code block, treat as normal content
                    pass
            
            if in_code_block:
                result_lines.append(line)
                i += 1
                continue
            
            # Process non-code lines
            processed_line = self._process_line(line)
            if processed_line is not None:
                result_lines.append(processed_line)
            
            i += 1
        
        return "".join(result_lines)
    
    def _process_line(self, line: str) -> str | None:
        """Process a single line outside of code blocks."""
        original_line = line
        
        # Remove pleasantries
        if any(phrase in line.lower() for phrase in ["hope this helps", "feel free to ask"]):
            return None  # Skip entire line
        
        # Apply phrase replacements
        for pattern, replacement in self._replacements:
            line = pattern.sub(replacement, line)
        
        # Strip filler phrases
        line = self._filler_phrases.sub("", line)
        
        # Remove meta-commentary at start of sentence
        for meta_pattern in self._meta_patterns:
            if meta_pattern.match(line) and not self._contains_code_token(line):
                return None  # Skip meta-commentary lines without code
        
        # Clean up extra whitespace
        line = re.sub(r"\s+", " ", line).strip()
        if not line and original_line.strip():
            return "\n"  # Preserve blank lines that were originally blank
        elif line:
            return line + "\n"
        else:
            return original_line  # Preserve original formatting for empty cases
    
    def _contains_code_token(self, text: str) -> bool:
        """Check if text contains likely code tokens."""
        code_indicators = ["{", "}", "(", ")", "[", "]", "=", ":", ";", "//", "#"]
        return any(indicator in text for indicator in code_indicators)


# --- AutoCompressor (spec 8.8) ---


class AutoCompressor:
    """Automatically compress chat history when approaching context limits."""
    
    def __init__(
        self,
        threshold: float = 0.7,
        keep_recent: int = 10,
        context_window: int = 200_000,
    ) -> None:
        self.threshold = threshold
        self.keep_recent = keep_recent
        self.context_window = context_window
    
    def compress(self, messages: list[dict]) -> tuple[list[dict], bool]:
        """Compress chat history if total size exceeds threshold.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Tuple of (compressed_messages, was_compressed)
        """
        if not messages:
            return messages, False
        
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        threshold_chars = self.threshold * self.context_window
        
        if total_chars <= threshold_chars:
            return messages, False
        
        # Identify messages that cannot be compressed
        system_messages = []
        tool_messages = []
        regular_messages = []
        
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_messages.append((i, msg))
            elif msg.get("role") == "tool" or "tool_calls" in msg:
                tool_messages.append((i, msg))
            else:
                regular_messages.append((i, msg))
        
        # Determine which messages to keep vs compress
        recent_keep_count = min(self.keep_recent, len(regular_messages))
        recent_messages = regular_messages[-recent_keep_count:] if recent_keep_count > 0 else []
        compressible_messages = regular_messages[:-recent_keep_count] if recent_keep_count > 0 else regular_messages
        
        if not compressible_messages:
            return messages, False
        
        # Create summary of compressible messages
        summary_parts = []
        for _, msg in compressible_messages:
            content = msg.get("content", "")
            if content.strip():
                # Extract first sentence
                sentences = re.split(r'[.!?]+', content.strip())
                first_sentence = sentences[0].strip() if sentences else content.strip()
                if first_sentence:
                    summary_parts.append(f"[{msg.get('role', 'unknown')}]: {first_sentence}")
        
        summary = "[compressed] " + " ".join(summary_parts[:150])  # Cap at ~1500 chars
        if len(summary) > 1500:
            summary = summary[:1497] + "..."
        
        # Reconstruct message list
        new_messages = []
        
        # Add system messages first (preserve order)
        system_msg_indices = {idx for idx, _ in system_messages}
        tool_msg_indices = {idx for idx, _ in tool_messages}
        recent_msg_indices = {idx for idx, _ in recent_messages}
        
        # Create a mapping of what to include at each position
        for i in range(len(messages)):
            if i in system_msg_indices:
                # Keep system messages
                new_messages.append(messages[i])
            elif i in tool_msg_indices:
                # Keep tool messages
                new_messages.append(messages[i])
            elif i in recent_msg_indices:
                # Keep recent messages
                new_messages.append(messages[i])
            elif i == compressible_messages[0][0]:
                # Add summary at position of first compressible message
                new_messages.append({"role": "system", "content": summary})
                # Skip all other compressible messages
                remaining_compressible = {idx for idx, _ in compressible_messages}
                while i < len(messages) and i in remaining_compressible:
                    i += 1
                if i < len(messages):
                    # Re-process current message since we advanced
                    if i in system_msg_indices:
                        new_messages.append(messages[i])
                    elif i in tool_msg_indices:
                        new_messages.append(messages[i])
                    elif i in recent_msg_indices:
                        new_messages.append(messages[i])
        
        return new_messages, True


# --- TokenDashboard (spec 8.13) ---


class TokenDashboard:
    """Track token usage statistics in memory."""
    
    def __init__(self) -> None:
        self.total_calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0
        self.by_model = {}
    
    def record(self, model: str, prompt_chars: int, completion_chars: int) -> None:
        """Record a token usage event."""
        self.total_calls += 1
        self.prompt_chars += prompt_chars
        self.completion_chars += completion_chars
        
        if model not in self.by_model:
            self.by_model[model] = {
                "calls": 0,
                "prompt_chars": 0,
                "completion_chars": 0,
            }
        
        self.by_model[model]["calls"] += 1
        self.by_model[model]["prompt_chars"] += prompt_chars
        self.by_model[model]["completion_chars"] += completion_chars
    
    def stats(self) -> dict[str, Any]:
        """Return current statistics (a deep-enough copy to mutate safely)."""
        import copy

        return {
            "total_calls": self.total_calls,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "est_tokens": (self.prompt_chars + self.completion_chars) // 4,
            "by_model": copy.deepcopy(self.by_model),
        }


# --- EffortLevels ---


def apply_effort(base_max_iterations: int, effort: str) -> int:
    """Apply effort multiplier to base iteration count.
    
    Args:
        base_max_iterations: Base iteration count
        effort: Effort level ('low', 'medium', 'high', 'xhigh')
        
    Returns:
        Adjusted iteration count (minimum 1)
        
    Raises:
        ValueError: If effort level is unknown
    """
    effort_multipliers = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
        "xhigh": 2.0,
    }
    
    if effort not in effort_multipliers:
        raise ValueError(f"Unknown effort level: {effort}")
    
    result = int(base_max_iterations * effort_multipliers[effort])
    return max(1, result)