"""Web tools: web_fetch (HTTP GET/POST with SSRF guards) and web_search.

Implements spec 6.2 tools #23 web_fetch and #24 web_search. Uses only the
standard library (urllib + html.parser) so the project does not gain another
HTTP-client dependency (rules 17).

SSRF defense: rejects file://, ftp://, etc. schemes and blocks private IP
ranges (localhost, 127.0.0.0/8, 10.*, 192.168.*, 172.16-31.*) unless
allow_private=True is set at tool construction time.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.request
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from synth.constants import (
    MAX_OUTPUT_SIZE,
)
from synth.tools import (
    ToolFunc,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    _truncate,
    default_registry,
)

# --- Constants ------------------------------------------------------------

# User-Agent header for all outgoing requests (identifies the agent).
USER_AGENT = "synth/0.1"

# Default cap on fetched content size (prevents OOM on huge responses).
MAX_FETCH_BYTES = 1_000_000

# Default timeout for fetch requests (seconds).
FETCH_TIMEOUT = 15

# Maximum number of search results to return (clamped 1..10).
SEARCH_RESULT_LIMIT_MAX = 10

# Private IP ranges that are blocked by default (SSRF defense).
PRIVATE_IP_PREFIXES = (
    "127.",      # localhost / loopback
    "10.",       # RFC1918 private
    "192.168.",  # RFC1918 private
    "172.16.",   # RFC1918 private (172.16.0.0 - 172.31.255.255)
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
    "::1",       # IPv6 localhost
    "fe80:",     # IPv6 link-local
    "fc00:",     # IPv6 unique local
    "fd00:",
)

# Content types that are treated as text and decoded to UTF-8.
HTML_CONTENT_TYPES = (
    "text/",
    "application/json",
    "+json",
    "application/xml",
    "+xml",
)


# --- HTML-to-text helper --------------------------------------------------

class _HTMLToTextParser(HTMLParser):
    """Strip HTML tags and normalize whitespace for readable text."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.in_skip_tag = False  # script/style/head title contents dropped

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style", "title"):
            self.in_skip_tag = True
        elif tag.lower() == "br":
            # A line break inside a paragraph reads as a space in text form.
            self.text_parts.append(" ")
        elif tag.lower() in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            # Add paragraph breaks for block elements
            if self.text_parts and not self.text_parts[-1].endswith("\n\n"):
                self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "title"):
            self.in_skip_tag = False
        elif tag.lower() in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            # Ensure block elements end with double newline
            if self.text_parts and not self.text_parts[-1].endswith("\n\n"):
                self.text_parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self.in_skip_tag:
            self.text_parts.append(data)

    def get_text(self) -> str:
        """Return the accumulated text with normalized whitespace."""
        text = "".join(self.text_parts)
        # Collapse multiple newlines to max 2, and strip leading/trailing
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    parser = _HTMLToTextParser()
    parser.feed(html)
    return parser.get_text()


# --- web_fetch ------------------------------------------------------------

WEB_FETCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "URL to fetch (http:// or https:// only)",
        },
        "method": {
            "type": "string",
            "description": "HTTP method (GET or POST, default GET)",
            "enum": ["GET", "POST"],
        },
        "max_bytes": {
            "type": "integer",
            "description": f"Maximum bytes to read (default {MAX_FETCH_BYTES})",
        },
        "timeout": {
            "type": "integer",
            "description": f"Timeout in seconds (default {FETCH_TIMEOUT})",
        },
    },
    "required": ["url"],
}


def _is_private_host(host: str) -> bool:
    """Return True when a host is loopback, private, or explicitly local."""
    host_lower = host.lower()
    # Name-level blocks first — 'localhost' is the common SSRF target and
    # resolving it requires no DNS round-trip.
    if host_lower in {"localhost", "localhost.localdomain"} or host_lower.endswith((".localhost", ".internal", ".local")):
        return True
    for prefix in PRIVATE_IP_PREFIXES:
        if host_lower.startswith(prefix):
            return True
    # Literal IP → proper is_private check.
    try:
        ip = ipaddress.ip_address(host_lower)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except ValueError:
        pass
    # Hostname → resolve and re-check (defeats 'evil.test → 127.0.0.1').
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False  # unresolvable: let the request fail naturally
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def make_web_fetch_tool(
    timeout: int = FETCH_TIMEOUT,
    max_bytes: int = MAX_FETCH_BYTES,
    allow_private: bool = False,
) -> ToolFunc:
    """Build the web_fetch tool with SSRF guards and size limits."""

    def _web_fetch(args: dict[str, Any]) -> ToolResult:
        url = args.get("url")
        method = args.get("method", "GET")
        actual_max_bytes = args.get("max_bytes", max_bytes)
        actual_timeout = args.get("timeout", timeout)

        if not isinstance(url, str) or not url.strip():
            return ToolResult("Error: url must be a non-empty string", is_error=True)
        
        if method not in ("GET", "POST"):
            return ToolResult("Error: method must be 'GET' or 'POST'", is_error=True)
        
        if not isinstance(actual_max_bytes, int) or actual_max_bytes <= 0:
            actual_max_bytes = max_bytes
        
        if not isinstance(actual_timeout, int) or actual_timeout <= 0:
            actual_timeout = timeout

        # Parse URL and validate scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                f"Error: URL scheme must be http or https, got {parsed.scheme!r}",
                is_error=True,
            )
        
        if not parsed.netloc:
            return ToolResult("Error: URL must have a host", is_error=True)
        
        # SSRF guard: block private IPs unless allowed
        if not allow_private:
            host = parsed.hostname
            if host and _is_private_host(host):
                return ToolResult(
                    f"Error: access to private IP ranges is blocked: {host}",
                    is_error=True,
                )
        
        # Build request
        headers = {"User-Agent": USER_AGENT}
        request = urllib.request.Request(url, headers=headers, method=method)

        try:
            with urllib.request.urlopen(request, timeout=actual_timeout) as response:
                # Check content type to decide how to handle response
                content_type = response.headers.get_content_type()

                # Determine if this is a text-like content type
                is_text_content = any(
                    content_type.startswith(ct) or ct in content_type
                    for ct in HTML_CONTENT_TYPES
                )

                if not is_text_content:
                    # Binary content - return placeholder without reading body
                    declared = response.headers.get("Content-Length", "0")
                    return ToolResult(
                        f"[binary content: {content_type}, {declared} bytes]"
                    )

                # Read the body with a hard cap (real HTTPResponse objects
                # have no peek(); read(max_bytes+1) detects overflow).
                content_bytes = response.read(actual_max_bytes + 1)
                was_truncated = len(content_bytes) > actual_max_bytes
                if was_truncated:
                    content_bytes = content_bytes[:actual_max_bytes]

                # Decode the content
                content = content_bytes.decode("utf-8", errors="replace")

                # If it's HTML, convert to readable text
                if content_type.startswith("text/html"):
                    content = _html_to_text(content)

                if was_truncated:
                    # Reserve room for the notice so the total respects the cap.
                    notice = f"\n...[truncated at {actual_max_bytes} bytes]"
                    content = content[: max(0, actual_max_bytes - len(notice))] + notice

                return ToolResult(content)
                
        except HTTPError as exc:
            return ToolResult(
                f"Error: HTTP {exc.code} {exc.reason}", is_error=True
            )
        except URLError as exc:
            return ToolResult(
                f"Error: cannot fetch: {exc.reason}", is_error=True
            )
        except OSError as exc:
            return ToolResult(
                f"Error: cannot fetch: {exc}", is_error=True
            )

    return _web_fetch


# --- web_search -----------------------------------------------------------

WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query string",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of results (1-{SEARCH_RESULT_LIMIT_MAX}, default 5)",
        },
    },
    "required": ["query"],
}


def _default_search_fn(query: str, limit: int) -> list[dict[str, str]]:
    """Default search implementation using DuckDuckGo HTML endpoint."""
    import urllib.parse
    
    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    
    headers = {"User-Agent": USER_AGENT}
    request = urllib.request.Request(search_url, headers=headers)
    
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("text/"):
            raise URLError("Unexpected content type from search")
        
        html = response.read(MAX_FETCH_BYTES).decode("utf-8", errors="replace")
    
    # Simple regex-based extraction (conservative)
    import re
    
    # Find result blocks - very basic parsing
    results: list[dict[str, str]] = []
    # This is a very simple extractor - in practice would use proper HTML parsing
    # but we're avoiding external dependencies
    
    # Look for links with titles and snippets
    link_pattern = r'<a class="result__a" href="([^"]+)"[^>]*>(.*?)</a>'
    snippet_pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'
    
    links = re.findall(link_pattern, html, re.DOTALL)
    snippets = re.findall(snippet_pattern, html, re.DOTALL)
    
    for i, (url, title) in enumerate(links):
        if len(results) >= limit:
            break
        
        # Clean up HTML entities and tags from title/snippet
        clean_title = re.sub(r"<[^>]+>", "", title).strip()
        clean_snippet = ""
        if i < len(snippets):
            clean_snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
        
        if clean_title:
            results.append({
                "title": clean_title,
                "url": url,
                "snippet": clean_snippet,
            })
    
    return results[:limit]


def make_web_search_tool(
    search_fn: Callable[[str, int], list[dict[str, str]]] | None = None,
    max_output_size: int = MAX_OUTPUT_SIZE,
) -> ToolFunc:
    """Build the web_search tool with pluggable search function."""
    
    actual_search_fn = search_fn or _default_search_fn

    def _web_search(args: dict[str, Any]) -> ToolResult:
        query = args.get("query")
        limit = args.get("limit", 5)
        
        if not isinstance(query, str) or not query.strip():
            return ToolResult("Error: query must be a non-empty string", is_error=True)
        
        if not isinstance(limit, int):
            limit = 5
        limit = max(1, min(limit, SEARCH_RESULT_LIMIT_MAX))
        
        try:
            results = actual_search_fn(query, limit)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                f"Error: search failed: {exc}\n"
                f"(attempted to search for: {query!r})\n"
                f"(expected: list of {{'title','url','snippet'}} dicts)\n"
                f"(happened: {type(exc).__name__}: {exc})\n"
                f"(fix: check network connectivity or provide a working search_fn)",
                is_error=True,
            )
        
        if not results:
            return ToolResult("No results found")
        
        # The provider may over-deliver; the limit is the contract.
        results = list(results)[:limit]
        
        # Format results as numbered plain text
        lines = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "").strip()
            url = result.get("url", "").strip()
            snippet = result.get("snippet", "").strip()
            
            lines.append(f"{i}. {title}")
            if url:
                lines.append(f"   {url}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")  # blank line between results
        
        output = "\n".join(lines).rstrip()
        return ToolResult(_truncate(output, max_output_size))
    
    return _web_search


# --- web registry ---------------------------------------------------------

def web_registry(
    timeout: int = FETCH_TIMEOUT,
    max_bytes: int = MAX_FETCH_BYTES,
    max_output_size: int = MAX_OUTPUT_SIZE,
    allow_private: bool = False,
) -> ToolRegistry:
    """Build a registry with web_fetch and web_search tools."""
    registry = default_registry(max_output_size=max_output_size)
    registry.register(
        ToolSpec(
            name="web_fetch",
            description="Fetch content from a URL (http/https only) with SSRF protection",
            parameters=WEB_FETCH_SCHEMA,
            func=make_web_fetch_tool(timeout, max_bytes, allow_private),
        )
    )
    registry.register(
        ToolSpec(
            name="web_search",
            description="Search the web and return results as plain text",
            parameters=WEB_SEARCH_SCHEMA,
            func=make_web_search_tool(max_output_size=max_output_size),
        )
    )
    return registry