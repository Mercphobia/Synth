"""Tests for web_tools.py."""

import contextlib
import io
from unittest import mock

import pytest

from synth.web_tools import (
    WEB_FETCH_SCHEMA,
    WEB_SEARCH_SCHEMA,
    _html_to_text,
    make_web_fetch_tool,
    make_web_search_tool,
    web_registry,
)


class MockHTTPResponse:
    """Mock HTTP response for testing (context-manager like the real one)."""
    
    def __init__(self, content: bytes, content_type: str = "text/html", status: int = 200):
        self.content = content
        self._content_type = content_type
        self.status = status
        self._read_pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        pass

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            result = self.content[self._read_pos:]
            self._read_pos = len(self.content)
            return result
        end = min(self._read_pos + size, len(self.content))
        result = self.content[self._read_pos:end]
        self._read_pos = end
        return result

    @property
    def headers(self):
        outer = self

        class Headers:
            def get_content_type(self):
                return outer._content_type

            def get(self, name, default=None):
                if name == "Content-Length":
                    return str(len(outer.content))
                return default

        return Headers()


def test_html_to_text():
    """Test HTML to plain text conversion."""
    html = """
    <html>
    <head><title>Test</title></head>
    <body>
        <h1>Main Title</h1>
        <p>First paragraph with <em>emphasis</em>.</p>
        <script>console.log('ignore me');</script>
        <style>body { color: red; }</style>
        <div>Second paragraph.</div>
        <p>Third paragraph<br>with line break.</p>
    </body>
    </html>
    """
    expected = "Main Title\n\nFirst paragraph with emphasis.\n\nSecond paragraph.\n\nThird paragraph with line break."
    assert _html_to_text(html) == expected


def test_web_fetch_schema():
    """Test web_fetch schema structure."""
    assert "url" in WEB_FETCH_SCHEMA["required"]
    assert WEB_FETCH_SCHEMA["properties"]["url"]["type"] == "string"


def test_web_search_schema():
    """Test web_search schema structure."""
    assert "query" in WEB_SEARCH_SCHEMA["required"]
    assert WEB_SEARCH_SCHEMA["properties"]["query"]["type"] == "string"


@mock.patch("urllib.request.urlopen")
def test_web_fetch_http_get_success(mock_urlopen):
    """Test successful HTTP GET fetch."""
    mock_response = MockHTTPResponse(b"Hello, world!", "text/plain")
    mock_urlopen.return_value = mock_response
    
    tool = make_web_fetch_tool()
    result = tool({"url": "https://example.com"})
    
    assert not result.is_error
    assert result.text == "Hello, world!"


@mock.patch("urllib.request.urlopen")
def test_web_fetch_http_post_success(mock_urlopen):
    """Test successful HTTP POST fetch."""
    mock_response = MockHTTPResponse(b"Posted successfully!", "text/plain")
    mock_urlopen.return_value = mock_response
    
    tool = make_web_fetch_tool()
    result = tool({"url": "https://example.com", "method": "POST"})
    
    assert not result.is_error
    assert result.text == "Posted successfully!"


@mock.patch("urllib.request.urlopen")
def test_web_fetch_binary_content(mock_urlopen):
    """Test binary content returns placeholder."""
    mock_response = MockHTTPResponse(b"\x89PNG\r\n\x1a\n", "image/png")
    mock_urlopen.return_value = mock_response
    
    tool = make_web_fetch_tool()
    result = tool({"url": "https://example.com/image.png"})
    
    assert not result.is_error
    assert "[binary content: image/png," in result.text


@mock.patch("urllib.request.urlopen")
def test_web_fetch_html_stripping(mock_urlopen):
    """Test HTML content is stripped to readable text."""
    html_content = b"<html><body><h1>Title</h1><p>Content</p></body></html>"
    mock_response = MockHTTPResponse(html_content, "text/html")
    mock_urlopen.return_value = mock_response
    
    tool = make_web_fetch_tool()
    result = tool({"url": "https://example.com"})
    
    assert not result.is_error
    assert "Title" in result.text
    assert "<h1>" not in result.text


@mock.patch("urllib.request.urlopen")
def test_web_fetch_truncation(mock_urlopen):
    """Test content truncation at max_bytes limit."""
    large_content = b"A" * 1500  # Larger than default 1000 byte limit for test
    mock_response = MockHTTPResponse(large_content, "text/plain")
    mock_urlopen.return_value = mock_response
    
    tool = make_web_fetch_tool(max_bytes=1000)
    result = tool({"url": "https://example.com"})
    
    assert not result.is_error
    assert len(result.text) <= 1000
    assert "[truncated at 1000 bytes]" in result.text


@mock.patch("urllib.request.urlopen")
def test_web_fetch_http_error(mock_urlopen):
    """Test HTTP error handling."""
    from urllib.error import HTTPError
    mock_urlopen.side_effect = HTTPError(
        "https://example.com", 404, "Not Found", {}, None
    )
    
    tool = make_web_fetch_tool()
    result = tool({"url": "https://example.com"})
    
    assert result.is_error
    assert "HTTP 404" in result.text


@mock.patch("urllib.request.urlopen")
def test_web_fetch_url_error(mock_urlopen):
    """Test URL error handling."""
    from urllib.error import URLError
    mock_urlopen.side_effect = URLError("connection failed")
    
    tool = make_web_fetch_tool()
    result = tool({"url": "https://example.com"})
    
    assert result.is_error
    assert "cannot fetch" in result.text


def test_web_fetch_invalid_url_scheme():
    """Test rejection of invalid URL schemes."""
    tool = make_web_fetch_tool()
    
    # Test file:// scheme
    result = tool({"url": "file:///etc/passwd"})
    assert result.is_error
    assert "scheme must be http or https" in result.text
    
    # Test ftp:// scheme  
    result = tool({"url": "ftp://example.com/file.txt"})
    assert result.is_error
    assert "scheme must be http or https" in result.text


def test_web_fetch_private_ip_blocking():
    """Test blocking of private IP addresses."""
    tool = make_web_fetch_tool(allow_private=False)
    
    # Test localhost
    result = tool({"url": "http://localhost:8080"})
    assert result.is_error
    assert "private IP ranges is blocked" in result.text
    
    # Test 127.0.0.1
    result = tool({"url": "http://127.0.0.1:8080"})
    assert result.is_error
    assert "private IP ranges is blocked" in result.text
    
    # Test 10.x.x.x
    result = tool({"url": "http://10.0.0.1:8080"})
    assert result.is_error
    assert "private IP ranges is blocked" in result.text
    
    # Test 192.168.x.x
    result = tool({"url": "http://192.168.1.1:8080"})
    assert result.is_error
    assert "private IP ranges is blocked" in result.text


def test_web_fetch_private_ip_allowed():
    """Test allow_private=True bypasses private IP blocking."""
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MockHTTPResponse(b"OK", "text/plain")
        mock_urlopen.return_value = mock_response
        
        tool = make_web_fetch_tool(allow_private=True)
        result = tool({"url": "http://localhost:8080"})
        
        assert not result.is_error
        assert result.text == "OK"


@mock.patch("urllib.request.urlopen")
def test_web_fetch_empty_url(mock_urlopen):
    """Test empty URL handling."""
    tool = make_web_fetch_tool()
    result = tool({"url": ""})
    assert result.is_error
    assert "non-empty string" in result.text


@mock.patch("urllib.request.urlopen")
def test_web_fetch_invalid_method(mock_urlopen):
    """Test invalid HTTP method handling."""
    tool = make_web_fetch_tool()
    result = tool({"url": "https://example.com", "method": "PUT"})
    assert result.is_error
    assert "method must be 'GET' or 'POST'" in result.text


def test_web_search_empty_query():
    """Test empty search query handling."""
    tool = make_web_search_tool()
    result = tool({"query": ""})
    assert result.is_error
    assert "non-empty string" in result.text


def test_web_search_limit_clamping():
    """Test search result limit clamping."""
    def fake_search_fn(query: str, limit: int) -> list[dict[str, str]]:
        # Return more results than requested to test clamping
        return [
            {"title": f"Result {i}", "url": f"http://example{i}.com", "snippet": f"Snippet {i}"}
            for i in range(1, 15)
        ]
    
    # Test limit above maximum
    tool = make_web_search_tool(search_fn=fake_search_fn)
    result = tool({"query": "test", "limit": 15})
    # Should be clamped to SEARCH_RESULT_LIMIT_MAX (10)
    assert result.text.count(". Result ") == 10
    
    # Test limit below minimum  
    result = tool({"query": "test", "limit": 0})
    assert result.text.count(". Result ") == 1


def test_web_search_results_formatting():
    """Test search results formatting."""
    def fake_search_fn(query: str, limit: int) -> list[dict[str, str]]:
        return [
            {
                "title": "Example Title",
                "url": "https://example.com",
                "snippet": "This is a snippet."
            }
        ]
    
    tool = make_web_search_tool(search_fn=fake_search_fn)
    result = tool({"query": "test"})
    
    assert not result.is_error
    assert "1. Example Title" in result.text
    assert "https://example.com" in result.text
    assert "This is a snippet." in result.text


def test_web_search_no_results():
    """Test handling of no search results."""
    def fake_search_fn(query: str, limit: int) -> list[dict[str, str]]:
        return []
    
    tool = make_web_search_tool(search_fn=fake_search_fn)
    result = tool({"query": "test"})
    
    assert not result.is_error
    assert result.text == "No results found"


def test_web_search_exception_handling():
    """Test search function exception handling."""
    def failing_search_fn(query: str, limit: int) -> list[dict[str, str]]:
        raise ValueError("Search service unavailable")
    
    tool = make_web_search_tool(search_fn=failing_search_fn)
    result = tool({"query": "test"})
    
    assert result.is_error
    assert "search failed" in result.text
    assert "attempted to search for" in result.text
    assert "expected:" in result.text
    assert "happened:" in result.text
    assert "fix:" in result.text


def test_web_registry_contains_both_tools():
    """Test web_registry contains both web tools."""
    registry = web_registry()
    tool_names = registry.names()
    
    assert "web_fetch" in tool_names
    assert "web_search" in tool_names
    assert len(tool_names) >= 2  # Should have MVP tools plus web tools


def test_web_registry_schemas_structure():
    """Test web registry schemas have correct OpenAI format."""
    registry = web_registry()
    schemas = registry.schemas()
    
    web_fetch_schema = None
    web_search_schema = None
    
    for schema in schemas:
        if schema["function"]["name"] == "web_fetch":
            web_fetch_schema = schema
        elif schema["function"]["name"] == "web_search":
            web_search_schema = schema
    
    assert web_fetch_schema is not None
    assert web_search_schema is not None
    assert web_fetch_schema["type"] == "function"
    assert web_search_schema["type"] == "function"
    assert "parameters" in web_fetch_schema["function"]
    assert "parameters" in web_search_schema["function"]


if __name__ == "__main__":
    pytest.main([__file__])