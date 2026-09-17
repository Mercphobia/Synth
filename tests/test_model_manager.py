"""Tests for model manager functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth.model_manager import (
    CATALOG,
    ManagerError,
    ModelManager,
    RecommendedModel,
    _validate_model_name,
)


def test_catalog_has_expected_models():
    """Test that catalog contains expected models."""
    assert "llama3.2:1b" in CATALOG
    assert "mistral" in CATALOG
    assert "phi3" in CATALOG
    
    model = CATALOG["llama3.2:1b"]
    assert isinstance(model, RecommendedModel)
    assert model.tier in {"fast", "balanced", "capable"}
    assert model.size_hint_gb > 0


def test_validate_model_name_valid():
    """Test valid model names pass validation."""
    assert _validate_model_name("llama3") == "llama3"
    assert _validate_model_name("llama3:8b") == "llama3:8b"
    assert _validate_model_name("qwen2.5:7b") == "qwen2.5:7b"


def test_validate_model_name_invalid():
    """Test invalid model names raise error."""
    with pytest.raises(ManagerError):
        _validate_model_name("")
    
    with pytest.raises(ManagerError):
        _validate_model_name("model;rm -rf")
    
    with pytest.raises(ManagerError):
        _validate_model_name("model|cat /etc/passwd")


def test_recommend_all_models():
    """Test recommend returns all models when no filters."""
    manager = ModelManager()
    recommendations = manager.recommend()
    
    assert len(recommendations) == len(CATALOG)
    # Should be sorted by size
    sizes = [m.size_hint_gb for m in recommendations]
    assert sizes == sorted(sizes)


def test_recommend_by_tier():
    """Test recommend filters by tier."""
    manager = ModelManager()
    fast_models = manager.recommend(tier="fast")
    
    assert len(fast_models) > 0
    assert all(m.tier == "fast" for m in fast_models)


def test_recommend_by_size():
    """Test recommend filters by available space."""
    manager = ModelManager()
    # Request models that fit in 5GB (more realistic threshold)
    small_models = manager.recommend(min_gb_free=5.0)
    
    assert len(small_models) > 0
    assert all(m.size_hint_gb <= 5.0 for m in small_models)


def test_recommend_unknown_tier():
    """Test recommend raises error for unknown tier."""
    manager = ModelManager()
    with pytest.raises(ManagerError, match="Unknown tier"):
        manager.recommend(tier="invalid")


@patch("synth.model_manager.OllamaClient")
def test_install_success(mock_client_class):
    """Test successful model installation."""
    mock_client = MagicMock()
    mock_client.pull_model.return_value = True
    mock_client.is_available.return_value = True
    mock_client_class.return_value = mock_client
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / ".synth" / "models.json"
        manager = ModelManager(client=mock_client, state_path=state_path)
        
        result = manager.install("test-model")
        
        assert result == "installed"
        mock_client.pull_model.assert_called_once_with("test-model", stream_callback=None)
        assert state_path.exists()


@patch("synth.model_manager.OllamaClient")
def test_install_failure(mock_client_class):
    """Test failed model installation."""
    mock_client = MagicMock()
    mock_client.pull_model.return_value = False
    mock_client_class.return_value = mock_client
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / ".synth" / "models.json"
        manager = ModelManager()
        manager.state_path = state_path
        
        result = manager.install("test-model")
        
        assert result == "failed"
        assert not state_path.exists()


@patch("synth.model_manager.OllamaClient")
def test_uninstall_success(mock_client_class):
    """Test successful model uninstallation."""
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["test-model"]
    mock_client.base_url = "http://localhost:11434"  # Add base_url
    mock_client_class.return_value = mock_client
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / ".synth" / "models.json"
        manager = ModelManager()
        manager.state_path = state_path
        
        # Install first
        with patch.object(manager, "_update_state"):
            manager.install("test-model")
        
        # Mock urllib for DELETE
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            result = manager.uninstall("test-model")
            
            assert result is True
            mock_urlopen.assert_called_once()


@patch("synth.model_manager.OllamaClient")
def test_uninstall_not_found(mock_client_class):
    """Test uninstall returns False for non-existent model."""
    mock_client = MagicMock()
    mock_client.list_models.return_value = []
    mock_client_class.return_value = mock_client
    
    manager = ModelManager()
    
    # Mock urllib to return 404
    with patch("urllib.request.urlopen") as mock_urlopen:
        from urllib.error import HTTPError
        
        mock_error = HTTPError("url", 404, "Not Found", {}, None)
        mock_urlopen.side_effect = mock_error
        
        result = manager.uninstall("nonexistent-model")
        
        assert result is False


@patch("synth.model_manager.OllamaClient")
def test_installed_merges_sources(mock_client_class):
    """Test installed() merges Ollama and state sources."""
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["ollama-model"]
    mock_client_class.return_value = mock_client
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / ".synth" / "models.json"
        # Create state with additional model
        state_path.parent.mkdir(parents=True)
        with open(state_path, "w") as f:
            json.dump({"installed": ["state-model"]}, f)
        
        manager = ModelManager()
        manager.state_path = state_path
        
        installed = manager.installed()
        
        assert "ollama-model" in installed
        assert "state-model" in installed
        assert len(installed) >= 2


@patch("synth.model_manager.shutil.disk_usage")
def test_status(mock_disk_usage):
    """Test status returns correct information."""
    mock_disk_usage.return_value.free = 10 * (1024**3)  # 10 GB free
    
    with patch("synth.model_manager.OllamaClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.list_models.return_value = ["model1", "model2"]
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        status = manager.status()
        
        assert status["available"] is True
        assert status["models"] == 2
        assert status["free_gb"] == 10.0


def test_state_file_atomic_write():
    """Test state file is written atomically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "models.json"
        manager = ModelManager()
        manager.state_path = state_path
        
        # Update state multiple times
        manager._update_state("model1", "installed")
        manager._update_state("model2", "installed")
        
        # Verify final state
        with open(state_path, "r") as f:
            state = json.load(f)
        
        assert "model1" in state["installed"]
        assert "model2" in state["installed"]


def test_corrupt_state_file_handled():
    """Test corrupt state file is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "models.json"
        # Write corrupt JSON
        with open(state_path, "w") as f:
            f.write("corrupt json {")
        
        manager = ModelManager()
        manager.state_path = state_path
        
        # Should not crash
        state = manager._load_state()
        assert state == {"installed": []}
