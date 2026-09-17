"""Model manager for Ollama models (spec 6.18).

Provides a curated catalog of recommended models with size hints and
installation management around the OllamaClient.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR
from synth.local import OllamaClient


@dataclass(frozen=True)
class RecommendedModel:
    """A recommended model from the curated catalog."""
    
    name: str
    description: str
    size_hint_gb: float
    tier: str  # 'fast', 'balanced', or 'capable'


# Curated catalog of well-known Ollama models
# Extend this list as needed with accurate size estimates
CATALOG = {
    "llama3.2:1b": RecommendedModel(
        name="llama3.2:1b",
        description="Fast, compact Llama 3.2 1B parameter model",
        size_hint_gb=0.6,
        tier="fast",
    ),
    "llama3.2:3b": RecommendedModel(
        name="llama3.2:3b",
        description="Balanced Llama 3.2 3B parameter model",
        size_hint_gb=2.0,
        tier="balanced",
    ),
    "qwen2.5:0.5b": RecommendedModel(
        name="qwen2.5:0.5b",
        description="Ultra-fast Qwen 2.5 0.5B parameter model",
        size_hint_gb=0.3,
        tier="fast",
    ),
    "qwen2.5:7b": RecommendedModel(
        name="qwen2.5:7b",
        description="Capable Qwen 2.5 7B parameter model",
        size_hint_gb=4.5,
        tier="capable",
    ),
    "mistral": RecommendedModel(
        name="mistral",
        description="Balanced Mistral 7B model",
        size_hint_gb=4.1,
        tier="balanced",
    ),
    "phi3": RecommendedModel(
        name="phi3",
        description="Fast Microsoft Phi-3 3.8B model",
        size_hint_gb=2.3,
        tier="fast",
    ),
    "gemma2:2b": RecommendedModel(
        name="gemma2:2b",
        description="Balanced Google Gemma 2 2B model",
        size_hint_gb=1.4,
        tier="balanced",
    ),
    "deepseek-r1:1.5b": RecommendedModel(
        name="deepseek-r1:1.5b",
        description="Capable DeepSeek R1 1.5B reasoning model",
        size_hint_gb=0.9,
        tier="capable",
    ),
}


class ManagerError(Exception):
    """Raised when model management operations fail."""


def _get_free_space() -> float | None:
    """Get free disk space in GB."""
    try:
        usage = shutil.disk_usage(Path.home())
        return usage.free / (1024**3)  # Convert to GB
    except OSError:
        return None


def _validate_model_name(name: str) -> str:
    """Validate and normalize model name."""
    if not isinstance(name, str) or not name.strip():
        raise ManagerError("Model name must be a non-empty string")
    
    # Allow alphanumeric, colon, dot, dash, underscore
    import re
    if not re.match(r"^[a-zA-Z0-9._:-]+$", name):
        raise ManagerError(f"Invalid model name: {name}")
    
    # Reject shell metacharacters
    dangerous = {";", "&", "|", "$", "`", "\\", "(", ")", "{", "}", "[", "]"}
    if any(c in name for c in dangerous):
        raise ManagerError(f"Model name contains dangerous characters: {name}")
    
    return name.strip()


class ModelManager:
    """Manages Ollama model installation and recommendations."""
    
    def __init__(self, client: OllamaClient | None = None, state_path: Path | None = None):
        """Initialize the model manager.
        
        Args:
            client: Optional OllamaClient instance. Creates one if None.
            state_path: Optional path to state file. Defaults to CONFIG_DIR/models.json.
        """
        self.client = client or OllamaClient()
        self.state_path = state_path or (Path(CONFIG_DIR) / "models.json")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
    
    def recommend(
        self, min_gb_free: float | None = None, tier: str | None = None
    ) -> list[RecommendedModel]:
        """Get recommended models based on constraints.
        
        Args:
            min_gb_free: Minimum free space required (in GB). If None, skip check.
            tier: Filter by tier ('fast', 'balanced', 'capable'). If None, no filter.
            
        Returns:
            List of RecommendedModel objects sorted by size (smallest first).
            
        Raises:
            ManagerError: If tier is unknown.
        """
        if tier is not None and tier not in {"fast", "balanced", "capable"}:
            raise ManagerError(f"Unknown tier: {tier}. Must be 'fast', 'balanced', or 'capable'")
        
        # Get free space if needed
        free_gb = _get_free_space() if min_gb_free is not None else None
        
        candidates = []
        for model in CATALOG.values():
            # Apply tier filter
            if tier is not None and model.tier != tier:
                continue
            
            # Apply size filter
            if free_gb is not None and model.size_hint_gb > free_gb:
                continue
            
            candidates.append(model)
        
        # Sort by size (smallest first)
        candidates.sort(key=lambda m: m.size_hint_gb)
        return candidates
    
    def install(self, name: str, callback=None) -> str:
        """Install a model.
        
        Args:
            name: Model name (must be in CATALOG or already available)
            callback: Optional callback for progress updates
            
        Returns:
            Status string: 'installed' or 'failed'
            
        Raises:
            ManagerError: If model name is invalid or installation fails.
        """
        name = _validate_model_name(name)
        
        # Check if already installed
        if name in self.installed():
            return "installed"
        
        # Pull the model
        try:
            success = self.client.pull_model(name, stream_callback=callback)
            if success:
                self._update_state(name, "installed")
                return "installed"
            else:
                return "failed"
        except Exception as exc:
            raise ManagerError(f"Failed to install {name}: {exc}") from exc
    
    def uninstall(self, name: str) -> bool:
        """Uninstall a model.
        
        Args:
            name: Model name to uninstall
            
        Returns:
            True if uninstalled, False if model was not found.
            
        Raises:
            ManagerError: If uninstallation fails for other reasons.
        """
        name = _validate_model_name(name)
        
        # Check if installed
        if name not in self.installed():
            return False
        
        # Use DELETE /api/delete endpoint
        import urllib.request
        import urllib.error
        
        url = f"{self.client.base_url}/api/delete"
        data = json.dumps({"name": name}).encode("utf-8")
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    self._update_state(name, "uninstalled")
                    return True
                elif response.status == 404:
                    return False
                else:
                    raise ManagerError(f"DELETE failed with status {response.status}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise ManagerError(f"HTTP error during uninstall: {exc}") from exc
        except Exception as exc:
            raise ManagerError(f"Failed to uninstall {name}: {exc}") from exc
    
    def installed(self) -> list[str]:
        """Get list of installed models.
        
        Returns:
            List of model names from both Ollama and local state.
        """
        try:
            ollama_models = self.client.list_models()
        except Exception:
            ollama_models = []
        
        state_models = self._load_state().get("installed", [])
        
        # Union of both sources
        all_models = set(ollama_models + state_models)
        return sorted(all_models)
    
    def status(self) -> dict[str, Any]:
        """Get overall status.
        
        Returns:
            Dictionary with availability, model count, and free space.
        """
        available = self.client.is_available()
        models = len(self.installed())
        free_gb = _get_free_space()
        
        return {
            "available": available,
            "models": models,
            "free_gb": free_gb,
        }
    
    def _load_state(self) -> dict[str, Any]:
        """Load state from JSON file."""
        if not self.state_path.exists():
            return {"installed": []}
        
        try:
            with open(self.state_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable state - treat as empty
            return {"installed": []}
    
    def _update_state(self, model_name: str, action: str) -> None:
        """Update state file atomically."""
        state = self._load_state()
        
        if action == "installed":
            if model_name not in state["installed"]:
                state["installed"].append(model_name)
        elif action == "uninstalled":
            if model_name in state["installed"]:
                state["installed"].remove(model_name)
        
        # Write atomically
        tmp_path = self.state_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            tmp_path.replace(self.state_path)
        except OSError:
            # Best effort - don't crash if state write fails
            pass
