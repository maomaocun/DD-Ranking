import yaml
import json
from typing import Dict, Any


class Config:
    """Configuration object to manage individual configurations."""
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with a configuration dictionary."""
        self.config = config or {}

    @classmethod
    def from_file(cls, filepath: str):
        """Load configuration from a YAML or JSON file."""
        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            with open(filepath, "r") as f:
                config = yaml.safe_load(f)
        elif filepath.endswith(".json"):
            with open(filepath, "r") as f:
                config = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use YAML or JSON.")
        return cls(config)

    def get(self, key: str, default: Any = None):
        """Get a value from the config."""
        return self.config.get(key, default)

    def update(self, overrides: Dict[str, Any]):
        """Update the configuration with overrides."""
        self.config.update(overrides)

    def __repr__(self):
        return f"Config({self.config})"