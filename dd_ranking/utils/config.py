import yaml
import json
from typing import Dict, Any


class ConfigRegistry:
    """Registry to manage configurations."""
    _configs = {}

    @classmethod
    def register_config(cls, name: str, config: Dict[str, Any]):
        """Register a new configuration."""
        if name in cls._configs:
            raise ValueError(f"Configuration {name} already exists.")
        cls._configs[name] = config

    @classmethod
    def get_config(cls, name: str):
        """Retrieve a registered configuration."""
        if name not in cls._configs:
            raise ValueError(f"Configuration {name} not found.")
        return cls._configs[name]

    @classmethod
    def list_configs(cls):
        """List all registered configurations."""
        return list(cls._configs.keys())

    @classmethod
    def load_config_from_file(cls, filepath: str, config_name: str):
        """Load configuration from a YAML or JSON file and register it."""
        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            with open(filepath, "r") as f:
                config = yaml.safe_load(f)
        elif filepath.endswith(".json"):
            with open(filepath, "r") as f:
                config = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use YAML or JSON.")
        
        cls.register_config(config_name, config)