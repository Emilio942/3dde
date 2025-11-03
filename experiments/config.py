"""Configuration loader and parser."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for diffusion model."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._config = config_dict
        
    def __getattr__(self, name: str) -> Any:
        """Get config value."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load config from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_default(cls) -> 'Config':
        """Load default configuration."""
        config_dir = Path(__file__).parent / "configs"
        return cls.from_yaml(config_dir / "default.yaml")
    
    def save(self, yaml_path: str):
        """Save config to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def update(self, updates: Dict[str, Any]):
        """Update config with new values."""
        self._config.update(updates)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config})"


def load_config(config_path: str = None) -> Config:
    """
    Load configuration.
    
    Args:
        config_path: Path to YAML config file. If None, uses default.
    
    Returns:
        Config object
    """
    if config_path is None:
        return Config.from_default()
    
    return Config.from_yaml(config_path)


if __name__ == "__main__":
    # Test config loading
    print("Loading default config...")
    config = load_config()
    
    print(f"\nModel config:")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Num layers: {config.model.num_layers}")
    
    print(f"\nDiffusion config:")
    print(f"  Num steps: {config.diffusion.num_steps}")
    print(f"  Beta schedule: {config.diffusion.beta_schedule}")
    
    print(f"\nTraining config:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    
    print("\n✓ Config loading works!")
