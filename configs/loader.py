"""Configuration loader for training experiments.

Provides flexible loading of TrainingConfig from Python files.
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.base import TrainingConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str, project_root: Path | None = None) -> "TrainingConfig":
    """Load config from Python file.

    Supports multiple formats:
    - Module path: 'configs.experiments.quick_val'
    - File path: 'configs/experiments/quick_val.py'
    - Absolute path: '/path/to/config.py'

    The config file must contain a 'config' variable that is a TrainingConfig instance.

    Args:
        config_path: Import path like 'configs.experiments.quick_val' or file path.
        project_root: Optional project root for relative paths.

    Returns:
        TrainingConfig instance.

    Raises:
        ValueError: If config file cannot be loaded or doesn't contain valid config.
    """
    # Import here to avoid circular dependency
    from configs.base import TrainingConfig

    try:
        config = _load_config_module(config_path, project_root)

        # Validate config type
        if not isinstance(config, TrainingConfig):
            raise ValueError(
                f"Config object must be TrainingConfig instance, got {type(config).__name__}"
            )

        return config

    except (ModuleNotFoundError, ImportError) as e:
        raise ValueError(f"Failed to import config module '{config_path}': {e}") from e
    except AttributeError as e:
        raise ValueError(f"Config file must contain a 'config' variable: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load config from '{config_path}': {e}") from e


def _load_config_module(config_path: str, project_root: Path | None = None):
    """Load the config module and extract the config variable.

    Args:
        config_path: Path or module name for the config.
        project_root: Optional project root for relative paths.

    Returns:
        The config object from the module.
    """
    if config_path.startswith("configs/") or config_path.startswith("configs."):
        # Module path - try importing as module
        return _load_as_module(config_path)
    else:
        # File path - exec and get 'config' variable
        return _load_as_file(config_path, project_root)


def _load_as_module(config_path: str):
    """Load config as a Python module import.

    Args:
        config_path: Module path like 'configs.experiments.quick_val'.

    Returns:
        The config object from the module.
    """
    module_path = config_path.replace(".py", "").replace("/", ".")

    try:
        # First, try to import the full path as a module
        module = importlib.import_module(module_path)
        return module.config
    except (ModuleNotFoundError, ImportError, AttributeError):
        # If that fails, try treating the last part as a variable name
        parts = module_path.split(".")
        module_name = ".".join(parts[:-1])
        var_name = parts[-1]

        module = importlib.import_module(module_name)
        return getattr(module, var_name)


def _load_as_file(config_path: str, project_root: Path | None = None):
    """Load config from a file path.

    Args:
        config_path: File path to the config.
        project_root: Optional project root for relative paths.

    Returns:
        The config object from the file.
    """
    config_file = Path(config_path)

    if not config_file.is_absolute() and project_root:
        config_file = project_root / config_path

    spec = importlib.util.spec_from_file_location("config_module", config_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config from path: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config
