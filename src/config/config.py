"""Unified configuration loading for DeepBioParts.

Supports layered configuration:
    configs/default.yaml  <  configs/bioparts/{biopart}.yaml  <  CLI arguments
         (lowest)                      (medium)                 (highest)

Usage:
    from config import load_config
    cfg = load_config("promoter", cli_overrides={"training.batch_size": 32})
    print(cfg["training"]["learning_rate"])

    # Get model paths
    from config import get_model_paths
    paths = get_model_paths("promoter")
    print(paths["predictor_dir"])  # Auto-discovered predictor path
    print(paths["diffusion_checkpoint"])  # Auto-discovered diffusion path
"""

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base (override wins on conflict)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict to dotted keys: {"a": {"b": 1}} -> {"a.b": 1}."""
    items: List = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(flat: Dict[str, Any], sep: str = ".") -> Dict:
    """Unflatten dotted keys back to nested dict."""
    result: Dict = {}
    for key, value in flat.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


def _resolve_config_dir() -> Path:
    """Find the configs/ directory."""
    # Try relative to this file first
    candidate = Path(__file__).parent.parent / "configs"
    if candidate.is_dir():
        return candidate

    # Try CWD
    candidate = Path.cwd() / "configs"
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        "Cannot find configs/ directory. "
        "Ensure it exists at codes/configs/ or the current working directory."
    )


def load_yaml(path: Path) -> Dict:
    """Load a single YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config(
    biopart: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load merged configuration from YAML files and overrides.

    Priority order (later overrides earlier):
        1. configs/default.yaml
        2. configs/bioparts/{biopart}.yaml
        3. explicit config overrides
        4. config_overrides dict
        5. cli_overrides dict (highest priority)

    Args:
        biopart: Biological part type (promoter, rbs, terminator).
        config_overrides: Dict of config values to override.
        cli_overrides: Dict of CLI-provided overrides (dotted keys supported).
        config_dir: Explicit path to configs directory.

    Returns:
        Merged configuration dictionary.
    """
    configs_path = Path(config_dir) if config_dir else _resolve_config_dir()

    # Layer 1: defaults
    config = load_yaml(configs_path / "default.yaml")

    # Layer 2: biopart-specific
    if biopart:
        biopart_config = load_yaml(configs_path / "bioparts" / f"{biopart}.yaml")
        config = _deep_merge(config, biopart_config)

    # Layer 3: explicit config overrides
    if config_overrides:
        config = _deep_merge(config, config_overrides)

    # Layer 4: CLI overrides (dotted keys like "training.batch_size")
    if cli_overrides:
        flat_config = _flatten_dict(config)
        flat_config.update(cli_overrides)
        config = _unflatten_dict(flat_config)

    return config


def config_to_namespace(config: Dict) -> argparse.Namespace:
    """Convert config dict to argparse.Namespace for backward compatibility.

    Args:
        config: Configuration dictionary.

    Returns:
        Nested argparse.Namespace object.
    """
    def _dict_to_ns(d):
        ns = argparse.Namespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(ns, key, _dict_to_ns(value))
            else:
                setattr(ns, key, value)
        return ns

    return _dict_to_ns(config)


def save_config(config: Dict, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        save_path: Output file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_model_paths(
    biopart: str,
    config_dir: Optional[Union[str, Path]] = None,
    auto_discover: bool = True,
) -> Dict[str, Any]:
    """Get model path configuration for a given biopart.

    Supports both auto-discovery and manual configuration:
    1. If model_paths is defined in the config file, use it directly.
    2. If auto_discover=True, search standard directories for the model.

    Args:
        biopart: biological part type (promoter, rbs, terminator)
        config_dir: path to the configs directory
        auto_discover: whether to auto-discover model paths

    Returns:
        Dict of model paths:
        {
            "predictor_dir": "path/to/predictor",
            "predictor_model_type": "conv",
            "predictor_encoding": "onehot",
            "diffusion_checkpoint": "path/to/diffusion.pth",
            "diffusion_checkpoint_latest": "path/to/latest.pth"
        }

    Raises:
        FileNotFoundError: could not find model paths
        ValueError: unsupported biopart type
    """
    # Supported biopart types
    SUPPORTED_BIOPARTS = ["promoter", "rbs", "terminator"]
    if biopart not in SUPPORTED_BIOPARTS:
        raise ValueError(
            f"Unsupported biopart type: {biopart}. "
            f"Supported types: {', '.join(SUPPORTED_BIOPARTS)}"
        )

    configs_path = Path(config_dir) if config_dir else _resolve_config_dir()

    # Load the config file
    config = load_config(biopart=biopart, config_dir=config_dir)

    # Initialize the return dict
    model_paths = {
        "predictor_dir": None,
        "predictor_model_type": "conv",
        "predictor_encoding": "onehot",
        "diffusion_checkpoint": None,
        "diffusion_checkpoint_latest": None,
    }

    # Try to read model paths from the config file
    if "model_paths" in config:
        for key in model_paths.keys():
            if key in config["model_paths"]:
                model_paths[key] = config["model_paths"][key]

    # Fallback: read the default_predictor field actually used by the config
    # (the legacy model_paths field is usually unset).
    if not model_paths["predictor_dir"]:
        default_pred = config.get("default_predictor")
        if default_pred:
            model_paths["predictor_dir"] = default_pred
            dp_lower = default_pred.lower()
            if "/evo/" in dp_lower or "evo_lora" in dp_lower:
                model_paths["predictor_model_type"] = "evo"

    # If auto-discovery is enabled and the config has no complete paths, search.
    if auto_discover:
        model_paths = _auto_discover_model_paths(biopart, model_paths)

    # Validate that paths exist
    _validate_model_paths(biopart, model_paths)

    return model_paths


def get_predictor_configs(
    biopart: str,
    config_dir: Optional[Union[str, Path]] = None,
    auto_discover: bool = True,
) -> Dict[str, Any]:
    """Get predictor configuration for a given biopart (supports multi-model ensembles).

    The new config format supports multiple predictors (a DL model + an Evo model).

    Args:
        biopart: biological part type (promoter, rbs, terminator)
        config_dir: path to the configs directory
        auto_discover: whether to auto-discover model paths

    Returns:
        Dict containing multiple predictor configurations:
        {
            "dl_model": {
                "predictor_dir": "path/to/dl_predictor",
                "model_type": "conv",
                "encoding": "onehot",
                "enabled": true,
                "weight": 0.5
            },
            "evo_model": {
                "predictor_dir": "path/to/evo_predictor",
                "model_type": "evo",
                "enabled": true,
                "weight": 0.5,
                "model_name": "evo-1.5-8k-base",
                "lora_rank": 16,
                "single_model": false
            },
            "diffusion": {
                "checkpoint_path": "path/to/diffusion.pth",
                "checkpoint_path_latest": "path/to/latest.pth"
            }
        }

    Raises:
        FileNotFoundError: could not find model paths
        ValueError: unsupported biopart type
    """
    # Supported biopart types
    SUPPORTED_BIOPARTS = ["promoter", "rbs", "terminator"]
    if biopart not in SUPPORTED_BIOPARTS:
        raise ValueError(
            f"Unsupported biopart type: {biopart}. "
            f"Supported types: {', '.join(SUPPORTED_BIOPARTS)}"
        )

    # Load the config file
    config = load_config(biopart=biopart, config_dir=config_dir)

    # Initialize the return dict (defaults)
    predictor_configs = {
        "dl_model": {
            "predictor_dir": None,
            "model_type": "conv",
            "encoding": "onehot",
            "enabled": True,
            "weight": 0.5,
        },
        "evo_model": {
            "predictor_dir": None,
            "checkpoint_path": None,
            "model_type": "evo",
            "enabled": True,
            "weight": 0.5,
            "model_name": "evo-1.5-8k-base",
            "lora_rank": 16,
            "single_model": False,
        },
        "diffusion": {
            "checkpoint_path": None,
            "checkpoint_path_latest": None,
        },
    }

    # Read from the new config format
    if "predictors" in config:
        # Update the DL model config
        if "dl_model" in config["predictors"]:
            predictor_configs["dl_model"].update(config["predictors"]["dl_model"])

        # Update the Evo model config
        if "evo_model" in config["predictors"]:
            predictor_configs["evo_model"].update(config["predictors"]["evo_model"])

        # Update the diffusion model config
        if "diffusion" in config:
            predictor_configs["diffusion"].update(config["diffusion"])

    # Backward compatibility with the legacy model_paths format
    elif "model_paths" in config:
        old_paths = config["model_paths"]
        predictor_configs["dl_model"]["predictor_dir"] = old_paths.get("predictor_dir")
        predictor_configs["dl_model"]["model_type"] = old_paths.get("predictor_model_type", "conv")
        predictor_configs["dl_model"]["encoding"] = old_paths.get("predictor_encoding", "onehot")
        predictor_configs["diffusion"]["checkpoint_path"] = old_paths.get("diffusion_checkpoint")
        predictor_configs["diffusion"]["checkpoint_path_latest"] = old_paths.get("diffusion_checkpoint_latest")

        # Disable the Evo model (legacy configs have no Evo entry)
        predictor_configs["evo_model"]["enabled"] = False

    # If auto-discovery is enabled, search for model paths
    if auto_discover:
        predictor_configs = _auto_discover_predictor_configs(biopart, predictor_configs)

    # Validate the enabled model paths
    _validate_predictor_configs(biopart, predictor_configs)

    return predictor_configs


def _auto_discover_model_paths(
    biopart: str,
    existing_paths: Dict[str, Any],
) -> Dict[str, Any]:
    """Auto-discover model paths.

    Searches the standard directories for model files, per project convention.

    Args:
        biopart: biological part type
        existing_paths: existing path configuration

    Returns:
        Updated path dict
    """
    import re

    result = existing_paths.copy()

    # Project root (two levels above the config directory)
    config_dir = _resolve_config_dir()
    project_root = config_dir.parent

    # Auto-discover the predictor path
    if not result["predictor_dir"]:
        predictor_base = project_root / "predictor_checkpoints" / "dl"
        if predictor_base.exists():
            # Search for a directory matching the biopart
            pattern = re.compile(rf"{biopart}_predictor_")
            for item in predictor_base.iterdir():
                if item.is_dir() and pattern.search(item.name):
                    result["predictor_dir"] = str(item)
                    logger.info(f"Auto-discovered predictor path: {item}")
                    break

    # Auto-discover the diffusion model path
    if not result["diffusion_checkpoint"]:
        diffusion_base = project_root / "diffusion_checkpoints"
        if diffusion_base.exists():
            # Search for a directory matching the biopart
            pattern = re.compile(rf"{biopart}_direct_")
            for item in diffusion_base.iterdir():
                if item.is_dir() and pattern.search(item.name):
                    best_checkpoint = item / "checkpoints" / "best.pth"
                    latest_checkpoint = item / "checkpoints" / "latest.pth"

                    if best_checkpoint.exists():
                        result["diffusion_checkpoint"] = str(best_checkpoint)
                        logger.info(f"Auto-discovered diffusion checkpoint (best): {best_checkpoint}")

                    if latest_checkpoint.exists():
                        result["diffusion_checkpoint_latest"] = str(latest_checkpoint)

                    break

    return result


def _validate_model_paths(biopart: str, model_paths: Dict[str, Any]) -> None:
    """Validate that model paths exist.

    Args:
        biopart: biological part type
        model_paths: model path dict

    Raises:
        FileNotFoundError: a required path does not exist
    """
    errors = []

    # Validate the predictor path
    if model_paths["predictor_dir"]:
        predictor_path = Path(model_paths["predictor_dir"])
        if not predictor_path.exists():
            errors.append(f"Predictor directory does not exist: {predictor_path}")
        else:
            # Check that fold_* subdirectories exist
            fold_dirs = list(predictor_path.glob("fold_*"))
            if not fold_dirs:
                errors.append(f"No fold_* subdirectory found in predictor directory: {predictor_path}")

    # Validate the diffusion model path
    if model_paths["diffusion_checkpoint"]:
        diffusion_path = Path(model_paths["diffusion_checkpoint"])
        if not diffusion_path.exists():
            errors.append(f"Diffusion model file does not exist: {diffusion_path}")

    if errors:
        error_msg = f"{biopart} model path validation failed:\n" + "\n".join(errors)
        logger.warning(error_msg)
        # Only raise when all paths are missing
        if not model_paths["predictor_dir"] and not model_paths["diffusion_checkpoint"]:
            raise FileNotFoundError(error_msg)


def _auto_discover_predictor_configs(
    biopart: str,
    existing_configs: Dict[str, Any],
) -> Dict[str, Any]:
    """Auto-discover predictor configurations (supports multiple models).

    Args:
        biopart: biological part type
        existing_configs: existing configuration dict

    Returns:
        Updated configuration dict
    """
    import re

    result = existing_configs.copy()

    # Project root
    config_dir = _resolve_config_dir()
    project_root = config_dir.parent

    # Auto-discover the DL model path
    if not result["dl_model"]["predictor_dir"]:
        predictor_base = project_root / "predictor_checkpoints" / "dl"
        if predictor_base.exists():
            pattern = re.compile(rf"{biopart}_predictor_")
            for item in predictor_base.iterdir():
                if item.is_dir() and pattern.search(item.name):
                    result["dl_model"]["predictor_dir"] = str(item)
                    logger.info(f"Auto-discovered DL predictor path: {item}")
                    break

    # Auto-discover the Evo model path
    if not result["evo_model"]["predictor_dir"] and not result["evo_model"]["checkpoint_path"]:
        evo_base = project_root / "predictor_checkpoints" / "evo"
        if evo_base.exists():
            # Prefer the newest Evo LoRA model
            evo_lora_pattern = re.compile(rf"{biopart}_evo_lora_")
            for item in sorted(evo_base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if item.is_dir() and evo_lora_pattern.search(item.name):
                    result["evo_model"]["predictor_dir"] = str(item)
                    result["evo_model"]["single_model"] = False
                    logger.info(f"Auto-discovered Evo LoRA predictor path: {item}")
                    break

            # If no LoRA model was found, fall back to a legacy Evo model
            if not result["evo_model"]["predictor_dir"]:
                evo_pattern = re.compile(rf"{biopart}_")
                for item in evo_base.iterdir():
                    if item.is_dir() and evo_pattern.search(item.name):
                        # Check whether this is a single model (best_model.pth)
                        best_model = item / "best_model.pth"
                        if best_model.exists():
                            result["evo_model"]["checkpoint_path"] = str(best_model)
                            result["evo_model"]["single_model"] = True
                            logger.info(f"Auto-discovered Evo single-model path: {best_model}")
                        else:
                            result["evo_model"]["predictor_dir"] = str(item)
                            result["evo_model"]["single_model"] = False
                            logger.info(f"Auto-discovered Evo cross-validation model path: {item}")
                        break

    # Auto-discover the diffusion model path
    if not result["diffusion"]["checkpoint_path"]:
        diffusion_base = project_root / "diffusion_checkpoints"
        if diffusion_base.exists():
            pattern = re.compile(rf"{biopart}_direct_")
            for item in diffusion_base.iterdir():
                if item.is_dir() and pattern.search(item.name):
                    best_checkpoint = item / "checkpoints" / "best.pth"
                    latest_checkpoint = item / "checkpoints" / "latest.pth"

                    if best_checkpoint.exists():
                        result["diffusion"]["checkpoint_path"] = str(best_checkpoint)
                        logger.info(f"Auto-discovered diffusion checkpoint (best): {best_checkpoint}")

                    if latest_checkpoint.exists():
                        result["diffusion"]["checkpoint_path_latest"] = str(latest_checkpoint)

                    break

    return result


def _validate_predictor_configs(biopart: str, configs: Dict[str, Any]) -> None:
    """Validate predictor configurations.

    Args:
        biopart: biological part type
        configs: predictor configuration dict

    Raises:
        FileNotFoundError: none of the enabled model paths exist
    """
    errors = []
    has_valid_model = False

    # Validate the DL model
    if configs["dl_model"]["enabled"]:
        dl_dir = configs["dl_model"]["predictor_dir"]
        if dl_dir:
            dl_path = Path(dl_dir)
            if dl_path.exists():
                fold_dirs = list(dl_path.glob("fold_*"))
                if fold_dirs:
                    has_valid_model = True
                else:
                    errors.append(f"No fold_* subdirectory found in DL model directory: {dl_dir}")
            else:
                errors.append(f"DL model directory does not exist: {dl_dir}")
        else:
            errors.append("DL model path not configured")

    # Validate the Evo model
    if configs["evo_model"]["enabled"]:
        if configs["evo_model"]["single_model"]:
            # Single-model validation
            evo_checkpoint = configs["evo_model"]["checkpoint_path"]
            if evo_checkpoint:
                evo_path = Path(evo_checkpoint)
                if evo_path.exists():
                    has_valid_model = True
                else:
                    errors.append(f"Evo model file does not exist: {evo_checkpoint}")
            else:
                errors.append("Evo single-model path not configured")
        else:
            # Cross-validation model validation
            evo_dir = configs["evo_model"]["predictor_dir"]
            if evo_dir:
                evo_path = Path(evo_dir)
                if evo_path.exists():
                    fold_dirs = list(evo_path.glob("fold_*"))
                    if fold_dirs:
                        has_valid_model = True
                    else:
                        errors.append(f"No fold_* subdirectory found in Evo model directory: {evo_dir}")
                else:
                    errors.append(f"Evo model directory does not exist: {evo_dir}")
            else:
                errors.append("Evo model path not configured")

    # Validate the diffusion model
    diffusion_path = configs["diffusion"]["checkpoint_path"]
    if diffusion_path:
        if not Path(diffusion_path).exists():
            errors.append(f"Diffusion model file does not exist: {diffusion_path}")

    # Log errors
    if errors:
        error_msg = f"{biopart} model path validation warnings:\n" + "\n".join(errors)
        logger.warning(error_msg)

    # Only raise when all enabled models are invalid
    if not has_valid_model:
        raise FileNotFoundError(
            f"{biopart}: no valid predictor found. Ensure at least one predictor (DL or Evo) is available."
        )


def list_available_bioparts(config_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict]:
    """List all available bioparts and their model paths.

    Args:
        config_dir: path to the configs directory

    Returns:
        Dict keyed by biopart name, with model path information as the value
    """
    SUPPORTED_BIOPARTS = ["promoter", "rbs", "terminator"]
    result = {}

    for biopart in SUPPORTED_BIOPARTS:
        try:
            paths = get_model_paths(biopart, config_dir=config_dir, auto_discover=True)
            result[biopart] = {
                "available": True,
                "predictor_dir": paths["predictor_dir"],
                "diffusion_checkpoint": paths["diffusion_checkpoint"],
            }
        except (FileNotFoundError, ValueError) as e:
            result[biopart] = {
                "available": False,
                "error": str(e),
            }

    return result


# Module-level logger
import logging
logger = logging.getLogger(__name__)
