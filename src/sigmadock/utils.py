import inspect
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import wandb

from sigmadock.diff.denoiser import SigmaDockDenoiser
from sigmadock.net.model import EquiformerV2
from sigmadock.trainer import SigmaLightningModule


def get_git_commit_hash() -> str | None:
    """
    Gets the git commit hash, prioritizing a pre-generated file
    and falling back to the git command.
    """
    commit_hash_file = Path("git_commit_hash.txt")

    # Try to read from the pre-generated file first. This will work in the sbatch job if we run:
    # GIT_HASH=$(git rev-parse --short HEAD)
    # echo $GIT_HASH > git_commit_hash.txt

    if commit_hash_file.exists():
        return commit_hash_file.read_text().strip()

    # Fall back to the git command if the file isn't found. This allows the function to still work on local machin
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WARN] Could not determine git commit hash.")
        return None


def download_wandb_artifact(
    artifact_name: str,
    version: str,
    entity: str = "sigma-dock",
    project: str = "SigmaDock",
    local_path: str | None = None,
) -> str:
    """
    Download a W&B artifact to a local path.

    Parameters
    ----------
    entity : str
        The W&B entity (user or team name).
    project : str
        The W&B project name.
    artifact_name : str
        The name of the artifact to download.
    version : str
        The version string of the artifact (e.g., 'v0', 'latest', or an explicit hash).
    local_path : str, optional
        The local path where the artifact will be downloaded. If None, a temporary W&B-managed path is used.

    Returns
    -------
    str
        The local path where the artifact was downloaded.
    """
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}:{version}", type="model")

    # If local_path is not provided, download to default managed directory
    artifact_dir = artifact.download(root=local_path)
    return artifact_dir


def get_config_from_checkpoint(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Extracts the configuration from the state_dict.
    """
    return {
        "equiformer": state_dict["hyper_parameters"].get("equiformer_config", None),
        "denoiser": state_dict["hyper_parameters"].get("denoiser_config", None),
        "all": state_dict["hyper_parameters"],
    }


def _strip_prefix_from_state_dict(state_dict: dict[str, Any], prefix: str = "model.") -> dict[str, Any]:
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def build_equiformer_from_config(cfg: dict[str, Any]) -> EquiformerV2:
    if "chemistry_edge_feature_dims" in cfg:
        assert "edge_feature_dims" not in cfg, "Edge feature dimensions already specified in the config."
        cfg["edge_feature_dims"] = cfg["chemistry_edge_feature_dims"]
        cfg.pop("chemistry_edge_feature_dims")
    else:
        assert "edge_feature_dims" in cfg, "Edge feature dimensions must be specified in the config."
    return EquiformerV2(**cfg)


def load_from_checkpoint(checkpoint: dict[str, Any], load_ema: bool = True) -> SigmaDockDenoiser:
    configs: dict[str, Any] = get_config_from_checkpoint(checkpoint)
    model = build_equiformer_from_config(configs["equiformer"])
    state_name = "ema_state_dict" if load_ema else "state_dict"
    state_dict = _strip_prefix_from_state_dict(checkpoint[state_name])
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters.")

    # Denoiser
    denoiser = SigmaDockDenoiser(model=model, **configs["denoiser"])
    denoiser.eval()
    print("Set model to evaluation mode.")
    return denoiser


def filter_cfg_for_cls(
    cls: type[Any],
    cfg: dict[str, Any],
    *,
    ignore: list[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    ignore = set(ignore or [])
    aliases = aliases or {}

    filtered: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name == "self" or param.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            continue
        if name in ignore:
            continue

        # Use alias if available
        source_key = aliases.get(name, name)
        if source_key in cfg:
            filtered[name] = cfg[source_key]

    return filtered


def load_from_scratch(  # noqa: C901
    ckpt: Path | str,
    *,
    denoiser_cfg: dict | None = None,
    equiformer_cfg: dict | None = None,
    enforced_cfg: dict | None = None,
    default_cfg: Path | None = None,
    load_ema: bool = True,
    strict: bool = True,
) -> SigmaLightningModule:
    """
    Load from checkpoint: build EquiformerV2 and SigmaDockDenoiser from
    hyper_parameters, load Lightning module; optionally load EMA into model.ema_model.
    """
    assert os.path.isfile(ckpt), f"Checkpoint {ckpt} does not exist or is not a file."

    # 1) load raw hparams
    try:
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint with torch.load: {e}. Attempting load with legacy module alias.")
        import importlib
        import sys
        sig = importlib.import_module("sigmadock")
        sys.modules["alphadock"] = sig
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)

    # 2) Equiformer config (from checkpoint)
    configs: dict[str, Any] = get_config_from_checkpoint(checkpoint.copy())
    if configs["equiformer"] is None:
        # Default EquiformerV2 config
        equiformer_cfg: dict = filter_cfg_for_cls(EquiformerV2, configs["all"])
    else:
        if equiformer_cfg is None:
            # Config from checkpoint
            equiformer_cfg: dict = configs["equiformer"]

    if configs["denoiser"] is None:
        denoiser_cfg: dict = filter_cfg_for_cls(SigmaDockDenoiser, configs["all"], ignore=["model"])
    else:
        if denoiser_cfg is None:
            # Denoiser config from checkpoint
            denoiser_cfg: dict = configs["denoiser"]

    # For nonetypes, we can use the default config
    for cfg in (equiformer_cfg, denoiser_cfg):
        for key, value in list(cfg.items()):
            if value is None:
                if default_cfg is not None and default_cfg.get(key, None) is not None:
                    print(f"Substituting {key} with default config.")
                    cfg[key] = default_cfg.get(key, None)
                else:
                    # Pop keys that are None
                    cfg.pop(key, None)

    # 3) enforce config
    if enforced_cfg is not None:
        for key, value in enforced_cfg.items():
            equiformer_cfg[key] = value
            denoiser_cfg[key] = value

    # 6) instantiate submodules
    equi = build_equiformer_from_config(equiformer_cfg)
    denoiser = SigmaDockDenoiser(model=equi, **denoiser_cfg)

    # 7) load LightningModule once, supplying denoiser
    model = SigmaLightningModule.load_from_checkpoint(str(ckpt), denoiser=denoiser, strict=strict, map_location="cpu")

    ema_model = deepcopy(model)
    if load_ema:
        if "ema_state_dict" in checkpoint:
            try:
                ema_model.model.load_state_dict(checkpoint["ema_state_dict"], strict=strict)
                print("Successfully loaded EMA model.")
            except Exception as e:
                ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=strict)
                print("Successfully loaded EMA model.")
        else:
            print("No EMA state_dict found in checkpoint. Skipping EMA model loading.")
    # Assign the EMA model to the model if load_ema is True
    model.ema_model = ema_model if load_ema else None
    return model
