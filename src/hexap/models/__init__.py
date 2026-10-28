"""Stage-2 networks and their checkpoint format.

A checkpoint is a dict with keys `model` (the class name), `config` (a plain dict
with the network config under `model` and the data config under `data`) and
`state_dict`.
"""

from typing import Any, Dict, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from baselines.social_transmotion import SocialTransmotion
from hexap.models.locoval_transformer import LocoValTransformer

MODELS = {
    "LocoValTransformer": LocoValTransformer,
    "SocialTransmotion": SocialTransmotion,
}


def build_model(name: str, config: DictConfig, device: str) -> torch.nn.Module:
    return MODELS[name](config).float().to(device)


def load_checkpoint(path: str, device: str) -> Tuple[torch.nn.Module, DictConfig]:
    """Builds the network stored in a checkpoint; returns it in eval mode with its full config."""
    ckpt: Dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    config = OmegaConf.create(ckpt["config"])
    model = build_model(ckpt["model"], config.model, device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), config


def save_checkpoint(
    path: str, model: torch.nn.Module, config: DictConfig, **extra: Any
) -> None:
    torch.save(
        {
            "model": type(model).__name__,
            "config": OmegaConf.to_container(config, resolve=True),
            "state_dict": model.state_dict(),
            **extra,
        },
        path,
    )
