"""Stage 2: fine-tune on a real dataset.

    python -m hexap.train dataset=jrdb model=hexap pretrained=weights/stage1/hexap_pretrain.ckpt

The best checkpoint on the validation split (best-of-M ADE) is written to
`<out_dir>/best.pt`, the last one to `<out_dir>/last.pt`.
"""

import logging
import os
import random
from typing import Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from hexap.data.dataset import TrajPoseDataset, batch_process_coords, collate_batch
from hexap.metrics import min_ade_loss
from hexap.models import build_model, save_checkpoint
from baselines.emloco_mlp import EmLocoMLP

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)


def step_decay(
    optimizer: torch.optim.Optimizer, lr: float, epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Multiplies the learning rate by 0.1 after four fifths of the epochs."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 0.1 ** (epoch // (epochs * 4.0 / 5.0))
    )


def load_pretrained(
    model: torch.nn.Module, path: str, peft_type: str, st_type: str
) -> None:
    """Loads the Stage-1 weights and freezes the layers that the fine-tuning scheme keeps fixed.

    The Physical Core is frozen except under `fft` (and trained from scratch under
    `none`); the LocoVal-Head is always frozen; the Path-Head is trained (and loaded
    together with the Physical Core or when the social module is disabled);
    the adapters and the social module are trained.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)["locoval"]
    load_core = peft_type in ("adapter", "fix_local", "fft")
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
        elif "predict_head" in name:
            if load_core or st_type == "none":
                param.data = state[name].to(param.device)
            param.requires_grad = True
        elif "reward" in name:
            if load_core:
                param.data = state[name].to(param.device)
            param.requires_grad = False
        elif "cross_modality_transformer" in name:
            if load_core:
                param.data = state[name].to(param.device)
            param.requires_grad = peft_type in ("none", "fft")
        else:
            param.requires_grad = st_type != "none"


def compute_loss(
    model: torch.nn.Module,
    config: DictConfig,
    batch,
    device: str,
    training: bool,
    emloco: Optional[EmLocoMLP],
) -> Dict[str, torch.Tensor]:
    tokens, masks, padding_mask = batch
    in_tokens, _, out_tokens, _, padding_mask = batch_process_coords(
        tokens, masks, padding_mask, config, training=training, device=device
    )
    in_F = in_tokens.shape[1]
    pred = model(
        in_tokens, padding_mask, nmode=config.model.nmode, random_masking=training
    )
    out = {"mse": min_ade_loss(pred[:, in_F:], out_tokens)}
    if emloco is not None:
        # EmLoco baseline: push the frozen EmLoco plausibility of the predictions towards 1.
        score = emloco(in_tokens, pred[:, in_F:])
        out["value"] = (
            torch.nn.functional.mse_loss(score, torch.ones_like(score)) * 1000
        )
    return out


def train(config: DictConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.out_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.out_dir, "config.yaml"))
    if config.wandb.enabled:
        import wandb

        wandb.init(
            project=config.wandb.project,
            name=config.exp_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    def loader(split: str, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TrajPoseDataset(
            config.data_root, config.data.dataset, split, max_windows=config.max_windows
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.train.num_workers,
            collate_fn=collate_batch,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=config.train.num_workers > 0,
        )

    train_loader = loader("train", config.train.batch_size, shuffle=True)
    val_loader = loader("val", config.train.batch_size * 3, shuffle=False)

    model = build_model("LocoValTransformer", config.model, device)
    emloco = None
    if config.train.valueloss_w > 0:
        emloco = EmLocoMLP(config.emloco).to(device)
        emloco.load_state_dict(
            torch.load(
                config.emloco_checkpoint, map_location=device, weights_only=False
            )["locoval"]
        )
        emloco.requires_grad_(False)

    optimizer_cls = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}[
        config.train.optimizer.type
    ]
    kwargs = (
        {"weight_decay": config.train.optimizer.weight_decay}
        if config.train.optimizer.type == "AdamW"
        else {}
    )
    optimizer = optimizer_cls(
        model.parameters(), lr=config.train.optimizer.lr, **kwargs
    )
    if config.pretrained is not None:
        load_pretrained(
            model, config.pretrained, config.model.peft_type, config.model.st_type
        )
    scheduler = step_decay(optimizer, config.train.optimizer.lr, config.train.epochs)
    log.info(
        "parameters: %d total, %d trainable",
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total, n = 0.0, 0
        for batch in val_loader:
            loss = compute_loss(
                model, config, batch, device, training=False, emloco=emloco
            )["mse"]
            total += loss.item() * batch[0].shape[0]
            n += batch[0].shape[0]
        return total / n / 100

    best = validate()
    log.info("initial validation minADE %.4f", best)
    w = config.train.valueloss_w
    for epoch in range(config.train.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            losses = compute_loss(
                model, config, batch, device, training=True, emloco=emloco
            )
            loss = (
                (1 - w) * losses["mse"] + w * losses["value"]
                if emloco is not None
                else losses["mse"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.max_grad_norm
            )
            optimizer.step()
            if config.wandb.enabled and step % 50 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )
        val = validate()
        log.info("epoch %d validation minADE %.4f", epoch, val)
        if config.wandb.enabled:
            wandb.log({"val/min_ade": val, "epoch": epoch + 1})
        save_checkpoint(
            os.path.join(config.out_dir, "last.pt"), model, config, epoch=epoch
        )
        if val < best:
            best = val
            save_checkpoint(
                os.path.join(config.out_dir, "best.pt"), model, config, epoch=epoch
            )
            log.info("best checkpoint updated")
        scheduler.step()
    log.info("best validation minADE %.4f", best)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    train(config)


if __name__ == "__main__":
    main()
