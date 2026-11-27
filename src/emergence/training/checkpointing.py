from typing import TypedDict

import torch
from torch.optim import Optimizer

from emergence.models.transformer.model import TransformerModel


class CheckpointMetadata(TypedDict):
    epoch: int
    loss: float
    best_loss: float


def save_checkpoint(
    model: TransformerModel,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    path: str,
    best_loss: float | None = None,
) -> None:
    checkpoint: dict[str, object] = {
        "epoch": epoch,
        "loss": loss,
        "best_loss": loss if best_loss is None else best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model: TransformerModel,
    optimizer: Optimizer,
    path: str,
    device: torch.device | None = None,
) -> CheckpointMetadata:
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    loss = float(checkpoint["loss"])

    return {
        "epoch": int(checkpoint["epoch"]),
        "loss": loss,
        "best_loss": float(checkpoint.get("best_loss", loss)),
    }
