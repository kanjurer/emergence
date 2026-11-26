import torch
from torch.optim import Optimizer

from emergence.models.transformer.model import TransformerModel


def save_checkpoint(
    model: TransformerModel,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    checkpoint: dict[str, object] = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model: TransformerModel,
    optimizer: Optimizer,
    path: str,
    device: torch.device | None = None,
) -> dict[str, object]:
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
    }
