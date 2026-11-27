from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from emergence.models.transformer.model import TransformerModel
from emergence.training.trainer import Trainer
from emergence.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)


@dataclass
class TrainingRun:
    model: TransformerModel
    trainer: Trainer
    history: list[float]
    best_loss: float
    completed_epochs: int


def train_model(
    model: TransformerModel,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 15,
    learning_rate: float = 3e-4,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> TrainingRun:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=learning_rate,
    )

    start_epoch = 0
    best_loss = float("inf")

    # Resume from an existing checkpoint.
    if resume_from is not None:
        metadata = load_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            path=str(resume_from),
            device=trainer.device,
        )

        start_epoch = metadata["epoch"]
        best_loss = metadata["best_loss"]

        print(f"Resuming from epoch {start_epoch} " f"(loss: {metadata['loss']:.4f})")

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[float] = []

    for epoch in range(start_epoch, epochs):
        loss = trainer.train_epoch()
        history.append(loss)

        # Determine whether this is the best checkpoint.
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)

        if checkpoint_dir is not None:
            # Save checkpoint for every epoch.
            save_checkpoint(
                model=trainer.model,
                optimizer=trainer.optimizer,
                epoch=epoch + 1,
                loss=loss,
                best_loss=best_loss,
                path=str(checkpoint_dir / f"checkpoint_{epoch + 1}.pt"),
            )

            # Save latest checkpoint.
            save_checkpoint(
                model=trainer.model,
                optimizer=trainer.optimizer,
                epoch=epoch + 1,
                loss=loss,
                best_loss=best_loss,
                path=str(checkpoint_dir / "latest.pt"),
            )

            # Save best checkpoint when loss improves.
            if is_best:
                save_checkpoint(
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    epoch=epoch + 1,
                    loss=loss,
                    best_loss=best_loss,
                    path=str(checkpoint_dir / "best.pt"),
                )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss:.4f} "
            f"Best: {best_loss:.4f}"
        )

    return TrainingRun(
        model=trainer.model,
        trainer=trainer,
        history=history,
        best_loss=best_loss,
        completed_epochs=start_epoch + len(history),
    )
