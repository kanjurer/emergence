import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emergence.models.transformer.model import TransformerModel
from emergence.training.optimizer import create_optimizer


class Trainer:
    def __init__(
        self,
        model: TransformerModel,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        learning_rate: float = 3e-4,
        device: torch.device | None = None,
        pad_token_id: int | None = None,
    ):
        self.model = model
        self.train_loader = train_loader

        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.device = device
        self.model.to(self.device)

        self.optimizer = create_optimizer(
            self.model,
            learning_rate=learning_rate,
        )

        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=pad_token_id if pad_token_id is not None else -100
        )

    def train_epoch(self) -> float:
        self.model.train()

        total_loss = 0.0

        for x, y in self.train_loader:  # comes from datasets.py __getitem__
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            logits = self.model(x)

            # Calculate prediction error
            loss = self.loss_function(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            # Clear gradients from previous iteration
            self.optimizer.zero_grad()

            # Calculate .grad
            loss.backward()

            # Update model weights based on .grad
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def train(self, epochs: int) -> list[float]:
        history: list[float] = []

        for epoch in range(epochs):
            loss = self.train_epoch()
            history.append(loss)

            print(f"Epoch {epoch + 1}/{epochs} | " f"Loss: {loss:.4f}")

        return history
