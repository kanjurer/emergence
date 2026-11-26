from pathlib import Path

from torch.utils.data import DataLoader

from emergence.data.tokenizer import CharacterTokenizer
from emergence.data.datasets import NextTokenDataset
from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.model import TransformerModel
from emergence.training.trainer import Trainer
from emergence.training.checkpointing import save_checkpoint


def main():
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "shakespeare.txt"
    checkpoint_dir = project_root / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load text
    text = dataset_path.read_text(encoding="utf-8")

    # Tokenize
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)

    token_ids = tokenizer.encode(text)

    print("Dataset characters:", len(text))
    print("Vocabulary size:", tokenizer.vocab_size)

    # Model configuration
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=256,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    )

    # Dataset and DataLoader
    dataset = NextTokenDataset(
        token_ids=token_ids,
        context_length=config.context_length,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    # Initialize model and trainer
    model = TransformerModel(config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=3e-4,
    )

    # Train
    epochs = 15
    history: list[float] = []
    best_loss = float("inf")

    for epoch in range(epochs):
        loss = trainer.train(epochs=1)[-1]
        history.append(loss)

        # Save latest checkpoint
        save_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            epoch=epoch + 1,
            loss=loss,
            path=str(checkpoint_dir / "latest.pt"),
        )

        # Save best checkpoint
        if loss < best_loss:
            best_loss = loss

            save_checkpoint(
                model=trainer.model,
                optimizer=trainer.optimizer,
                epoch=epoch + 1,
                loss=loss,
                path=str(checkpoint_dir / "best.pt"),
            )

        print(f"Epoch [{epoch + 1}/{epochs}] " f"Loss: {loss:.4f}")

    print("\nTraining complete!")
    print("Final loss:", history[-1])
    print("Best loss:", best_loss)
    print("Latest checkpoint:", checkpoint_dir / "latest.pt")
    print("Best checkpoint:", checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
