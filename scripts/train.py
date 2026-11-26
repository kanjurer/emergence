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
    dataset_path = project_root / "data" / "my_dataset.txt"
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
        context_length=128,
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
        batch_size=16,
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
    epochs = 10
    history = trainer.train(epochs=epochs)

    # Save checkpoint
    save_checkpoint(
        model=trainer.model,
        optimizer=trainer.optimizer,
        epoch=epochs,
        loss=history[-1],
        path=str(checkpoint_dir / "final.pt"),
    )

    print("\nTraining complete!")
    print("Final loss:", history[-1])
    print("Checkpoint saved:", checkpoint_dir / "final.pt")


if __name__ == "__main__":
    main()
