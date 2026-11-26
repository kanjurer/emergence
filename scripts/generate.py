from pathlib import Path

import torch

from emergence.data.tokenizer import CharacterTokenizer
from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.model import TransformerModel
from emergence.models.transformer.generation import generate
from emergence.training.checkpointing import load_checkpoint


def main():
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = project_root / "data" / "shakespeare.txt"
    checkpoint_path = project_root / "checkpoints" / "best.pt"

    # Load training text and reconstruct tokenizer
    text = dataset_path.read_text(encoding="utf-8")

    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)

    # Reconstruct model configuration
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=256,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    )

    # Initialize model
    model = TransformerModel(config)

    # Initialize optimizer required by checkpoint loader
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Select device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)

    # Load trained checkpoint
    load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=str(checkpoint_path),
        device=device,
    )

    print("\n=== Transformer Text Generation ===")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("Prompt: ")

        if prompt.strip().lower() == "exit":
            break

        if not prompt:
            continue

        max_new_tokens = 128
        temperature = 0.8

        # Encode prompt
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device=device,
        )

        # Generate
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Decode only newly generated tokens
        generated_ids: list[int] = output_ids[0].tolist()  # type: ignore (annoying)
        generated_text = tokenizer.decode(generated_ids)

        print("\n=== Generated Text ===\n")
        print(generated_text)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
