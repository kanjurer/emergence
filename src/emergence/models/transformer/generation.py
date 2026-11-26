import torch

from emergence.models.transformer.model import TransformerModel


@torch.no_grad()
def generate(
    model: TransformerModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    model.eval()

    for _ in range(max_new_tokens):
        # Keep only the most recent context_length tokens
        input_ids = input_ids[
            :, -model.token_embeddings.position_embedding.num_embeddings :
        ]

        # Get predictions
        logits = model(input_ids)

        # Use predictions from the final position
        logits = logits[:, -1, :]

        # Apply temperature
        logits = logits / temperature

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Sample the next token
        next_token = torch.multinomial(probabilities, num_samples=1)

        # Append it to the sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids
