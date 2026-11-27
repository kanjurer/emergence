import torch

from emergence.models.transformer.model import TransformerModel


@torch.no_grad()
def generate(
    model: TransformerModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    model.eval()

    for _ in range(max_new_tokens):
        input_ids = input_ids[
            :, -model.token_embeddings.position_embedding.num_embeddings :
        ]

        logits = model(input_ids)
        logits = logits[:, -1, :]
        logits = logits / temperature

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return input_ids
