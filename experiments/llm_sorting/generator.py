from pathlib import Path

import torch
from torch import Tensor

from experiments.llm_sorting.tokenizer import SortingTokenizer
from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.model import TransformerModel
from emergence.models.transformer.generation import generate
from emergence.training.checkpointing import load_checkpoint


class TextGenerator:
    def __init__(
        self,
        model: TransformerModel,
        tokenizer: SortingTokenizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        config: TransformerConfig,
        tokenizer: SortingTokenizer,
        device: torch.device | None = None,
    ) -> "TextGenerator":
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        model = TransformerModel(config)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
        )

        model.to(device)

        load_checkpoint(
            model=model,
            optimizer=optimizer,
            path=str(checkpoint_path),
            device=device,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        if isinstance(self.tokenizer, SortingTokenizer):
            prompt_ids = self.tokenizer.encode_prompt(prompt)
        else:
            prompt_ids = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )

        eos_token_id = (
            self.tokenizer.vocab["<EOS>"]
            if isinstance(self.tokenizer, SortingTokenizer)
            else None
        )

        output_ids: Tensor = generate(
            model=self.model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )

        generated_ids: list[int] = output_ids[0].tolist()

        return self.tokenizer.decode(generated_ids)
