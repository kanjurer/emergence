import torch
from torch.utils.data import Dataset


class SortingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        encoded_examples: list[list[int]],
        context_length: int,
        pad_token_id: int,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        if not encoded_examples:
            raise ValueError("Dataset contains no examples.")

        self.examples: list[torch.Tensor] = []

        for example in encoded_examples:
            if len(example) - 1 > context_length:
                raise ValueError("An example exceeds context_length.")

            self.examples.append(torch.tensor(example, dtype=torch.long))

        self.context_length = context_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]

        x = example[:-1]
        y = example[1:]

        padding_length = self.context_length - len(x)

        x = torch.nn.functional.pad(x, (0, padding_length), value=self.pad_token_id)
        y = torch.nn.functional.pad(y, (0, padding_length), value=self.pad_token_id)

        return x, y
