import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        if len(token_ids) <= context_length:
            raise ValueError("Token sequence must be longer than context_length")

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length  # sliding window approach

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")

        x = self.token_ids[index : index + self.context_length]

        y = self.token_ids[index + 1 : index + self.context_length + 1]

        return x, y
