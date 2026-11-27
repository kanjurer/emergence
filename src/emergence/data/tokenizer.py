class CharacterTokenizer:
    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def fit(self, text: str) -> None:
        characters = sorted(set(text))

        self.char_to_id = {char: index for index, char in enumerate(characters)}

        self.id_to_char = {index: char for char, index in self.char_to_id.items()}

    def encode(self, text: str) -> list[int]:
        if not self.char_to_id:
            raise ValueError("Tokenizer has not been fitted yet.")

        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as error:
            raise ValueError(
                f"Unknown character: {error.args[0]!r}. "
                "The character was not present during fitting."
            ) from error

    def decode(self, token_ids: list[int]) -> str:
        if not self.id_to_char:
            raise ValueError("Tokenizer has not been fitted yet.")

        try:
            return "".join(self.id_to_char[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"Unknown token ID: {error.args[0]}") from error

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)


class SortingTokenizer:
    def __init__(self, min_value: int = 0, max_value: int = 99) -> None:
        if min_value > max_value:
            raise ValueError("min_value cannot exceed max_value.")

        self.min_value = min_value
        self.max_value = max_value

        self.special_tokens = ["|", "<EOS>", "<PAD>"]

        self.vocab = {
            str(value): value - min_value for value in range(min_value, max_value + 1)
        }

        offset = len(self.vocab)

        for index, token in enumerate(self.special_tokens):
            self.vocab[token] = offset + index

        self.inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[list[int]]:
        encoded_examples: list[list[int]] = []

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue

            tokens = line.split()

            if tokens.count("|") != 1:
                raise ValueError(f"Line {line_number} must contain exactly one '|'.")

            if "<EOS>" in tokens:
                raise ValueError(f"Line {line_number} contains a reserved token.")

            example: list[int] = []

            for token in tokens:
                if token not in self.vocab:
                    raise ValueError(f"Unknown token {token!r} on line {line_number}.")

                example.append(self.vocab[token])

            example.append(self.vocab["<EOS>"])
            encoded_examples.append(example)

        return encoded_examples

    def decode(self, token_ids: list[int]) -> str:
        tokens: list[str] = []

        for token_id in token_ids:
            if token_id not in self.inverse_vocab:
                raise ValueError(f"Unknown token ID: {token_id}")

            tokens.append(self.inverse_vocab[token_id])

        lines: list[str] = []
        current_line: list[str] = []

        for token in tokens:
            if token == "<EOS>":
                lines.append(" ".join(current_line))
                current_line = []
            else:
                current_line.append(token)

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def encode_prompt(self, text: str) -> list[int]:
        tokens = text.split()
        encoded: list[int] = []

        for token in tokens:
            if token not in self.vocab:
                raise ValueError(f"Unknown token: {token!r}")

            if token in ("<EOS>", "<PAD>"):
                raise ValueError(f"Reserved token in prompt: {token!r}")

            encoded.append(self.vocab[token])

        return encoded
