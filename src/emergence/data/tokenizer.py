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
