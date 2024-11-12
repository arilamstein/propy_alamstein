from collections import defaultdict, Counter
import random
from typing import DefaultDict, Counter as TypingCounter


class SmallLanguageModel:
    def __init__(self) -> None:
        self.char_map: DefaultDict[str, TypingCounter[str, int]] = defaultdict(Counter)

    def train(self, text: str) -> None:
        if len(text) < 2:
            raise Exception("Text is too short")

        for i in range(len(text) - 1):
            current_char = text[i]
            next_char = text[i + 1]
            self.char_map[current_char][next_char] += 1

    def predict_next(self, current_char: str) -> str | None:
        if current_char not in self.char_map:
            return None
        next_chars, weights = zip(*self.char_map[current_char].items())
        if next_chars:
            chosen_char: str | None = random.choices(next_chars, weights=weights)[0]
        else:
            chosen_char = None
        return chosen_char


# Example usage
if __name__ == "__main__":
    slm = SmallLanguageModel()
    slm.train("hello")
    print(slm.char_map)
    print(slm.predict_next("l"))
    print(slm.predict_next("l"))
