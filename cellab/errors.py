from dataclasses import dataclass


@dataclass
class Problem:
    message: str
    sentence: str | None = None  # the user's original sentence, if any
    line: int | None = None

    def __str__(self) -> str:
        if self.sentence is None:
            return self.message
        where = f"line {self.line}: " if self.line else ""
        return f'{where}"{self.sentence}" -> {self.message}'


class CelLabError(Exception):
    """Base class: carries a list of problems so the UI can show all of them."""

    def __init__(self, problems: list[Problem]):
        self.problems = problems
        super().__init__("\n".join(str(p) for p in problems))


class DescriptionError(CelLabError):
    """Some sentences of the description could not be parsed."""


class ValidationError(CelLabError):
    """The description was parsed, but the model it describes is not valid."""


class StorageError(CelLabError):
    """A saved model or simulation could not be read back."""
