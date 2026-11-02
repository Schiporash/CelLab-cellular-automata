"""CelLab: cellular automata described in (restricted) English."""

from .errors import CelLabError, DescriptionError, Problem, ValidationError
from .parser import parse_description, wolfram_table
from .validation import satisfying_counts, validate, validate_or_raise


def load_description(text: str) -> dict:
    """Parse + validate. Returns the model dict or raises a CelLabError
    whose .problems lists everything that is wrong."""
    return validate_or_raise(parse_description(text))


__all__ = [
    "CelLabError", "DescriptionError", "Problem", "ValidationError",
    "load_description", "parse_description", "satisfying_counts",
    "validate", "validate_or_raise", "wolfram_table",
]
