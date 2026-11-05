"""CelLab: cellular automata described in (restricted) English."""

from .errors import CelLabError, DescriptionError, Problem, StorageError, ValidationError
from .parser import parse_description, wolfram_table
from .simulation import Simulation, initial_grid, neighbour_counts, render, step
from .storage import load_grid, load_model, load_simulation, save_model, save_simulation
from .validation import satisfying_counts, validate, validate_or_raise


def load_description(text: str) -> dict:
    """Parse + validate. Returns the model dict or raises a CelLabError
    whose .problems lists everything that is wrong."""
    return validate_or_raise(parse_description(text))


__all__ = [
    "CelLabError", "DescriptionError", "Problem", "StorageError", "ValidationError",
    "Simulation", "initial_grid", "load_description", "load_grid", "load_model", "load_simulation",
    "neighbour_counts", "parse_description", "render", "satisfying_counts",
    "save_model", "save_simulation", "step", "validate", "validate_or_raise",
    "wolfram_table",
]
