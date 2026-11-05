"""Saving and loading models and simulations.

A model is saved as JSON (the same format the parser produces).
A simulation is saved as two files next to each other:

  name.json  the model, the current generation, which generations were kept
             and the state of the random generator (so a resumed simulation
             continues exactly as if it had never stopped);
  name.npz   the kept generations, as an array of shape (k, rows, cols) with
             the state index of every cell.

keep="all" stores every generation from the start, keep="last" only the
current one. Either way, load_simulation() returns a Simulation that can go on
with sim.run(...).
"""

import json
from pathlib import Path

import numpy as np

from .errors import Problem, StorageError
from .simulation import Simulation, state_chars
from .validation import validate

FORMAT_VERSION = 1
KEEP = ("all", "last")


# --- model -------------------------------------------------------------------

def save_model(model: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return path


def load_model(path) -> dict:
    """Read a model from JSON and validate it (raises a CelLabError)."""
    data = _read_json(Path(path))
    return _checked_model(data)


# --- starting grid -----------------------------------------------------------

def load_grid(path, model: dict) -> np.ndarray:
    """Read a starting grid from a text file, one line per row and one character
    per cell. It can be what the program prints (render) and, when the first
    letters of the states are different, those letters. Blank lines and spaces
    around a line are ignored. Raises a StorageError listing all the problems."""
    path = Path(path)
    symbols = dict(zip(state_chars(model["states"]), range(len(model["states"]))))
    letters = [s[0] for s in model["states"]]
    if len(set(letters)) == len(letters):
        symbols.update(zip(letters, range(len(letters))))

    lines = [(number, line.strip()) for number, line in enumerate(_read_text(path).splitlines(), 1) if line.strip()]
    problems = []
    if len(lines) != model["rows"]:
        problems.append(f"{path} has {len(lines)} lines of cells, the model has {model['rows']} rows")
    # only the first line of each kind of mistake, a wrong file would give hundreds
    for number, line in lines:
        if len(line) != model["cols"]:
            problems.append(f"{path}, line {number}: {len(line)} cells, the model has {model['cols']} columns")
            break
    for number, line in lines:
        if unknown := [c for c in dict.fromkeys(line) if c not in symbols]:
            listed = ", ".join(f"'{c}'" for c in unknown)
            problems.append(f"{path}, line {number}: unknown symbols {listed} (it can be {', '.join(symbols)})")
            break
    if problems:
        raise StorageError([Problem(p) for p in problems])
    return np.array([[symbols[c] for c in line] for _, line in lines], dtype=np.uint8)


# --- simulation --------------------------------------------------------------

def save_simulation(sim: Simulation, path, keep: str = "all") -> tuple[Path, Path]:
    """Save sim as <path>.json + <path>.npz; returns the two paths."""
    if keep not in KEEP:
        raise ValueError(f"keep must be one of {KEEP}, not {keep!r}")
    if keep == "all" and not sim.keep_history:
        raise ValueError("this simulation does not keep its history, save it with keep='last'")
    json_path, npz_path = _paths(path)

    if keep == "all":
        grids, first = np.stack(sim.history), sim.first_generation
    else:
        grids, first = sim.grid[np.newaxis], sim.generation
    np.savez_compressed(npz_path, grids=grids.astype(np.uint8))

    meta = {
        "cellab": FORMAT_VERSION,
        "model": sim.model,
        "generation": sim.generation,
        "keep": keep,
        "first_generation": first,
        "generations_file": npz_path.name,
        "rng": sim.rng.bit_generator.state,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return json_path, npz_path


def load_simulation(path) -> Simulation:
    """Load a simulation saved with save_simulation(); path can be the .json,
    the .npz or the name without extension. Raises a CelLabError if the files
    are missing, the model is invalid or the generations do not match it."""
    json_path, _ = _paths(path)
    meta = _read_json(json_path)
    if not isinstance(meta, dict) or "model" not in meta:
        raise StorageError([Problem(f"{json_path} is not a saved simulation (no 'model')")])
    model = _checked_model(meta["model"])

    npz_path = json_path.with_name(meta.get("generations_file", json_path.with_suffix(".npz").name))
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            grids = data["grids"]
    except (OSError, KeyError, ValueError) as err:
        raise StorageError([Problem(f"cannot read the generations from {npz_path}: {err}")]) from err

    generation, first = meta.get("generation"), meta.get("first_generation")
    problems = []
    if grids.ndim != 3 or grids.shape[1:] != (model["rows"], model["cols"]):
        problems.append(f"the saved grids have shape {grids.shape[1:]}, the model is {model['rows']}x{model['cols']}")
    elif grids.size and grids.max() >= len(model["states"]):
        problems.append("the saved grids contain states that the model does not declare")
    if not isinstance(generation, int) or not isinstance(first, int) or first + len(grids) - 1 != generation:
        problems.append(f"{len(grids)} saved generations do not match generations {first}..{generation}")
    if problems:
        raise StorageError([Problem(p) for p in problems])

    rng = np.random.default_rng()
    if "rng" in meta:
        try:
            rng.bit_generator.state = meta["rng"]
        except (TypeError, ValueError, KeyError) as err:
            raise StorageError([Problem(f"the saved random generator state is not valid: {err}")]) from err

    sim = Simulation(model, grid=grids[-1], generation=generation, rng=rng)
    sim.history = list(grids)
    sim.first_generation = first
    return sim


# --- helpers -----------------------------------------------------------------

def _paths(path) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix in (".json", ".npz"):
        path = path.with_suffix("")
    return path.with_name(path.name + ".json"), path.with_name(path.name + ".npz")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")  # Notepad may add a BOM
    except OSError as err:
        raise StorageError([Problem(f"cannot read {path}: {err.strerror or err}")]) from err
    except UnicodeDecodeError as err:
        raise StorageError([Problem(f"cannot read {path}: it is not UTF-8 text")]) from err


def _read_json(path: Path):
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        raise StorageError([Problem(f"{path} is not valid JSON (line {err.lineno}: {err.msg})")]) from err


def _checked_model(model) -> dict:
    if not isinstance(model, dict):
        raise StorageError([Problem("the saved model must be a JSON object")])
    problems = validate(model)
    if problems:
        raise StorageError(problems)
    return model
