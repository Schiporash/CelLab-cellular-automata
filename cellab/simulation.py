"""Running a validated model: the initial state and the step from one
generation to the next.

A grid is a 2D numpy array of state indices (the position of the state in
model["states"]), so grid == 0 marks the cells in the first declared state.

  * neighbours are counted for the whole grid at once with convolve2d
    (8 or 4 neighbours; with a fixed border the cells outside the grid count
    as no neighbour, with "wrap" the grid continues on the other side);
  * a cell for which no rule applies keeps its state;
  * rules with a probability use one random number per cell and generation,
    drawn from a generator seeded with model["seed"], so the same model and
    seed always give the same simulation;
  * elementary automata (Rule 30 etc.) use their 8-case table; outside a
    fixed border the cells are in state 0.
"""

import numpy as np
from scipy.signal import convolve2d

DEFAULT_GENERATIONS = 100

KERNELS = {
    8: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16),
    4: np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int16),
}

_COMPARE = {
    ">=": np.greater_equal, "<=": np.less_equal, "==": np.equal,
    ">": np.greater, "<": np.less,
}


# --- initial state -----------------------------------------------------------

def initial_grid(model: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """The generation 0 grid.

    With "initial" percentages, that many cells (rounded, at least one when
    the percentage is not 0) get each state, placed at random. The remaining
    cells get the first declared state that has no percentage.
    Without "initial", every cell is in the first state and the cell in the
    middle is in the second one (one fire in the forest, one ON cell for
    Rule 30), as in Deliverable 1.
    """
    rows, cols = model["rows"], model["cols"]
    states = model["states"]
    initial = model.get("initial")

    if not initial:
        grid = np.zeros((rows, cols), dtype=np.uint8)
        grid[rows // 2, cols // 2] = 1
        return grid

    if rng is None:
        rng = np.random.default_rng(model.get("seed"))
    total = rows * cols
    rest = next((i for i, s in enumerate(states) if s not in initial), 0)
    cells = np.full(total, rest, dtype=np.uint8)
    order = rng.permutation(total)
    start = 0
    for state, percent in initial.items():
        count = round(total * percent / 100)
        if percent > 0:
            count = max(count, 1)
        count = min(count, total - start)
        cells[order[start:start + count]] = states.index(state)
        start += count
    return cells.reshape(rows, cols)


# --- one generation ----------------------------------------------------------

def neighbour_counts(grid: np.ndarray, state: int, neighbours: int = 8, border: str = "fixed") -> np.ndarray:
    """For every cell, how many of its neighbours are in the given state."""
    boundary = "wrap" if border == "wrap" else "fill"
    mask = (grid == state).astype(np.int16)
    return convolve2d(mask, KERNELS[neighbours], mode="same", boundary=boundary, fillvalue=0)


def _condition_mask(cond: list, counts: np.ndarray) -> np.ndarray:
    _, op, value = cond
    if op == "in":
        return np.isin(counts, value)
    return _COMPARE[op](counts, value)


def step(grid: np.ndarray, model: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the next generation; the given grid is not changed.

    rng is only needed for models with probabilistic rules."""
    if "table" in model:
        return _elementary_step(grid, model)

    states = model["states"]
    neighbours = model.get("neighbours", 8)
    border = model.get("border", "fixed")
    rules = model.get("rules", [])

    counts = {}
    for rule in rules:
        if "if" in rule and rule["if"][0] not in counts:
            counts[rule["if"][0]] = neighbour_counts(grid, states.index(rule["if"][0]), neighbours, border)

    random = None
    if any("prob" in rule for rule in rules):
        if rng is None:
            raise ValueError("this model has rules with a probability, pass a random generator")
        random = rng.random(grid.shape)
    # for each cell, how much of [0, 1) the probabilistic rules that hold for
    # it have used already, so each rule gets its own slice of the same draw
    used = np.zeros(grid.shape) if random is not None else None

    new = grid.copy()
    for rule in rules:
        applies = grid == states.index(rule["state"])
        if "if" in rule:
            applies &= _condition_mask(rule["if"], counts[rule["if"][0]])
        if "prob" in rule:
            holds = applies
            applies = holds & (random >= used) & (random < used + rule["prob"])
            used[holds] += rule["prob"]
        new[applies] = states.index(rule["next"])
    return new


def _elementary_step(grid: np.ndarray, model: dict) -> np.ndarray:
    lookup = np.zeros(8, dtype=grid.dtype)
    for pattern, result in model["table"].items():
        lookup[int(pattern, 2)] = result
    cells = grid.astype(np.int16)
    if model.get("border", "fixed") == "wrap":
        left, right = np.roll(cells, 1, axis=1), np.roll(cells, -1, axis=1)
    else:
        padded = np.pad(cells, ((0, 0), (1, 1)))
        left, right = padded[:, :-2], padded[:, 2:]
    return lookup[left * 4 + cells * 2 + right]


# --- a whole simulation ------------------------------------------------------

class Simulation:
    """A model together with its current generation.

    sim = Simulation(model)       # generation 0
    sim.run(50)                   # 50 more generations
    sim.grid, sim.generation      # where it is now
    sim.history                   # every generation so far (if keep_history)
    """

    def __init__(self, model: dict, grid: np.ndarray | None = None, generation: int = 0,
                 rng: np.random.Generator | None = None, keep_history: bool = True):
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng(model.get("seed"))
        self.grid = initial_grid(model, self.rng) if grid is None else np.asarray(grid, dtype=np.uint8)
        self.generation = generation
        self.keep_history = keep_history
        # history[i] is generation first_generation + i
        self.first_generation = generation
        self.history = [self.grid] if keep_history else []

    def step(self) -> np.ndarray:
        self.grid = step(self.grid, self.model, self.rng)
        self.generation += 1
        if self.keep_history:
            self.history.append(self.grid)
        return self.grid

    def run(self, generations: int | None = None) -> np.ndarray:
        """Run that many more generations (default: the model's
        "generations", or DEFAULT_GENERATIONS)."""
        if generations is None:
            generations = self.model.get("generations", DEFAULT_GENERATIONS)
        for _ in range(generations):
            self.step()
        return self.grid

    def counts(self) -> dict[str, int]:
        """How many cells are in each state now."""
        found = np.bincount(self.grid.ravel(), minlength=len(self.model["states"]))
        return {state: int(n) for state, n in zip(self.model["states"], found)}

    def render(self) -> str:
        return render(self.grid, self.model["states"])


def render(grid: np.ndarray, states: list[str]) -> str:
    """The grid as text, one character per cell: '.'/'#' for two states,
    otherwise the first letter of each state (or its index if those clash)."""
    if len(states) == 2:
        chars = ".#"
    elif len({s[0] for s in states}) == len(states):
        chars = [s[0] for s in states]
    else:
        chars = [str(i) if i < 10 else chr(ord("a") + i - 10) for i in range(len(states))]
    return "\n".join("".join(chars[v] for v in row) for row in grid)
