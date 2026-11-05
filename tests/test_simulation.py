import numpy as np
import pytest

from cellab import Simulation, initial_grid, load_description, neighbour_counts, step
from conftest import EXAMPLES


def example(name):
    return load_description((EXAMPLES / f"{name}.txt").read_text())


def model_of(text):
    return load_description(text)


def grid_from(text, states):
    """'.#' style picture -> grid of state indices (chars = first letters)."""
    chars = {s[0]: i for i, s in enumerate(states)}
    return np.array([[chars[c] for c in line.strip()] for line in text.strip().splitlines()], dtype=np.uint8)


# --- neighbours --------------------------------------------------------------

def test_neighbour_counts_fixed_and_wrap():
    grid = np.ones((3, 3), dtype=np.uint8)
    assert neighbour_counts(grid, 1, 8, "fixed").tolist() == [[3, 5, 3], [5, 8, 5], [3, 5, 3]]
    assert neighbour_counts(grid, 1, 4, "fixed").tolist() == [[2, 3, 2], [3, 4, 3], [2, 3, 2]]
    assert (neighbour_counts(grid, 1, 8, "wrap") == 8).all()
    assert (neighbour_counts(grid, 1, 4, "wrap") == 4).all()


def test_neighbour_counts_only_the_given_state():
    grid = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.uint8)
    assert neighbour_counts(grid, 1, 8)[1, 1] == 4
    assert neighbour_counts(grid, 2, 8)[1, 1] == 2
    assert neighbour_counts(grid, 0, 8)[1, 1] == 2   # the cell itself is not counted


# --- initial state -----------------------------------------------------------

def test_default_start_is_one_cell_in_the_middle():
    grid = initial_grid(example("forest_fire"))
    assert grid.shape == (5, 5)
    assert grid[2, 2] == 1 and (np.delete(grid.ravel(), 12) == 0).all()   # one FIRE among TREEs
    row = initial_grid(example("rule30"))
    assert row.shape == (1, 61) and row.sum() == 1 and row[0, 30] == 1


def test_initial_percentages_and_rest_state():
    model = example("game_of_life")   # 30% ALIVE, the rest is the first state without a percentage: DEAD
    grid = initial_grid(model)
    assert (grid == 0).sum() == 120 and (grid == 1).sum() == 280


def test_initial_percentages_that_add_up_to_100():
    grid = initial_grid(example("cyclic"))
    assert np.bincount(grid.ravel()).tolist() == [225, 225, 225, 225]


def test_a_small_percentage_still_gives_one_cell():
    model = model_of("The world is a 5x5 grid. Cells can be TREE, FIRE or EMPTY. "
                     "Initially 60% of cells are TREE and 1% are FIRE.")
    counts = np.bincount(initial_grid(model).ravel(), minlength=3)
    assert counts.tolist() == [15, 1, 9]


def test_same_seed_same_start():
    model = example("game_of_life")
    assert (initial_grid(model) == initial_grid(model)).all()
    other = dict(model, seed=43)
    assert not (initial_grid(model) == initial_grid(other)).all()


# --- deterministic rules -----------------------------------------------------

def test_forest_fire_from_deliverable_1():
    sim = Simulation(example("forest_fire"))
    sim.step()
    assert sim.render().splitlines() == ["TTTTT", "TFFFT", "TFEFT", "TFFFT", "TTTTT"]
    sim.run(2)
    assert sim.counts() == {"TREE": 0, "FIRE": 0, "EMPTY": 25}
    sim.step()
    assert (sim.history[3] == sim.history[4]).all()   # stable from here on


def test_game_of_life_blinker():
    model = model_of((EXAMPLES / "game_of_life.txt").read_text().replace("20x20", "5x5"))
    states = model["states"]
    blinker = grid_from(".....\n..A..\n..A..\n..A..\n.....".replace(".", "D"), states)
    once = step(blinker, model)
    assert (once == grid_from(".....\n.....\n.AAA.\n.....\n.....".replace(".", "D"), states)).all()
    assert (step(once, model) == blinker).all()


def test_game_of_life_glider_wraps_around():
    model = model_of((EXAMPLES / "game_of_life.txt").read_text().replace("20x20", "6x6"))
    glider = np.ones((6, 6), dtype=np.uint8)   # DEAD everywhere
    for r, c in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        glider[r, c] = 0                        # ALIVE
    sim = Simulation(model, grid=glider)
    sim.run(4 * 6)                              # moves one cell diagonally every 4 steps
    assert (sim.grid == glider).all()


def test_no_rule_keeps_the_cell():
    model = model_of("The world is a 3x3 grid. Cells can be A or B. A A cell becomes B if exactly 8 neighbours are B.")
    grid = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.uint8)
    assert (step(grid, model) == grid).all()


def test_in_condition_and_von_neumann():
    model = model_of("The world is a 3x3 grid. Each cell has 4 neighbours. Cells can be A or B. "
                     "A A cell becomes B if 2 or 3 neighbours are B.")
    grid = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)
    # only the corners are B: the edge cells see 2 of them, the centre none
    assert step(grid, model).tolist() == [[1, 1, 1], [1, 0, 1], [1, 1, 1]]


def test_step_does_not_change_its_input():
    model = example("forest_fire")
    grid = initial_grid(model)
    before = grid.copy()
    step(grid, model)
    assert (grid == before).all()


def test_cyclic_ca_advances_states():
    model = example("cyclic")
    grid = np.zeros((30, 30), dtype=np.uint8)
    grid[0, 0] = 1
    new = step(grid, model)
    # the four (wrapped) von Neumann neighbours of (0, 0) become S1
    assert {(r, c) for r, c in zip(*np.nonzero(new == 1))} == {(0, 0), (0, 1), (1, 0), (0, 29), (29, 0)}


# --- Rule 30 -----------------------------------------------------------------

def naive_elementary(row, rule, wrap):
    n = len(row)
    out = []
    for i in range(n):
        left = row[(i - 1) % n] if wrap or i > 0 else 0
        right = row[(i + 1) % n] if wrap or i < n - 1 else 0
        out.append((rule >> (left * 4 + row[i] * 2 + right)) & 1)
    return out


def test_rule30_centre_column():
    sim = Simulation(example("rule30"))
    sim.run(15)
    centre = [int(g[0, 30]) for g in sim.history]
    # OEIS A051023
    assert centre == [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1]


def test_rule30_triangle():
    sim = Simulation(example("rule30"))
    sim.run(3)
    rows = ["".join(".#"[v] for v in g[0, 26:35]) for g in sim.history]
    assert rows == ["....#....", "...###...", "..##..#..", ".##.####."]


@pytest.mark.parametrize("border", ["fixed", "wrap"])
def test_rule30_matches_the_definition(border):
    model = dict(example("rule30"), cols=20, border=border)
    rng = np.random.default_rng(0)
    for _ in range(20):
        row = rng.integers(0, 2, size=(1, 20), dtype=np.uint8)
        assert step(row, model)[0].tolist() == naive_elementary(row[0].tolist(), 30, border == "wrap")


def test_rule30_table_example_is_the_same_automaton():
    a, b = Simulation(example("rule30")), Simulation(example("rule30_table"))
    a.run(30)
    b.run(30)
    assert (np.stack(a.history) == np.stack(b.history)).all()


# --- probabilities -----------------------------------------------------------

def test_same_seed_same_simulation():
    a, b = Simulation(example("sir_epidemic")), Simulation(example("sir_epidemic"))
    a.run(40)
    b.run(40)
    assert (np.stack(a.history) == np.stack(b.history)).all()
    c = Simulation(dict(example("sir_epidemic"), seed=8))
    c.run(40)
    assert not (c.grid == a.grid).all()


def test_probability_is_respected():
    model = model_of("The world is a 100x100 grid. Cells can be A or B. A A cell becomes B with probability 0.3. The seed is 1.")
    grid = np.zeros((100, 100), dtype=np.uint8)
    share = (step(grid, model, np.random.default_rng(1)) == 1).mean()
    assert 0.27 < share < 0.33


def test_probability_one_always_applies():
    model = model_of("The world is a 10x10 grid. Cells can be A or B. A A cell becomes B with probability 1.")
    assert (step(np.zeros((10, 10), dtype=np.uint8), model, np.random.default_rng()) == 1).all()


def test_two_random_rules_share_one_draw():
    model = model_of("The world is a 100x100 grid. Cells can be A, B or C. "
                     "A A cell becomes B with probability 0.5. A A cell becomes C with probability 0.5.")
    new = step(np.zeros((100, 100), dtype=np.uint8), model, np.random.default_rng(3))
    assert (new != 0).all()                    # 0.5 + 0.5: every cell changes
    assert 0.45 < (new == 1).mean() < 0.55


def test_random_rules_with_conditions_that_exclude_each_other():
    # never both true, so each rule keeps its full probability
    model = model_of("The world is a 1x3 grid. Cells can be A, B or C. "
                     "A A cell becomes B with probability 0.8 if no neighbours are C. "
                     "A A cell becomes C with probability 0.8 if at least 1 neighbour is C.")
    rng = np.random.default_rng(5)
    alone = np.array([[0, 0, 0]], dtype=np.uint8)
    near_c = np.array([[2, 0, 2]], dtype=np.uint8)
    became_b = sum(step(alone, model, rng)[0, 1] == 1 for _ in range(2000)) / 2000
    became_c = sum(step(near_c, model, rng)[0, 1] == 2 for _ in range(2000)) / 2000
    assert 0.76 < became_b < 0.84 and 0.76 < became_c < 0.84


def test_probabilistic_step_needs_a_generator():
    with pytest.raises(ValueError):
        step(np.zeros((40, 40), dtype=np.uint8), example("sir_epidemic"))


# --- Simulation --------------------------------------------------------------

def test_run_uses_the_generations_from_the_description():
    sim = Simulation(example("rule30"))
    sim.run()
    assert sim.generation == 30 and len(sim.history) == 31


def test_without_history():
    sim = Simulation(example("game_of_life"), keep_history=False)
    sim.run(10)
    assert sim.generation == 10 and sim.history == []


@pytest.mark.parametrize("name", ["forest_fire", "game_of_life", "cyclic", "sir_epidemic", "rule30", "rule30_table"])
def test_every_example_runs(name):
    sim = Simulation(example(name))
    sim.run(20)
    assert sum(sim.counts().values()) == sim.model["rows"] * sim.model["cols"]


@pytest.mark.parametrize("name", ["game_of_life", "cyclic", "sir_epidemic"])
def test_examples_with_a_random_start_repeat(name):
    assert (Simulation(example(name)).grid == Simulation(example(name)).grid).all()
