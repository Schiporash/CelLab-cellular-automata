import io
import json

import numpy as np
import pytest

from cellab import (Simulation, StorageError, load_description, load_grid, load_model, load_simulation,
                    save_model, save_simulation)
from cellab.__main__ import main
from conftest import EXAMPLES


def example(name):
    return load_description((EXAMPLES / f"{name}.txt").read_text())


def history(sim):
    return np.stack(sim.history)


# --- model -------------------------------------------------------------------

def test_model_round_trip(tmp_path):
    model = example("sir_epidemic")
    save_model(model, tmp_path / "sir.json")
    assert load_model(tmp_path / "sir.json") == model


def test_loaded_model_is_validated(tmp_path):
    model = dict(example("forest_fire"), rules=[{"state": "SMOKE", "next": "EMPTY"}])
    save_model(model, tmp_path / "bad.json")
    with pytest.raises(StorageError) as err:
        load_model(tmp_path / "bad.json")
    assert "SMOKE" in str(err.value)


def test_broken_json(tmp_path):
    (tmp_path / "broken.json").write_text("{\"rows\": 5,")
    with pytest.raises(StorageError, match="not valid JSON"):
        load_model(tmp_path / "broken.json")


def test_missing_file(tmp_path):
    with pytest.raises(StorageError, match="cannot read"):
        load_model(tmp_path / "nothing.json")


# --- simulation --------------------------------------------------------------

def test_save_all_and_load(tmp_path):
    sim = Simulation(example("game_of_life"))
    sim.run(15)
    json_path, npz_path = save_simulation(sim, tmp_path / "gol")
    assert json_path.name == "gol.json" and npz_path.name == "gol.npz"
    with np.load(npz_path) as data:
        assert data["grids"].shape == (16, 20, 20)

    loaded = load_simulation(tmp_path / "gol")
    assert loaded.model == sim.model and loaded.generation == 15
    assert (history(loaded) == history(sim)).all()


@pytest.mark.parametrize("name", ["sir_epidemic", "game_of_life", "rule30"])
@pytest.mark.parametrize("keep", ["all", "last"])
def test_resuming_gives_the_same_simulation(tmp_path, name, keep):
    whole = Simulation(example(name))
    whole.run(30)

    first = Simulation(example(name))
    first.run(12)
    save_simulation(first, tmp_path / "half", keep=keep)
    resumed = load_simulation(tmp_path / "half")
    resumed.run(18)

    assert resumed.generation == 30
    assert (resumed.grid == whole.grid).all()
    if keep == "all":
        assert (history(resumed) == history(whole)).all()
    else:
        assert resumed.first_generation == 12 and len(resumed.history) == 19


def test_keep_last_stores_one_generation(tmp_path):
    sim = Simulation(example("cyclic"))
    sim.run(25)
    _, npz_path = save_simulation(sim, tmp_path / "cyc", keep="last")
    with np.load(npz_path) as data:
        assert data["grids"].shape == (1, 30, 30)
    meta = json.loads((tmp_path / "cyc.json").read_text())
    assert meta["generation"] == 25 and meta["first_generation"] == 25 and meta["keep"] == "last"


def test_any_of_the_two_file_names_works(tmp_path):
    sim = Simulation(example("forest_fire"))
    sim.run(2)
    save_simulation(sim, tmp_path / "ff.json")
    for name in ["ff", "ff.json", "ff.npz"]:
        assert load_simulation(tmp_path / name).generation == 2


def test_cannot_save_all_without_history(tmp_path):
    sim = Simulation(example("forest_fire"), keep_history=False)
    with pytest.raises(ValueError):
        save_simulation(sim, tmp_path / "ff", keep="all")
    save_simulation(sim, tmp_path / "ff", keep="last")


def test_generations_that_do_not_match_the_model(tmp_path):
    sim = Simulation(example("forest_fire"))
    sim.run(3)
    save_simulation(sim, tmp_path / "ff")
    np.savez_compressed(tmp_path / "ff.npz", grids=np.zeros((4, 6, 6), dtype=np.uint8))
    with pytest.raises(StorageError, match="shape"):
        load_simulation(tmp_path / "ff")
    np.savez_compressed(tmp_path / "ff.npz", grids=np.full((4, 5, 5), 7, dtype=np.uint8))
    with pytest.raises(StorageError, match="does not declare"):
        load_simulation(tmp_path / "ff")
    np.savez_compressed(tmp_path / "ff.npz", grids=np.zeros((2, 5, 5), dtype=np.uint8))
    with pytest.raises(StorageError, match="do not match"):
        load_simulation(tmp_path / "ff")


def test_missing_generations_file(tmp_path):
    sim = Simulation(example("forest_fire"))
    save_simulation(sim, tmp_path / "ff")
    (tmp_path / "ff.npz").unlink()
    with pytest.raises(StorageError, match="cannot read the generations"):
        load_simulation(tmp_path / "ff")


# --- command line ------------------------------------------------------------

def test_cli_run_save_resume(tmp_path, capsys):
    name = str(tmp_path / "sir")
    assert main([str(EXAMPLES / "sir_epidemic.txt"), "--run", "50", "--save", name]) == 0
    assert main(["--resume", name, "--run", "150"]) == 0
    resumed = capsys.readouterr().out.splitlines()[-41]
    assert main([str(EXAMPLES / "sir_epidemic.txt"), "--run"]) == 0
    whole = capsys.readouterr().out.splitlines()[0]
    assert resumed.startswith("generation 200") and resumed == whole


def test_cli_reports_problems(tmp_path, capsys):
    assert main(["--resume", str(tmp_path / "nothing")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_cli_missing_description_file(tmp_path, capsys):
    assert main([str(tmp_path / "nothing.txt")]) == 1
    assert "nothing.txt" in capsys.readouterr().err


def test_cli_save_in_a_folder_that_does_not_exist(tmp_path, capsys):
    assert main([str(EXAMPLES / "forest_fire.txt"), "--run", "1", "--save", str(tmp_path / "no" / "x")]) == 1
    assert "x.npz" in capsys.readouterr().err


def test_cli_write_model_in_a_folder_that_does_not_exist(tmp_path, capsys):
    assert main([str(EXAMPLES / "forest_fire.txt"), "-o", str(tmp_path / "no" / "x.json")]) == 1
    assert "x.json" in capsys.readouterr().err


def test_cli_description_that_is_not_utf8(tmp_path, capsys):
    (tmp_path / "ansi.txt").write_bytes(b"The world is a 5x5 grid. \xe3\n")
    assert main([str(tmp_path / "ansi.txt")]) == 1
    assert "UTF-8" in capsys.readouterr().err


# --- starting grid from a file -----------------------------------------------

FOREST = "The world is a 2x3 grid. Cells can be TREE, FIRE or EMPTY. A FIRE cell becomes EMPTY."


def grid_file(tmp_path, content):
    path = tmp_path / "start.grid"
    path.write_bytes(content.encode("utf-8"))
    return path


def test_load_grid_with_first_letters(tmp_path):
    grid = load_grid(grid_file(tmp_path, "TFE\nEET\n"), load_description(FOREST))
    assert grid.tolist() == [[0, 1, 2], [2, 2, 0]]


def test_load_grid_ignores_blank_lines_spaces_and_bom(tmp_path):
    grid = load_grid(grid_file(tmp_path, "﻿\n TFE \n\nEET\n\n"), load_description(FOREST))
    assert grid.tolist() == [[0, 1, 2], [2, 2, 0]]


def test_load_grid_with_two_states_takes_letters_or_the_printed_symbols(tmp_path):
    model = load_description("The world is a 1x4 grid. Cells can be ALIVE or DEAD.")
    assert load_grid(grid_file(tmp_path, "ADDA"), model).tolist() == [[0, 1, 1, 0]]
    assert load_grid(grid_file(tmp_path, ".##."), model).tolist() == [[0, 1, 1, 0]]


def test_load_grid_with_indices_when_first_letters_clash(tmp_path):
    grid = load_grid(grid_file(tmp_path, "0123\n3210"),
                     load_description("The world is a 2x4 grid. Cells can be S0, S1, S2 or S3."))
    assert grid.tolist() == [[0, 1, 2, 3], [3, 2, 1, 0]]


def test_load_grid_reports_every_problem(tmp_path):
    with pytest.raises(StorageError) as err:
        load_grid(grid_file(tmp_path, "TFX\nEE\nTTT\n"), load_description(FOREST))
    text = str(err.value)
    assert "line 1" in text and "'X'" in text      # unknown symbol
    assert "line 2" in text and "2 cells" in text  # short line
    assert "3 lines" in text                       # the model has 2 rows


def test_load_grid_from_the_wrong_file_gives_a_short_message():
    model = load_description(FOREST)
    with pytest.raises(StorageError) as err:
        load_grid(EXAMPLES / "game_of_life.txt", model)
    assert len(err.value.problems) == 3  # number of lines, width, unknown symbols


def test_load_grid_missing_file(tmp_path):
    with pytest.raises(StorageError, match="cannot read"):
        load_grid(tmp_path / "nothing.grid", load_description(FOREST))


def test_cli_start_from_the_figure_in_deliverable_1(capsys):
    # the start and generations 1-3 are the ones drawn in the document
    fire, grid = str(EXAMPLES / "forest_fire.txt"), str(EXAMPLES / "forest_fire_l1.grid")
    expected = {1: "TTTET TEFFT TFEFE EFFFT TTETT",
                2: "TFFEF FEEEF FEEEE EEEEF FFEFF",
                3: "FEEEE EEEEE EEEEE EEEEE EEEEE",
                4: "EEEEE EEEEE EEEEE EEEEE EEEEE"}
    for generation, rows in expected.items():
        assert main([fire, "--grid", grid, "--run", str(generation)]) == 0
        assert capsys.readouterr().out.splitlines()[1:] == rows.split()


def test_cli_grid_without_run_shows_generation_0(tmp_path, capsys):
    (tmp_path / "forest.txt").write_text(FOREST)
    path = grid_file(tmp_path, "TFE\nEET\n")
    assert main([str(tmp_path / "forest.txt"), "--grid", str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == ["generation 0: TREE 2, FIRE 1, EMPTY 3", "TFE", "EET"]


def test_cli_grid_is_saved_and_resumed(tmp_path, capsys):
    (tmp_path / "forest.txt").write_text(FOREST)
    path = grid_file(tmp_path, "TFE\nEET\n")
    assert main([str(tmp_path / "forest.txt"), "--grid", str(path), "--save", str(tmp_path / "f")]) == 0
    capsys.readouterr()
    assert main(["--resume", str(tmp_path / "f"), "--run", "1"]) == 0
    assert capsys.readouterr().out.splitlines() == ["generation 1: TREE 2, FIRE 0, EMPTY 4", "TEE", "EET"]


def test_cli_grid_that_does_not_fit(tmp_path, capsys):
    (tmp_path / "forest.txt").write_text(FOREST)
    path = grid_file(tmp_path, "TF\n")
    assert main([str(tmp_path / "forest.txt"), "--grid", str(path)]) == 1
    assert "2 cells" in capsys.readouterr().err


def test_cli_grid_cannot_go_with_resume(tmp_path):
    with pytest.raises(SystemExit):
        main(["--resume", str(tmp_path / "x"), "--grid", str(tmp_path / "g")])


# --- --show ------------------------------------------------------------------

def test_cli_show_prints_every_generation(tmp_path, capsys):
    (tmp_path / "forest.txt").write_text(FOREST)
    path = grid_file(tmp_path, "TFE\nEET\n")
    assert main([str(tmp_path / "forest.txt"), "--grid", str(path), "--run", "1", "--show"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "generation 0: TREE 2, FIRE 1, EMPTY 3", "TFE", "EET", "",
        "generation 1: TREE 2, FIRE 0, EMPTY 4", "TEE", "EET",
    ]


def test_cli_show_without_run_uses_the_generations_from_the_description(capsys):
    assert main([str(EXAMPLES / "rule30.txt"), "--show"]) == 0
    headers = [line for line in capsys.readouterr().out.splitlines() if line.startswith("generation")]
    assert len(headers) == 31 and headers[0].startswith("generation 0:") and headers[-1].startswith("generation 30:")


def test_cli_show_continues_a_resumed_simulation(tmp_path, capsys):
    name = str(tmp_path / "ff")
    assert main([str(EXAMPLES / "forest_fire.txt"), "--run", "2", "--save", name]) == 0
    capsys.readouterr()
    assert main(["--resume", name, "--run", "1", "--show"]) == 0
    headers = [line for line in capsys.readouterr().out.splitlines() if line.startswith("generation")]
    assert [h.split(":")[0] for h in headers] == ["generation 2", "generation 3"]


def test_cli_show_saves_all_the_generations_it_showed(tmp_path, capsys):
    name = str(tmp_path / "ff")
    assert main([str(EXAMPLES / "forest_fire.txt"), "--run", "3", "--show", "--save", name]) == 0
    assert load_simulation(name).generation == 3 and len(load_simulation(name).history) == 4


# --- description from the keyboard -------------------------------------------

def type_in(monkeypatch, text):
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def test_cli_description_from_the_keyboard(monkeypatch, capsys):
    type_in(monkeypatch, FOREST + "\n\n")
    assert main(["--run", "0"]) == 0
    assert capsys.readouterr().out.splitlines() == ["generation 0: TREE 5, FIRE 1, EMPTY 0", "TTT", "TFT"]


def test_cli_keyboard_asks_for_the_description_in_a_real_terminal(monkeypatch, capsys):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Terminal(FOREST + "\n\n"))
    assert main(["--run", "0"]) == 0
    out = capsys.readouterr().out
    assert "empty line" in out and "> " in out


def test_cli_keyboard_stops_at_the_first_empty_line(monkeypatch, capsys):
    type_in(monkeypatch, "The world is a 2x3 grid.\nCells can be TREE or FIRE.\n\nthis is not read\n")
    assert main([]) == 0
    assert '"rows": 2' in capsys.readouterr().out


def test_cli_keyboard_works_until_the_input_ends(monkeypatch, capsys):
    type_in(monkeypatch, "The world is a 2x3 grid.\nCells can be TREE or FIRE.")
    assert main([]) == 0
    assert '"cols": 3' in capsys.readouterr().out


def test_cli_keyboard_mistakes_are_reported(monkeypatch, capsys):
    type_in(monkeypatch, "The world is a 5x5 grid. Cells can be A or B. A A cell becomes C.\n\n")
    assert main(["--run", "1"]) == 1
    assert "state C" in capsys.readouterr().err


def test_cli_keyboard_with_nothing_typed(monkeypatch, capsys):
    type_in(monkeypatch, "\n")
    assert main([]) == 1
    assert "world size" in capsys.readouterr().err


def test_cli_keyboard_cancelled_with_ctrl_c(monkeypatch, capsys):
    def interrupted(prompt=""):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", interrupted)
    assert main([]) == 1
    assert "cancelled" in capsys.readouterr().err


def test_cli_description_and_resume_together_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "d.txt"), "--resume", str(tmp_path / "x")])


# --- --from ------------------------------------------------------------------

@pytest.fixture
def saved_fire(tmp_path, capsys):
    """Forest Fire saved at generation 3, with generations 0-3 kept."""
    name = str(tmp_path / "ff")
    assert main([str(EXAMPLES / "forest_fire.txt"), "--run", "3", "--save", name]) == 0
    capsys.readouterr()
    return name


def test_cli_from_shows_an_earlier_generation(saved_fire, capsys):
    assert main(["--resume", saved_fire, "--from", "1"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "generation 1: TREE 16, FIRE 8, EMPTY 1", "TTTTT", "TFFFT", "TFEFT", "TFFFT", "TTTTT",
    ]


def test_cli_from_continues_from_that_generation(saved_fire, capsys):
    assert main(["--resume", saved_fire, "--from", "1", "--run", "1"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "generation 2: TREE 0, FIRE 16, EMPTY 9"


def test_cli_from_with_show_starts_at_that_generation(saved_fire, capsys):
    assert main(["--resume", saved_fire, "--from", "2", "--run", "1", "--show"]) == 0
    headers = [line.split(":")[0] for line in capsys.readouterr().out.splitlines() if line.startswith("generation")]
    assert headers == ["generation 2", "generation 3"]


def test_cli_from_then_save_drops_the_later_generations(saved_fire, tmp_path, capsys):
    other = str(tmp_path / "other")
    assert main(["--resume", saved_fire, "--from", "1", "--save", other]) == 0
    loaded = load_simulation(other)
    assert loaded.generation == 1 and len(loaded.history) == 2
    assert loaded.grid.tolist()[1] == [0, 1, 1, 1, 0]   # TFFFT, generation 1


def test_cli_from_leaves_the_original_save_alone(saved_fire, capsys):
    assert main(["--resume", saved_fire, "--from", "1", "--run", "1"]) == 0
    assert load_simulation(saved_fire).generation == 3


def test_cli_from_a_generation_after_the_last_one(saved_fire, capsys):
    assert main(["--resume", saved_fire, "--from", "9"]) == 1
    assert "generations 0..3" in capsys.readouterr().err


def test_cli_from_a_negative_generation(saved_fire, capsys):
    assert main(["--resume", saved_fire, "--from=-1"]) == 1
    assert "generations 0..3" in capsys.readouterr().err


def test_cli_from_a_generation_that_was_not_kept(tmp_path, capsys):
    name = str(tmp_path / "last")
    assert main([str(EXAMPLES / "forest_fire.txt"), "--run", "3", "--save", name, "--keep", "last"]) == 0
    capsys.readouterr()
    assert main(["--resume", name, "--from", "1"]) == 1
    assert "generations 3..3" in capsys.readouterr().err


def test_cli_from_the_same_save_twice_gives_the_same_run(tmp_path, capsys):
    name = str(tmp_path / "sir")
    assert main([str(EXAMPLES / "sir_epidemic.txt"), "--run", "30", "--save", name]) == 0
    capsys.readouterr()
    assert main(["--resume", name, "--from", "10", "--run", "5"]) == 0
    first = capsys.readouterr().out
    assert main(["--resume", name, "--from", "10", "--run", "5"]) == 0
    assert capsys.readouterr().out == first


def test_cli_from_needs_resume(capsys):
    with pytest.raises(SystemExit):
        main([str(EXAMPLES / "forest_fire.txt"), "--from", "1"])
