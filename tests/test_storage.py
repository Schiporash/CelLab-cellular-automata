import json

import numpy as np
import pytest

from cellab import (Simulation, StorageError, load_description, load_model, load_simulation,
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
