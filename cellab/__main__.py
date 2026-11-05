"""python -m cellab [description.txt] [-o model.json] [--grid START] [--run [N]] [--show] [--save NAME]

Parses and validates a description (from a file or, without one, typed in the
terminal) and prints the resulting model as JSON (or the list of problems, with
exit code 1). With --run it also simulates the automaton and prints the last
generation, --show prints all of them; --grid starts from a grid read from a text
file, --save keeps the simulation (NAME.json + NAME.npz) and --resume NAME
continues a saved one, from its last generation or from an earlier one (--from N)."""

import argparse
import json
import sys
from pathlib import Path

from . import (CelLabError, Simulation, load_description, load_grid, load_simulation, save_model,
               save_simulation)
from .simulation import DEFAULT_GENERATIONS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cellab", description=__doc__.splitlines()[0])
    ap.add_argument("description", type=Path, nargs="?",
                    help="text file with the automaton description (without one, you type it in the terminal)")
    ap.add_argument("-o", "--output", type=Path, help="write the model JSON here")
    ap.add_argument("--grid", type=Path, metavar="FILE", help="start from this grid (one line per row)")
    ap.add_argument("--resume", type=Path, metavar="NAME", help="continue a simulation saved with --save")
    ap.add_argument("--from", type=int, dest="start", metavar="N",
                    help="with --resume: go back to generation N, if it was kept (with probabilities the "
                         "simulation then continues differently from the first time)")
    ap.add_argument("--run", type=int, nargs="?", const=-1, metavar="N",
                    help="simulate N generations (default: the number in the description)")
    ap.add_argument("--show", action="store_true", help="print every generation, not only the last one")
    ap.add_argument("--save", type=Path, metavar="NAME", help="save the simulation as NAME.json + NAME.npz")
    ap.add_argument("--keep", choices=("all", "last"), default="all",
                    help="with --save: keep every generation or only the last one (default: all)")
    args = ap.parse_args(argv)
    if args.description and args.resume:
        ap.error("give either a description file or --resume NAME, not both")
    if args.start is not None and not args.resume:
        ap.error("--from only works with --resume NAME")
    if args.grid and args.resume:
        ap.error("--grid only works with a description, a resumed simulation continues from its own grid")

    try:
        return _run(args)
    except CelLabError as err:
        print(f"{type(err).__name__}:", file=sys.stderr)
        for problem in err.problems:
            print(f"  - {problem}", file=sys.stderr)
    except OSError as err:
        print(f"error: {err.filename or 'file'}: {err.strerror or err}", file=sys.stderr)
    except UnicodeDecodeError:
        print(f"error: {args.description} is not UTF-8 text, save it as UTF-8", file=sys.stderr)
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
    return 1


def _type_description() -> str:
    """Read lines from the terminal until an empty line (or the end of the input)."""
    terminal = sys.stdin.isatty()
    if terminal:
        print("Type the description, then an empty line to finish:")
    lines = []
    try:
        while line := input("> " if terminal else "").strip():
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def _print_generation(sim: Simulation) -> None:
    print(f"generation {sim.generation}: " + ", ".join(f"{s} {n}" for s, n in sim.counts().items()))
    print(sim.render())


def _run(args) -> int:
    keep_history = args.keep == "all"
    if args.resume:
        sim = load_simulation(args.resume)
        model = sim.model
        if args.start is not None:
            sim.rewind(args.start)
    else:
        text = args.description.read_text(encoding="utf-8") if args.description else _type_description()
        model = load_description(text)
        sim = Simulation(model, grid=load_grid(args.grid, model), keep_history=keep_history) if args.grid else None

    if args.output:
        save_model(model, args.output)
        print(f"model written to {args.output}")
    elif args.run is None and args.save is None and sim is None and not args.show:
        print(json.dumps(model, indent=2))
        return 0

    if sim is None:
        sim = Simulation(model, keep_history=keep_history)
    if args.show:
        generations = model.get("generations", DEFAULT_GENERATIONS) if args.run is None or args.run < 0 else args.run
        _print_generation(sim)
        for _ in range(generations):
            sim.step()
            print()
            _print_generation(sim)
    else:
        if args.run is not None:
            sim.run(None if args.run < 0 else args.run)
        if args.run is not None or args.resume or args.grid:
            _print_generation(sim)
    if args.save:
        json_path, npz_path = save_simulation(sim, args.save, keep=args.keep)
        print(f"simulation saved to {json_path} and {npz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
