"""python -m cellab description.txt [-o model.json] [--run [N]] [--save NAME]

Parses and validates a description and prints the resulting model as JSON
(or the list of problems, with exit code 1). With --run it also simulates the
automaton and prints the last generation; --save keeps the simulation
(NAME.json + NAME.npz) and --resume NAME continues a saved one."""

import argparse
import json
import sys
from pathlib import Path

from . import CelLabError, Simulation, load_description, load_simulation, save_model, save_simulation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cellab", description=__doc__.splitlines()[0])
    ap.add_argument("description", type=Path, nargs="?", help="text file with the automaton description")
    ap.add_argument("-o", "--output", type=Path, help="write the model JSON here")
    ap.add_argument("--resume", type=Path, metavar="NAME", help="continue a simulation saved with --save")
    ap.add_argument("--run", type=int, nargs="?", const=-1, metavar="N",
                    help="simulate N generations (default: the number in the description)")
    ap.add_argument("--save", type=Path, metavar="NAME", help="save the simulation as NAME.json + NAME.npz")
    ap.add_argument("--keep", choices=("all", "last"), default="all",
                    help="with --save: keep every generation or only the last one (default: all)")
    args = ap.parse_args(argv)
    if (args.description is None) == (args.resume is None):
        ap.error("give either a description file or --resume NAME")

    try:
        if args.resume:
            sim = load_simulation(args.resume)
            model = sim.model
        else:
            model = load_description(args.description.read_text(encoding="utf-8"))
            sim = None
    except CelLabError as err:
        print(f"{type(err).__name__}:", file=sys.stderr)
        for problem in err.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if args.output:
        save_model(model, args.output)
        print(f"model written to {args.output}")
    elif args.run is None and args.save is None and sim is None:
        print(json.dumps(model, indent=2))
        return 0

    if sim is None:
        sim = Simulation(model, keep_history=args.keep == "all")
    if args.run is not None:
        sim.run(None if args.run < 0 else args.run)
    if args.run is not None or args.resume:
        print(f"generation {sim.generation}: " + ", ".join(f"{s} {n}" for s, n in sim.counts().items()))
        print(sim.render())
    if args.save:
        json_path, npz_path = save_simulation(sim, args.save, keep=args.keep)
        print(f"simulation saved to {json_path} and {npz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
