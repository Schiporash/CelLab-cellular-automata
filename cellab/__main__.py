"""python -m cellab description.txt [-o model.json]

Parses and validates a description and prints the resulting model as JSON
(or the list of problems, with exit code 1)."""

import argparse
import json
import sys
from pathlib import Path

from . import CelLabError, load_description


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cellab", description=__doc__.splitlines()[0])
    ap.add_argument("description", type=Path, help="text file with the automaton description")
    ap.add_argument("-o", "--output", type=Path, help="write the model JSON here")
    args = ap.parse_args(argv)

    try:
        model = load_description(args.description.read_text(encoding="utf-8"))
    except CelLabError as err:
        print(f"{type(err).__name__}:", file=sys.stderr)
        for problem in err.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    text = json.dumps(model, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"model written to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
