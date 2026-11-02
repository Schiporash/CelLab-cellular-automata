"""English description -> automaton model (a JSON-serialisable dict).

Every sentence is parsed on its own with the Lark grammar in grammar.lark,
so that when something is not understood we can tell the user exactly which
sentence it was. The pieces are then merged into a single model:

    {"rows": 5, "cols": 5, "neighbours": 8, "border": "fixed",
     "states": ["TREE", "FIRE", "EMPTY"],
     "rules": [
       {"state": "FIRE", "next": "EMPTY"},
       {"state": "TREE", "next": "FIRE", "if": ["FIRE", ">=", 1]}
     ]}

Optional keys: "prob" on a rule, "table" / "wolfram" for elementary (1D)
automata, "seed", "generations" and "initial" (percent per state).
"""

from pathlib import Path

from lark import Lark, Transformer, v_args
from lark.exceptions import UnexpectedInput, VisitError

from .errors import DescriptionError, Problem
from .normalize import split_sentences

_GRAMMAR = (Path(__file__).parent / "grammar.lark").read_text(encoding="utf-8")
_lark = Lark(_GRAMMAR, parser="earley", ambiguity="resolve")

DEFAULTS = {"neighbours": 8, "border": "fixed"}
ELEMENTARY_STATES = ["OFF", "ON"]


class _SentenceError(Exception):
    pass


@v_args(inline=True)
class _ToModel(Transformer):
    """Turns the parse tree of one sentence into a (kind, value) pair."""

    def INT(self, tok):
        return int(tok)

    def DECIMAL(self, tok):
        return float(tok)

    def NAME(self, tok):
        return str(tok).upper()

    # --- world -----------------------------------------------------------
    def world(self, rows, cols):
        return "size", (rows, cols)

    @v_args(inline=False)
    def name_list(self, names):
        return names

    def states(self, names):
        return "states", names

    def neighbourhood(self, value):
        if isinstance(value, int):
            return "neighbours", value
        return "neighbours", 8 if value == "moore" else 4

    def border(self, kind):
        return "border", str(kind)

    # --- rules -----------------------------------------------------------
    def becomes(self, target):
        return target

    def stays(self, target=None):
        return target  # None = the cell keeps its own state

    def prob_decimal(self, p):
        return "prob", p

    def prob_percent(self, p):
        return "prob", p / 100

    def condition(self, count, state):
        op, value = count
        return "if", [state, op, value]

    def cond_none(self, state):
        return "if", [state, "==", 0]

    def cond_any(self, state):
        return "if", [state, ">=", 1]

    def ge(self, n): return ">=", n
    def le(self, n): return "<=", n
    def eq(self, n): return "==", n
    def gt(self, n): return ">", n
    def lt(self, n): return "<", n

    @v_args(inline=False)
    def one_of(self, values):
        return "in", sorted(set(values))

    def between(self, lo, hi):
        if lo > hi:
            raise _SentenceError(f"'between {lo} and {hi}': the first number must be the smaller one")
        return "in", list(range(lo, hi + 1))

    def rule(self, state, target, *modifiers):
        rule = {"state": state, "next": target or state}
        for key, value in modifiers:
            if key in rule:
                raise _SentenceError(f"a rule can have only one '{'if' if key == 'if' else 'probability'}' part")
            rule[key] = value
        return "rule", rule

    # --- elementary (1D) automata ----------------------------------------
    def elementary(self, number):
        return "wolfram", number

    def pattern(self, bits, value):
        bits = str(bits).zfill(3)
        if len(bits) != 3 or set(bits) - {"0", "1"} or value not in (0, 1):
            raise _SentenceError("a pattern looks like 'Pattern 110 becomes 1' (three bits -> 0 or 1)")
        return "pattern", (bits, value)

    # --- experiment parameters -------------------------------------------
    def seed(self, value):
        return "seed", value

    def generations(self, value):
        return "generations", value

    def percent_state(self, percent, state):
        return state, percent

    @v_args(inline=False)
    def initial(self, pairs):
        return "initial", dict(pairs)


_transformer = _ToModel()


def wolfram_table(number: int) -> dict[str, int]:
    """Rule number (0-255) -> {"111": b7, "110": b6, ..., "000": b0}."""
    return {format(i, "03b"): (number >> i) & 1 for i in range(7, -1, -1)}


def _describe_parse_error(err: UnexpectedInput, text: str) -> str:
    if hasattr(err, "char"):
        near = text[max(0, err.column - 1):].split(" ")[0] if err.column else ""
        return f"I don't understand the part starting at '{near}'" if near else "sentence not recognised"
    return "the sentence ends too early"


def parse_description(description: str) -> dict:
    """Parse a whole description. Raises DescriptionError listing every
    sentence that was not understood (not only the first one)."""
    problems: list[Problem] = []
    model: dict = {}
    rules: list[dict] = []
    patterns: dict[str, int] = {}

    for sentence in split_sentences(description):
        try:
            kind, value = _transformer.transform(_lark.parse(sentence.normalized))
        except UnexpectedInput as err:
            problems.append(Problem(_describe_parse_error(err, sentence.normalized), sentence.text, sentence.line))
            continue
        except VisitError as err:
            problems.append(Problem(str(err.orig_exc), sentence.text, sentence.line))
            continue

        if kind == "rule":
            rules.append(value)
        elif kind == "pattern":
            patterns[value[0]] = value[1]
        elif kind in model or (kind == "size" and "rows" in model):
            problems.append(Problem(f"'{kind}' is given more than once", sentence.text, sentence.line))
        elif kind == "size":
            model["rows"], model["cols"] = value
        else:
            model[kind] = value

    if problems:
        raise DescriptionError(problems)

    for key, value in DEFAULTS.items():
        model.setdefault(key, value)
    model["rules"] = rules

    if "wolfram" in model or patterns:
        table = wolfram_table(model["wolfram"]) if "wolfram" in model else {}
        table.update(patterns)
        model["table"] = table
        model.setdefault("states", ELEMENTARY_STATES)
    return _ordered(model)


_KEY_ORDER = ["rows", "cols", "neighbours", "border", "states", "rules",
              "wolfram", "table", "seed", "generations", "initial"]


def _ordered(model: dict) -> dict:
    return {k: model[k] for k in _KEY_ORDER if k in model}
