"""Checks that a model (from the parser or from a saved JSON) makes sense.

All problems are collected and reported together. Checked:
  * the world size is given and positive;
  * states are declared, at least two, no duplicates;
  * every state used in a rule / condition / initial setting is declared;
  * a condition never asks for more neighbours than a cell has;
  * probabilities are in (0, 1], percentages add up to at most 100;
  * no two rules contradict each other (same cell state, a neighbourhood
    where both apply, different results);
  * elementary automata (Rule 30 etc.) have one row, two states and a
    complete 8-case table.
A cell for which no rule applies keeps its state, so rules do not have to
cover every case.
"""

from itertools import combinations

from .errors import Problem, ValidationError

BORDERS = ("fixed", "wrap")
NEIGHBOURS = (4, 8)
OPERATORS = (">=", "<=", "==", ">", "<", "in")
RESERVED = {"cell", "cells", "if", "or", "and", "with", "becomes", "stays"}


def validate(model: dict) -> list[Problem]:
    """Return the list of problems (empty if the model is valid)."""
    problems: list[str] = []
    err = problems.append

    # --- world -----------------------------------------------------------
    rows, cols = model.get("rows"), model.get("cols")
    if rows is None or cols is None:
        err("the world size is missing (e.g. 'The world is a 5x5 grid.')")
    elif not (_is_int(rows) and _is_int(cols)) or rows <= 0 or cols <= 0:
        err(f"the grid size must be positive whole numbers, got {rows}x{cols}")

    neighbours = model.get("neighbours", 8)
    if neighbours not in NEIGHBOURS:
        err(f"a cell can have 4 or 8 neighbours, not {neighbours}")
        neighbours = 8
    if model.get("border", "fixed") not in BORDERS:
        err(f"the border must be 'fixed' or 'wrap', not '{model.get('border')}'")

    # --- states ----------------------------------------------------------
    states = model.get("states")
    if not states:
        states = []
    else:
        if len(states) < 2:
            err("an automaton needs at least two states")
        for s in sorted({s for s in states if states.count(s) > 1}):
            err(f"state {s} is declared more than once")
        for s in states:
            if not isinstance(s, str) or not s.replace("_", "").isalnum() or s.lower() in RESERVED:
                err(f"'{s}' cannot be used as a state name")
    declared = set(states)
    undeclared: dict[str, str] = {}  # state -> where it is first used

    def check_state(name, where):
        if name not in declared:
            undeclared.setdefault(name, where)

    # --- rules -----------------------------------------------------------
    rules = model.get("rules", [])
    for rule in rules:
        text = describe_rule(rule)
        check_state(rule.get("state"), f"rule '{text}'")
        check_state(rule.get("next"), f"rule '{text}'")
        if "prob" in rule and not (isinstance(rule["prob"], (int, float)) and 0 < rule["prob"] <= 1):
            err(f"rule '{text}': the probability must be between 0 and 1")
        if "if" in rule:
            cond = rule["if"]
            if not (isinstance(cond, list) and len(cond) == 3 and cond[1] in OPERATORS):
                err(f"rule '{text}': malformed condition {cond}")
                continue
            check_state(cond[0], f"rule '{text}'")
            values = cond[2] if cond[1] == "in" else [cond[2]]
            if any(not _is_int(v) or v < 0 for v in values):
                err(f"rule '{text}': neighbour counts must be whole numbers >= 0")
            elif not satisfying_counts(cond, neighbours):
                err(f"rule '{text}' can never apply: a cell has only {neighbours} neighbours")

    if not problems:
        problems.extend(_conflicts(rules, neighbours))

    # --- elementary automata ---------------------------------------------
    if "table" in model or "wolfram" in model:
        problems.extend(_check_elementary(model, states))

    # --- experiment parameters -------------------------------------------
    if "seed" in model and not (_is_int(model["seed"]) and model["seed"] >= 0):
        err("the seed must be a whole number >= 0")
    if "generations" in model and not (_is_int(model["generations"]) and model["generations"] > 0):
        err("the number of generations must be a whole number > 0")
    initial = model.get("initial", {})
    for state, percent in initial.items():
        check_state(state, "the initial configuration")
        if not isinstance(percent, (int, float)) or not 0 <= percent <= 100:
            err(f"the percentage for {state} must be between 0 and 100")
    if sum(p for p in initial.values() if isinstance(p, (int, float))) > 100:
        err("the initial percentages add up to more than 100%")

    # one message per missing state, not one for every place it appears
    if not states:
        used = ", ".join(undeclared) or "ALIVE or DEAD"
        problems.insert(0, f"the cell states are missing, add a sentence like 'Cells can be {used}.'")
    else:
        for name, where in undeclared.items():
            err(f"state {name} is used in {where} but was not declared")

    return [Problem(p) for p in problems]


def validate_or_raise(model: dict) -> dict:
    problems = validate(model)
    if problems:
        raise ValidationError(problems)
    return model


# --- helpers ---------------------------------------------------------------

def satisfying_counts(cond: list, neighbours: int) -> set[int]:
    """The neighbour counts (0..neighbours) for which the condition holds."""
    _, op, value = cond
    tests = {
        ">=": lambda n: n >= value, "<=": lambda n: n <= value,
        "==": lambda n: n == value, ">": lambda n: n > value,
        "<": lambda n: n < value, "in": lambda n: n in value,
    }
    return {n for n in range(neighbours + 1) if tests[op](n)}


def _can_both_apply(r1: dict, r2: dict, neighbours: int) -> bool:
    if "if" not in r1 or "if" not in r2:
        return True
    (s1, *_), (s2, *_) = r1["if"], r2["if"]
    c1, c2 = satisfying_counts(r1["if"], neighbours), satisfying_counts(r2["if"], neighbours)
    if s1 == s2:
        return bool(c1 & c2)
    # different neighbour states: need a neighbours of one kind and b of the
    # other, and they must fit around the cell together
    return min(c1) + min(c2) <= neighbours


def _conflicts(rules: list[dict], neighbours: int) -> list[str]:
    found = []
    for r1, r2 in combinations(rules, 2):
        if r1["state"] != r2["state"] or r1["next"] == r2["next"]:
            continue
        if not _can_both_apply(r1, r2, neighbours):
            continue
        p1, p2 = r1.get("prob"), r2.get("prob")
        if p1 is not None and p2 is not None and p1 + p2 <= 1:
            continue  # two random outcomes that fit in the same draw
        found.append(f"rules '{describe_rule(r1)}' and '{describe_rule(r2)}' contradict each other")
    return found


def _check_elementary(model: dict, states: list) -> list[str]:
    problems = []
    table = model.get("table", {})
    expected = {format(i, "03b") for i in range(8)}
    if "wolfram" in model and not (_is_int(model["wolfram"]) and 0 <= model["wolfram"] <= 255):
        problems.append(f"an elementary rule number must be between 0 and 255, not {model['wolfram']}")
    if missing := sorted(expected - set(table), reverse=True):
        problems.append(f"the rule table is missing the patterns {', '.join(missing)}")
    if extra := sorted(set(table) - expected):
        problems.append(f"unknown patterns in the rule table: {', '.join(extra)}")
    if any(v not in (0, 1) for v in table.values()):
        problems.append("the rule table can only produce 0 or 1")
    if model.get("rows") not in (None, 1):
        problems.append("an elementary automaton (like Rule 30) needs a grid with a single row")
    if len(states) != 2:
        problems.append("an elementary automaton needs exactly two states")
    if model.get("rules"):
        problems.append("an elementary automaton is described by its table; other rules are not allowed")
    return problems


def describe_rule(rule: dict) -> str:
    text = f"{rule.get('state')} -> {rule.get('next')}"
    if "if" in rule:
        state, op, value = rule["if"]
        text += f" if {state} {op} {value}"
    if "prob" in rule:
        text += f" (p={rule['prob']})"
    return text


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
