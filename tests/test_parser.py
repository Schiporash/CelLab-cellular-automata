import pytest

from cellab import DescriptionError, parse_description, wolfram_table
from conftest import EXAMPLES


def parse(text):
    return parse_description(text)


def rules_of(text):
    return parse("The world is a 3x3 grid. Cells can be A, B or C. " + text)["rules"]


def test_forest_fire_matches_deliverable_1():
    model = parse((EXAMPLES / "forest_fire.txt").read_text())
    assert model == {
        "rows": 5, "cols": 5, "neighbours": 8, "border": "fixed",
        "states": ["TREE", "FIRE", "EMPTY"],
        "rules": [
            {"state": "FIRE", "next": "EMPTY"},
            {"state": "TREE", "next": "FIRE", "if": ["FIRE", ">=", 1]},
            {"state": "EMPTY", "next": "EMPTY"},
        ],
    }


@pytest.mark.parametrize("sentence, expected", [
    ("A A cell becomes B if at least 2 neighbours are C.", ["C", ">=", 2]),
    ("A A cell becomes B if at most 2 neighbours are C.", ["C", "<=", 2]),
    ("A A cell becomes B if exactly 3 neighbours are C.", ["C", "==", 3]),
    ("A A cell becomes B if 3 neighbours are C.", ["C", "==", 3]),
    ("A A cell becomes B if more than 3 neighbours are C.", ["C", ">", 3]),
    ("A A cell becomes B if greater than 3 neighbours are C.", ["C", ">", 3]),
    ("A A cell becomes B if fewer than 2 neighbours are C.", ["C", "<", 2]),
    ("A A cell becomes B if less than 2 neighbors are C.", ["C", "<", 2]),
    ("A A cell becomes B if 2, 3 or 6 neighbours are C.", ["C", "in", [2, 3, 6]]),
    ("A A cell becomes B if between 2 and 4 neighbours are C.", ["C", "in", [2, 3, 4]]),
    ("A A cell becomes B if no neighbours are C.", ["C", "==", 0]),
    ("A A cell becomes B if any neighbour is C.", ["C", ">=", 1]),
    ("A A cell becomes B if at least two neighbours are C.", ["C", ">=", 2]),
])
def test_conditions(sentence, expected):
    assert rules_of(sentence) == [{"state": "A", "next": "B", "if": expected}]


@pytest.mark.parametrize("verb", ["becomes", "turns into", "changes to", "changes into", "turns", "transforms into"])
def test_becomes_synonyms(verb):
    assert rules_of(f"A A cell {verb} B.") == [{"state": "A", "next": "B"}]


@pytest.mark.parametrize("sentence", [
    "A B cell stays B.", "A B cell remains B.", "A B cell stays the same.", "A B cell is unchanged.", "A B cell stays.",
])
def test_stays(sentence):
    assert rules_of(sentence) == [{"state": "B", "next": "B"}]


@pytest.mark.parametrize("sentence, prob", [
    ("A A cell becomes B with probability 0.25.", 0.25),
    ("A A cell becomes B with probability 25%.", 0.25),
    ("A A cell becomes B with 25% probability.", 0.25),
    ("A A cell becomes B with probability 1.", 1),
])
def test_probability(sentence, prob):
    assert rules_of(sentence) == [{"state": "A", "next": "B", "prob": prob}]


def test_probability_and_condition_in_any_order():
    a = rules_of("A A cell becomes B with probability 0.5 if any neighbour is C.")
    b = rules_of("A A cell becomes B if any neighbour is C with probability 0.5.")
    assert a == b == [{"state": "A", "next": "B", "prob": 0.5, "if": ["C", ">=", 1]}]


def test_case_insensitive_and_lowercase_states():
    model = parse("the world is a 2 by 3 grid. cells can be alive or dead. "
                  "a dead cell becomes alive if 3 neighbours are alive.")
    assert (model["rows"], model["cols"]) == (2, 3)
    assert model["states"] == ["ALIVE", "DEAD"]
    assert model["rules"][0]["state"] == "DEAD"


@pytest.mark.parametrize("sentence", [
    "Cells can be TREE, FIRE or EMPTY.",
    "A cell can be TREE, FIRE or EMPTY.",
    "Each cell can be TREE, FIRE or EMPTY.",
    "Every cell can be TREE, FIRE, or EMPTY.",
    "The states are TREE, FIRE and EMPTY.",
    "The possible states are TREE, FIRE and EMPTY.",
])
def test_states_spellings(sentence):
    assert parse("The world is a 3x3 grid. " + sentence)["states"] == ["TREE", "FIRE", "EMPTY"]


@pytest.mark.parametrize("size",["5x7", "5 x 7", "5×7", "5 by 7"])
def test_world_size_spellings(size):
    model = parse(f"The world is a {size} grid. Cells can be A or B.")
    assert (model["rows"], model["cols"]) == (5, 7)


@pytest.mark.parametrize("sentence, neighbours", [
    ("Each cell has 4 neighbours.", 4),
    ("Cells have 8 neighbors.", 8),
    ("The neighbourhood is von Neumann.", 4),
    ("The neighborhood is Moore.", 8),
])
def test_neighbourhood(sentence, neighbours):
    assert parse("The world is a 3x3 grid. Cells can be A or B. " + sentence)["neighbours"] == neighbours


@pytest.mark.parametrize("sentence, border", [
    ("The grid wraps around.", "wrap"),
    ("The border is toroidal.", "wrap"),
    ("The edges are periodic.", "wrap"),
    ("The border is fixed.", "fixed"),
    ("The edges are closed.", "fixed"),
])
def test_border(sentence, border):
    assert parse("The world is a 3x3 grid. Cells can be A or B. " + sentence)["border"] == border


def test_defaults():
    model = parse("The world is a 3x3 grid. Cells can be A or B.")
    assert model["neighbours"] == 8 and model["border"] == "fixed" and model["rules"] == []


def test_experiment_parameters():
    model = parse("The world is a 3x3 grid. Cells can be A, B or C. The seed is 42. Run for 50 generations. "
                  "Initially 60% of cells are A, 10% are B and 5% are C.")
    assert model["seed"] == 42
    assert model["generations"] == 50
    assert model["initial"] == {"A": 60, "B": 10, "C": 5}


def test_sentences_on_separate_lines_without_periods():
    model = parse("The world is a 3x3 grid\nCells can be A or B\nA A cell becomes B")
    assert model["rules"] == [{"state": "A", "next": "B"}]


def test_comments_are_ignored():
    model = parse("# a comment. with dots.\nThe world is a 3x3 grid. Cells can be A or B.  # trailing")
    assert model["states"] == ["A", "B"]


def test_wolfram_table():
    assert wolfram_table(30) == {"111": 0, "110": 0, "101": 0, "100": 1,
                                 "011": 1, "010": 1, "001": 1, "000": 0}
    assert wolfram_table(110) == {"111": 0, "110": 1, "101": 1, "100": 0,
                                  "011": 1, "010": 1, "001": 1, "000": 0}


def test_rule_30_by_number_and_by_table_agree():
    by_number = parse((EXAMPLES / "rule30.txt").read_text())
    by_table = parse((EXAMPLES / "rule30_table.txt").read_text())
    assert by_number["table"] == by_table["table"] == wolfram_table(30)
    assert by_table["states"] == ["OFF", "ON"]  # default states for 1D automata


def test_pattern_overrides_rule_number():
    model = parse("The world is a 1x9 grid. Use rule 30. Pattern 000 becomes 1.")
    assert model["table"]["000"] == 1 and model["wolfram"] == 30


# --- errors ----------------------------------------------------------------

def test_unknown_sentence_is_reported_with_its_text_and_line():
    with pytest.raises(DescriptionError) as exc:
        parse("The world is a 3x3 grid.\nCells can be A or B.\nA A cell explodes loudly.")
    [problem] = exc.value.problems
    assert problem.sentence == "A A cell explodes loudly"
    assert problem.line == 3
    assert "explodes" in problem.message


def test_all_bad_sentences_are_reported():
    with pytest.raises(DescriptionError) as exc:
        parse("The world is a 3x3 grid. Hello there. Cells can be A or B. The sky is blue.")
    assert [p.sentence for p in exc.value.problems] == ["Hello there", "The sky is blue"]


def test_incomplete_sentence():
    with pytest.raises(DescriptionError) as exc:
        parse("The world is a 3x3 grid. A A cell becomes B if at least")
    assert "too early" in exc.value.problems[0].message


def test_world_given_twice():
    with pytest.raises(DescriptionError, match="more than once"):
        parse("The world is a 3x3 grid. The world is a 4x4 grid.")


def test_two_probabilities_in_one_rule():
    with pytest.raises(DescriptionError, match="only one"):
        rules_of("A A cell becomes B with probability 0.5 with probability 0.2.")


def test_bad_between_order():
    with pytest.raises(DescriptionError, match="smaller"):
        rules_of("A A cell becomes B if between 5 and 2 neighbours are C.")


def test_bad_pattern():
    with pytest.raises(DescriptionError, match="three bits"):
        parse("The world is a 1x9 grid. Pattern 121 becomes 1.")
