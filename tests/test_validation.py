import json

import pytest

from cellab import ValidationError, load_description, parse_description, validate

BASE = "The world is a 5x5 grid. Cells can be A, B or C. "


def messages(text):
    return [p.message for p in validate(parse_description(text))]


def assert_problem(text, fragment):
    found = messages(text)
    assert any(fragment in m for m in found), found


def test_valid_model_has_no_problems():
    assert messages(BASE + "A A cell becomes B if at least 1 neighbour is C.") == []


def test_missing_world():
    assert_problem("Cells can be A or B.", "world size is missing")


def test_missing_states():
    assert_problem("The world is a 5x5 grid.", "states are missing")


def test_zero_size():
    assert_problem("The world is a 0x5 grid. Cells can be A or B.", "positive")


def test_single_state():
    assert_problem("The world is a 5x5 grid. Cells can be A.", "at least two states")


def test_duplicate_state():
    assert_problem("The world is a 5x5 grid. Cells can be A, B or A.", "declared more than once")


def test_undeclared_state_in_rule():
    assert_problem(BASE + "A D cell becomes A.", "D, which was not declared")
    assert_problem(BASE + "A A cell becomes D.", "D, which was not declared")
    assert_problem(BASE + "A A cell becomes B if any neighbour is D.", "D, which was not declared")


def test_undeclared_state_in_initial():
    assert_problem(BASE + "Initially 10% of cells are D.", "D, which was not declared")


def test_too_many_neighbours_requested():
    assert_problem(BASE + "A A cell becomes B if at least 9 neighbours are C.", "only 8 neighbours")
    assert_problem(BASE + "Each cell has 4 neighbours. A A cell becomes B if exactly 5 neighbours are C.",
                   "only 4 neighbours")
    assert_problem(BASE + "A A cell becomes B if more than 8 neighbours are C.", "can never apply")


def test_bad_neighbour_count():
    assert_problem(BASE + "Each cell has 6 neighbours.", "4 or 8 neighbours")


def test_probability_out_of_range():
    assert_problem(BASE + "A A cell becomes B with probability 1.5.", "between 0 and 1")


def test_percentages_over_100():
    assert_problem(BASE + "Initially 70% of cells are A and 40% are B.", "more than 100%")


# --- contradicting rules ---------------------------------------------------

def test_two_unconditional_rules_contradict():
    assert_problem(BASE + "A A cell becomes B. A A cell becomes C.", "contradict")


def test_overlapping_conditions_on_same_state_contradict():
    assert_problem(BASE + "A A cell becomes B if at least 2 neighbours are C. "
                          "A A cell becomes C if at most 3 neighbours are C.", "contradict")


def test_disjoint_conditions_do_not_contradict():
    assert messages(BASE + "A A cell becomes B if fewer than 2 neighbours are C. "
                           "A A cell becomes C if more than 3 neighbours are C.") == []


def test_conditions_on_different_states_that_can_hold_together_contradict():
    assert_problem(BASE + "A A cell becomes B if at least 4 neighbours are B. "
                          "A A cell becomes C if at least 4 neighbours are C.", "contradict")


def test_conditions_on_different_states_that_cannot_hold_together():
    # 5 B neighbours + 5 C neighbours would need 10 neighbours
    assert messages(BASE + "A A cell becomes B if at least 5 neighbours are B. "
                           "A A cell becomes C if at least 5 neighbours are C.") == []


def test_same_result_is_not_a_contradiction():
    assert messages(BASE + "A A cell becomes B if any neighbour is B. "
                           "A A cell becomes B if any neighbour is C.") == []


def test_probabilistic_outcomes_that_fit_together():
    assert messages(BASE + "A A cell becomes B with probability 0.3. "
                           "A A cell becomes C with probability 0.5.") == []
    assert_problem(BASE + "A A cell becomes B with probability 0.7. "
                          "A A cell becomes C with probability 0.5.", "contradict")


# --- elementary automata ---------------------------------------------------

def test_elementary_needs_one_row():
    assert_problem("The world is a 3x9 grid. Use rule 30.", "single row")


def test_elementary_rule_number_range():
    assert_problem("The world is a 1x9 grid. Use rule 300.", "between 0 and 255")


def test_incomplete_table():
    assert_problem("The world is a 1x9 grid. Pattern 111 becomes 0. Pattern 000 becomes 1.",
                   "missing the patterns 110, 101, 100, 011, 010, 001")


def test_elementary_needs_two_states():
    assert_problem("The world is a 1x9 grid. Cells can be A, B or C. Use rule 30.", "exactly two states")


# --- models loaded from JSON (e.g. a saved simulation) ---------------------

def test_validate_plain_json_model():
    model = json.loads('{"rows": 5, "cols": 5, "states": ["X", "Y"],'
                       ' "rules": [{"state": "X", "next": "Z", "if": ["Y", "??", 1]}]}')
    found = [p.message for p in validate(model)]
    assert any("Z, which was not declared" in m for m in found)
    assert any("malformed condition" in m for m in found)


def test_load_description_raises_with_all_problems():
    with pytest.raises(ValidationError) as exc:
        load_description("Cells can be A. A B cell becomes A.")
    assert len(exc.value.problems) == 3  # no world, one state, B undeclared
