"""benchmark/paired_tests.py: the main table's sign test, Holm, and the unscorable rule.

Run:  python tests/test_paired_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import paired_tests as pt                              # noqa: E402

NAN = float("nan")


def test_outcome_rounding_and_direction():
    assert pt.outcome(1.0, 2.0) == "win"
    assert pt.outcome(2.0, 1.0) == "loss"
    assert pt.outcome(1.0001, 1.0004) == "tie"                     # equal at 3 decimals
    assert pt.outcome(0.9, 0.5, lower_is_better=False) == "win"


def test_outcome_unscorable_against_ours():
    assert pt.outcome(None, 1.0) == "loss"                         # ours refused, A0 scored
    assert pt.outcome(NAN, 1.0) == "loss"
    assert pt.outcome(1.0, None) == "excluded"                     # A0 refused: out of n'
    assert pt.outcome(None, NAN) == "excluded"


def test_sign_test_brief_numbers():
    r = pt.sign_test(["win"] * 12 + ["loss"] * 3)
    assert r["n_prime"] == 15 and abs(r["p_one_sided"] - 576 / 32768) < 1e-12      # 0.018
    r = pt.sign_test(["win"] * 8 + ["loss"] + ["tie", "excluded"])
    assert (r["n_prime"], r["ties"], r["excluded"]) == (9, 1, 1)
    assert abs(r["p_one_sided"] - 10 / 512) < 1e-12                               # 0.020
    assert pt.sign_test([])["p_one_sided"] == 1.0


def test_wins_needed():
    assert pt.wins_needed(15, 0.025) == 12 and pt.wins_needed(15, 0.05) == 12
    assert pt.wins_needed(9, 0.05) == 8
    assert pt.wins_needed(4, 0.05) is None                         # both-gates line: descriptive only


def test_holm_step_down():
    assert pt.holm({"a": 0.01, "b": 0.04}) == {"a": (0.025, True), "b": (0.05, True)}
    assert pt.holm({"a": 0.03, "b": 0.04}) == {"a": (0.025, False), "b": (0.05, False)}
    r = pt.holm({"cam": 0.001, "obj": 0.2})
    assert r["cam"] == (0.025, True) and r["obj"] == (0.05, False)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
