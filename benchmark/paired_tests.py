"""Paired one-sided sign test for the experiment set's main table: ours (A3) against its backbone (A0).

Rules fixed before any per-pair number existed (docs/grader-response-experiment-v1.md §2):
  win          ours < A0 after rounding both to `decimals` (the reported precision); lower is better
  tie          equal after rounding: removed from n'
  unscorable   decided against ours: ours unscorable and A0 scorable is a LOSS; A0 unscorable removes
               the pair from n' ('excluded'), whether ours is scorable or not. None / NaN = unscorable.
  p            P(X >= wins | n', 1/2), one-sided
  Holm         step-down across the primary metrics: the smaller p against alpha/2, then alpha
"""
from __future__ import annotations

from math import comb, isfinite


def outcome(ours, base, lower_is_better: bool = True, decimals: int = 3) -> str:
    """'win' | 'loss' | 'tie' | 'excluded' for one pair and one metric."""
    ok = lambda v: v is not None and isfinite(v)                              # noqa: E731
    if not ok(base):
        return "excluded"
    if not ok(ours):
        return "loss"
    a, b = round(ours, decimals), round(base, decimals)
    if a == b:
        return "tie"
    return "win" if (a < b) == lower_is_better else "loss"


def tail_p(wins: int, n: int) -> float:
    """P(X >= wins) for X ~ Binomial(n, 1/2); 1.0 when n = 0."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0


def sign_test(outcomes) -> dict:
    o = list(outcomes)
    w, l = o.count("win"), o.count("loss")
    return {"wins": w, "losses": l, "ties": o.count("tie"), "excluded": o.count("excluded"),
            "n_prime": w + l, "p_one_sided": tail_p(w, w + l)}


def wins_needed(n: int, alpha: float):
    """Smallest win count with p <= alpha at n' = n; None when no count reaches alpha."""
    return next((w for w in range(n + 1) if tail_p(w, n) <= alpha), None)


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """{name: p} -> {name: (threshold, rejected)}. Step-down: once one fails, the rest fail."""
    order = sorted(pvals, key=lambda k: pvals[k])
    out, go = {}, True
    for i, k in enumerate(order):
        thr = alpha / (len(order) - i)
        go = go and pvals[k] <= thr
        out[k] = (thr, go)
    return out
