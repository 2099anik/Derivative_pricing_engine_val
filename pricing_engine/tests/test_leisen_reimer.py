"""Validation tests for the Leisen-Reimer tree (Phase 4b).

Run with:  python -m pytest -q   (or just  python tests/test_leisen_reimer.py)
"""

from pricing_engine.models.black76 import Black76
from pricing_engine.models.crr import CRRBinomial
from pricing_engine.models.leisen_reimer import LeisenReimer

PARAMS = dict(F=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25)


def test_leisen_reimer_converges_to_black76():
    for option_type in ("call", "put"):
        euro = Black76(**PARAMS, option_type=option_type).price()
        for n in (51, 101, 201):
            lr = LeisenReimer(**PARAMS, option_type=option_type, n_steps=n, american=False).price()
            assert abs(lr - euro) < 1e-3


def test_leisen_reimer_faster_than_crr_at_equal_steps():
    # Same step count, both European (no early exercise): LR's Peizer-Pratt
    # probabilities track Black-76 far more tightly than CRR's sqrt(dt)
    # parametrization, which oscillates and only converges at O(1/n).
    for option_type in ("call", "put"):
        euro = Black76(**PARAMS, option_type=option_type).price()
        for n in (25, 51, 101, 201):
            crr_err = abs(
                CRRBinomial(**PARAMS, option_type=option_type, n_steps=n, american=False).price() - euro
            )
            lr_err = abs(
                LeisenReimer(**PARAMS, option_type=option_type, n_steps=n, american=False).price() - euro
            )
            assert lr_err < crr_err


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")
