"""Validation tests for the CRR binomial tree (Phase 4).

Run with:  python -m pytest -q   (or just  python tests/test_crr.py)
"""

from pricing_engine.models.black76 import Black76
from pricing_engine.models.crr import CRRBinomial

PARAMS = dict(F=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25)


def test_american_geq_european_call():
    euro = Black76(**PARAMS, option_type="call").price()
    amer = CRRBinomial(**PARAMS, option_type="call", n_steps=500, american=True).price()
    assert amer >= euro - 1e-9


def test_american_geq_european_put():
    euro = Black76(**PARAMS, option_type="put").price()
    amer = CRRBinomial(**PARAMS, option_type="put", n_steps=500, american=True).price()
    assert amer >= euro - 1e-9


def test_crr_converges_to_black76_without_early_exercise():
    # With early exercise disabled, CRR prices the European contract, so its
    # error against the closed-form Black-76 price should shrink as the tree
    # gets finer.
    for option_type in ("call", "put"):
        euro = Black76(**PARAMS, option_type=option_type).price()
        diffs = []
        for n in (50, 200, 800, 3200):
            crr = CRRBinomial(**PARAMS, option_type=option_type, n_steps=n, american=False).price()
            diffs.append(abs(crr - euro))
        assert diffs[-1] < 1e-3
        assert diffs[-1] < diffs[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")
