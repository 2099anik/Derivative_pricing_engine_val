"""Validation tests for the Monte Carlo engine (Phase 6).

Run with:  python -m pytest -q   (or just  python tests/test_monte_carlo.py)
"""

from pricing_engine.models.black76 import Black76
from pricing_engine.models.monte_carlo import AsianOption, geometric_asian_price

PARAMS = dict(F=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25)


def test_mc_european_payoff_matches_black76_within_ci():
    # n_steps=1 collapses the "average" to a single observation F_T, i.e. a
    # plain European payoff -- so this checks the MC engine itself against
    # the closed-form Black-76 price.
    for option_type in ("call", "put"):
        euro = Black76(**PARAMS, option_type=option_type).price()
        mc = AsianOption(**PARAMS, option_type=option_type, n_steps=1, n_paths=200_000, seed=7)
        price = mc.price()
        lo, hi = mc.confidence_interval()
        assert lo < hi
        assert lo <= euro <= hi


def test_geometric_asian_matches_closed_form():
    n_steps = 50
    for option_type in ("call", "put"):
        analytic = geometric_asian_price(
            PARAMS["F"], PARAMS["K"], PARAMS["T"], PARAMS["r"], PARAMS["sigma"], n_steps, option_type
        )
        mc = AsianOption(
            **PARAMS, option_type=option_type, average="geometric",
            n_steps=n_steps, n_paths=200_000, seed=11,
        )
        price = mc.price()
        lo, hi = mc.confidence_interval()
        assert lo < hi
        assert lo <= analytic <= hi


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")
