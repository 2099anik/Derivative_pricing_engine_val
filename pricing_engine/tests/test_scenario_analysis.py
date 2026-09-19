"""Validation tests for the scenario analysis engine (Phase 9).

Run with:  python -m pytest -q   (or just  python tests/test_scenario_analysis.py)
"""

from pricing_engine.models.black76 import Black76
from pricing_engine.models.crr import CRRBinomial
from pricing_engine.models.monte_carlo import AsianOption
from pricing_engine.scenario_analysis import scenario_table

PARAMS = dict(F=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25)


def _models():
    return [
        Black76(**PARAMS, option_type="call"),
        Black76(**PARAMS, option_type="put"),
        CRRBinomial(**PARAMS, option_type="put", n_steps=200),
        AsianOption(**PARAMS, option_type="call", n_steps=20, n_paths=20_000, seed=3),
    ]


def test_zero_shock_reproduces_base_price_exactly():
    # _bumped() with no changes is a full copy with identical fields (same
    # random seed for the Monte Carlo model too), so it should reprice
    # bit-for-bit identically to model.price(), not just approximately.
    for model in _models():
        table = scenario_table(model)
        base_row = table[table["shock"] == "Base (0 shock)"].iloc[0]
        assert base_row["price"] == model.price()
        assert base_row["pnl"] == 0.0


def test_small_underlying_shock_pnl_matches_delta():
    # For a small move, P&L should be close to delta * dF; the residual is
    # the second-order (gamma) term, which is small for a 1% move.
    for model in _models():
        table = scenario_table(model)
        base_price = model.price()
        delta = model.delta()
        row = table[table["shock"] == "F +1%"].iloc[0]
        dF = model.F * 0.01
        linear_estimate = delta * dF
        assert abs(row["pnl"] - linear_estimate) < 0.05
        assert row["price"] == base_price + row["pnl"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")
