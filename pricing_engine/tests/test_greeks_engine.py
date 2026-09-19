"""Validation tests for the Greeks aggregation engine (Phase 8).

Run with:  python -m pytest -q   (or just  python tests/test_greeks_engine.py)
"""

from pricing_engine.models.black76 import Black76
from pricing_engine.models.crr import CRRBinomial
from pricing_engine.models.monte_carlo import AsianOption
from pricing_engine.greeks import greeks_table

PARAMS = dict(F=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25)


def _models():
    return [
        Black76(**PARAMS, option_type="call"),
        Black76(**PARAMS, option_type="put"),
        CRRBinomial(**PARAMS, option_type="call", n_steps=200),
        AsianOption(**PARAMS, option_type="put", n_steps=20, n_paths=20_000, seed=5),
    ]


def test_table_matches_each_models_own_greeks():
    # AsianOption is Monte Carlo, but its seed is a dataclass field, so
    # calling .greeks() again here replays identical random draws --
    # the comparison should be exact, not just approximate.
    models = _models()
    table = greeks_table(models)
    assert len(table) == len(models)
    for model, (_, row) in zip(models, table.iterrows()):
        expected = model.greeks()
        for key, value in expected.items():
            assert abs(row[key] - value) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")
