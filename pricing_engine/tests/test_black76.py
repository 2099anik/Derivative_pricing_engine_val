"""Validation tests for the Black-76 core (Phase 13 starts here).

Run with:  python -m pytest -q   (or just  python tests/test_black76.py)
"""

import math

from pricing_engine.models.black76 import Black76
from pricing_engine.calibration.implied_vol import implied_vol

TOL = 1e-6


def _params(option_type="call"):
    return dict(F=100.0, K=100.0, T=0.5, r=0.03, sigma=0.25, option_type=option_type)


def test_put_call_parity():
    # C - P = exp(-rT) * (F - K)
    c = Black76(**_params("call")).price()
    p = Black76(**_params("put")).price()
    lhs = c - p
    rhs = math.exp(-0.03 * 0.5) * (100.0 - 100.0)
    assert abs(lhs - rhs) < TOL


def test_delta_signs():
    assert Black76(**_params("call")).delta() > 0
    assert Black76(**_params("put")).delta() < 0


def test_gamma_vega_positive():
    m = Black76(**_params("call"))
    assert m.gamma() > 0
    assert m.vega() > 0


def test_implied_vol_roundtrip():
    true_sigma = 0.32
    p = _params("call")
    p["sigma"] = true_sigma
    price = Black76(**p).price()
    iv = implied_vol(price, p["F"], p["K"], p["T"], p["r"], "call")
    assert abs(iv - true_sigma) < 1e-6


def test_analytic_matches_finite_difference():
    """Analytic Greeks should match the base-class FD Greeks closely."""
    m = Black76(**_params("call"))
    # base-class FD versions via unbound methods
    from pricing_engine.models.base import OptionModel
    fd_delta = OptionModel.delta(m)
    fd_gamma = OptionModel.gamma(m)
    fd_vega = OptionModel.vega(m)
    fd_rho = OptionModel.rho(m)
    assert abs(m.delta() - fd_delta) < 1e-4
    assert abs(m.gamma() - fd_gamma) < 1e-3
    assert abs(m.vega() - fd_vega) < 1e-3
    assert abs(m.rho() - fd_rho) < 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")
