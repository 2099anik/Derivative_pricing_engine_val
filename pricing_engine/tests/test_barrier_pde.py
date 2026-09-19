"""Validation tests for the Crank-Nicolson barrier-option PDE (Phase 7).

Run with:  python -m pytest -q   (or just  python tests/test_barrier_pde.py)
"""

from pricing_engine.models.black76 import Black76
from pricing_engine.models.barrier_pde import BarrierOption

PARAMS = dict(F=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25)

# (barrier_type, matching barrier level) -- close enough to F0 that the PDE's
# knock-out boundary condition is actually doing meaningful work.
BARRIER_CASES = [
    ("up-and-out", 120.0),
    ("down-and-out", 80.0),
]


def test_in_out_parity():
    # knock-in = vanilla - knock-out by construction, so this holds to
    # floating-point precision regardless of grid resolution -- it's a
    # check that the parity wiring (and barrier_type string handling) is
    # correct, not a check of PDE accuracy.
    for option_type in ("call", "put"):
        vanilla = Black76(**PARAMS, option_type=option_type).price()
        for out_type, barrier in BARRIER_CASES:
            in_type = out_type.replace("-out", "-in")
            knock_out = BarrierOption(**PARAMS, option_type=option_type, barrier=barrier, barrier_type=out_type).price()
            knock_in = BarrierOption(**PARAMS, option_type=option_type, barrier=barrier, barrier_type=in_type).price()
            assert abs((knock_in + knock_out) - vanilla) < 1e-9


def test_far_barrier_converges_to_vanilla():
    # A barrier several sigma away from F0 is essentially never touched, so
    # a knock-out there should price almost identically to the vanilla.
    for option_type in ("call", "put"):
        vanilla = Black76(**PARAMS, option_type=option_type).price()
        for out_type, barrier in (("up-and-out", 500.0), ("down-and-out", 10.0)):
            price = BarrierOption(**PARAMS, option_type=option_type, barrier=barrier, barrier_type=out_type).price()
            assert abs(price - vanilla) < 0.01 * vanilla


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")
