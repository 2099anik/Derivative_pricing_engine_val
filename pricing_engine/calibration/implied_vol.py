"""Implied volatility solver for Black-76.

Newton-Raphson (fast, uses vega) with a bisection fallback so it stays
robust for deep ITM/OTM quotes where vega is tiny.
"""

from __future__ import annotations

import math

from ..models.black76 import Black76


def implied_vol(
    market_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
    sigma_init: float = 0.20,
) -> float:
    """Return the Black-76 implied volatility, or nan if it can't be solved."""

    def price_at(sig: float) -> float:
        return Black76(F=F, K=K, T=T, r=r, sigma=sig, option_type=option_type).price()

    # --- Newton-Raphson ---------------------------------------------------
    sigma = sigma_init
    for _ in range(max_iter):
        model = Black76(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
        diff = model.price() - market_price
        if abs(diff) < tol:
            return sigma
        v = model.vega()
        if v < 1e-10:
            break  # vega too small — hand off to bisection
        sigma -= diff / v
        if sigma <= 0:
            break

    # --- bisection fallback ----------------------------------------------
    lo, hi = 1e-6, 5.0
    if (price_at(lo) - market_price) * (price_at(hi) - market_price) > 0:
        return float("nan")  # target price not bracketed
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if abs(price_at(mid) - market_price) < tol:
            return mid
        if (price_at(lo) - market_price) * (price_at(mid) - market_price) < 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
