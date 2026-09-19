"""Scenario analysis engine (Phase 9).

Reprices any OptionModel -- or a portfolio (list) of them -- under the
spec's shock grid and returns a tidy table of (shock, price, P&L vs base).
Every shock is just OptionModel._bumped() with one field replaced, so the
same grid works uniformly across every model in the engine (analytic,
tree, Monte Carlo, PDE), plus product-specific shocks layered on top:
barrier-level shocks for BarrierOption, correlation shocks for
BasketOption (detected via hasattr, no per-model-type branching needed).

Shock grid:
    F        +/- 1%, 5%, 10%
    vol      +/- 5, 10 vol points
    rate     +/- 50, 100 bps
    time     -1, -7, -30 days (decay only; T floored at 0)
    barrier  +/- 1%, 5%, 10%   (BarrierOption only)
    corr     +/- 0.10, 0.20   (BasketOption only, off-diagonal entries)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd

from ..models.base import OptionModel

Mutator = Callable[[OptionModel], OptionModel]

F_PCT_SHOCKS = (0.01, 0.05, 0.10)
VOL_POINT_SHOCKS = (0.05, 0.10)
RATE_BPS_SHOCKS = (0.0050, 0.0100)
TIME_DECAY_DAYS = (1, 7, 30)
BARRIER_PCT_SHOCKS = (0.01, 0.05, 0.10)
CORR_SHOCKS = (0.10, 0.20)


def _relative(value, pct: float):
    """Scale `value` by (1+pct); handles both a scalar field and a per-leg tuple (basket)."""
    if isinstance(value, tuple):
        return tuple(v * (1.0 + pct) for v in value)
    return value * (1.0 + pct)


def _absolute(value, delta: float):
    """Shift `value` by `delta`; handles both a scalar field and a per-leg tuple (basket)."""
    if isinstance(value, tuple):
        return tuple(v + delta for v in value)
    return value + delta


def _shift_corr(corr, delta: float):
    """Shift every off-diagonal correlation by `delta`, clipped to stay a valid correlation."""
    n = len(corr)
    return tuple(
        tuple(1.0 if i == j else min(0.999, max(-0.999, corr[i][j] + delta)) for j in range(n))
        for i in range(n)
    )


def _signed(magnitude: float, sign: int) -> float:
    return magnitude * sign


def _shock_grid(model: OptionModel) -> List[Tuple[str, Mutator]]:
    shocks: List[Tuple[str, Mutator]] = [("Base (0 shock)", lambda m: m._bumped())]

    for pct in F_PCT_SHOCKS:
        for sign, sym in ((1, "+"), (-1, "-")):
            shocks.append((f"F {sym}{pct:.0%}", lambda m, pct=_signed(pct, sign): m._bumped(F=_relative(m.F, pct))))

    for vp in VOL_POINT_SHOCKS:
        for sign, sym in ((1, "+"), (-1, "-")):
            shocks.append(
                (f"vol {sym}{vp * 100:.0f}vp", lambda m, d=_signed(vp, sign): m._bumped(sigma=_absolute(m.sigma, d)))
            )

    for bps in RATE_BPS_SHOCKS:
        for sign, sym in ((1, "+"), (-1, "-")):
            shocks.append(
                (f"rate {sym}{bps * 10_000:.0f}bps", lambda m, d=_signed(bps, sign): m._bumped(r=m.r + d))
            )

    for days in TIME_DECAY_DAYS:
        shocks.append((f"time -{days}d", lambda m, d=days: m._bumped(T=max(m.T - d / 365.0, 0.0))))

    if hasattr(model, "barrier"):
        for pct in BARRIER_PCT_SHOCKS:
            for sign, sym in ((1, "+"), (-1, "-")):
                shocks.append(
                    (f"barrier {sym}{pct:.0%}", lambda m, pct=_signed(pct, sign): m._bumped(barrier=m.barrier * (1.0 + pct)))
                )

    if hasattr(model, "corr"):
        for delta in CORR_SHOCKS:
            for sign, sym in ((1, "+"), (-1, "-")):
                shocks.append(
                    (f"corr {sym}{delta:.2f}", lambda m, d=_signed(delta, sign): m._bumped(corr=_shift_corr(m.corr, d)))
                )

    return shocks


def scenario_table(model: OptionModel, shock_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Tidy (shock, price, pnl) table for one model.

    `shock_names`, if given, fixes the row set (used by portfolio_scenario_table
    to align instruments with different natural grids); a shock not applicable
    to this model leaves it unchanged (price=base, pnl=0).
    """
    grid = dict(_shock_grid(model))
    names = list(grid.keys()) if shock_names is None else list(shock_names)
    base_price = model.price()

    rows = []
    for name in names:
        mutate = grid.get(name)
        price = mutate(model).price() if mutate is not None else base_price
        rows.append({"shock": name, "price": price, "pnl": price - base_price})
    return pd.DataFrame(rows, columns=["shock", "price", "pnl"])


def portfolio_scenario_table(models: Sequence[OptionModel], positions: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Position-weighted sum of scenario_table() across a portfolio of models."""
    positions = list(positions) if positions is not None else [1.0] * len(models)
    if len(positions) != len(models):
        raise ValueError("positions must have the same length as models")

    all_shocks: List[str] = []
    seen = set()
    for model in models:
        for name in dict(_shock_grid(model)).keys():
            if name not in seen:
                seen.add(name)
                all_shocks.append(name)

    total: Optional[pd.DataFrame] = None
    for model, position in zip(models, positions):
        t = scenario_table(model, shock_names=all_shocks)
        t["price"] *= position
        t["pnl"] *= position
        total = t if total is None else total.assign(price=total["price"] + t["price"], pnl=total["pnl"] + t["pnl"])
    return total
