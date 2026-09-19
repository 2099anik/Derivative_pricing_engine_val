"""Greeks aggregation engine (Phase 8).

Any list of OptionModel instances -- analytic, tree, Monte Carlo, PDE, it
doesn't matter which -- can be combined into one risk view, because they
all share the same interface: .greeks() -> price + delta/gamma/vega/theta/rho.

    greeks_table(models)             -- one row per instrument
    greeks_surface(model, strikes)   -- that instrument's Greeks recomputed
                                         across a range of strikes, for a heatmap
    plot_greeks_heatmap(surface)     -- matplotlib heatmap of that surface
    sensitivity_report(models, ...)  -- per-instrument + portfolio-total risk,
                                         in per-convention units
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ..models.base import OptionModel

_GREEK_COLUMNS = ["price", "delta", "gamma", "vega", "theta", "rho"]


def greeks_table(models: Sequence[OptionModel]) -> pd.DataFrame:
    """One row per model (price + all Greeks), labeled by the model's own repr."""
    records = [{"instrument": repr(m), **m.greeks()} for m in models]
    return pd.DataFrame.from_records(records, columns=["instrument"] + _GREEK_COLUMNS).set_index("instrument")


def greeks_surface(
    model: OptionModel, strikes: Sequence[float], greek_names: Sequence[str] = _GREEK_COLUMNS
) -> pd.DataFrame:
    """`model`'s Greeks recomputed across a range of strikes K (rows=K, cols=Greek)."""
    rows = [model._bumped(K=k).greeks() for k in strikes]
    return pd.DataFrame(rows, index=pd.Index(list(strikes), name="K"))[list(greek_names)]


def plot_greeks_heatmap(surface: pd.DataFrame, ax=None):
    """Render a strikes x Greeks heatmap from greeks_surface()'s output."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(surface.values, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(surface.columns)))
    ax.set_xticklabels(surface.columns)
    ax.set_yticks(range(len(surface.index)))
    ax.set_yticklabels([f"{k:g}" for k in surface.index])
    ax.set_xlabel("Greek")
    ax.set_ylabel("Strike")
    ax.figure.colorbar(im, ax=ax, label="value")
    return ax


def sensitivity_report(models: Sequence[OptionModel], positions: Optional[Sequence[float]] = None) -> str:
    """Per-instrument + portfolio-total Greeks, in per-convention units
    (vega per 1 vol point, theta per calendar day, rho per 1%), scaled by
    each instrument's position size (default 1 each).
    """
    positions = list(positions) if positions is not None else [1.0] * len(models)
    if len(positions) != len(models):
        raise ValueError("positions must have the same length as models")

    display = greeks_table(models).copy()
    display["vega"] /= 100.0
    display["theta"] /= 365.0
    display["rho"] /= 100.0
    display.insert(0, "position", positions)
    for col in _GREEK_COLUMNS:
        display[col] = display[col].to_numpy() * display["position"].to_numpy()

    totals = display[["position"] + _GREEK_COLUMNS].sum()
    totals.name = "TOTAL"

    lines = [
        "Sensitivity report (vega/1 vol pt, theta/day, rho/1%)",
        "=" * 56,
        display.to_string(float_format=lambda x: f"{x:,.4f}"),
        "-" * 56,
        totals.to_string(float_format=lambda x: f"{x:,.4f}"),
    ]
    return "\n".join(lines)
