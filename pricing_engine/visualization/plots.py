"""Risk visualization (Phase 10): matplotlib figures saved as PNGs.

Every function takes an already-built OptionModel instance (any model in
the engine -- analytic, tree, Monte Carlo, PDE) and reprices it across a
sweep via OptionModel._bumped()/._u(), the same generic mechanism used by
the scenario and Greeks engines. That's what lets a single implementation
handle both a scalar-underlying model and BasketOption's per-leg tuple
without branching on type.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe: this module's job is to save PNGs, not pop up windows

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from ..greeks import greeks_surface, plot_greeks_heatmap
from ..models.base import OptionModel
from ..scenario_analysis import scenario_table

FIGURES_DIR_DEFAULT = "figures"


def _scalar(value):
    """Average a per-leg tuple field (basket) down to one number for display."""
    return sum(value) / len(value) if isinstance(value, tuple) else value


def _save(fig, filename: str, figures_dir: str) -> str:
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- option payoff diagram --------------------------------------------------
def payoff_diagram(
    model: OptionModel, F_range: Optional[Tuple[float, float]] = None, n: int = 200,
    filename: str = "payoff_diagram.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    u0 = model._u()
    if F_range is None:
        F_range = (0.5 * u0, 1.5 * u0)
    Fs = np.linspace(*F_range, n)
    payoff = [model._bumped(F=f, T=0.0).price() for f in Fs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Fs, payoff, color="tab:blue", label="payoff at expiry")
    ax.axvline(u0, color="gray", linestyle="--", linewidth=1, label=f"current F={u0:g}")
    if hasattr(model, "K"):
        ax.axvline(model.K, color="tab:red", linestyle=":", linewidth=1, label=f"K={model.K:g}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Futures price F at expiry")
    ax.set_ylabel("Payoff")
    ax.set_title(f"{type(model).__name__} payoff diagram  (price today = {model.price():.4f})")
    ax.legend()
    return _save(fig, filename, figures_dir)


# --- P&L surface: price x time ----------------------------------------------
def pnl_surface(
    model: OptionModel, F_range: Optional[Tuple[float, float]] = None,
    T_range: Optional[Tuple[float, float]] = None, n_F: int = 40, n_T: int = 40,
    filename: str = "pnl_surface.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    u0 = model._u()
    base_price = model.price()
    if F_range is None:
        F_range = (0.6 * u0, 1.4 * u0)
    if T_range is None:
        T_range = (max(model.T * 0.02, 1e-4), model.T)

    Fs = np.linspace(*F_range, n_F)
    Ts = np.linspace(*T_range, n_T)
    FF, TT = np.meshgrid(Fs, Ts)
    pnl = np.array(
        [model._bumped(F=f, T=t).price() - base_price for f, t in zip(FF.ravel(), TT.ravel())]
    ).reshape(FF.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(FF, TT, pnl, cmap="viridis")
    ax.set_xlabel("Futures price F")
    ax.set_ylabel("Time to expiry T (years)")
    ax.set_zlabel("P&L vs base")
    ax.set_title(f"{type(model).__name__} P&L surface (price x time)")
    return _save(fig, filename, figures_dir)


# --- Greeks heatmap across strikes ------------------------------------------
def greeks_heatmap(
    model: OptionModel, strikes: Optional[Sequence[float]] = None,
    filename: str = "greeks_heatmap.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    u0 = model._u()
    if strikes is None:
        strikes = np.linspace(0.7 * u0, 1.3 * u0, 9)
    surface = greeks_surface(model, strikes)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_greeks_heatmap(surface, ax=ax)
    ax.set_title(f"{type(model).__name__} Greeks heatmap across strikes")
    return _save(fig, filename, figures_dir)


# --- delta-gamma approximation vs actual reprice ----------------------------
def delta_gamma_vs_actual(
    model: OptionModel, dF_range: Optional[Tuple[float, float]] = None, n: int = 100,
    filename: str = "delta_gamma_vs_actual.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    u0 = model._u()
    base_price = model.price()
    delta, gamma = model.delta(), model.gamma()
    if dF_range is None:
        dF_range = (-0.3 * u0, 0.3 * u0)
    dFs = np.linspace(*dF_range, n)
    actual = [model._bumped(F=u0 + dF).price() for dF in dFs]
    approx = [base_price + delta * dF + 0.5 * gamma * dF ** 2 for dF in dFs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(u0 + dFs, actual, color="tab:blue", label="actual reprice")
    ax.plot(u0 + dFs, approx, color="tab:red", linestyle="--", label="delta-gamma approx")
    ax.axvline(u0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Futures price F")
    ax.set_ylabel("Option price")
    ax.set_title(f"{type(model).__name__}: delta-gamma approximation vs actual reprice")
    ax.legend()
    return _save(fig, filename, figures_dir)


# --- scenario shock bar chart ------------------------------------------------
def scenario_shock_bars(
    model: OptionModel, filename: str = "scenario_shocks.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    table = scenario_table(model)
    table = table[table["shock"] != "Base (0 shock)"]
    colors = ["tab:green" if p >= 0 else "tab:red" for p in table["pnl"]]

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.3 * len(table))))
    ax.barh(table["shock"], table["pnl"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()  # first shock at the top
    ax.set_xlabel("P&L vs base")
    ax.set_title(f"{type(model).__name__} scenario shock P&L")
    return _save(fig, filename, figures_dir)


# --- Black-76 spec charts ----------------------------------------------------
def price_vs_F(
    model: OptionModel, F_range: Optional[Tuple[float, float]] = None, n: int = 200,
    filename: str = "price_vs_F.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    u0 = model._u()
    if F_range is None:
        F_range = (0.5 * u0, 1.5 * u0)
    Fs = np.linspace(*F_range, n)
    prices = [model._bumped(F=f).price() for f in Fs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Fs, prices, color="tab:blue")
    ax.axvline(u0, color="gray", linestyle="--", linewidth=1, label=f"current F={u0:g}")
    ax.set_xlabel("Futures price F")
    ax.set_ylabel("Option price")
    ax.set_title(f"{type(model).__name__} price vs F  (K={_scalar(model.K):g}, T={model.T:.3f}, "
                 f"sigma={_scalar(model.sigma):.2%})")
    ax.legend()
    return _save(fig, filename, figures_dir)


def delta_vs_F(
    model: OptionModel, F_range: Optional[Tuple[float, float]] = None, n: int = 200,
    filename: str = "delta_vs_F.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    u0 = model._u()
    if F_range is None:
        F_range = (0.5 * u0, 1.5 * u0)
    Fs = np.linspace(*F_range, n)
    deltas = [model._bumped(F=f).delta() for f in Fs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Fs, deltas, color="tab:green")
    ax.axvline(u0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Futures price F")
    ax.set_ylabel("Delta")
    ax.set_title(f"{type(model).__name__} delta vs F")
    return _save(fig, filename, figures_dir)


def vega_vs_vol(
    model: OptionModel, vol_range: Optional[Tuple[float, float]] = None, n: int = 200,
    filename: str = "vega_vs_vol.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    s0 = _scalar(model.sigma)
    if vol_range is None:
        vol_range = (max(0.01, 0.2 * s0), 2.0 * s0)
    vols = np.linspace(*vol_range, n)
    vegas = [model._bumped(sigma=v).vega() / 100.0 for v in vols]  # per 1 vol point

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(vols * 100, vegas, color="tab:purple")
    ax.axvline(s0 * 100, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Vega (per 1 vol point)")
    ax.set_title(f"{type(model).__name__} vega vs vol")
    return _save(fig, filename, figures_dir)


def theta_decay(
    model: OptionModel, n: int = 200, filename: str = "theta_decay.png", figures_dir: str = FIGURES_DIR_DEFAULT,
) -> str:
    T0 = model.T
    Ts = np.linspace(max(T0 * 0.001, 1e-6), T0, n)
    prices = [model._bumped(T=t).price() for t in Ts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ts * 365.0, prices, color="tab:orange")
    ax.invert_xaxis()  # time flows left (far from expiry) to right (expiry)
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("Option price")
    ax.set_title(f"{type(model).__name__} theta decay: price vs. days to expiry")
    return _save(fig, filename, figures_dir)
