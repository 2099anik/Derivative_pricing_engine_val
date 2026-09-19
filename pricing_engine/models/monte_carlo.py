"""Monte Carlo engine for path-dependent and multi-asset futures options
(Black-76, Phase 6).

Futures are driftless martingales, so every path is simulated with the
*exact* (no discretization bias) lognormal step:

    F_t = F_0 * exp(-0.5*sigma^2*t + sigma*sqrt(dt)*Z),  Z ~ N(0,1) i.i.d.

applied cumulatively step by step, and every price is
``exp(-r*T) * mean(payoff)``. Antithetic variates (Z and -Z paired up) cut
variance for free since the payoffs here are monotone-ish in Z, and every
price comes with a 95% confidence interval computed from the sample
standard error of the discounted payoffs.

Classes:
    AsianOption   -- fixed-strike, arithmetic- or geometric-average price.
    LookbackOption -- floating-strike (payoff keyed off the path min/max).
    BasketOption  -- fixed-strike option on a weighted basket of futures,
                     correlated via Cholesky factorization of a correlation
                     matrix.

All three only implement price(), so delta/gamma/vega/theta/rho come free
from OptionModel's finite-difference bumps. Because every subclass reuses
the same `seed`, a bump and its unbumped counterpart draw identical random
numbers (common random numbers), which is what keeps the FD Greeks from
being swamped by Monte Carlo noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np

from .base import OptionModel
from .black76 import Black76

_Z_95 = NormalDist().inv_cdf(0.975)  # two-sided 95% normal quantile


def _sign(option_type: str) -> float:
    return 1.0 if option_type == "call" else -1.0


def _confidence_interval(discounted_payoffs: np.ndarray) -> tuple:
    """(price, ci_low, ci_high) for a 95% CI from a sample of discounted payoffs."""
    n = discounted_payoffs.size
    price = float(discounted_payoffs.mean())
    se = float(discounted_payoffs.std(ddof=1)) / math.sqrt(n)
    half_width = _Z_95 * se
    return price, price - half_width, price + half_width


def geometric_asian_price(
    F: float, K: float, T: float, r: float, sigma: float, n_steps: int, option_type: str = "call",
) -> float:
    """Closed-form price of a geometric-average Asian option on a driftless
    Black-76 futures process, monitored at `n_steps` equally spaced dates.

    ln(geometric average) is normal, so the payoff reduces to a Black-76
    price on an adjusted (forward, vol) pair. As n_steps -> infinity this
    converges to the classic continuous-averaging result
    F*exp(-sigma^2*T/12), sigma/sqrt(3).
    """
    n = n_steps
    mean_shift = -sigma ** 2 * T * (n + 1) / (4 * n)
    var_ln = sigma ** 2 * T * (n + 1) * (2 * n + 1) / (6 * n * n)
    sigma_eff = math.sqrt(var_ln / T)
    F_eff = F * math.exp(mean_shift + 0.5 * var_ln)
    return Black76(F=F_eff, K=K, T=T, r=r, sigma=sigma_eff, option_type=option_type).price()


class _MonteCarloResult:
    """Shared 95% CI bookkeeping for the Monte Carlo pricers below."""

    def _finalize(self, discounted_payoffs: np.ndarray) -> float:
        price, lo, hi = _confidence_interval(discounted_payoffs)
        self._ci = (lo, hi)
        return price

    def confidence_interval(self) -> tuple:
        """95% CI for the price, from the paths drawn by the last price() call."""
        if not hasattr(self, "_ci"):
            self.price()
        return self._ci


@dataclass
class _MonteCarloOption(OptionModel, _MonteCarloResult):
    """Shared fields + path simulator for the single-underlying MC pricers."""

    F: float = 0.0           # futures price
    K: float = 0.0           # strike
    T: float = 0.0           # time to expiry (years)
    r: float = 0.0           # risk-free rate (decimal)
    sigma: float = 0.0       # volatility (decimal)
    option_type: str = "call"    # "call" or "put"
    n_paths: int = 100_000
    n_steps: int = 50
    seed: int = 0
    antithetic: bool = True
    _underlying: str = field(default="F")

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    def _simulate_paths(self) -> np.ndarray:
        """Driftless GBM futures paths, shape (n_sim_paths, n_steps+1); column 0 is F0."""
        dt = self.T / self.n_steps
        half = self.n_paths // 2 if self.antithetic else self.n_paths
        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal((half, self.n_steps))
        if self.antithetic:
            z = np.concatenate([z, -z], axis=0)
        increments = -0.5 * self.sigma ** 2 * dt + self.sigma * math.sqrt(dt) * z
        log_paths = np.cumsum(increments, axis=1)
        paths = self.F * np.exp(log_paths)
        f0 = np.full((paths.shape[0], 1), self.F)
        return np.concatenate([f0, paths], axis=1)


@dataclass
class AsianOption(_MonteCarloOption):
    """Fixed-strike Asian option. `average="arithmetic"` (default) or "geometric".

    With n_steps=1 the "average" is just F_T, i.e. a plain European payoff --
    useful for checking the MC engine itself against Black-76.
    """

    average: str = "arithmetic"

    def __post_init__(self):
        super().__post_init__()
        if self.average not in ("arithmetic", "geometric"):
            raise ValueError("average must be 'arithmetic' or 'geometric'")

    def price(self) -> float:
        samples = self._simulate_paths()[:, 1:]  # the n_steps monitoring dates
        if self.average == "arithmetic":
            avg = samples.mean(axis=1)
        else:
            avg = np.exp(np.log(samples).mean(axis=1))
        payoff = np.maximum(_sign(self.option_type) * (avg - self.K), 0.0)
        discounted = math.exp(-self.r * self.T) * payoff
        return self._finalize(discounted)


@dataclass
class LookbackOption(_MonteCarloOption):
    """Floating-strike lookback: payoff is keyed off the path min/max, not K."""

    def price(self) -> float:
        paths = self._simulate_paths()
        terminal = paths[:, -1]
        if self.option_type == "call":
            payoff = terminal - paths.min(axis=1)
        else:
            payoff = paths.max(axis=1) - terminal
        discounted = math.exp(-self.r * self.T) * payoff
        return self._finalize(discounted)


@dataclass
class BasketOption(OptionModel, _MonteCarloResult):
    """Fixed-strike option on a weighted basket of correlated futures.

    F, sigma, weights are per-asset sequences of equal length; corr is their
    correlation matrix (Cholesky-factorized to correlate the driftless GBM
    draws). Weights default to equal weighting. The basket payoff only
    depends on the terminal value, so n_steps is purely a compute/precision
    knob, not a source of discretization bias (each step is an exact
    lognormal increment).
    """

    F: Sequence[float] = (0.0,)
    sigma: Sequence[float] = (0.0,)
    corr: Sequence[Sequence[float]] = ((1.0,),)
    weights: Optional[Sequence[float]] = None
    K: float = 0.0
    T: float = 0.0
    r: float = 0.0
    option_type: str = "call"
    n_paths: int = 100_000
    n_steps: int = 1
    seed: int = 0
    antithetic: bool = True
    _underlying: str = field(default="F")

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        n = len(self.F)
        if self.weights is None:
            self.weights = tuple(1.0 / n for _ in range(n))
        if len(self.sigma) != n or len(self.weights) != n or len(self.corr) != n or any(
            len(row) != n for row in self.corr
        ):
            raise ValueError("F, sigma, weights and corr must all have consistent length n_assets")

    # delta/gamma bump a single scalar field (F) via _u()/_bumped(); F is a
    # per-leg tuple here, so reinterpret the bump as a uniform parallel
    # shift of every leg, centered on the tuple's average -- delta reads as
    # sensitivity to a basket-wide forward move.
    def _u(self) -> float:
        return sum(self.F) / len(self.F)

    def _bumped(self, **changes) -> "BasketOption":
        for key, new_value in list(changes.items()):
            current = getattr(self, key)
            if isinstance(current, tuple) and not isinstance(new_value, tuple):
                shift = new_value - sum(current) / len(current)
                changes[key] = tuple(v + shift for v in current)
        return super()._bumped(**changes)

    # base.vega() bumps self.sigma directly (self.sigma + h), which assumes
    # a scalar; sigma is a per-leg tuple here, so bump every leg uniformly.
    def vega(self) -> float:
        h = 1e-4
        up = self._bumped(sigma=tuple(s + h for s in self.sigma)).price()
        dn = self._bumped(sigma=tuple(s - h for s in self.sigma)).price()
        return (up - dn) / (2 * h)

    def price(self) -> float:
        n_assets = len(self.F)
        F0 = np.asarray(self.F, dtype=float)
        sigmas = np.asarray(self.sigma, dtype=float)
        L = np.linalg.cholesky(np.asarray(self.corr, dtype=float))

        dt = self.T / self.n_steps
        half = self.n_paths // 2 if self.antithetic else self.n_paths
        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal((half, self.n_steps, n_assets)) @ L.T
        if self.antithetic:
            z = np.concatenate([z, -z], axis=0)

        log_moves = (-0.5 * sigmas ** 2 * dt + sigmas * math.sqrt(dt) * z).sum(axis=1)
        terminal = F0 * np.exp(log_moves)                       # (n_sim, n_assets)
        basket = terminal @ np.asarray(self.weights, dtype=float)

        payoff = np.maximum(_sign(self.option_type) * (basket - self.K), 0.0)
        discounted = math.exp(-self.r * self.T) * payoff
        return self._finalize(discounted)
