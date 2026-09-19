"""Barrier options via Crank-Nicolson finite differences (Black-76, Phase 7).

Solves the Black-76 PDE on a grid in (F, t):

    dV/dt + 0.5*sigma^2*F^2*d2V/dF2 - r*V = 0

Substituting tau = T - t turns this into the forward-in-tau diffusion

    dV/dtau = 0.5*sigma^2*F^2*d2V/dF2 - r*V

(no first-derivative/drift term -- F is a driftless martingale under
Black-76), which Crank-Nicolson steps forward from tau=0 (V = payoff, i.e.
t=T) to tau=T (today's price). There's no advection term, so the tridiagonal
system only needs central differences for the second derivative.

The barrier is a boundary condition: for a knock-out, the grid is truncated
at the barrier and V=0 is imposed there for every tau > 0. Knock-in prices
come from in-out parity instead of a second PDE solve:

    knock-in + knock-out = vanilla  =>  knock-in = Black76 price - knock-out
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .base import OptionModel
from .black76 import Black76

_OUT_TYPES = {"up-and-out", "down-and-out"}
_IN_TYPES = {"up-and-in", "down-and-in"}
_BARRIER_TYPES = _OUT_TYPES | _IN_TYPES


def _thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system (Thomas algorithm).

    `lower[i]`/`upper[i]` are row i's sub-/super-diagonal entries (row 0 has
    no sub-diagonal, the last row no super-diagonal -- those slots are
    unused, never read).
    """
    n = len(diag)
    c = upper.copy()
    d = diag.copy()
    b = rhs.copy()
    for i in range(1, n):
        w = lower[i] / d[i - 1]
        d[i] -= w * c[i - 1]
        b[i] -= w * b[i - 1]
    x = np.empty(n)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / d[i]
    return x


@dataclass
class BarrierOption(OptionModel):
    F: float = 0.0             # futures price
    K: float = 0.0             # strike
    T: float = 0.0             # time to expiry (years)
    r: float = 0.0             # risk-free rate (decimal)
    sigma: float = 0.0         # volatility (decimal)
    option_type: str = "call"      # "call" or "put"
    barrier: float = 0.0
    barrier_type: str = "up-and-out"   # up-and-out / down-and-out / up-and-in / down-and-in
    f_steps: int = 200         # grid points in F
    t_steps: int = 200         # time steps
    _underlying: str = field(default="F")

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        self.barrier_type = self.barrier_type.lower()
        if self.barrier_type not in _BARRIER_TYPES:
            raise ValueError(f"barrier_type must be one of {sorted(_BARRIER_TYPES)}")
        is_up = self.barrier_type.startswith("up")
        if is_up and self.barrier <= self.F:
            raise ValueError("an up barrier must be above the current futures price")
        if not is_up and self.barrier >= self.F:
            raise ValueError("a down barrier must be below the current futures price")

    def _intrinsic(self, f):
        return np.maximum(f - self.K, 0.0) if self.option_type == "call" else np.maximum(self.K - f, 0.0)

    # --- knock-out pricer: Crank-Nicolson on the barrier-truncated domain --
    def _knock_out_price(self) -> float:
        is_up = self.barrier_type.startswith("up")
        B = self.barrier

        if is_up:
            # domain [0, B]; F=0 is an absorbing boundary for driftless GBM,
            # so V(0, tau) = exp(-r*tau) * intrinsic(0) is exact.
            f_min, f_max = 0.0, B
        else:
            # domain [B, f_max]; push f_max out with the diffusion's own
            # length scale so it stays a negligible-probability truncation.
            far = max(self.F, self.K, B)
            f_max = far * max(4.0, math.exp(5.0 * self.sigma * math.sqrt(max(self.T, 1e-8))))
            f_min = B

        n, m = self.f_steps, self.t_steps
        dF = (f_max - f_min) / n
        dt = self.T / m
        grid = f_min + dF * np.arange(n + 1)

        V = self._intrinsic(grid)
        if is_up:
            V[-1] = 0.0   # knocked out at the barrier, even at expiry
        else:
            V[0] = 0.0

        a = 0.5 * self.sigma ** 2 * grid ** 2 / dF ** 2   # diffusion coefficient per node
        alpha = 0.5 * dt * a
        beta = 0.5 * dt * self.r

        lower = -alpha[1:n]
        diag = 1.0 + 2.0 * alpha[1:n] + beta
        upper = -alpha[1:n]

        for step in range(1, m + 1):
            tau = step * dt
            if is_up:
                v_low = math.exp(-self.r * tau) * float(self._intrinsic(np.array(0.0)))
                v_high = 0.0
            else:
                v_low = 0.0
                v_high = math.exp(-self.r * tau) * float(self._intrinsic(np.array(f_max)))

            rhs = alpha[1:n] * V[0:n - 1] + (1.0 - 2.0 * alpha[1:n] - beta) * V[1:n] + alpha[1:n] * V[2:n + 1]
            rhs[0] += alpha[1] * v_low
            rhs[-1] += alpha[n - 1] * v_high

            V[1:n] = _thomas_solve(lower, diag, upper, rhs)
            V[0], V[-1] = v_low, v_high

        return float(np.interp(self.F, grid, V))

    def price(self) -> float:
        if self.barrier_type in _OUT_TYPES:
            return self._knock_out_price()
        # in-out parity: knock-in = vanilla - knock-out, same up/down side
        out_type = self.barrier_type.replace("-in", "-out")
        vanilla = Black76(
            F=self.F, K=self.K, T=self.T, r=self.r, sigma=self.sigma, option_type=self.option_type
        ).price()
        knock_out = self._bumped(barrier_type=out_type)._knock_out_price()
        return vanilla - knock_out
