"""Leisen-Reimer binomial tree: faster-converging tree pricer (Phase 4b).

Same driftless-futures setup as the CRR tree (models/crr.py), but u/d/p
come from the Peizer-Pratt inversion of the terminal Black-76 d1/d2
instead of the sqrt(dt) CRR parametrization:

    d1 = (ln(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T));  d2 = d1 - sigma*sqrt(T)
    p  = PeizerPratt(d2, n)          # risk-neutral up probability
    p' = PeizerPratt(d1, n)
    u  = p'/p                        # driftless -> no exp(r*dt) growth term
    d  = (1-p')/(1-p)

Backward induction is identical to CRR: discount each step at exp(-r*dt)
and, when early exercise is allowed, take max(continuation, intrinsic) at
every node. The Peizer-Pratt inversion makes the tree match the target
d1/d2 by construction, which removes CRR's O(1/n) oscillation and gives
O(1/n^2) convergence -- so LR needs far fewer steps for the same accuracy
on a European contract. ``n_steps`` must be odd (bumped up by one if even)
so the tree is centered on the strike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import OptionModel


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt inversion (method 2) mapping z into a probability in (0, 1)."""
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    inner = -((z / denom) ** 2) * (n + 1.0 / 6.0)
    sign = 1.0 if z >= 0 else -1.0
    return 0.5 + sign * math.sqrt(max(0.25 - 0.25 * math.exp(inner), 0.0))


@dataclass
class LeisenReimer(OptionModel):
    F: float = 0.0          # futures price
    K: float = 0.0          # strike
    T: float = 0.0          # time to expiry (years)
    r: float = 0.0          # risk-free rate (decimal)
    sigma: float = 0.0      # volatility (decimal)
    option_type: str = "call"   # "call" or "put"
    n_steps: int = 101
    american: bool = True   # False => European (no early exercise), for convergence checks
    _underlying: str = field(default="F")

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.n_steps % 2 == 0:
            self.n_steps += 1   # LR requires an odd step count

    def _intrinsic(self, f: float) -> float:
        return max(f - self.K, 0.0) if self.option_type == "call" else max(self.K - f, 0.0)

    def price(self) -> float:
        if self.T <= 0 or self.sigma <= 0:
            return math.exp(-self.r * max(self.T, 0.0)) * self._intrinsic(self.F)

        n = self.n_steps
        dt = self.T / n
        vsqrt = self.sigma * math.sqrt(self.T)
        d1 = (math.log(self.F / self.K) + 0.5 * self.sigma ** 2 * self.T) / vsqrt
        d2 = d1 - vsqrt

        p = _peizer_pratt(d2, n)
        p_prime = _peizer_pratt(d1, n)

        u = p_prime / p
        d = (1.0 - p_prime) / (1.0 - p)
        disc = math.exp(-self.r * dt)

        # values at expiry: node i has i down-moves out of n
        values = [self._intrinsic(self.F * u ** (n - i) * d ** i) for i in range(n + 1)]

        for step in range(n - 1, -1, -1):
            for i in range(step + 1):
                continuation = disc * (p * values[i] + (1 - p) * values[i + 1])
                if self.american:
                    f_node = self.F * u ** (step - i) * d ** i
                    values[i] = max(continuation, self._intrinsic(f_node))
                else:
                    values[i] = continuation
        return values[0]
