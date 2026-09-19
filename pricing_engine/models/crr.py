"""CRR binomial tree: American options on futures (Black-76, Phase 4).

Futures are driftless martingales under the risk-neutral measure, so the
usual Cox-Ross-Rubinstein tree drops its growth term:

    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (1 - d) / (u - d)          # no drift -- futures are martingales
    discount each step by exp(-r * dt)

At every node the value is max(discounted continuation, intrinsic) when
early exercise is allowed. Set ``american=False`` to price the European
contract on the same tree (used to check convergence against Black-76).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import OptionModel


@dataclass
class CRRBinomial(OptionModel):
    F: float = 0.0          # futures price
    K: float = 0.0          # strike
    T: float = 0.0          # time to expiry (years)
    r: float = 0.0          # risk-free rate (decimal)
    sigma: float = 0.0      # volatility (decimal)
    option_type: str = "call"   # "call" or "put"
    n_steps: int = 500
    american: bool = True   # False => European (no early exercise), for convergence checks
    _underlying: str = field(default="F")

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")

    def _intrinsic(self, f: float) -> float:
        return max(f - self.K, 0.0) if self.option_type == "call" else max(self.K - f, 0.0)

    def price(self) -> float:
        if self.T <= 0 or self.sigma <= 0:
            return math.exp(-self.r * max(self.T, 0.0)) * self._intrinsic(self.F)

        n = self.n_steps
        dt = self.T / n
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (1.0 - d) / (u - d)
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
