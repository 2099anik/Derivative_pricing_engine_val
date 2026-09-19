"""Black-76 model: European options on futures.

Under the risk-neutral measure the futures price is a driftless
martingale, dF = sigma * F dW, and payoffs are discounted at exp(-rT).
That single fact is what lets the rest of the engine (trees, MC, PDE)
reuse the same structure with drift = 0.

    d1 = (ln(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    call = exp(-rT) * (F*N(d1) - K*N(d2))
    put  = exp(-rT) * (K*N(-d2) - F*N(-d1))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import OptionModel

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Black76(OptionModel):
    F: float = 0.0          # futures price
    K: float = 0.0          # strike
    T: float = 0.0          # time to expiry (years)
    r: float = 0.0          # risk-free rate (decimal)
    sigma: float = 0.0      # volatility (decimal)
    option_type: str = "call"   # "call" or "put"
    _underlying: str = field(default="F")

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    # --- intermediate quantities ------------------------------------------
    def _d1_d2(self):
        vsqrt = self.sigma * math.sqrt(self.T)
        d1 = (math.log(self.F / self.K) + 0.5 * self.sigma ** 2 * self.T) / vsqrt
        d2 = d1 - vsqrt
        return d1, d2

    # --- price ------------------------------------------------------------
    def price(self) -> float:
        if self.T <= 0 or self.sigma <= 0:
            # intrinsic value at expiry / degenerate vol
            intrinsic = (self.F - self.K if self.option_type == "call"
                         else self.K - self.F)
            return math.exp(-self.r * max(self.T, 0.0)) * max(intrinsic, 0.0)
        disc = math.exp(-self.r * self.T)
        d1, d2 = self._d1_d2()
        if self.option_type == "call":
            return disc * (self.F * _norm_cdf(d1) - self.K * _norm_cdf(d2))
        return disc * (self.K * _norm_cdf(-d2) - self.F * _norm_cdf(-d1))

    # --- analytic Greeks --------------------------------------------------
    def delta(self) -> float:
        disc = math.exp(-self.r * self.T)
        d1, _ = self._d1_d2()
        if self.option_type == "call":
            return disc * _norm_cdf(d1)
        return -disc * _norm_cdf(-d1)

    def gamma(self) -> float:
        disc = math.exp(-self.r * self.T)
        d1, _ = self._d1_d2()
        return disc * _norm_pdf(d1) / (self.F * self.sigma * math.sqrt(self.T))

    def vega(self) -> float:
        disc = math.exp(-self.r * self.T)
        d1, _ = self._d1_d2()
        return self.F * disc * _norm_pdf(d1) * math.sqrt(self.T)

    def theta(self) -> float:
        disc = math.exp(-self.r * self.T)
        d1, d2 = self._d1_d2()
        decay = -self.F * disc * _norm_pdf(d1) * self.sigma / (2 * math.sqrt(self.T))
        # r * V term (r enters only through discounting in Black-76)
        return decay + self.r * self.price()

    def rho(self) -> float:
        # In Black-76, r appears only via the discount factor, so rho = -T * V.
        return -self.T * self.price()
