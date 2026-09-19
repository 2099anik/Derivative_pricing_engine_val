"""Base class for every pricing model.

Design goal (from the spec): *every* model exposes the same interface —
price(), delta(), gamma(), vega(), theta(), rho(). This ABC provides
working **finite-difference** Greeks by default, so any new model
(trees, Monte Carlo, PDE) gets Greeks for free just by implementing
price(). Analytic models (Black-76) override them for speed/accuracy.

Conventions (used everywhere in the engine):
    T      : time to expiry in YEARS
    r      : continuously-compounded risk-free rate (decimal, e.g. 0.05)
    sigma  : volatility (decimal, e.g. 0.20)
    Greeks are 'raw' / per unit change:
        vega  -> per 1.0 change in sigma  (divide by 100 for per 1 vol point)
        rho   -> per 1.0 change in r      (divide by 100 for per 1%)
        theta -> per 1.0 YEAR             (divide by 365 for per calendar day)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace


@dataclass
class OptionModel(ABC):
    """All models are dataclasses holding their market inputs as fields.

    Subclasses must define, at minimum, the fields used by the FD bumps
    below: an underlying level, T, r, sigma. The name of the underlying
    field can differ (F for futures, S for spot), so subclasses set
    ``_underlying`` to point the FD delta/gamma at the right field.
    """

    _underlying: str = "F"  # field name to bump for delta/gamma

    @abstractmethod
    def price(self) -> float:
        """Return the option price. The only thing a new model must implement."""
        raise NotImplementedError

    # --- helpers -----------------------------------------------------------
    def _bumped(self, **changes) -> "OptionModel":
        """Return a copy of this model with some fields replaced."""
        return replace(self, **changes)

    def _u(self) -> float:
        return getattr(self, self._underlying)

    # --- default finite-difference Greeks ----------------------------------
    def delta(self) -> float:
        u = self._u()
        h = max(u * 1e-4, 1e-6)
        up = self._bumped(**{self._underlying: u + h}).price()
        dn = self._bumped(**{self._underlying: u - h}).price()
        return (up - dn) / (2 * h)

    def gamma(self) -> float:
        u = self._u()
        h = max(u * 1e-4, 1e-6)
        up = self._bumped(**{self._underlying: u + h}).price()
        mid = self.price()
        dn = self._bumped(**{self._underlying: u - h}).price()
        return (up - 2 * mid + dn) / (h * h)

    def vega(self) -> float:
        h = 1e-4
        up = self._bumped(sigma=self.sigma + h).price()
        dn = self._bumped(sigma=self.sigma - h).price()
        return (up - dn) / (2 * h)

    def theta(self) -> float:
        # theta = dV/dt = -dV/dT
        h = min(1e-4, self.T / 2)
        up = self._bumped(T=self.T + h).price()
        dn = self._bumped(T=self.T - h).price()
        return -(up - dn) / (2 * h)

    def rho(self) -> float:
        h = 1e-4
        up = self._bumped(r=self.r + h).price()
        dn = self._bumped(r=self.r - h).price()
        return (up - dn) / (2 * h)

    def greeks(self) -> dict:
        """Convenience: all Greeks + price in one dict."""
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
        }
