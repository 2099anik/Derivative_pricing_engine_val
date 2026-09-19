"""Market data loader (Phase 1).

Reads a CME settlement CSV (strike, call_settle, put_settle, cme_vol)
plus the scalar market inputs (F, T, r) and returns a clean, typed slice
ready for the pricer / IV solver.

The risk-free rate: on your own machine you can pull it live from FRED
(uncomment `fetch_rate_fred` below and set a free FRED API key). In this
demo we just pass r in, since the container has no FRED access.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date


@dataclass
class OptionQuote:
    strike: float
    call_settle: float
    put_settle: float
    cme_vol: float        # implied vol as a DECIMAL (0.4997), not percent


@dataclass
class MarketSlice:
    F: float              # futures price
    T: float              # time to expiry in years
    r: float              # risk-free rate (decimal)
    quotes: list          # list[OptionQuote]


def year_fraction(valuation: date, expiry: date, basis: int = 365) -> float:
    return (expiry - valuation).days / basis


def load_slice(csv_path: str, F: float, T: float, r: float) -> MarketSlice:
    quotes = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            quotes.append(
                OptionQuote(
                    strike=float(row["strike"]),
                    call_settle=float(row["call_settle"]),
                    put_settle=float(row["put_settle"]),
                    cme_vol=float(row["cme_vol"]) / 100.0,   # percent -> decimal
                )
            )
    quotes.sort(key=lambda q: q.strike)
    return MarketSlice(F=F, T=T, r=r, quotes=quotes)


# --- optional: live rate from FRED (run on your own machine) ---------------
# from fredapi import Fred
# def fetch_rate_fred(api_key: str, series: str = "DGS3MO") -> float:
#     fred = Fred(api_key=api_key)
#     latest = fred.get_series(series).dropna().iloc[-1]
#     return float(latest) / 100.0