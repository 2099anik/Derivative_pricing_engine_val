"""Validate the Black-76 engine against real CME LOZ6 settlements.

Loop:
  1. solve implied vol from each settlement price -> compare to CME's vol
  2. price with CME's vol -> compare to the settlement
  3. check put-call parity holds on the settlements
"""

import os
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from pricing_engine.models.black76 import Black76
from pricing_engine.calibration.implied_vol import implied_vol
from pricing_engine.market_data import load_slice

# --- market inputs (from the CME header / calendar) ---
F = 93.9                      # December WTI futures price
T = 68 / 365                  # 10 Sep 2026 -> 17 Nov 2026
r = 0.042                     # ~3M T-bill (plug the live FRED value on your machine)

CSV = os.path.join(os.path.dirname(__file__), "market_data", "loz6_settlements.csv")
slice_ = load_slice(CSV, F=F, T=T, r=r)

print(f"F={F}  T={T:.4f}  r={r:.3%}\n")
print(f"{'K':>5} {'type':>4} {'mkt':>7} {'B76(cmeσ)':>10} {'solvedσ':>9} {'cmeσ':>7} {'Δσ':>7}")
print("-" * 56)

for q in slice_.quotes:
    for otype, mkt in (("call", q.call_settle), ("put", q.put_settle)):
        model = Black76(F=F, K=q.strike, T=T, r=r, sigma=q.cme_vol, option_type=otype)
        price_at_cme_vol = model.price()
        solved = implied_vol(mkt, F, q.strike, T, r, otype)
        dvol = solved - q.cme_vol
        print(f"{q.strike:>5.0f} {otype:>4} {mkt:>7.2f} {price_at_cme_vol:>10.3f} "
              f"{solved:>9.4f} {q.cme_vol:>7.4f} {dvol:>+7.4f}")

print("\nPut-call parity check  C - P  vs  e^(-rT)(F-K):")
import math
disc = math.exp(-r * T)
for q in slice_.quotes:
    lhs = q.call_settle - q.put_settle
    rhs = disc * (F - q.strike)
    print(f"  K={q.strike:>5.0f}   C-P={lhs:>7.3f}   e^(-rT)(F-K)={rhs:>7.3f}   diff={lhs-rhs:>+6.3f}")