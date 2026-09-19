"""Generate the Phase 10 figure set for the LOZ6 example.

Run from the project root:  python -m pricing_engine.generate_loz6_figures

Builds a Black76 model from the ATM CME LOZ6 settlement (same market
inputs as validate_loz6.py) and saves every visualization/ chart to
figures/ so they can be eyeballed.
"""

import os

from pricing_engine.market_data import load_slice
from pricing_engine.models.black76 import Black76
from pricing_engine import visualization as viz

# --- market inputs (same as validate_loz6.py) ---
F = 93.9                      # December WTI futures price
T = 68 / 365                  # 10 Sep 2026 -> 17 Nov 2026
r = 0.042                     # ~3M T-bill

CSV = os.path.join(os.path.dirname(__file__), "market_data", "loz6_settlements.csv")
slice_ = load_slice(CSV, F=F, T=T, r=r)

# ATM strike: closest to F
atm = min(slice_.quotes, key=lambda q: abs(q.strike - F))
model = Black76(F=F, K=atm.strike, T=T, r=r, sigma=atm.cme_vol, option_type="call")

print(f"LOZ6 ATM call: F={F}  K={atm.strike}  T={T:.4f}  r={r:.3%}  sigma={atm.cme_vol:.4f}")
print(f"price = {model.price():.4f}\n")

figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")

generators = [
    viz.payoff_diagram,
    viz.pnl_surface,
    viz.greeks_heatmap,
    viz.delta_gamma_vs_actual,
    viz.scenario_shock_bars,
    viz.price_vs_F,
    viz.delta_vs_F,
    viz.vega_vs_vol,
    viz.theta_decay,
]

for generate in generators:
    path = generate(model, figures_dir=figures_dir)
    print(f"saved {path}")
