# Derivatives Pricing Engine — Options on Futures (Black-76)

A modular Python engine for pricing options on futures, computing Greeks,
running scenarios, and validating models. Built around one idea:

> Under Black-76 the futures price is a **driftless martingale**
> (`dF = σF dW`) and payoffs discount at `exp(-rT)`. Every model in this
> engine is therefore the standard model with **drift set to 0** and **F**
> as the underlying — so the whole thing reuses one interface.

## Structure

```
pricing_engine/
├── models/            # pricing models — all share the OptionModel interface
│   ├── base.py        # ABC: price() + default finite-difference Greeks
│   └── black76.py     # Black-76 with analytic price + Greeks   ✅ done
├── calibration/
│   └── implied_vol.py # Newton-Raphson IV solver (+ bisection fallback) ✅
├── greeks/            # aggregated Greeks table / heatmap / report   ⬜
├── market_data/       # yfinance / CBOE / Polygon loaders            ⬜
├── scenario_analysis/ # price / vol / rate / time shocks             ⬜
├── visualization/     # payoff, P&L surface, sensitivity plots       ⬜
├── dashboard/         # Streamlit app                                ⬜
└── tests/             # model validation                       ✅ (Black-76)
```

## The interface every model implements

```python
price()  delta()  gamma()  vega()  theta()  rho()   greeks()  # all-in-one dict
```

`OptionModel` (base.py) provides **working finite-difference Greeks by
default**, so a new model only has to implement `price()` and it gets Greeks
for free. Analytic models override them.

## Usage

```python
from pricing_engine import Black76, implied_vol

opt = Black76(F=78.5, K=80, T=45/365, r=0.045, sigma=0.35, option_type="call")
opt.price()      # 3.16
opt.greeks()     # {'price':..., 'delta':..., 'gamma':..., ...}

implied_vol(opt.price(), opt.F, opt.K, opt.T, opt.r, "call")   # -> 0.35
```

Conventions: `T` in years, `r`/`sigma` as decimals. Greeks are raw
(vega per 1.0 vol → ÷100 for a vol point; theta per year → ÷365 for a day;
rho per 1.0 → ÷100 for 1%).

Run tests / demo:
```
python -m pricing_engine.tests.test_black76
python -m pricing_engine.example
```

## Build order (what to do next)

The core (Black-76 + IV + tests) is done. Recommended order — each new model
just subclasses `OptionModel` and implements `price()`:

1. **American on futures — trees** (`models/crr.py`, `models/leisen_reimer.py`)
   CRR with `u=exp(σ√Δt)`, `d=1/u`, and — because futures are martingales —
   risk-neutral prob `p=(1-d)/(u-d)` (no growth term). Discount steps at
   `exp(-rΔt)`, apply early-exercise at each node. FD Greeks come free.
2. **Monte Carlo** (`models/monte_carlo.py`) for Asian / lookback / basket:
   `F_t = F0·exp(-½σ²t + σW_t)`, `price = exp(-rT)·mean(payoff)`. Add
   antithetic / control-variate variance reduction and confidence intervals.
3. **Barrier — PDE** (`models/barrier_pde.py`): Crank-Nicolson on the Black-76
   PDE `∂V/∂t + ½σ²F²∂²V/∂F² − rV = 0`; barrier as a boundary condition.
4. **greeks/** aggregator → table + heatmap + report (loops the interface).
5. **scenario_analysis/** → loop the spec's shocks (±1/5/10% F, ±5/10 vol pts,
   ±50/100 bps, 1/7/30-day decay) through the same interface.
6. **visualization/** + **dashboard/** (Streamlit + Plotly).
7. **tests/** → extend validation: American ≥ European, tree/MC convergence.
```
