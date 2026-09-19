"""Streamlit dashboard (Phase 12).

Run with:  streamlit run pricing_engine/dashboard/app.py

Pure UI wiring: every price, Greek, and scenario number comes straight
from the existing models/, greeks/, and scenario_analysis/ modules -- this
file only builds sidebar widgets and Plotly figures around them, it does
not reimplement any pricing, Greeks, or scenario logic.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# `streamlit run` only puts this script's own directory (dashboard/) on
# sys.path, not the project root -- so `import pricing_engine` fails
# regardless of the shell's current directory unless we add it ourselves.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pricing_engine.greeks import greeks_table
from pricing_engine.models.barrier_pde import BarrierOption
from pricing_engine.models.black76 import Black76
from pricing_engine.models.crr import CRRBinomial
from pricing_engine.models.leisen_reimer import LeisenReimer
from pricing_engine.models.monte_carlo import AsianOption, BasketOption, LookbackOption
from pricing_engine.scenario_analysis import scenario_table

st.set_page_config(page_title="Pricing Engine", layout="wide")
st.title("Derivatives Pricing Engine — Options on Futures (Black-76)")

# ============================================================== sidebar ===
st.sidebar.header("Instrument")
category = st.sidebar.selectbox("Option type", ["European", "American", "Asian", "Lookback", "Basket", "Barrier"])
option_type = st.sidebar.selectbox("Call / put", ["call", "put"])

MODEL_CHOICES = {
    "European": ["Black-76 (analytic)", "CRR binomial", "Leisen-Reimer", "Monte Carlo"],
    "American": ["CRR binomial", "Leisen-Reimer"],
    "Asian": ["Monte Carlo (arithmetic)", "Monte Carlo (geometric)"],
    "Lookback": ["Monte Carlo"],
    "Basket": ["Monte Carlo"],
    "Barrier": ["Finite-difference PDE"],
}
model_choice = st.sidebar.selectbox("Model", MODEL_CHOICES[category])

st.sidebar.header("Market data")
F = st.sidebar.number_input("Futures price F", value=100.0, min_value=0.01)
K = st.sidebar.number_input("Strike K", value=100.0, min_value=0.0)
days = st.sidebar.number_input("Days to expiry", value=90, min_value=1)
T = days / 365.0
r = st.sidebar.number_input("Risk-free rate r (%)", value=5.0) / 100.0
sigma = st.sidebar.number_input("Volatility sigma (%)", value=25.0, min_value=0.01) / 100.0

barrier = barrier_type = None
if category == "Barrier":
    st.sidebar.header("Barrier")
    barrier_type = st.sidebar.selectbox("Barrier type", ["up-and-out", "down-and-out", "up-and-in", "down-and-in"])
    default_barrier = F * 1.2 if barrier_type.startswith("up") else F * 0.8
    barrier = st.sidebar.number_input("Barrier level", value=default_barrier, min_value=0.01)

basket_F = basket_sigma = basket_weights = corr_matrix = None
if category == "Basket":
    st.sidebar.header("Basket legs")
    n_assets = st.sidebar.selectbox("Number of assets", [2, 3], index=0)
    basket_F, basket_sigma, basket_weights = [], [], []
    for i in range(n_assets):
        st.sidebar.markdown(f"**Leg {i + 1}**")
        basket_F.append(st.sidebar.number_input(f"F{i + 1}", value=F, min_value=0.01, key=f"basket_F{i}"))
        basket_sigma.append(
            st.sidebar.number_input(f"sigma{i + 1} (%)", value=sigma * 100, min_value=0.01, key=f"basket_sig{i}") / 100.0
        )
        basket_weights.append(st.sidebar.number_input(f"weight{i + 1}", value=1.0 / n_assets, key=f"basket_w{i}"))

    st.sidebar.markdown("**Correlations**")
    corr_matrix = [[1.0] * n_assets for _ in range(n_assets)]
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            rho = st.sidebar.slider(f"corr({i + 1},{j + 1})", -0.99, 0.99, 0.50, key=f"corr_{i}_{j}")
            corr_matrix[i][j] = corr_matrix[j][i] = rho

with st.sidebar.expander("Advanced (tree / Monte Carlo / PDE)"):
    n_steps_tree = st.number_input("Tree steps (CRR / Leisen-Reimer)", value=200, min_value=1)
    n_paths_mc = st.number_input("Monte Carlo paths", value=20_000, min_value=1_000, step=1_000)
    n_steps_mc = st.number_input("Monte Carlo time steps", value=50, min_value=1)
    seed_mc = st.number_input("Monte Carlo seed", value=0, min_value=0)
    f_steps_pde = st.number_input("Barrier PDE F-steps", value=200, min_value=10)
    t_steps_pde = st.number_input("Barrier PDE t-steps", value=200, min_value=10)


# ========================================================= build model ====
def build_model():
    if category == "European" and model_choice == "Black-76 (analytic)":
        return Black76(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
    if model_choice == "CRR binomial":
        return CRRBinomial(
            F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type,
            n_steps=n_steps_tree, american=(category == "American"),
        )
    if model_choice == "Leisen-Reimer":
        return LeisenReimer(
            F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type,
            n_steps=n_steps_tree, american=(category == "American"),
        )
    if category == "European" and model_choice == "Monte Carlo":
        # n_steps=1 collapses the Asian average to F_T -- a plain European payoff.
        return AsianOption(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type,
                            n_steps=1, n_paths=n_paths_mc, seed=seed_mc)
    if category == "Asian":
        average = "arithmetic" if "arithmetic" in model_choice else "geometric"
        return AsianOption(
            F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type, average=average,
            n_steps=n_steps_mc, n_paths=n_paths_mc, seed=seed_mc,
        )
    if category == "Lookback":
        return LookbackOption(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type,
                               n_steps=n_steps_mc, n_paths=n_paths_mc, seed=seed_mc)
    if category == "Basket":
        return BasketOption(
            F=tuple(basket_F), sigma=tuple(basket_sigma), corr=tuple(tuple(row) for row in corr_matrix),
            weights=tuple(basket_weights), K=K, T=T, r=r, option_type=option_type,
            n_paths=n_paths_mc, n_steps=n_steps_mc, seed=seed_mc,
        )
    if category == "Barrier":
        return BarrierOption(
            F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type,
            barrier=barrier, barrier_type=barrier_type, f_steps=f_steps_pde, t_steps=t_steps_pde,
        )
    raise ValueError("no model matched the current sidebar selection")


model = build_model()

# ========================================================== main panel ====
st.subheader("Pricing")
price = model.price()
cols = st.columns(3)
cols[0].metric("Price", f"{price:.4f}")
if hasattr(model, "confidence_interval"):
    lo, hi = model.confidence_interval()
    cols[1].metric("95% CI low", f"{lo:.4f}")
    cols[2].metric("95% CI high", f"{hi:.4f}")

st.subheader("Greeks")
st.dataframe(greeks_table([model]), use_container_width=True)

st.subheader("Payoff diagram")
u0 = model._u()
Fs = np.linspace(0.5 * u0, 1.5 * u0, 150)
if category == "Barrier":
    # Terminal payoff structure only (ignores path history): the vanilla
    # payoff, masked to zero on the knocked-out side of the barrier (or the
    # mirror image for a knock-in, via the same in = vanilla - out logic
    # used everywhere else in the engine).
    is_up, is_out = barrier_type.startswith("up"), barrier_type.endswith("out")

    def _payoff(f):
        intrinsic = max(f - K, 0.0) if option_type == "call" else max(K - f, 0.0)
        beyond_barrier = f >= barrier if is_up else f <= barrier
        return (0.0 if beyond_barrier else intrinsic) if is_out else (intrinsic if beyond_barrier else 0.0)

    payoff = [_payoff(f) for f in Fs]
    st.caption("Barrier payoff shown is the terminal structure only (path history before expiry is ignored).")
else:
    payoff = [model._bumped(F=f, T=0.0).price() for f in Fs]

fig_payoff = go.Figure()
fig_payoff.add_trace(go.Scatter(x=Fs, y=payoff, mode="lines", name="payoff at expiry"))
fig_payoff.add_vline(x=u0, line_dash="dash", annotation_text=f"F={u0:g}")
fig_payoff.update_layout(xaxis_title="Futures price F at expiry", yaxis_title="Payoff", height=420)
st.plotly_chart(fig_payoff, use_container_width=True)

st.subheader("Scenario analysis")
st.dataframe(scenario_table(model), use_container_width=True)

st.subheader("CRR vs Leisen-Reimer convergence")
st.caption("European convergence check against Black-76, using the market-data inputs above (independent of the selected instrument type/model).")
steps_grid = [5, 11, 21, 41, 81, 161, 321, 601]
benchmark = Black76(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type).price()
crr_prices = [
    CRRBinomial(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type, n_steps=n, american=False).price()
    for n in steps_grid
]
lr_prices = [
    LeisenReimer(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type, n_steps=n, american=False).price()
    for n in steps_grid
]
fig_conv = go.Figure()
fig_conv.add_trace(go.Scatter(x=steps_grid, y=crr_prices, mode="lines+markers", name="CRR"))
fig_conv.add_trace(go.Scatter(x=steps_grid, y=lr_prices, mode="lines+markers", name="Leisen-Reimer"))
fig_conv.add_hline(y=benchmark, line_dash="dash", annotation_text=f"Black-76 = {benchmark:.4f}")
fig_conv.update_layout(xaxis_title="n_steps", yaxis_title="Price", xaxis_type="log", height=420)
st.plotly_chart(fig_conv, use_container_width=True)
