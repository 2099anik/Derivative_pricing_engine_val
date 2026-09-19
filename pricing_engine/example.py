"""Quick demo. Run from the project root:  python -m pricing_engine.example"""

from pricing_engine import Black76, implied_vol

# WTI-crude-style option on futures
opt = Black76(F=78.50, K=80.0, T=45 / 365, r=0.045, sigma=0.35, option_type="call")

print("Black-76 call on futures")
print(f"  price : {opt.price():.4f}")
for name, val in opt.greeks().items():
    if name != "price":
        print(f"  {name:6s}: {val:+.4f}")

# per-convention scaling
print(f"\n  vega per 1 vol point : {opt.vega() / 100:.4f}")
print(f"  theta per day        : {opt.theta() / 365:.4f}")
print(f"  rho per 1%           : {opt.rho() / 100:.4f}")

# implied vol round-trip
mkt = opt.price()
iv = implied_vol(mkt, opt.F, opt.K, opt.T, opt.r, "call")
print(f"\n  recovered implied vol: {iv:.4%}  (input was 35.00%)")
