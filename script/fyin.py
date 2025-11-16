from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Binance API Client
# -----------------------------
client = Client()

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

def get_binance_data(symbol, interval=Client.KLINE_INTERVAL_1DAY, lookback="1 month ago UTC"):
    klines = client.get_historical_klines(symbol, interval, lookback)
    
    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_base_vol","taker_quote_vol","ignore"
    ])
    
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df = df[["time","close"]].set_index("time")
    return df

# -----------------------------
# 2) Download crypto data
# -----------------------------
dfs = [get_binance_data(s) for s in symbols]
data = pd.concat(dfs, axis=1)
data.columns = symbols

# -----------------------------
# 3) Compute returns
# -----------------------------
returns = np.log(data / data.shift(1)).dropna()

mu = returns.mean() * 365           # annualized return
cov = returns.cov() * 365           # annualized covariance
inv_cov = np.linalg.inv(cov)

# -----------------------------
# 4) Monte Carlo Frontier
# -----------------------------
N = 20000
vols = []
rets = []

for _ in range(N):
    w = np.random.random(3)
    w /= w.sum()
    
    r = np.dot(w, mu)
    vol = np.sqrt(np.dot(w.T, cov @ w))
    
    vols.append(vol)
    rets.append(r)

# -----------------------------
# 5) Special portfolios
# -----------------------------

# (A) Individual cryptos (100% allocations)
asset_points = []
for i, sym in enumerate(symbols):
    w = np.zeros(3)
    w[i] = 1.0
    r = np.dot(w, mu)
    vol = np.sqrt(np.dot(w.T, cov @ w))
    asset_points.append((vol, r))

# (B) Minimum Variance Portfolio (MVP)
ones = np.ones(3)
w_mvp = inv_cov @ ones / (ones @ inv_cov @ ones)
mvp_return = np.dot(w_mvp, mu)
mvp_vol = np.sqrt(np.dot(w_mvp.T, cov @ w_mvp))

# (C) Tangency Portfolio (Max Sharpe, risk-free = 0)
w_tan = inv_cov @ mu / (ones @ inv_cov @ mu)
w_tan /= w_tan.sum()
tan_return = np.dot(w_tan, mu)
tan_vol = np.sqrt(np.dot(w_tan.T, cov @ w_tan))
tan_sharpe = tan_return / tan_vol

# -----------------------------
# 6) Plot everything
# -----------------------------
plt.figure(figsize=(12,8))

# Frontier cloud
plt.scatter(vols, rets, s=5, alpha=0.4, label="Random Portfolios")

# Asset points
for i, (vol, ret) in enumerate(asset_points):
    plt.scatter(vol, ret, s=100, marker="X", label=f"{symbols[i]}")

# MVP
plt.scatter(mvp_vol, mvp_return, s=150, marker="D", color="black", label="Minimum Variance")

# Tangency portfolio
plt.scatter(tan_vol, tan_return, s=200, marker="P", color="red", label="Max Sharpe Ratio")

plt.title("Efficient Frontier (Crypto, Binance, 1 Month)")
plt.xlabel("Volatility (annualized)")
plt.ylabel("Return (annualized)")
plt.grid(True)
plt.legend()
plt.show()

# Print portfolio weights
print("\n=== Special Portfolios ===")
print("\nMinimum Variance Portfolio weights:")
print(pd.Series(w_mvp, index=symbols))

print("\nTangency Portfolio (Max Sharpe) weights:")
print(pd.Series(w_tan, index=symbols))

print(f"\nSharpe Ratio: {tan_sharpe:.4f}")
