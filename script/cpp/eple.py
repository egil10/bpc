import yfinance as yf
import matplotlib.pyplot as plt
import requests

# --- FIX RATE LIMITS ---
session = requests.Session()
session.headers['User-Agent'] = 'Mozilla/5.0'

# --- DOWNLOAD BTC DATA ---
btc = yf.download("BTC-USD", period="1mo", threads=False, session=session)

# OPTIONAL: cache once
# btc.to_csv("btc_cached.csv")

# Plot
plt.plot(btc.index, btc["Close"])
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.title("Bitcoin (BTC-USD) Close Price – Last 1 Month")
plt.grid(True)
plt.show()
