import yfinance as yf
import matplotlib.pyplot as plt

# download data
tsla = yf.download("IBM", period="1mo")

# plot
plt.plot(tsla.index, tsla["Close"])
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("TSLA Close Price - Last 3 Months")
plt.grid(True)
plt.show()
