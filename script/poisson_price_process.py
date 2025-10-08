
import numpy as np

def poisson_price_paths(price0=100, lam=5, T=10, sigma=0.01):
    """
    price0 : initial price
    lam    : rate (expected trades per unit time)
    T      : total time horizon
    sigma  : price volatility per trade
    """
    # Step 1: simulate event times
    inter_arrivals = np.random.exponential(1/lam, size=int(lam*T*2))  # oversample
    times = np.cumsum(inter_arrivals)
    times = times[times <= T]

    # Step 2: simulate price jumps
    n = len(times)
    prices = [price0]
    for _ in range(n):
        prices.append(prices[-1] * np.exp(np.random.normal(0, sigma)))
    
    return np.column_stack((np.r_[0, times], prices))
