
import numpy as np
import random

def price_paths(price=100, n=100, v=None):
    if v is None:
        v = random.random()/100
    prices = [price]
    for i in range(1, n+1):
        price *= np.random.normal(1, v)
        prices.append(price)
    return prices