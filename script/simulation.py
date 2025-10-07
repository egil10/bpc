"""
Trading Simulation Runner

Runs trading simulations and visualizes results.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from price_process import price_paths
from typing import Callable, Optional, List, Tuple
import os


def run_simulation(trading_algo: Callable,
                   initial_price: float = 100, 
                   n_steps: int = 10000, 
                   initial_cash: float = 10000, 
                   volatility: Optional[float] = None, 
                   save_path: Optional[str] = None) -> Tuple[List[float], List[float]]:
    """
    Run a complete trading simulation and plot results.
    
    Parameters:
    -----------
    trading_algo : Callable
        Trading algorithm function that takes (cash, prices) and returns portfolio_values
    initial_price : float
        Starting stock price
    n_steps : int
        Number of time steps
    initial_cash : float
        Starting cash amount
    volatility : float, optional
        Price volatility (if None, randomized)
    save_path : str, optional
        Where to save the plot (if None, doesn't save)
        Example: 'plots/momentum_trading.pdf'
    
    Returns:
    --------
    Tuple[List[float], List[float]]
        (prices, portfolio_values)
    
    Example:
    --------
    >>> def my_algo(cash=10000, prices=None):
    ...     # your strategy
    ...     return portfolio_values
    >>> prices, portfolio = run_simulation(
    ...     my_algo, 
    ...     initial_price=100, 
    ...     n_steps=1000,
    ...     save_path='plots/my_strategy.pdf'
    ... )
    """
    if volatility is None:
        volatility = random.random() / 100
    
    # Generate data
    prices = price_paths(price=initial_price, n=n_steps, v=volatility)
    portfolio_values = trading_algo(cash=initial_cash, prices=prices)
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax1.plot(prices, color="blue", label="Stock Price", linewidth=1.5)
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("Stock Price", color="blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(portfolio_values, color="red", label="Portfolio Value", linewidth=1.5)
    ax2.set_ylabel("Portfolio Value", color="red", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="red")
    
    plt.title(f"Stock Price and Portfolio Value Over Time (Volatility: {volatility:.4f})", 
              fontsize=14, pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    # Save plot if path provided
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    final_value = portfolio_values[-1]
    return_pct = (final_value - initial_cash) / initial_cash * 100
    
    print(f"\nInitial cash: ${initial_cash:,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Return: {return_pct:.2f}%")
    
    return prices, portfolio_values


if __name__ == "__main__":
    # Example usage when running this script directly
    def simple_buy_hold(cash=10000, prices=None):
        if prices is None:
            prices = price_paths()
        shares = cash // prices[0]
        remaining = cash - (shares * prices[0])
        return [remaining + shares * p for p in prices]
    
    print("Running example simulation with buy-and-hold...")
    prices, portfolio = run_simulation(
        trading_algo=simple_buy_hold,
        initial_price=100,
        n_steps=1000,
        initial_cash=10000,
        save_path='plots/example_buy_hold.pdf'
    )