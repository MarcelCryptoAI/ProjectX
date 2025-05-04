import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical data
def load_data(file_path):
    return pd.read_csv(file_path)  # Assuming CSV file with 'Date', 'Close' columns

# Backtest the strategy (SMA crossover)
def backtest_strategy(data, short_window=20, long_window=50):
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    # Buy when short MA crosses above long MA, Sell when short MA crosses below long MA
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    data['Position'] = data['Signal'].diff()

    return data

# Simulate trading
def simulate_trading(data):
    data['Daily_Return'] = data['Close'].pct_change() * data['Position']
    data['Strategy_Return'] = data['Daily_Return'].cumsum()
    return data

# Plot results
def plot_results(data):
    plt.figure(figsize=(10,5))
    plt.plot(data['Date'], data['Strategy_Return'], label='Strategy Return')
    plt.plot(data['Date'], data['Close'], label='Market Price')
    plt.legend()
    plt.show()

# Main function to run backtest
def run_backtest(file_path):
    data = load_data(file_path)
    data = backtest_strategy(data)
    data = simulate_trading(data)
    plot_results(data)
    print("Backtest completed successfully!")
