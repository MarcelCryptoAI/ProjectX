import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

def calculate_rsi(data, window=14):
    rsi = RSIIndicator(data['Close'], window)
    return rsi.rsi()

def calculate_macd(data):
    macd = MACD(data['Close'])
    return macd.macd_signal()

def calculate_bollinger_bands(data, window=20):
    bollinger = BollingerBands(data['Close'], window=window)
    return bollinger.bollinger_hband(), bollinger.bollinger_lband()

# Update strategy to include the indicators
def update_strategy(data):
    data['RSI'] = calculate_rsi(data)
    data['MACD'] = calculate_macd(data)
    data['Bollinger_Upper'], data['Bollinger_Lower'] = calculate_bollinger_bands(data)

    # Trading logic with multiple indicators
    if data['RSI'][-1] < 30 and data['Close'][-1] < data['Bollinger_Lower'][-1]:
        return "buy"
    elif data['RSI'][-1] > 70 and data['Close'][-1] > data['Bollinger_Upper'][-1]:
        return "sell"
    return None
