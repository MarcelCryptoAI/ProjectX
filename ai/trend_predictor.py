import talib
import numpy as np

class TrendPredictor:
    def __init__(self, session):
        self.session = session  # The session object (fetch market data)

    def get_rsi(self, prices, time_period=14):
        """
        Calculate the Relative Strength Index (RSI) for a given price list.
        """
        return talib.RSI(np.array(prices), timeperiod=time_period)

    def get_macd(self, prices, fastperiod=12, slowperiod=26, signalperiod=9):
        """
        Calculate the Moving Average Convergence Divergence (MACD).
        """
        macd, macdsignal, macdhist = talib.MACD(np.array(prices), fastperiod, slowperiod, signalperiod)
        return macd, macdsignal, macdhist

    def predict_trend(self, symbol):
        """
        Predict the trend based on multiple indicators.
        """
        # Get historical market data for the symbol
        prices = self.session.get_historical_prices(symbol)
        
        # Calculate indicators
        rsi = self.get_rsi(prices)
        macd, macdsignal, macdhist = self.get_macd(prices)
        
        trend = {
            "rsi": rsi[-1],
            "macd": macd[-1],
            "macdsignal": macdsignal[-1],
            "direction": "buy" if rsi[-1] < 30 and macd[-1] > macdsignal[-1] else "sell",  # Example condition
            "confidence": 0.85  # Add logic for confidence calculation
        }
        return trend
