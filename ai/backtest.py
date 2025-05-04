# ai/backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ai.trader import Trader
from ai.sentiment_analyzer import SentimentAnalyzer
from ai.trend_predictor import TrendPredictor

class Backtester:
    def __init__(self, historical_data, settings):
        self.historical_data = historical_data
        self.settings = settings
        self.trader = Trader(settings, None)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_predictor = TrendPredictor()

    def run_backtest(self):
        balance = 1000  # Starting with $1000
        initial_balance = balance
        positions = []
        returns = []
        for i in range(len(self.historical_data)):
            # Step 1: Get market data
            current_data = self.historical_data.iloc[i]
            symbol = current_data['symbol']
            price = current_data['close']
            
            # Step 2: Get sentiment and trend predictions
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(current_data['news'])
            trend = self.trend_predictor.predict_trend(symbol)

            # Step 3: Evaluate if we should trade
            if self.trader.should_trade(sentiment_score, trend):
                # Example: Buy or sell logic (simplified for backtest)
                if trend["direction"] == "buy":
                    balance -= price  # Simulate a buy
                    positions.append("buy")
                elif trend["direction"] == "sell":
                    balance += price  # Simulate a sell
                    positions.append("sell")

            # Step 4: Log returns
            returns.append(balance - initial_balance)

        self.plot_results(returns)

    def plot_results(self, returns):
        plt.plot(returns)
        plt.title('Backtest Results')
        plt.xlabel('Time')
        plt.ylabel('Balance Change')
        plt.show()

