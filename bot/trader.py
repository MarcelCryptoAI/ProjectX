import asyncio
from bot.session_manager import SessionManager
from bot.risk_manager import RiskManager
from ai.sentiment_analyzer import SentimentAnalyzer
from ai.trend_predictor import TrendPredictor
from utils.news_scraper import NewsScraper
from utils.trade_logger import TradeLogger
import datetime
from ai.reinforcement_learning import QLearningBot  # Import QLearningBot
from ai.trader import Trader
import time

class Trader:
    def __init__(self, settings, telegram):
        self.settings = settings
        self.telegram = telegram
        self.session = SessionManager(settings)
        self.risk_manager = RiskManager(settings)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_predictor = TrendPredictor()
        self.news_scraper = NewsScraper(settings)
        self.logger = TradeLogger()

        # Initialize the Q-learning bot
        self.qlearning_bot = QLearningBot(self.session)  # Initialize Q-learning bot with session as environment

    async def run(self):
        await self.session.connect()

        while True:
            try:
                # 1. Fetch news sentiment
                sentiment_score = await self.news_scraper.fetch_sentiment(self.sentiment_analyzer)

                # 2. Predict trend using Prophet
                top_symbol = await self.session.coin_selector.get_top_coin()
                trend = self.trend_predictor.predict_trend(top_symbol)

                print(f"\n🧠 AI Decision — {top_symbol} | Sentiment: {sentiment_score} | Trend: {trend}")

                # 3. Use Q-learning to decide if we should trade
                current_state = self.get_current_state(sentiment_score, trend, top_symbol)  # Get current state based on sentiment, trend, price
                action = self.qlearning_bot.choose_action(current_state)  # Use Q-learning to decide on action

                if action:  # If the action is to trade, execute the trade
                    await self.execute_trade(action, sentiment_score, trend, top_symbol)

                await asyncio.sleep(60)  # Every 60 seconds
            except Exception as e:
                print(f"❌ Error in main trading loop: {e}")
                await asyncio.sleep(30)

    def get_current_state(self, sentiment_score, trend, symbol):
        """Create a state representation from sentiment, trend, and market data (e.g., price)."""
        # Get market price (e.g., latest closing price)
        market_price = self.session.get_price(symbol)
        
        # Create a dynamic state based on sentiment, trend, and market data
        return (sentiment_score, trend["direction"], trend["confidence"], market_price)

    async def execute_trade(self, action, sentiment_score, trend, symbol):
        """Execute the trade based on the Q-learning decision."""
        side = "Buy" if action == "buy" else "Sell"

        # 1. Place order (execute trade)
        await self.session.open_position(side)

        # 2. Log the trade
        self.logger.log_trade(symbol, side, sentiment_score, trend["confidence"])

        # 3. Send alert to telegram
        await self.telegram.send_trade_alert(side, sentiment_score, trend["confidence"])

        # Update Q-learning after executing the trade
        reward = self.calculate_trade_reward(action, sentiment_score, trend, symbol)  # Reward based on trade outcome
        next_state = self.get_current_state(sentiment_score, trend, symbol)  # Next state (could use updated market data)
        self.qlearning_bot.update_q_table(current_state, action, reward, next_state)  # Update Q-table with the result

    def calculate_trade_reward(self, action, sentiment_score, trend, symbol):
        """Calculate reward based on trade performance (including slippage and fees)."""
        market_price = self.session.get_price(symbol)  # Current price when the trade happens
        trade_profit = self.calculate_trade_profit(action, market_price)

        # Calculate reward with slippage and fees taken into account
        reward = trade_profit - self.calculate_slippage(symbol) - self.calculate_fees(symbol)

        # Positive reward for good trade, negative for bad trade
        if reward > 0:
            return 1  # Positive reward for profitable trade
        else:
            return -1  # Negative reward for losing trade

    def calculate_trade_profit(self, action, market_price):
        """Calculate the profit of the trade based on the action (buy/sell)."""
        # Placeholder logic: calculate profit based on market price changes
        if action == "buy":
            # For simplicity, assume a 2% gain
            return market_price * 0.02  # 2% profit from buying
        elif action == "sell":
            return market_price * -0.02  # 2% loss from selling
        return 0

    def calculate_slippage(self, symbol):
        """Calculate slippage (difference between expected and actual price)."""
        # Placeholder logic: assume a fixed slippage of 0.1%
        return 0.001  # 0.1% slippage

    def calculate_fees(self, symbol):
        """Calculate transaction fees."""
        # Placeholder logic: assume a fixed fee of 0.1%
        return 0.001  # 0.1% fee per transaction

