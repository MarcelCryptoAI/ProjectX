import asyncio
from bot.session_manager import SessionManager
from bot.risk_manager import RiskManager
from ai.sentiment_analyzer import SentimentAnalyzer
from ai.trend_predictor import TrendPredictor
from utils.news_scraper import NewsScraper
from utils.trade_logger import TradeLogger
import datetime

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

                # 3. Determine if we should trade
                if self.should_trade(sentiment_score, trend):
                    await self.execute_trade(sentiment_score, trend, top_symbol)

                await asyncio.sleep(60)  # Every 60 seconds
            except Exception as e:
                print(f"❌ Error in main trading loop: {e}")
                await asyncio.sleep(30)

    def should_trade(self, sentiment_score, trend):
        confidence = trend["confidence"]
        direction = trend["direction"]

        if confidence >= self.settings.ai_confidence_threshold:
            if sentiment_score > 0 and direction == "buy":
                return True
            elif sentiment_score < 0 and direction == "sell":
                return True
        return False

    async def execute_trade(self, sentiment_score, trend, symbol):
        side = "Buy" if trend["direction"] == "buy" else "Sell"

        # 1. Place order
        await self.session.open_position(side)

        # 2. Log trade
        self.logger.log_trade(symbol, side, sentiment_score, trend["confidence"])

        # 3. Send alert
        await self.telegram.send_trade_alert(side, sentiment_score, trend["confidence"])
