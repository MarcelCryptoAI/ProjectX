import requests
import pandas as pd
from prophet import Prophet
import datetime
import time

class TrendPredictor:
    def __init__(self):
        self.symbol = "BTCUSDT"  # Default, will update
        self.interval = "60"  # 1h candles
        self.limit = 100
        self.base_url = "https://api.bybit.com/v5/market/kline"

    def get_ohlc(self, symbol):
        params = {
            "symbol": symbol,
            "interval": self.interval,
            "limit": self.limit,
            "category": "linear"
        }
        r = requests.get(self.base_url, params=params)
        data = r.json()["result"]["list"]

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df["close"] = pd.to_numeric(df["close"])
        return df[["timestamp", "close"]].rename(columns={"timestamp": "ds", "close": "y"})

    def predict_trend(self, symbol="BTCUSDT"):
        try:
            df = self.get_ohlc(symbol)
            m = Prophet(daily_seasonality=False)
            m.fit(df)

            future = m.make_future_dataframe(periods=2, freq='H')
            forecast = m.predict(future)

            last = forecast.iloc[-1]
            prev = forecast.iloc[-2]

            direction = "buy" if last['yhat'] > prev['yhat'] else "sell"
            confidence = round(abs(last['yhat'] - prev['yhat']) / prev['yhat'] * 100, 2)

            return {
                "direction": direction,
                "confidence": min(confidence, 99),
                "from": prev['yhat'],
                "to": last['yhat']
            }
        except Exception as e:
            print("⚠️ Trend prediction failed:", e)
            return {"direction": "hold", "confidence": 0}
