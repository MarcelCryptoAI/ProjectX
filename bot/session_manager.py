import time
import hmac
import hashlib
import aiohttp
import json

from utils.coin_selector import CoinSelector  # ✅ Must be at top

class SessionManager:
    def __init__(self, settings):
        self.settings = settings
        self.api_key = settings.bybit_api_key
        self.api_secret = settings.bybit_api_secret
        self.base_url = "https://api.bybit.com"
        self.coin_selector = CoinSelector()
        self.symbol = "BTCUSDT"  # Will update dynamically

    async def connect(self):
        print("✅ Connected to ByBit API")

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def _sign(self, params: dict):
        sorted_params = sorted(params.items())
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def open_position(self, side: str):
        self.symbol = await self.coin_selector.get_top_coin()

        timestamp = self._get_timestamp()
        recv_window = "5000"
        qty = "0.01"  # Default trade size (you can make dynamic)

        order_data = {
            "category": "linear",
            "symbol": self.symbol,
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": qty,
            "timestamp": timestamp,
            "recvWindow": recv_window
        }

        sign = self._sign(order_data)
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}/v5/order/create"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=order_data) as resp:
                response = await resp.json()
                print(f"\n📈 TRADE EXECUTED → {self.symbol} | {side.upper()} | QTY: {qty}")
                print(json.dumps(response, indent=2))
