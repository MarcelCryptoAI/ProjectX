import aiohttp

class CoinSelector:
    def __init__(self):
        self.url = "https://api.bybit.com/v5/market/tickers"
        self.cached_top_symbol = "BTCUSDT"

    async def get_top_coin(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, params={"category": "linear"}) as resp:
                    data = await resp.json()

                    tickers = data.get("result", {}).get("list", [])
                    usdt_pairs = [t for t in tickers if t["symbol"].endswith("USDT")]

                    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x["turnover24h"]), reverse=True)
                    top = sorted_pairs[:5]

                    for coin in top:
                        if coin["symbol"] != "BTCUSDT":
                            self.cached_top_symbol = coin["symbol"]
                            break

            return self.cached_top_symbol
        except Exception as e:
            print("⚠️ Coin selector error:", e)
            return self.cached_top_symbol
