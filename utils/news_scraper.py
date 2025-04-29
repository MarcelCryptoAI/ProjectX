import aiohttp

class NewsScraper:
    def __init__(self, settings):
        self.api_key = settings.cryptopanic_api_key
        self.endpoint = "https://cryptopanic.com/api/v1/posts/"

    async def fetch_sentiment(self, sentiment_analyzer):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.endpoint,
                    params={"auth_token": self.api_key, "filter": "important"},
                    timeout=10
                ) as resp:
                    data = await resp.json()
                    headlines = [item["title"] for item in data.get("results", [])]

                    sentiment_scores = []
                    for headline in headlines:
                        score = sentiment_analyzer.analyze(headline)
                        sentiment_scores.append(score)

                    if sentiment_scores:
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                        return avg_sentiment
                    else:
                        return 0
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return 0
