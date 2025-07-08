import requests
import time
from utils.settings_loader import Settings

class NewsScraper:
    def __init__(self, settings=None):
        if settings is None:
            try:
                settings = Settings.load('config/settings.yaml')
                self.api_key = settings.cryptopanic_api_key
            except:
                # Fallback to direct API key if settings loading fails
                self.api_key = 'defb6fad0b3f58d8369f3cbcafea0c0e8727025f'
        else:
            self.api_key = settings.cryptopanic_api_key
            
        self.endpoint = "https://cryptopanic.com/api/v1/posts/"
        self.last_request_time = 0
        self.rate_limit_delay = 1  # 1 second between requests

    def get_recent_news(self, limit=10, symbol=None):
        """Synchronous method to get recent crypto news"""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
            
            # Prepare parameters
            params = {
                "auth_token": self.api_key,
                "filter": "important",
                "public": "true"
            }
            
            if symbol:
                params["currencies"] = symbol.replace('USDT', '').replace('USDC', '')
            
            # Make request
            response = requests.get(self.endpoint, params=params, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for item in data.get("results", [])[:limit]:
                    # Basic sentiment analysis based on keywords
                    title = item.get("title", "")
                    sentiment = self._analyze_title_sentiment(title)
                    
                    articles.append({
                        'id': item.get('id'),
                        'title': title,
                        'url': item.get('url', ''),
                        'created_at': item.get('created_at', ''),
                        'domain': item.get('domain', ''),
                        'sentiment': sentiment,
                        'currencies': item.get('currencies', [])
                    })
                
                return articles
            else:
                print(f"CryptoPanic API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def _analyze_title_sentiment(self, title):
        """Basic sentiment analysis based on keyword matching"""
        if not title:
            return 0
        
        title_lower = title.lower()
        
        # Positive keywords
        positive_words = ['bullish', 'surge', 'rally', 'gains', 'up', 'rise', 'moon', 'rocket', 
                         'green', 'profit', 'wins', 'success', 'breakthrough', 'adoption']
        
        # Negative keywords
        negative_words = ['bearish', 'crash', 'dump', 'fall', 'drop', 'down', 'red', 'loss', 
                         'fear', 'panic', 'liquidation', 'hack', 'scam', 'ban']
        
        # Neutral/caution keywords
        neutral_words = ['stable', 'sideways', 'consolidation', 'analysis', 'report', 'update']
        
        positive_count = sum(1 for word in positive_words if word in title_lower)
        negative_count = sum(1 for word in negative_words if word in title_lower)
        neutral_count = sum(1 for word in neutral_words if word in title_lower)
        
        # Calculate sentiment score (-1 to 1)
        if positive_count > negative_count:
            return min(0.8, positive_count * 0.3)
        elif negative_count > positive_count:
            return max(-0.8, -negative_count * 0.3)
        else:
            return 0

    def fetch_sentiment(self, sentiment_analyzer=None):
        """Sync method for sentiment analysis"""
        try:
            params = {
                "auth_token": self.api_key,
                "filter": "important",
                "public": "true"
            }
            
            response = requests.get(self.endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                headlines = [item["title"] for item in data.get("results", [])]

                sentiment_scores = []
                for headline in headlines:
                    if sentiment_analyzer:
                        score = sentiment_analyzer.analyze(headline)
                    else:
                        score = self._analyze_title_sentiment(headline)
                    sentiment_scores.append(score)

                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    return avg_sentiment
                else:
                    return 0
            else:
                print(f"API error: {response.status_code}")
                return 0
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return 0
