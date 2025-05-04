from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        return self.analyzer(text)

# Example usage:
# analyzer = SentimentAnalyzer()
# sentiment = analyzer.analyze_sentiment("Bitcoin is going to the moon!")
