from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def analyze(self, text):
        result = self.classifier(text)[0]
        label = result['label']

        if label == "positive":
            return 1
        elif label == "negative":
            return -1
        else:
            return 0
