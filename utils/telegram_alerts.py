import requests

class TelegramAlert:
    def __init__(self, settings):
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id

    async def send_trade_alert(self, side, sentiment, confidence):
        message = f"🚀 New Trade Signal:\n\nSide: {side}\nSentiment: {sentiment}\nConfidence: {confidence}%"
        self._send(message)

    def _send(self, message):
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message}
            requests.post(url, data=data)
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")
