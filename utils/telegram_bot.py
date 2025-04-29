import aiohttp

class TelegramBot:
    def __init__(self, settings):
        self.enabled = settings.telegram_enabled
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"

    async def send_trade_alert(self, side, sentiment_score, confidence):
        if not self.enabled:
            return

        message = (
            f"📢 New Trade Signal\n\n"
            f"Side: {side.upper()}\n"
            f"Sentiment Score: {sentiment_score}\n"
            f"Confidence: {confidence}%\n"
        )

        await self._send_message(message)

    async def send_message(self, text):
        if not self.enabled:
            return

        await self._send_message(text)

    async def _send_message(self, text):
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": text
            }
            async with aiohttp.ClientSession() as session:
                await session.post(f"{self.api_url}/sendMessage", json=payload)
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
