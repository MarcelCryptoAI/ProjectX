import asyncio
from bot.trader import Trader
from utils.telegram_alerts import TelegramAlert
from config.settings import Settings

settings = Settings()

async def main():
    telegram = TelegramAlert(settings)
    trader = Trader(settings, telegram)

    await trader.run()

if __name__ == "__main__":
    asyncio.run(main())
