from bot.trader import Trader
from utils.settings_loader import Settings
from utils.telegram_bot import TelegramBot

import asyncio
import uvicorn
import threading

def start_dashboard(port):
    uvicorn.run("dashboard.main:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    settings = Settings.load("config/settings.yaml")
    telegram = TelegramBot(settings)

    # Start dashboard in background
    dashboard_thread = threading.Thread(target=start_dashboard, args=(settings.dashboard_port,))
    dashboard_thread.daemon = True
    dashboard_thread.start()

    # Start bot loop
    trader = Trader(settings, telegram)
    asyncio.run(trader.run())
