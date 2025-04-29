import yaml

class Settings:
    def __init__(self, data):
        self.bot = data.get('bot', {})
        self.dashboard = data.get('dashboard', {})
        self.telegram = data.get('telegram', {})
        self.api_keys = data.get('api_keys', {})
        self.system = data.get('system', {})

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return Settings(data)

    @property
    def risk_per_trade_percent(self):
        return self.bot.get('risk_per_trade_percent', 1.0)

    @property
    def take_profit_percent(self):
        return self.bot.get('take_profit_percent', 3.0)

    @property
    def stop_loss_percent(self):
        return self.bot.get('stop_loss_percent', 2.0)

    @property
    def max_concurrent_trades(self):
        return self.bot.get('max_concurrent_trades', 5)

    @property
    def ai_confidence_threshold(self):
        return self.bot.get('ai_confidence_threshold', 75)

    @property
    def max_daily_loss_percent(self):
        return self.bot.get('max_daily_loss_percent', 10)

    @property
    def dashboard_username(self):
        return self.dashboard.get('username', "admin")

    @property
    def dashboard_password(self):
        return self.dashboard.get('password', "changeme123")

    @property
    def dashboard_port(self):
        return self.dashboard.get('port', 8000)

    @property
    def telegram_enabled(self):
        return self.telegram.get('enabled', False)

    @property
    def telegram_bot_token(self):
        return self.telegram.get('bot_token', "")

    @property
    def telegram_chat_id(self):
        return self.telegram.get('chat_id', "")

    @property
    def bybit_api_key(self):
        return self.api_keys.get('bybit_api_key', "")

    @property
    def bybit_api_secret(self):
        return self.api_keys.get('bybit_api_secret', "")

    @property
    def cryptopanic_api_key(self):
        return self.api_keys.get('cryptopanic_api_key', "")

    @property
    def system_currency(self):
        return self.system.get('currency', "EUR")

    @property
    def use_autostart(self):
        return self.system.get('use_autostart', True)
