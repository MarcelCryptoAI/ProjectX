class RiskManager:
    def __init__(self, settings):
        self.settings = settings

    def calculate_position_size(self, balance):
        risk_amount = (self.settings.risk_per_trade_percent / 100) * balance
        return risk_amount
