import json
import os
from datetime import datetime

class TradeLogger:
    def __init__(self, filepath="trades.json"):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump([], f)

    def log_trade(self, symbol, side, sentiment_score, confidence, pnl=0):
        with open(self.filepath, "r+") as f:
            data = json.load(f)
            data.append({
                "time": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "sentiment": sentiment_score,
                "confidence": confidence,
                "pnl": pnl
            })
            f.seek(0)
            json.dump(data, f, indent=2)

    def calculate_stats(self):
        with open(self.filepath, "r") as f:
            data = json.load(f)

        total_trades = len(data)
        total_profit = sum(trade.get("pnl", 0) for trade in data)
        wins = len([t for t in data if t.get("pnl", 0) > 0])
        losses = len([t for t in data if t.get("pnl", 0) <= 0])

        win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_profit": round(total_profit, 2)
        }
