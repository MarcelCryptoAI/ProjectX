from flask import Flask, jsonify, render_template
import random
import json
import os
import subprocess  # To run the bot start/stop commands

app = Flask(__name__)

# Simulate balance
@app.route('/api/balance', methods=['GET'])
def get_balance():
    return jsonify({'balance': round(random.uniform(100, 5000), 2)})

# Simulate market data
@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    time = ['12:00', '12:05', '12:10', '12:15', '12:20']
    prices = [random.randint(30000, 40000) for _ in range(5)]
    return jsonify({'time': time, 'prices': prices})

# Simulate trade data
@app.route('/api/trades', methods=['GET'])
def get_trades():
    trades = [
        {"symbol": "BTCUSDT", "side": "buy", "timestamp": "2025-04-29 12:20:00"},
        {"symbol": "ETHUSDT", "side": "sell", "timestamp": "2025-04-29 12:30:00"}
    ]
    return jsonify(trades)

# Start bot endpoint
@app.route('/start-bot', methods=['GET'])
def start_bot():
    try:
        # Start the bot process
        subprocess.Popen(["python3", "run_bot.py"])
        return jsonify({"status": "success", "message": "Bot started!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Stop bot endpoint
@app.route('/stop-bot', methods=['GET'])
def stop_bot():
    try:
        # Stop the bot process (can be improved)
        os.system("pkill -f run_bot.py")
        return jsonify({"status": "success", "message": "Bot stopped!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/')
def index():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
