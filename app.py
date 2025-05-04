from flask import Flask, jsonify, render_template
import random

app = Flask(__name__)

@app.route('/api/balance', methods=['GET'])
def get_balance():
    # Simulate fetching balance from your trading bot
    return jsonify({'balance': round(random.uniform(100, 5000), 2)})

@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    # Simulate fetching market data (e.g., price history)
    time = ['12:00', '12:05', '12:10', '12:15', '12:20']
    prices = [random.randint(30000, 40000) for _ in range(5)]
    return jsonify({'time': time, 'prices': prices})

@app.route('/')
def index():
    return render_template('dashboard.html')  # The FastAPI app will handle the rendering

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Flask running on port 5001
