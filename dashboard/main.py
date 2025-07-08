from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json
import os
from datetime import datetime
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

USERNAME = "admin"
PASSWORD = "changeme123"

# Mock configuration data
CONFIG_DATA = {
    'ai_confidence_threshold': 85,
    'max_positions': 20,
    'risk_per_trade': 2.0,
    'auto_execute': True,
    'api_key': 'demo_key_***',
    'api_secret': 'demo_secret_***'
}

# Routes
@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == USERNAME and password == PASSWORD:
        session['authenticated'] = True
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html', error='Invalid credentials')

@app.route('/dashboard')
def dashboard():
    if not session.get('authenticated'):
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

@app.route('/settings')
def settings():
    if not session.get('authenticated'):
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')  # For now, settings is part of dashboard

# API Routes
@app.route('/api/analytics_data')
def analytics_data():
    # Enhanced data that matches the current configuration
    data = {
        'success': True,
        'total_balance': 12543.67,
        'unrealized_pnl': 234.89,
        'total_volume': 45678.90,
        'total_trades': 156,
        'win_rate': CONFIG_DATA['ai_confidence_threshold'],  # Use actual config value
        'sharpe_ratio': 1.45,
        'max_drawdown': -8.2,
        'profit_factor': 1.32,
        'total_fees': 45.67,
        'ai_confidence': CONFIG_DATA['ai_confidence_threshold'],
        'max_positions': CONFIG_DATA['max_positions'],
        'current_positions': 3,
        'consecutive_wins': 7,
        'consecutive_losses': 2,
        'monthly_returns': [
            {'month': 'Jan', 'return': 2.1},
            {'month': 'Feb', 'return': 1.8},
            {'month': 'Mar', 'return': 3.2},
            {'month': 'Apr', 'return': 0.9},
            {'month': 'May', 'return': 2.7},
            {'month': 'Jun', 'return': 1.4},
            {'month': 'Jul', 'return': 2.3},
            {'month': 'Aug', 'return': 1.1},
            {'month': 'Sep', 'return': 1.9},
            {'month': 'Oct', 'return': 2.5},
            {'month': 'Nov', 'return': 1.7},
            {'month': 'Dec', 'return': 2.0}
        ],
        'trade_durations': [
            {'duration': '< 1h', 'count': 47},
            {'duration': '1-4h', 'count': 39},
            {'duration': '4-12h', 'count': 31},
            {'duration': '12-24h', 'count': 23},
            {'duration': '1-3d', 'count': 12},
            {'duration': '> 3d', 'count': 4}
        ],
        'ai_accuracy_history': [
            {'date': datetime(2024, 12, 1).isoformat(), 'accuracy': 78.5},
            {'date': datetime(2024, 12, 2).isoformat(), 'accuracy': 79.2},
            {'date': datetime(2024, 12, 3).isoformat(), 'accuracy': 81.1},
            {'date': datetime(2024, 12, 4).isoformat(), 'accuracy': 82.7},
            {'date': datetime(2024, 12, 5).isoformat(), 'accuracy': 83.9},
            {'date': datetime(2024, 12, 6).isoformat(), 'accuracy': 85.2},
            {'date': datetime(2024, 12, 7).isoformat(), 'accuracy': 84.8},
            {'date': datetime(2024, 12, 8).isoformat(), 'accuracy': 86.1}
        ]
    }
    return jsonify(data)

@app.route('/api/balance_header')
def balance_header():
    return jsonify({
        'total_balance': 12543.67,
        'unrealized_pnl': 234.89,
        'status': 'online'
    })

@app.route('/api/config')
def get_config():
    return jsonify(CONFIG_DATA)

@app.route('/api/config', methods=['POST'])
def update_config():
    global CONFIG_DATA
    data = request.get_json()
    CONFIG_DATA.update(data)
    return jsonify({'success': True, 'message': 'Configuration updated'})

@app.route('/start_bot', methods=['POST'])
def start_bot():
    return jsonify({'success': True, 'message': 'Bot started successfully'})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    return jsonify({'success': True, 'message': 'Bot stopped successfully'})

@app.route('/api/trading_signals')
def trading_signals():
    # Mock trading signals data
    signals = [
        {
            'id': 1,
            'symbol': 'BTCUSDT',
            'signal': 'BUY',
            'confidence': 87.5,
            'price': 43250.00,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        },
        {
            'id': 2,
            'symbol': 'ETHUSDT',
            'signal': 'SELL',
            'confidence': 82.3,
            'price': 2580.50,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        },
        {
            'id': 3,
            'symbol': 'ADAUSDT',
            'signal': 'BUY',
            'confidence': 78.9,
            'price': 0.485,
            'timestamp': datetime.now().isoformat(),
            'status': 'executed'
        }
    ]
    
    # Filter by minimum confidence threshold
    min_confidence = CONFIG_DATA['ai_confidence_threshold']
    filtered_signals = [s for s in signals if s['confidence'] >= min_confidence]
    
    return jsonify({
        'success': True,
        'signals': filtered_signals,
        'total_signals': len(filtered_signals),
        'auto_execute': CONFIG_DATA['auto_execute']
    })

@app.route('/api/execute_trade', methods=['POST'])
def execute_trade():
    data = request.get_json()
    symbol = data.get('symbol', 'BTCUSDT')
    signal = data.get('signal', 'BUY')
    confidence = data.get('confidence', 0)
    
    # Simulate trade execution
    if CONFIG_DATA['auto_execute'] and confidence >= CONFIG_DATA['ai_confidence_threshold']:
        return jsonify({
            'success': True,
            'message': f'Trade executed: {signal} {symbol} at {confidence}% confidence',
            'trade_id': f'trade_{random.randint(1000, 9999)}',
            'executed_at': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'success': False,
            'message': f'Trade not executed: Auto-execute disabled or confidence too low ({confidence}%)'
        })

# Status endpoints
@app.route('/api/status/api')
def api_status():
    return jsonify({'status': 'online', 'last_check': datetime.now().isoformat()})

@app.route('/api/status/db')
def db_status():
    return jsonify({'status': 'online', 'last_check': datetime.now().isoformat()})

@app.route('/api/status/ai')
def ai_status():
    return jsonify({'status': 'online', 'last_check': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
