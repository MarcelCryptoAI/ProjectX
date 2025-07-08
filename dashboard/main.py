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

# API Routes
@app.route('/api/analytics_data')
def analytics_data():
    # Mock data that matches the current configuration
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
        'current_positions': 3
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
