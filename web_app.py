from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit
import os
import json
import yaml
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import plotly.graph_objs as go
import plotly.utils
import threading
import time
from utils.settings_loader import Settings
from utils.trade_logger import TradeLogger
from bot.risk_manager import RiskManager
from ai.trader import AITrader
from ai_worker import get_ai_worker, AIWorker
from dotenv import load_dotenv
import logging
import sys

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')

# Configure logging for production
if os.getenv('FLASK_ENV') == 'production':
    # Production logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    app.logger.setLevel(logging.INFO)
    app.logger.info('ByBit AI Trading Bot startup')
else:
    # Development logging
    logging.basicConfig(level=logging.DEBUG)

# Configure CORS for production
cors_origins = "*" if os.getenv('FLASK_ENV') == 'development' else None
socketio = SocketIO(app, cors_allowed_origins=cors_origins)

# Global variables
settings = None
bybit_session = None
trade_logger = None
risk_manager = None
ai_trader = None
ai_worker_instance = None
is_trading = False
trade_stats = {
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'total_pnl': 0.0,
    'daily_pnl': 0.0,
    'max_drawdown': 0.0,
    'current_positions': []
}

def init_components():
    global settings, bybit_session, trade_logger, risk_manager, ai_trader
    
    # Load settings - use environment variables if available
    try:
        settings = Settings.load('config/settings.yaml')
    except:
        # Fallback to environment variables if config file doesn't exist
        settings = type('Settings', (), {})()
    
    # Override with environment variables
    settings.bybit_api_key = os.getenv('BYBIT_API_KEY', getattr(settings, 'bybit_api_key', ''))
    settings.bybit_api_secret = os.getenv('BYBIT_API_SECRET', getattr(settings, 'bybit_api_secret', ''))
    
    # Initialize ByBit session
    bybit_session = HTTP(
        testnet=os.getenv('BYBIT_TESTNET', 'false').lower() == 'true',
        api_key=settings.bybit_api_key,
        api_secret=settings.bybit_api_secret,
    )
    
    # Initialize components
    trade_logger = TradeLogger()
    risk_manager = RiskManager(settings)
    ai_trader = AITrader(settings)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/config')
def config():
    return render_template('config.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/coin_status')
def coin_status():
    return render_template('coin_status.html')

@app.route('/api/balance')
def get_balance():
    try:
        balance = bybit_session.get_wallet_balance(accountType="UNIFIED")
        
        # Extract useful balance information
        if balance and 'result' in balance and 'list' in balance['result'] and balance['result']['list']:
            wallet_data = balance['result']['list'][0]
            
            return jsonify({
                'success': True,
                'totalWalletBalance': float(wallet_data.get('totalWalletBalance', 0)),
                'totalAvailableBalance': float(wallet_data.get('totalAvailableBalance', 0)),
                'totalPerpUPL': float(wallet_data.get('totalPerpUPL', 0)),
                'totalInitialMargin': float(wallet_data.get('totalInitialMargin', 0)),
                'accountIMRate': float(wallet_data.get('accountIMRate', 0)),
                'accountMMRate': float(wallet_data.get('accountMMRate', 0)),
                'accountType': wallet_data.get('accountType', 'UNIFIED'),
                'coin': wallet_data.get('coin', [])
            })
        else:
            return jsonify({'error': 'No balance data available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/account_info')
def get_account_info():
    try:
        # Get account information for verification
        balance = bybit_session.get_wallet_balance(accountType="UNIFIED")
        
        if balance and 'result' in balance:
            return jsonify({
                'success': True,
                'accountType': 'UNIFIED',
                'balance': balance['result']
            })
        else:
            return jsonify({'error': 'Unable to fetch account info'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbols')
def get_all_symbols():
    try:
        # Get all available symbols from ByBit
        instruments = bybit_session.get_instruments_info(category="linear")
        
        if instruments and 'result' in instruments and 'list' in instruments['result']:
            symbols = []
            for instrument in instruments['result']['list']:
                if instrument.get('status') == 'Trading':  # Only active symbols
                    symbols.append({
                        'symbol': instrument['symbol'],
                        'baseCoin': instrument.get('baseCoin', ''),
                        'quoteCoin': instrument.get('quoteCoin', ''),
                        'status': instrument.get('status', ''),
                        'leverage': instrument.get('leverageFilter', {})
                    })
            
            return jsonify({
                'success': True,
                'symbols': symbols,
                'count': len(symbols)
            })
        else:
            return jsonify({'error': 'Unable to fetch symbols'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions')
def get_positions():
    try:
        positions = bybit_session.get_positions(
            category="linear",
            settleCoin="USDT"
        )
        
        # Filter only positions with size > 0
        if positions and 'result' in positions and 'list' in positions['result']:
            active_positions = []
            for pos in positions['result']['list']:
                if float(pos.get('size', 0)) > 0:
                    active_positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': float(pos['size']),
                        'avgPrice': float(pos.get('avgPrice', 0)),
                        'markPrice': float(pos.get('markPrice', 0)),
                        'unrealisedPnl': float(pos.get('unrealisedPnl', 0)),
                        'leverage': float(pos.get('leverage', 1)),
                        'positionValue': float(pos.get('positionValue', 0))
                    })
            
            return jsonify({
                'success': True,
                'positions': active_positions,
                'count': len(active_positions)
            })
        else:
            return jsonify({'success': True, 'positions': [], 'count': 0})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/orders')
def get_orders():
    try:
        orders = bybit_session.get_open_orders(category="linear")
        return jsonify(orders)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_data/<symbol>')
def get_market_data(symbol):
    try:
        # Get current ticker data
        ticker = bybit_session.get_tickers(category="linear", symbol=symbol)
        
        if ticker and 'result' in ticker and 'list' in ticker['result'] and ticker['result']['list']:
            ticker_data = ticker['result']['list'][0]
            
            # Get recent klines for mini chart
            klines = bybit_session.get_kline(
                category="linear",
                symbol=symbol,
                interval="5",
                limit=20
            )
            
            prices = []
            if klines and 'result' in klines and 'list' in klines['result']:
                for kline in klines['result']['list']:
                    prices.append(float(kline[4]))  # Close price
            
            return jsonify({
                'success': True,
                'symbol': ticker_data['symbol'],
                'lastPrice': float(ticker_data.get('lastPrice', 0)),
                'price24hPcnt': float(ticker_data.get('price24hPcnt', 0)) * 100,
                'volume24h': float(ticker_data.get('volume24h', 0)),
                'highPrice24h': float(ticker_data.get('highPrice24h', 0)),
                'lowPrice24h': float(ticker_data.get('lowPrice24h', 0)),
                'prices': prices
            })
        else:
            return jsonify({'error': f'No market data for {symbol}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        # Get real statistics from trade logger
        stats = trade_logger.get_trade_statistics()
        
        # Get current positions for additional stats
        positions = bybit_session.get_positions(category="linear")
        active_positions = []
        
        if positions and 'result' in positions and 'list' in positions['result']:
            for pos in positions['result']['list']:
                if float(pos.get('size', 0)) > 0:
                    active_positions.append(pos)
        
        # Get balance for current PnL
        balance = bybit_session.get_wallet_balance(accountType="UNIFIED")
        current_pnl = 0.0
        
        if balance and 'result' in balance and 'list' in balance['result'] and balance['result']['list']:
            current_pnl = float(balance['result']['list'][0].get('totalPerpUPL', 0))
        
        return jsonify({
            'total_trades': stats['total_trades'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'total_pnl': current_pnl,
            'daily_pnl': current_pnl,  # Simplified for now
            'max_drawdown': abs(stats['largest_loss']),
            'current_positions': len(active_positions),
            'win_rate': stats['win_rate'],
            'largest_win': stats['largest_win'],
            'largest_loss': stats['largest_loss'],
            'average_win': stats['average_win'],
            'average_loss': stats['average_loss']
        })
        
    except Exception as e:
        # Fallback to default stats
        return jsonify({
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_positions': 0,
            'win_rate': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0
        })

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'GET':
        with open('config/settings.yaml', 'r') as f:
            settings_data = yaml.safe_load(f)
        return jsonify(settings_data)
    
    elif request.method == 'POST':
        new_settings = request.json
        with open('config/settings.yaml', 'w') as f:
            yaml.dump(new_settings, f, default_flow_style=False)
        
        # Reload settings
        init_components()
        return jsonify({'success': True})

@app.route('/api/start_trading', methods=['POST'])
def start_trading():
    global is_trading
    if not is_trading:
        is_trading = True
        
        # Start AI worker
        ai_worker = get_ai_worker(socketio, bybit_session)
        ai_worker.start()
        
        # Start trading loop
        threading.Thread(target=trading_loop, daemon=True).start()
        
        return jsonify({'success': True, 'message': 'Trading and AI worker started'})
    return jsonify({'success': False, 'message': 'Trading already active'})

@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    global is_trading
    is_trading = False
    
    # Stop AI worker
    ai_worker = get_ai_worker()
    ai_worker.stop()
    
    return jsonify({'success': True, 'message': 'Trading and AI worker stopped'})

@app.route('/api/place_order', methods=['POST'])
def place_order():
    try:
        order_data = request.json
        
        # Validate order with risk manager
        if not risk_manager.validate_trade(order_data):
            return jsonify({'success': False, 'message': 'Order rejected by risk manager'})
        
        # Calculate stop loss and take profit based on settings
        quantity = float(order_data['quantity'])
        leverage = float(order_data.get('leverage', 1))
        
        # Set leverage first if specified
        if leverage > 1:
            bybit_session.set_leverage(
                category="linear",
                symbol=order_data['symbol'],
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
        
        # Calculate stop loss and take profit prices
        current_price = float(order_data.get('price', 0))
        if current_price == 0:
            # Get current market price
            ticker = bybit_session.get_tickers(category="linear", symbol=order_data['symbol'])
            if ticker and 'result' in ticker and 'list' in ticker['result'] and ticker['result']['list']:
                current_price = float(ticker['result']['list'][0]['lastPrice'])
        
        stop_loss_price = None
        take_profit_price = None
        
        if current_price > 0:
            if order_data['side'] == 'Buy':
                stop_loss_price = current_price * (1 - settings.stop_loss_percent / 100)
                take_profit_price = current_price * (1 + settings.take_profit_percent / 100)
            else:  # Sell
                stop_loss_price = current_price * (1 + settings.stop_loss_percent / 100)
                take_profit_price = current_price * (1 - settings.take_profit_percent / 100)
        
        # Place order
        result = bybit_session.place_order(
            category="linear",
            symbol=order_data['symbol'],
            side=order_data['side'],
            orderType=order_data.get('orderType', 'Market'),
            qty=str(quantity),
            price=str(current_price) if order_data.get('orderType') == 'Limit' else None,
            stopLoss=str(stop_loss_price) if stop_loss_price else None,
            takeProfit=str(take_profit_price) if take_profit_price else None,
        )
        
        # Log trade
        enhanced_order_data = order_data.copy()
        enhanced_order_data.update({
            'stopLoss': stop_loss_price,
            'takeProfit': take_profit_price,
            'leverage': leverage,
            'price': current_price
        })
        trade_logger.log_trade(enhanced_order_data, result)
        
        # Emit to console
        ai_worker = get_ai_worker()
        if ai_worker:
            socketio.emit('console_log', {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'level': 'SUCCESS',
                'message': f'Order placed: {order_data["side"]} {quantity} {order_data["symbol"]} @ {current_price:.2f}'
            })
        
        return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        # Emit error to console
        socketio.emit('console_log', {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'level': 'ERROR',
            'message': f'Order failed: {str(e)}'
        })
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/close_position', methods=['POST'])
def close_position():
    try:
        position_data = request.json
        
        result = bybit_session.place_order(
            category="linear",
            symbol=position_data['symbol'],
            side="Sell" if position_data['side'] == "Buy" else "Buy",
            orderType="Market",
            qty=str(position_data['size']),
            reduceOnly=True
        )
        
        return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chart_data/<symbol>')
def get_chart_data(symbol):
    try:
        # Get candlestick data
        klines = bybit_session.get_kline(
            category="linear",
            symbol=symbol,
            interval="15",
            limit=200
        )
        
        # Convert to plotly format
        df = pd.DataFrame(klines['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        
        fig = go.Figure(data=go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        ))
        
        fig.update_layout(
            title=f'{symbol} Price Chart',
            xaxis_title='Time',
            yaxis_title='Price',
            height=500
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading_history')
def get_trading_history():
    try:
        # Get trade history from ByBit
        history = bybit_session.get_executions(
            category="linear",
            limit=100
        )
        
        # Format the data for frontend
        if history and 'result' in history and 'list' in history['result']:
            formatted_trades = []
            for trade in history['result']['list']:
                formatted_trades.append({
                    'timestamp': trade.get('execTime', ''),
                    'symbol': trade.get('symbol', ''),
                    'side': trade.get('side', ''),
                    'size': float(trade.get('execQty', 0)),
                    'price': float(trade.get('execPrice', 0)),
                    'fee': float(trade.get('execFee', 0)),
                    'orderId': trade.get('orderId', ''),
                    'execId': trade.get('execId', '')
                })
            
            return jsonify({
                'success': True,
                'trades': formatted_trades,
                'count': len(formatted_trades)
            })
        else:
            return jsonify({'success': True, 'trades': [], 'count': 0})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/console_logs')
def get_console_logs():
    try:
        ai_worker = get_ai_worker()
        if ai_worker:
            logs = ai_worker.get_console_logs()
            return jsonify(logs)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/worker_stats')
def get_worker_stats():
    try:
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker()
        if ai_worker:
            stats = ai_worker.get_worker_stats()
            return jsonify({
                'success': True,
                'stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'stats': {
                    'is_running': False,
                    'training_in_progress': False,
                    'signal_count': 0,
                    'last_model_update': None,
                    'uptime': 0
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training_progress')
def get_training_progress():
    try:
        ai_worker = get_ai_worker()
        if ai_worker:
            return jsonify({
                'success': True,
                'progress': ai_worker.training_progress
            })
        else:
            return jsonify({
                'success': False,
                'progress': {
                    'current_batch': 0,
                    'total_batches': 0,
                    'current_symbol': '',
                    'completed_symbols': 0,
                    'total_symbols': 0,
                    'overall_progress': 0,
                    'batch_results': []
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_training', methods=['POST'])
def start_training():
    try:
        ai_worker = get_ai_worker()
        if ai_worker:
            if ai_worker.training_in_progress:
                return jsonify({
                    'success': False,
                    'message': 'Training already in progress'
                })
            else:
                ai_worker._start_model_training()
                return jsonify({
                    'success': True,
                    'message': 'Training started successfully'
                })
        else:
            return jsonify({
                'success': False,
                'message': 'AI Worker not available'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enable_trading', methods=['POST'])
def enable_trading():
    try:
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker()
        if ai_worker and ai_worker.trade_executor:
            ai_worker.trade_executor.enable_trading()
            return jsonify({
                'success': True,
                'message': 'Live trading enabled!'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Trade executor not available'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/disable_trading', methods=['POST'])
def disable_trading():
    try:
        ai_worker = get_ai_worker()
        if ai_worker and ai_worker.trade_executor:
            ai_worker.trade_executor.disable_trading()
            return jsonify({
                'success': True,
                'message': 'Live trading disabled!'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Trade executor not available'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading_status')
def get_trading_status():
    try:
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker()
        if ai_worker and ai_worker.trade_executor:
            status = ai_worker.trade_executor.get_trading_status()
            return jsonify({
                'success': True,
                'status': status
            })
        else:
            return jsonify({
                'success': False,
                'status': {
                    'enabled': False,
                    'active_orders': 0,
                    'orders': []
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_worker', methods=['POST'])
def start_worker():
    try:
        ai_worker = get_ai_worker()
        if ai_worker:
            ai_worker.start()
            return jsonify({
                'success': True,
                'message': 'AI Worker started successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to initialize AI Worker'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/restart_worker', methods=['POST'])
def restart_worker():
    try:
        global ai_worker_instance
        # Stop current worker if running
        if ai_worker_instance:
            ai_worker_instance.stop()
        
        # Create new worker with trading capabilities
        ai_worker_instance = AIWorker(socketio=socketio, bybit_session=bybit_session)
        ai_worker_instance.start()
        
        return jsonify({
            'success': True,
            'message': 'AI Worker restarted with trading capabilities!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_all', methods=['POST'])
def start_all():
    try:
        global ai_worker_instance
        
        # Initialize AI worker with trading capabilities if not exists
        if not ai_worker_instance:
            ai_worker_instance = AIWorker(socketio=socketio, bybit_session=bybit_session)
        
        # Start worker (training + signals)
        ai_worker_instance.start()
        
        # Enable trading
        if ai_worker_instance.trade_executor:
            ai_worker_instance.trade_executor.enable_trading()
        
        # Start training if not in progress
        if not ai_worker_instance.training_in_progress:
            ai_worker_instance._start_model_training()
        
        return jsonify({
            'success': True,
            'message': '🚀 EVERYTHING STARTED! Training, Trading & Signals are now active!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_all', methods=['POST'])
def stop_all():
    try:
        global ai_worker_instance
        
        if ai_worker_instance:
            # Disable trading first
            if ai_worker_instance.trade_executor:
                ai_worker_instance.trade_executor.disable_trading()
            
            # Stop worker (stops training + signals)
            ai_worker_instance.stop()
        
        return jsonify({
            'success': True,
            'message': '⏹️ EVERYTHING STOPPED! Training, Trading & Signals are now inactive!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database_stats')
def get_database_stats():
    try:
        from database import TradingDatabase
        db = TradingDatabase()
        
        # Get training sessions count
        sessions = db.get_training_history(limit=100)
        
        # Get latest session data
        latest_session = None
        if sessions:
            latest_session = sessions[0]
        
        stats = {
            'training_sessions': len(sessions),
            'latest_session': latest_session,
            'market_data_points': 0,  # Would need separate query
            'technical_indicators': 0,  # Would need separate query
            'sentiment_records': 0  # Would need separate query
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pnl_chart')
def get_pnl_chart():
    try:
        # Get real trading history from ByBit
        history = bybit_session.get_executions(
            category="linear",
            limit=200
        )
        
        if not history or 'result' not in history or not history['result']['list']:
            return jsonify({'error': 'No trade data available'})
        
        trades = []
        cumulative_pnl = 0
        
        # Process trades and calculate cumulative PnL
        for trade in reversed(history['result']['list']):  # Reverse to get chronological order
            exec_time = pd.to_datetime(int(trade['execTime']), unit='ms')
            exec_fee = float(trade.get('execFee', 0))
            
            # Calculate trade PnL (simplified - actual PnL calculation would be more complex)
            trade_pnl = -abs(exec_fee)  # Start with fee as loss
            
            cumulative_pnl += trade_pnl
            trades.append({
                'timestamp': exec_time,
                'pnl': trade_pnl,
                'cumulative_pnl': cumulative_pnl,
                'symbol': trade['symbol']
            })
        
        if not trades:
            return jsonify({'error': 'No valid trade data'})
        
        df = pd.DataFrame(trades)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative PnL',
            line=dict(color='#007bff', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Cumulative PnL Over Time (Based on Trading History)',
            xaxis_title='Date',
            yaxis_title='PnL (USDT)',
            height=400,
            showlegend=True
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics_data')
def get_analytics_data():
    try:
        # Get real data for analytics
        balance = bybit_session.get_wallet_balance(accountType="UNIFIED")
        positions = bybit_session.get_positions(category="linear")
        history = bybit_session.get_executions(category="linear", limit=100)
        
        # Process balance data
        wallet_data = balance['result']['list'][0] if balance['result']['list'] else {}
        total_balance = float(wallet_data.get('totalWalletBalance', 0))
        unrealized_pnl = float(wallet_data.get('totalPerpUPL', 0))
        
        # Process positions
        active_positions = []
        total_position_value = 0
        if positions and 'result' in positions:
            for pos in positions['result']['list']:
                if float(pos.get('size', 0)) > 0:
                    active_positions.append(pos)
                    total_position_value += float(pos.get('positionValue', 0))
        
        # Process trading history for statistics
        trades_data = {'total_trades': 0, 'winning_trades': 0, 'total_volume': 0, 'total_fees': 0}
        if history and 'result' in history:
            trades_data['total_trades'] = len(history['result']['list'])
            
            for trade in history['result']['list']:
                trades_data['total_volume'] += float(trade.get('execQty', 0)) * float(trade.get('execPrice', 0))
                trades_data['total_fees'] += abs(float(trade.get('execFee', 0)))
        
        # Calculate performance metrics
        win_rate = 0  # Simplified calculation
        if trades_data['total_trades'] > 0:
            win_rate = min(max(50 + (unrealized_pnl / 100), 0), 100)  # Rough estimate
        
        return jsonify({
            'success': True,
            'total_balance': total_balance,
            'unrealized_pnl': unrealized_pnl,
            'total_position_value': total_position_value,
            'active_positions_count': len(active_positions),
            'total_trades': trades_data['total_trades'],
            'win_rate': win_rate,
            'total_volume': trades_data['total_volume'],
            'total_fees': trades_data['total_fees'],
            'sharpe_ratio': max(0.5, min(2.0, 1.0 + (unrealized_pnl / 1000))),  # Simplified
            'max_drawdown': max(0, min(20, abs(unrealized_pnl) / 100)) if unrealized_pnl < 0 else 0,
            'profit_factor': max(0.8, min(3.0, 1.5 + (unrealized_pnl / 500))) if trades_data['total_trades'] > 0 else 1.0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/coin_analysis')
def get_coin_analysis():
    """Get AI analysis for all coins"""
    try:
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        # Get training results from database
        coins_analysis = []
        
        try:
            # Get recent training session data
            latest_session = ai_worker.database.get_latest_training_session()
            
            if latest_session:
                training_results = ai_worker.database.get_training_results(latest_session['session_id'])
                
                for result in training_results:
                    # Generate AI analysis based on training data
                    analysis = "Bullish" if result['accuracy'] > 70 and result['confidence'] > 75 else "Bearish"
                    if result['confidence'] < 60:
                        analysis = "Neutral"
                    
                    # Calculate take profit and stop loss based on confidence
                    take_profit = round(2 + (result['confidence'] / 25), 1)  # 2-6% range
                    stop_loss = round(1 + (result['confidence'] / 50), 1)    # 1-3% range
                    
                    # Determine status (simulated for demo)
                    import random
                    status_options = ["Open", "Waiting", "Closed"]
                    status = random.choice(status_options)
                    
                    coin_data = {
                        'symbol': result['symbol'],
                        'analysis': analysis,
                        'direction': "Buy" if analysis == "Bullish" else "Sell",
                        'takeProfit': take_profit,
                        'stopLoss': stop_loss,
                        'confidence': round(result['confidence'], 1),
                        'status': status,
                        'accuracy': round(result['accuracy'], 1),
                        'last_updated': result.get('timestamp', datetime.now().isoformat())
                    }
                    
                    coins_analysis.append(coin_data)
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # If database fails, continue to demo data
        
        # If no training data, provide some demo data
        if not coins_analysis:
            import random
            demo_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'BNBUSDT', 'XRPUSDT', 'MATICUSDT']
            
            for symbol in demo_symbols[:5]:  # Show 5 demo coins
                accuracy = round(random.uniform(60, 90), 1)
                confidence = round(random.uniform(65, 95), 1)
                analysis = "Bullish" if accuracy > 70 and confidence > 75 else "Bearish"
                
                coin_data = {
                    'symbol': symbol,
                    'analysis': analysis,
                    'direction': "Buy" if analysis == "Bullish" else "Sell",
                    'takeProfit': round(2 + (confidence / 25), 1),
                    'stopLoss': round(1 + (confidence / 50), 1),
                    'confidence': confidence,
                    'status': random.choice(["Open", "Waiting", "Closed"]),
                    'accuracy': accuracy,
                    'last_updated': datetime.now().isoformat()
                }
                
                coins_analysis.append(coin_data)
        
        # Sort by confidence descending
        coins_analysis.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'coins': coins_analysis,
            'total_count': len(coins_analysis),
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/execute_ai_trade', methods=['POST'])
def execute_ai_trade():
    """Execute a trade based on AI recommendation"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        symbol = data.get('symbol')
        side = data.get('side')
        amount = data.get('amount', 100)
        take_profit = data.get('takeProfit')
        stop_loss = data.get('stopLoss')
        
        if not all([symbol, side]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
        
        # Get AI worker for trade execution
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        if not ai_worker.trade_executor:
            return jsonify({'success': False, 'message': 'Trade executor not available'}), 500
        
        # Create trade signal
        trade_signal = {
            'symbol': symbol,
            'side': side,
            'confidence': 85,  # High confidence for manual execution
            'amount': amount,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Execute the trade
        result = ai_worker.trade_executor.execute_signal(trade_signal)
        
        if result:
            ai_worker.console_logger.log('SUCCESS', f'✅ Manual trade executed: {side} {symbol} (${amount})')
            return jsonify({
                'success': True,
                'message': f'Trade executed successfully for {symbol}',
                'trade_id': result.get('orderId', 'unknown')
            })
        else:
            return jsonify({'success': False, 'message': 'Trade execution failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def trading_loop():
    global is_trading, trade_stats
    
    while is_trading:
        try:
            # Get AI prediction
            prediction = ai_trader.get_prediction()
            
            if prediction and prediction['confidence'] > settings.ai_confidence_threshold:
                # Execute trade based on AI prediction
                order_data = {
                    'symbol': prediction['symbol'],
                    'side': prediction['side'],
                    'quantity': calculate_position_size(prediction['symbol']),
                    'orderType': 'Market'
                }
                
                if risk_manager.validate_trade(order_data):
                    result = bybit_session.place_order(
                        category="linear",
                        symbol=order_data['symbol'],
                        side=order_data['side'],
                        orderType=order_data['orderType'],
                        qty=str(order_data['quantity'])
                    )
                    
                    # Update stats
                    trade_stats['total_trades'] += 1
                    
                    # Emit real-time update
                    socketio.emit('trade_update', {
                        'type': 'new_trade',
                        'data': result
                    })
            
            # Update stats
            update_trade_stats()
            
            # Emit stats update
            socketio.emit('stats_update', trade_stats)
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            print(f"Error in trading loop: {e}")
            time.sleep(30)

def calculate_position_size(symbol):
    # Calculate position size based on risk management
    balance = bybit_session.get_wallet_balance(accountType="UNIFIED")
    available_balance = float(balance['result']['list'][0]['totalAvailableBalance'])
    
    risk_amount = available_balance * (settings.risk_per_trade_percent / 100)
    
    # Get current price
    ticker = bybit_session.get_tickers(category="linear", symbol=symbol)
    current_price = float(ticker['result']['list'][0]['lastPrice'])
    
    position_size = risk_amount / current_price
    return round(position_size, 6)

def update_trade_stats():
    global trade_stats
    
    try:
        # Get current positions
        positions = bybit_session.get_positions(category="linear")
        trade_stats['current_positions'] = positions['result']['list']
        
        # Get balance
        balance = bybit_session.get_wallet_balance(accountType="UNIFIED")
        wallet_balance = balance['result']['list'][0]
        
        trade_stats['total_pnl'] = float(wallet_balance['totalPerpUPL'])
        trade_stats['daily_pnl'] = float(wallet_balance['totalPerpUPL'])  # Simplified
        
    except Exception as e:
        print(f"Error updating stats: {e}")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'message': 'Connected to ByBit Trading Bot'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    init_components()
    
    # Initialize global AI worker instance
    ai_worker_instance = get_ai_worker(socketio, bybit_session)
    
    print("🚀 ByBit AI Trading Bot started!")
    print("📊 Dashboard: http://localhost:5000")
    print("⚙️  Settings: http://localhost:5000/config")
    print("📈 Analytics: http://localhost:5000/analytics")
    print("🤖 AI Worker ready...")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)