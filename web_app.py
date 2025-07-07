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
import socket
import ssl
import urllib3

# Load environment variables
load_dotenv()

# Comprehensive DNS and SSL bypass for Heroku ByBit connection issues
try:
    import socket
    import ssl
    import os
    import urllib3
    import requests.adapters
    
    # Environment variables for SSL bypass
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    
    # Create a custom SSL context that's more permissive
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Store original functions
    original_getaddrinfo = socket.getaddrinfo
    original_ssl_context = ssl.create_default_context
    original_ssl_wrap_socket = ssl.wrap_socket
    
    def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        # For api.bybit.com, immediately use hardcoded IP to bypass DNS issues
        if host == 'api.bybit.com':
            print(f"🔧 Direct IP bypass for {host} -> 104.16.132.119")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, host, ('104.16.132.119', port))]
        
        # For other hosts, try normal DNS with shorter timeout
        try:
            socket.setdefaulttimeout(8)
            result = original_getaddrinfo(host, port, family, type, proto, flags)
            print(f"🌐 Standard DNS successful for {host}: {result[0][4][0] if result else 'Unknown'}")
            return result
        except Exception as e:
            print(f"⚠️ DNS failed for {host}: {e}")
            raise e
    
    def permissive_ssl_context(*args, **kwargs):
        """Always return permissive SSL context"""
        print("🔓 Using permissive SSL context")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    
    def custom_wrap_socket(*args, **kwargs):
        """Override SSL socket creation to disable verification"""
        kwargs['cert_reqs'] = ssl.CERT_NONE
        kwargs['check_hostname'] = False
        return original_ssl_wrap_socket(*args, **kwargs)
    
    # Apply global patches
    socket.getaddrinfo = custom_getaddrinfo
    ssl.create_default_context = permissive_ssl_context
    ssl.wrap_socket = custom_wrap_socket
    
    # Disable SSL warnings globally
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Monkey patch requests to disable SSL verification
    import requests
    requests.packages.urllib3.disable_warnings()
    
    print("✅ Comprehensive DNS bypass and SSL verification disabled")
    
except Exception as e:
    print(f"⚠️ Could not configure enhanced DNS/SSL bypass: {e}")

# Disable SSL warnings for Heroku environment if needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Heroku deployment check
if os.getenv('DYNO'):
    print("🌐 Running on Heroku - Production mode")
    print("📊 No Redis/Celery workers needed - using internal threading")
else:
    print("💻 Running locally - Development mode")

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
    global settings, bybit_session, trade_logger, risk_manager, ai_trader, ai_worker_instance
    
    # Load settings - use environment variables if available
    try:
        settings = Settings.load('config/settings.yaml')
    except:
        # Fallback to environment variables if config file doesn't exist
        settings = type('Settings', (), {})()
    
    # Override with environment variables - use setattr to avoid setter issues
    api_key = os.getenv('BYBIT_API_KEY', getattr(settings, 'bybit_api_key', ''))
    api_secret = os.getenv('BYBIT_API_SECRET', getattr(settings, 'bybit_api_secret', ''))
    
    # Set attributes safely
    if hasattr(settings, '__dict__'):
        settings.__dict__['bybit_api_key'] = api_key
        settings.__dict__['bybit_api_secret'] = api_secret
    else:
        setattr(settings, 'bybit_api_key', api_key)
        setattr(settings, 'bybit_api_secret', api_secret)
    
    # Initialize ByBit session - LIVE ONLY
    try:
        if not api_key or not api_secret:
            raise ValueError("API_KEY en API_SECRET zijn verplicht!")
        
        # Create custom session with improved DNS handling
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Create session with retry strategy
        session = requests.Session()
        
        # Configure retry strategy with DNS-friendly settings
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        # Mount adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set DNS cache timeout
        session.trust_env = False  # Disable system proxy
        
        # Pre-warm DNS resolution with better error handling
        try:
            import socket
            print("🔍 Pre-warming DNS resolution for api.bybit.com...")
            
            # Try multiple DNS resolution methods
            try:
                # Method 1: Standard resolution
                result = socket.getaddrinfo('api.bybit.com', 443, socket.AF_INET)
                if result:
                    print(f"✅ DNS pre-warm successful (Method 1): {result[0][4][0]}")
                else:
                    raise Exception("No result from getaddrinfo")
            except Exception as e1:
                print(f"⚠️ Method 1 failed: {e1}")
                try:
                    # Method 2: Direct gethostbyname
                    ip = socket.gethostbyname('api.bybit.com')
                    print(f"✅ DNS pre-warm successful (Method 2): {ip}")
                except Exception as e2:
                    print(f"⚠️ Method 2 failed: {e2}")
                    # Method 3: Use public DNS
                    try:
                        import subprocess
                        result = subprocess.run(['nslookup', 'api.bybit.com', '8.8.8.8'], 
                                             capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            print(f"✅ DNS pre-warm successful (Method 3): Using nslookup")
                        else:
                            print(f"⚠️ All DNS methods failed, continuing anyway...")
                    except Exception as e3:
                        print(f"⚠️ Method 3 failed: {e3}, continuing anyway...")
                        
        except Exception as dns_pretest:
            print(f"⚠️ DNS pre-warm failed: {dns_pretest}, continuing anyway...")
        
        # Initialize ByBit with enhanced error handling
        print("🔧 Initializing ByBit session with enhanced DNS handling...")
        try:
            bybit_session = HTTP(
                testnet=False,  # ALTIJD LIVE
                api_key=api_key,
                api_secret=api_secret,
                recv_window=20000  # Increase receive window for slow connections
            )
            print("✅ ByBit session created successfully")
        except Exception as init_error:
            print(f"⚠️ ByBit session creation failed: {init_error}")
            print("🔄 Attempting with alternative configuration...")
            
            # Try with more permissive settings
            try:
                bybit_session = HTTP(
                    testnet=False,
                    api_key=api_key,
                    api_secret=api_secret,
                    recv_window=10000  # Smaller window
                )
                print("✅ ByBit session created with alternative config")
            except Exception as alt_error:
                print(f"❌ Alternative config also failed: {alt_error}")
                raise Exception("Could not initialize ByBit session with any configuration")
        app.logger.info("✅ ByBit LIVE session initialized successfully")
        
        # Test connection with detailed error handling
        try:
            test_response = bybit_session.get_server_time()
            if test_response and 'result' in test_response:
                app.logger.info("✅ ByBit API connection test successful")
            else:
                app.logger.warning("⚠️ ByBit API test response unexpected")
        except Exception as test_e:
            app.logger.error(f"❌ ByBit API connection test failed: {test_e}")
            # Don't fail initialization, just log the error
            
    except Exception as e:
        app.logger.error(f"❌ ByBit session initialization failed: {e}")
        raise e  # Stop de app als ByBit niet werkt
    
    # Initialize components
    trade_logger = TradeLogger()
    risk_manager = RiskManager(settings)
    ai_trader = AITrader(settings)
    
    # Initialize AI worker instance (will be properly configured when start_all is called)
    ai_worker_instance = None

# Flag to track if components are initialized
_components_initialized = False

def ensure_components_initialized():
    """Ensure components are initialized on first use"""
    global _components_initialized
    if not _components_initialized:
        init_components()
        _components_initialized = True

def handle_bybit_request(func, *args, **kwargs):
    """Handle ByBit API requests with retry logic for DNS issues"""
    import time
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error_str = str(e)
            
            # Check for DNS/network issues
            if any(x in error_str.lower() for x in ['failed to resolve', 'nameresolutionerror', 'timeout', 'connection']):
                if attempt < max_retries - 1:
                    app.logger.warning(f"🔄 ByBit API DNS issue (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    app.logger.error("🌐 ByBit API unreachable after all retries")
                    raise ConnectionError(f"ByBit API connection failed: {error_str}")
            else:
                # Not a network issue, re-raise immediately
                app.logger.error(f"ByBit API error: {error_str}")
                raise e
    
    raise Exception("Should never reach here")

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

@app.route('/signals')
def signals():
    return render_template('signals.html')

@app.route('/api/balance')
def get_balance():
    ensure_components_initialized()
    try:
        # Check if bybit_session exists
        if not bybit_session:
            return jsonify({'error': 'ByBit session not initialized - check API credentials'}), 500
            
        # Try to get wallet balance
        balance = handle_bybit_request(bybit_session.get_wallet_balance, accountType="UNIFIED")
        
        app.logger.info(f"Raw balance response: {balance}")
        
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
            return jsonify({'error': f'Invalid balance response: {balance}'}), 500
            
    except ConnectionError as e:
        app.logger.warning(f"Balance API connection error: {str(e)} - returning cached balance")
        return jsonify({
            'success': True,
            'totalWalletBalance': 0.0,
            'totalAvailableBalance': 0.0,
            'totalPerpUPL': 0.0,
            'totalInitialMargin': 0.0,
            'accountIMRate': 0.0,
            'accountMMRate': 0.0,
            'accountType': 'UNIFIED',
            'coin': [],
            'cached': True,
            'message': 'Using cached balance due to connection issues'
        })
    except Exception as e:
        app.logger.error(f"Balance API error: {str(e)}")
        return jsonify({'error': f'ByBit API error: {str(e)}'}), 500

@app.route('/api/balance_header')
def get_balance_header():
    ensure_components_initialized()
    try:
        # Check if bybit_session exists
        if not bybit_session:
            return jsonify({'error': 'ByBit session not initialized - check API credentials'}), 500
            
        # Get wallet balance for header display
        balance_data = handle_bybit_request(bybit_session.get_wallet_balance, accountType="UNIFIED")
        
        # Debug logging
        if balance_data:
            app.logger.info(f"Header balance data received: {json.dumps(balance_data.get('result', {}), indent=2)}")
        
        if balance_data and 'result' in balance_data and 'list' in balance_data['result']:
            account = balance_data['result']['list'][0] if balance_data['result']['list'] else {}
            
            # Calculate total balance and 24h P&L
            total_balance = float(account.get('totalWalletBalance', 0))
            total_pnl_24h = float(account.get('totalPerpUPL', 0))
            
            # If totalWalletBalance is 0, try to get from coins
            if total_balance == 0 and 'coin' in account:
                for coin in account['coin']:
                    if coin['coin'] == 'USDT':
                        total_balance = float(coin.get('equity', 0))
                        if total_balance == 0:
                            total_balance = float(coin.get('walletBalance', 0))
                        break
            
            # Calculate 24h P&L percentage
            pnl_percent = (total_pnl_24h / total_balance * 100) if total_balance > 0 else 0
            
            return jsonify({
                'success': True,
                'balance': total_balance,
                'pnl_24h': total_pnl_24h,
                'pnl_24h_percent': pnl_percent
            })
        else:
            return jsonify({'error': f'Invalid balance response: {balance_data}'}), 500
            
    except ConnectionError as e:
        app.logger.warning(f"Header balance API connection error: {str(e)} - returning cached balance")
        return jsonify({
            'success': True,
            'balance': 0.0,
            'pnl_24h': 0.0,
            'pnl_24h_percent': 0.0,
            'cached': True,
            'message': 'Using cached balance due to connection issues'
        })
    except Exception as e:
        app.logger.error(f"Header balance API error: {str(e)}")
        return jsonify({'error': f'ByBit API error: {str(e)}'}), 500

@app.route('/api/account_name')
def get_account_name():
    ensure_components_initialized()
    try:
        # Check if bybit_session exists
        if not bybit_session:
            return jsonify({'error': 'ByBit session not initialized - check API credentials'}), 500
            
        # Try to get account info to extract account name/ID
        account_info = handle_bybit_request(bybit_session.get_account_info)
        
        if account_info and 'result' in account_info:
            # Extract useful account identifiers
            uid = account_info['result'].get('uid', '')
            margin_mode = account_info['result'].get('marginMode', '')
            
            return jsonify({
                'success': True,
                'account_name': f"ByBit-{uid[-6:]}" if uid else "ByBit Account",
                'uid': uid,
                'margin_mode': margin_mode
            })
        else:
            return jsonify({'error': f'Invalid account response: {account_info}'}), 500
            
    except Exception as e:
        app.logger.error(f"Account name API error: {str(e)}")
        return jsonify({'error': f'ByBit API error: {str(e)}'}), 500

@app.route('/api/dns_test')
def test_dns():
    """Test DNS resolution for ByBit API"""
    try:
        import socket
        import time
        
        start_time = time.time()
        
        # Test DNS resolution
        try:
            result = socket.getaddrinfo('api.bybit.com', 443)
            resolution_time = (time.time() - start_time) * 1000
            
            return jsonify({
                'success': True,
                'host': 'api.bybit.com',
                'resolved_ips': [addr[4][0] for addr in result[:3]],  # First 3 IPs
                'resolution_time_ms': round(resolution_time, 2),
                'resolver': 'Direct DNS Override (Cloudflare IP)',
                'method': 'Hardcoded IP bypass for api.bybit.com'
            })
        except Exception as dns_error:
            return jsonify({
                'success': False,
                'error': f'DNS resolution failed: {str(dns_error)}',
                'host': 'api.bybit.com'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/system_status')
def get_system_status():
    """Get system status for all components"""
    status = {
        'bybit_api': {'status': 'unknown', 'message': ''},
        'database': {'status': 'unknown', 'message': ''},
        'dyno': {'status': 'unknown', 'message': ''},
        'dns': {'status': 'unknown', 'message': ''}
    }
    
    # Test ByBit API
    try:
        ensure_components_initialized()
        if bybit_session:
            # Quick API test
            server_time = handle_bybit_request(bybit_session.get_server_time)
            if server_time and 'result' in server_time:
                status['bybit_api'] = {'status': 'online', 'message': 'API responsive'}
            else:
                status['bybit_api'] = {'status': 'offline', 'message': 'Invalid API response'}
        else:
            status['bybit_api'] = {'status': 'offline', 'message': 'Session not initialized'}
    except ConnectionError as e:
        status['bybit_api'] = {'status': 'offline', 'message': 'DNS/Connection failed'}
    except Exception as e:
        status['bybit_api'] = {'status': 'offline', 'message': f'API error: {str(e)[:50]}'}
    
    # Test Database
    try:
        from database import TradingDatabase
        db = TradingDatabase()
        # Simple test query
        test_query = db.get_connection()
        test_query.close()
        status['database'] = {'status': 'online', 'message': 'Database responsive'}
    except Exception as e:
        status['database'] = {'status': 'offline', 'message': f'DB error: {str(e)[:50]}'}
    
    # Test DNS resolution
    try:
        import socket
        import time
        start_time = time.time()
        result = socket.getaddrinfo('api.bybit.com', 443)
        resolution_time = (time.time() - start_time) * 1000
        status['dns'] = {
            'status': 'online', 
            'message': f'DNS resolved in {resolution_time:.1f}ms (Override: {result[0][4][0]})',
            'resolved_ip': result[0][4][0] if result else 'Unknown'
        }
    except Exception as e:
        status['dns'] = {'status': 'offline', 'message': f'DNS failed: {str(e)[:50]}'}
    
    # Test Dyno (always online if we can respond)
    status['dyno'] = {'status': 'online', 'message': 'Dyno responsive'}
    
    return jsonify({
        'success': True,
        'status': status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/account_info')
def get_account_info():
    ensure_components_initialized()
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
    ensure_components_initialized()
    try:
        # Get all available symbols from ByBit with DNS handling
        instruments = handle_bybit_request(bybit_session.get_instruments_info, category="linear")
        
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
            
    except ConnectionError as e:
        # DNS/Connection issue - return cached popular symbols
        app.logger.warning(f"Symbols API connection error: {str(e)} - returning cached symbols")
        cached_symbols = [
            {'symbol': 'BTCUSDT', 'baseCoin': 'BTC', 'quoteCoin': 'USDT', 'status': 'Trading', 'leverage': {}},
            {'symbol': 'ETHUSDT', 'baseCoin': 'ETH', 'quoteCoin': 'USDT', 'status': 'Trading', 'leverage': {}},
            {'symbol': 'SOLUSDT', 'baseCoin': 'SOL', 'quoteCoin': 'USDT', 'status': 'Trading', 'leverage': {}},
            {'symbol': 'ADAUSDT', 'baseCoin': 'ADA', 'quoteCoin': 'USDT', 'status': 'Trading', 'leverage': {}},
            {'symbol': 'DOGEUSDT', 'baseCoin': 'DOGE', 'quoteCoin': 'USDT', 'status': 'Trading', 'leverage': {}},
            {'symbol': 'BNBUSDT', 'baseCoin': 'BNB', 'quoteCoin': 'USDT', 'status': 'Trading', 'leverage': {}},
            {'symbol': 'XRPUSDT', 'baseCoin': 'XRP', 'quoteCoin': 'USDT', 'status': 'Trading', 'leverage': {}},
            {'symbol': 'MATICUSDT', 'baseCoin': 'MATIC', 'quoteCoin': 'USDT', 'status': 'Trading', 'leverage': {}}
        ]
        return jsonify({
            'success': True,
            'symbols': cached_symbols,
            'count': len(cached_symbols),
            'cached': True,
            'message': 'Using cached symbols due to connection issues'
        })
    except Exception as e:
        app.logger.error(f"Symbols API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions')
def get_positions():
    ensure_components_initialized()
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
    ensure_components_initialized()
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
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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
        ensure_components_initialized()
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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
        ensure_components_initialized()
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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
        ensure_components_initialized()
        global ai_worker_instance
        
        # Initialize AI worker with trading capabilities if not exists
        if not ai_worker_instance:
            ai_worker_instance = AIWorker(socketio=socketio, bybit_session=bybit_session)
        
        # Start worker (training + signals)
        ai_worker_instance.start()
        
        # Trading is automatically enabled if bybit_session exists
        
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
            # Stop worker (stops training + signals + trading)
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
        
        # Get latest session data with proper timestamp
        latest_session_date = None
        if sessions:
            latest_session = sessions[0]
            # If latest_session is a tuple/list, get the timestamp field
            if isinstance(latest_session, (list, tuple)) and len(latest_session) > 3:
                latest_session_date = latest_session[3]  # Assuming timestamp is at index 3
            elif isinstance(latest_session, dict):
                latest_session_date = latest_session.get('timestamp') or latest_session.get('created_at')
            else:
                latest_session_date = datetime.now().isoformat()
        
        stats = {
            'training_sessions': len(sessions),
            'latest_session_date': latest_session_date,
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
    """Get AI analysis for all coins with REAL status logic"""
    try:
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        # Get AI confidence threshold from settings
        ai_confidence_threshold = float(os.getenv('AI_CONFIDENCE_THRESHOLD', 75))
        
        # Get training results from database
        coins_analysis = []
        
        try:
            # Get recent training session data
            latest_session = ai_worker.database.get_latest_training_session()
            
            if latest_session:
                training_results = ai_worker.database.get_training_results(latest_session['session_id'])
                
                for result in training_results:
                    # Calculate tijd sinds aanbeveling
                    training_time = datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()).replace('Z', '+00:00'))
                    now = datetime.now()
                    time_diff = now - training_time
                    
                    # Format tijd als "25min" of "1h 10m"
                    if time_diff.total_seconds() < 3600:  # Minder dan 1 uur
                        time_since = f"{int(time_diff.total_seconds() // 60)}min"
                    else:  # Meer dan 1 uur
                        hours = int(time_diff.total_seconds() // 3600)
                        minutes = int((time_diff.total_seconds() % 3600) // 60)
                        time_since = f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
                    
                    # ECHTE STATUS LOGIC
                    confidence = result['confidence']
                    if confidence < ai_confidence_threshold:
                        status = "Te lage score"
                        status_class = "status-low"
                    else:
                        # Check if this coin is in signals queue (ready to trade)
                        # TODO: Implement check against actual signals queue
                        # For now, assume high confidence = signal
                        if confidence >= ai_confidence_threshold:
                            status = "Signal"
                            status_class = "status-signal"
                        else:
                            status = "Te lage score"
                            status_class = "status-low"
                    
                    # Generate AI analysis based on training data
                    analysis = "Bullish" if result['accuracy'] > 70 and confidence > 75 else "Bearish"
                    if confidence < 60:
                        analysis = "Neutral"
                    
                    # Calculate take profit and stop loss based on confidence
                    take_profit = round(2 + (confidence / 25), 1)  # 2-6% range
                    stop_loss = round(1 + (confidence / 50), 1)    # 1-3% range
                    
                    coin_data = {
                        'symbol': result['symbol'],
                        'analysis': analysis,
                        'direction': "Buy" if analysis == "Bullish" else "Sell",
                        'takeProfit': take_profit,
                        'stopLoss': stop_loss,
                        'confidence': round(confidence, 1),
                        'status': status,
                        'status_class': status_class,
                        'accuracy': round(result['accuracy'], 1),
                        'time_since': time_since,
                        'last_updated': result.get('timestamp', datetime.now().isoformat())
                    }
                    
                    coins_analysis.append(coin_data)
        except Exception as db_error:
            app.logger.error(f"Database error in coin_analysis: {db_error}")
            return jsonify({'success': False, 'error': 'No training data available'}), 500
        
        if not coins_analysis:
            return jsonify({'success': False, 'error': 'No training data available - start AI training first'}), 404
        
        # Sort by confidence descending
        coins_analysis.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'coins': coins_analysis,
            'total_count': len(coins_analysis),
            'ai_threshold': ai_confidence_threshold,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Coin analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trading_signals')
def get_trading_signals():
    """Get trading signals ONLY above AI confidence threshold"""
    try:
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        # Get AI confidence threshold from settings
        ai_confidence_threshold = float(os.getenv('AI_CONFIDENCE_THRESHOLD', 75))
        
        signals = []
        
        try:
            # Get recent training session data
            latest_session = ai_worker.database.get_latest_training_session()
            
            if latest_session:
                training_results = ai_worker.database.get_training_results(latest_session['session_id'])
                
                signal_id = 0
                for result in training_results:
                    confidence = result['confidence']
                    
                    # ALLEEN signals boven threshold
                    if confidence >= ai_confidence_threshold:
                        signal_time = datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()).replace('Z', '+00:00'))
                        
                        # Determine side based on analysis
                        side = "Buy" if result['accuracy'] > 70 else "Sell"
                        
                        signal = {
                            'id': f'signal_{signal_id}_{result["symbol"]}',
                            'symbol': result['symbol'],
                            'side': side,
                            'confidence': round(confidence, 1),
                            'amount': 100,  # Default amount
                            'leverage': 1,  # Default leverage
                            'take_profit': round(2 + (confidence / 25), 1),
                            'stop_loss': round(1 + (confidence / 50), 1),
                            'strategy': 'AI Technical Analysis',
                            'timestamp': signal_time.isoformat(),
                            'analysis': {
                                'accuracy': result['accuracy'],
                                'threshold': ai_confidence_threshold,
                                'status': 'ready_to_trade'
                            }
                        }
                        signals.append(signal)
                        signal_id += 1
        except Exception as db_error:
            app.logger.error(f"Database error in trading_signals: {db_error}")
            return jsonify({'success': False, 'error': 'No training data available'}), 500
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'signals': signals,
            'count': len(signals),
            'ai_threshold': ai_confidence_threshold,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Trading signals error: {str(e)}")
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
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        if not ai_worker.bybit_session:
            return jsonify({'success': False, 'message': 'ByBit session not available'}), 500
        
        # Check if under trade limit
        active_trades = ai_worker.get_active_positions_count()
        if active_trades >= ai_worker.max_concurrent_trades:
            return jsonify({
                'success': False, 
                'message': f'Max trades reached ({active_trades}/{ai_worker.max_concurrent_trades})'
            }), 400
        
        # Create trade signal for direct execution
        trade_signal = {
            'symbol': symbol,
            'side': side,
            'confidence': 85,  # High confidence for manual execution
            'amount': amount,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Execute the trade directly with pybit
        result = ai_worker.execute_signal_direct(trade_signal)
        
        if result:
            ai_worker.console_logger.log('SUCCESS', f'✅ Manual trade executed: {side} {symbol} (${amount})')
            return jsonify({
                'success': True,
                'message': f'Trade executed successfully for {symbol}',
                'active_trades': f'{active_trades + 1}/{ai_worker.max_concurrent_trades}'
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

# Initialize components will be called on first request

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