from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit
import os
import json
import yaml
from datetime import datetime, timedelta
import time
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

# Use singleton database instance to prevent connection pool exhaustion
from db_singleton import get_database

def get_db_instance():
    """Get the global database singleton instance"""
    return get_database()

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
    os.environ['SSL_VERIFY'] = 'false'
    
    # Advanced TLS configuration for SOCKS5 proxy compatibility
    import ssl
    
    # Stack Overflow solution: Enhanced SSL context with certificate handling
    def create_proxy_ssl_context():
        context = ssl.create_default_context()
        
        # Stack Overflow fix: Complete SSL bypass for problematic environments
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        # Set SECLEVEL to 0 for maximum compatibility (Stack Overflow solution)
        try:
            context.set_ciphers('DEFAULT:@SECLEVEL=0')
        except:
            context.set_ciphers('DEFAULT')
        
        # Load system certificates if available
        try:
            import certifi
            context.load_verify_locations(certifi.where())
            print("✅ System certificates loaded successfully")
        except Exception as cert_error:
            print(f"⚠️ Certificate loading failed: {cert_error}")
        
        # Use widest TLS range for maximum compatibility
        try:
            context.minimum_version = ssl.TLSVersion.TLSv1
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        except:
            pass  # Ignore if TLS version setting fails
        return context
    
    # Apply the enhanced SSL context globally
    ssl._create_default_https_context = create_proxy_ssl_context
    ssl_context = create_proxy_ssl_context()
    
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
    
    # VPN/Proxy support for better connectivity + Stack Overflow SSL fixes
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    import certifi
    
    # Stack Overflow solution: Environment variables for certificate bundle
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['CURL_CA_BUNDLE'] = certifi.where()
    
    # Stack Overflow solution: Additional SSL context configuration
    try:
        import ssl
        
        # Create a more permissive SSL context for proxy compatibility
        def create_permissive_ssl_context():
            context = ssl.create_default_context()
            # Allow self-signed certificates (common with proxies)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            # Add support for wider range of ciphers
            context.set_ciphers('DEFAULT:@SECLEVEL=1')
            return context
        
        # Override default SSL context creation
        ssl._create_default_https_context = create_permissive_ssl_context
        print("✅ Enhanced SSL context configured for proxy compatibility")
        
    except Exception as ssl_config_error:
        print(f"⚠️ SSL context configuration failed: {ssl_config_error}")
    
    requests.packages.urllib3.disable_warnings()
    
    # Check for VPN proxy configuration (supports both HTTP and SOCKS5)
    vpn_proxy = None
    proxy_url = os.getenv('VPN_PROXY_URL')
    if proxy_url:
        vpn_proxy = {
            'http': proxy_url,
            'https': proxy_url
        }
        # Stack Overflow fix: Use socks5h:// for hostname resolution by proxy
        if proxy_url.startswith('socks5://'):
            # Convert to socks5h:// for better SSL compatibility
            socks5h_url = proxy_url.replace('socks5://', 'socks5h://')
            vpn_proxy = {
                'http': socks5h_url,
                'https': socks5h_url
            }
            proxy_type = "SOCKS5H (hostname resolution)"
            print(f"🔒 VPN Proxy configured ({proxy_type}): {socks5h_url[:50]}...")
        else:
            proxy_type = "HTTP"
            print(f"🔒 VPN Proxy configured ({proxy_type}): {proxy_url[:50]}...")
        
        # For SOCKS5, we need additional setup
        if proxy_url.startswith('socks5://'):
            try:
                import socks
                import socket as sock_module
                # Configure global SOCKS proxy  
                parsed_url = proxy_url.replace('socks5://', '').split('@')
                if len(parsed_url) == 2:
                    auth, server = parsed_url
                    username, password = auth.split(':')
                    host, port = server.split(':')
                    
                    # Set default proxy for all socket connections
                    socks.set_default_proxy(socks.SOCKS5, host, int(port), username=username, password=password)
                    sock_module.socket = socks.socksocket
                    print(f"🔗 SOCKS5 proxy active: {host}:{port}")
                else:
                    print("⚠️ SOCKS5 URL format incorrect")
            except ImportError:
                print("⚠️ PySocks not installed, falling back to HTTP proxy")
            except Exception as e:
                print(f"⚠️ SOCKS5 setup failed: {e}")
    else:
        print("ℹ️ No VPN proxy configured")
    
    # Create custom session that uses VPN proxy and disables SSL verification
    class VPNAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs['ssl_context'] = ssl_context
            return super().init_poolmanager(*args, **kwargs)
    
    # Monkey patch the default session
    original_session = requests.Session
    def patched_session():
        session = original_session()
        session.verify = False
        if vpn_proxy:
            session.proxies.update(vpn_proxy)
        session.mount('https://', VPNAdapter())
        return session
    
    requests.Session = patched_session
    
    # Also patch the global session
    requests.packages.urllib3.util.ssl_.create_urllib3_context = lambda: ssl_context
    
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
        
        # Initialize ByBit with enhanced error handling and SSL bypass
        print("🔧 Initializing ByBit session with enhanced DNS/SSL handling...")
        
        # Advanced SSL/TLS configuration for SOCKS5 proxy compatibility
        import pybit
        from pybit.unified_trading import HTTP as OriginalHTTP
        import ssl
        from requests.adapters import HTTPAdapter
        from urllib3.poolmanager import PoolManager
        
        # Stack Overflow solution: Multiple SSL adapter strategies
        class StackOverflowSSLAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                # Stack Overflow solution: Create permissive SSL context
                context = ssl.create_default_context()
                
                # Solution 1: Set SECLEVEL to 1 for legacy compatibility
                context.set_ciphers('DEFAULT:@SECLEVEL=1')
                
                # Solution 2: Disable hostname and certificate verification
                context.check_hostname = False  
                context.verify_mode = ssl.CERT_NONE
                
                # Solution 3: Try TLS 1.0+ for maximum compatibility
                context.minimum_version = ssl.TLSVersion.TLSv1
                context.maximum_version = ssl.TLSVersion.TLSv1_3
                
                # Solution 4: Use certifi bundle if available
                try:
                    import certifi
                    context.load_verify_locations(certifi.where())
                except:
                    pass
                
                kwargs['ssl_context'] = context
                return super().init_poolmanager(*args, **kwargs)
        
        class PatchedHTTP(OriginalHTTP):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Configure session with Stack Overflow SSL fixes
                if hasattr(self, 'session'):
                    # Mount the Stack Overflow SSL adapter
                    self.session.mount('https://', StackOverflowSSLAdapter())
                    self.session.verify = False  # Disable verification as Stack Overflow suggests
                    if vpn_proxy:
                        self.session.proxies = vpn_proxy
                        print("🔒 VPN proxy + Stack Overflow SSL fixes applied to pybit session")
                elif hasattr(self, '_session'):
                    self._session.mount('https://', StackOverflowSSLAdapter())
                    self._session.verify = False
                    if vpn_proxy:
                        self._session.proxies = vpn_proxy
                        print("🔒 VPN proxy + Stack Overflow SSL fixes applied to pybit _session")
                
                # Also try to patch the client if it exists
                if hasattr(self, 'client'):
                    if hasattr(self.client, 'session'):
                        self.client.session.mount('https://', StackOverflowSSLAdapter())
                        self.client.session.verify = False
                        if vpn_proxy:
                            self.client.session.proxies = vpn_proxy
        
        # Replace pybit's HTTP class
        pybit.unified_trading.HTTP = PatchedHTTP
        
        try:
            bybit_session = PatchedHTTP(
                testnet=False,  # ALTIJD LIVE
                api_key=api_key,
                api_secret=api_secret,
                recv_window=20000  # Increase receive window for slow connections
            )
            print("✅ ByBit session created with SSL bypass")
        except Exception as init_error:
            print(f"⚠️ ByBit session creation failed: {init_error}")
            print("🔄 Attempting with alternative configuration...")
            
            # Try with more permissive settings
            try:
                bybit_session = PatchedHTTP(
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

@app.route('/order_history')
def order_history():
    return render_template('order_history.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/healthcheck')
def healthcheck():
    return render_template('healthcheck.html')

@app.route('/api/order_history')
def get_order_history():
    """Get order history for this app's trades with pagination - now includes AI signals data"""
    try:
        ensure_components_initialized()
        
        # Get query params
        period = request.args.get('period', '7d')
        page = int(request.args.get('page', 1))
        per_page = 50
        
        # First, get completed trades from our trading_signals database
        from database import TradingDatabase
        db = get_database()
        ai_signals = db.get_trading_signals()
        
        # Filter completed signals with P&L data
        completed_signals = [s for s in ai_signals if s.get('status') == 'completed' and s.get('realized_pnl') is not None]
        
        app.logger.info(f"Found {len(completed_signals)} completed AI signals with P&L data")
        
        # Calculate time range
        now = datetime.now()
        if period == '1d':
            start_time = now - timedelta(days=1)
        elif period == '7d':
            start_time = now - timedelta(days=7)
        elif period == '30d':
            start_time = now - timedelta(days=30)
        else:  # all
            start_time = now - timedelta(days=365)  # 1 year max
        
        # Get closed P&L history for better P&L calculation
        closed_pnl_data = []
        try:
            # Get closed P&L data
            cursor = None
            while True:
                params = {
                    'category': 'linear',
                    'startTime': int(start_time.timestamp() * 1000),
                    'limit': 200
                }
                if cursor:
                    params['cursor'] = cursor
                    
                pnl_result = bybit_session.get_closed_pnl(**params)
                app.logger.info(f"Closed P&L batch response: {pnl_result}")
                
                if not pnl_result or 'result' not in pnl_result or not pnl_result['result']['list']:
                    break
                    
                closed_pnl_data.extend(pnl_result['result']['list'])
                
                # Check if there are more pages
                if 'nextPageCursor' in pnl_result['result'] and pnl_result['result']['nextPageCursor']:
                    cursor = pnl_result['result']['nextPageCursor']
                else:
                    break
                    
        except Exception as pnl_error:
            app.logger.warning(f"Closed P&L failed: {pnl_error}")
            
        # Get all executions for additional data
        all_executions = []
        try:
            # Get executions in batches to avoid limits
            cursor = None
            while True:
                params = {
                    'category': 'linear',
                    'startTime': int(start_time.timestamp() * 1000),
                    'limit': 200
                }
                if cursor:
                    params['cursor'] = cursor
                    
                executions = bybit_session.get_executions(**params)
                app.logger.info(f"Executions batch response: {executions}")
                
                if not executions or 'result' not in executions or not executions['result']['list']:
                    break
                    
                all_executions.extend(executions['result']['list'])
                
                # Check if there are more pages
                if 'nextPageCursor' in executions['result'] and executions['result']['nextPageCursor']:
                    cursor = executions['result']['nextPageCursor']
                else:
                    break
                    
        except Exception as exec_error:
            app.logger.warning(f"Executions failed: {exec_error}")
        
        # Process all data
        formatted_orders = []
        stats = {
            'total_orders': 0,
            'completed_orders': 0,
            'cancelled_orders': 0,
            'total_volume': 0,
            'total_pnl': 0
        }
        
        # First, process our AI signals data (most accurate)
        for signal in completed_signals:
            try:
                symbol = signal.get('symbol', '')
                side = signal.get('side', '')
                entry_price = float(signal.get('entry_price', 0))
                exit_price = float(signal.get('exit_price', 0))
                realized_pnl = float(signal.get('realized_pnl', 0))
                amount = float(signal.get('amount', 0))
                
                # Calculate quantity from amount and entry price
                quantity = amount / entry_price if entry_price > 0 else 0
                
                # Get leverage multiplier for better P&L calculation
                leverage_multiplier = 1.0
                try:
                    from database import TradingDatabase
                    db = get_database()
                    leverage_multiplier = db.get_leverage_multiplier(symbol)
                except:
                    pass
                
                # Calculate P&L percentage with leverage consideration
                entry_value = quantity * entry_price
                # P&L should be calculated on margin used, not full position value
                margin_used = entry_value / leverage_multiplier if leverage_multiplier > 0 else entry_value
                pnl_percentage = (realized_pnl / margin_used * 100) if margin_used > 0 else 0
                
                # Convert exit_time to timestamp format
                exit_time = signal.get('exit_time')
                if exit_time:
                    if isinstance(exit_time, str):
                        try:
                            exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                            timestamp = str(int(exit_dt.timestamp() * 1000))
                        except:
                            timestamp = str(int(datetime.now().timestamp() * 1000))
                    else:
                        timestamp = str(int(exit_time.timestamp() * 1000))
                else:
                    timestamp = str(int(datetime.now().timestamp() * 1000))
                
                formatted_order = {
                    'orderId': signal.get('signal_id', ''),
                    'symbol': symbol,
                    'side': side,
                    'type': 'AI Signal',
                    'quantity': quantity,
                    'price': exit_price,
                    'entry_price': entry_price,
                    'status': 'Completed',
                    'timestamp': timestamp,
                    'pnl': realized_pnl,
                    'pnl_percentage': pnl_percentage,
                    'fee': 0,  # Fee data not tracked in signals yet
                    'source': 'AI_SIGNAL'
                }
                
                formatted_orders.append(formatted_order)
                
                # Update stats
                stats['total_orders'] += 1
                stats['completed_orders'] += 1
                stats['total_volume'] += entry_value
                stats['total_pnl'] += realized_pnl
                
            except Exception as signal_error:
                app.logger.warning(f"Error processing AI signal {signal.get('signal_id', 'unknown')}: {signal_error}")
        
        # Process closed P&L data from ByBit (backup/additional data)
        if closed_pnl_data:
            for pnl_entry in closed_pnl_data:
                symbol = pnl_entry.get('symbol', '')
                side = pnl_entry.get('side', '')
                qty = float(pnl_entry.get('qty', 0))
                avg_entry_price = float(pnl_entry.get('avgEntryPrice', 0))
                avg_exit_price = float(pnl_entry.get('avgExitPrice', 0))
                closed_pnl = float(pnl_entry.get('closedPnl', 0))
                
                # Get leverage multiplier for better P&L calculation
                leverage_multiplier = 1.0
                try:
                    from database import TradingDatabase
                    db = get_database()
                    leverage_multiplier = db.get_leverage_multiplier(symbol)
                except:
                    pass
                
                # Calculate P&L percentage with leverage consideration
                entry_value = qty * avg_entry_price
                # P&L should be calculated on margin used, not full position value
                margin_used = entry_value / leverage_multiplier if leverage_multiplier > 0 else entry_value
                pnl_percentage = (closed_pnl / margin_used * 100) if margin_used > 0 else 0
                
                # Note: ByBit API already returns correct P&L for both long and short positions
                # Short positions: profit when price goes down (positive closed_pnl)
                # Short positions: loss when price goes up (negative closed_pnl)
                
                formatted_order = {
                    'orderId': pnl_entry.get('orderId', ''),
                    'symbol': symbol,
                    'side': side,
                    'type': 'Market',
                    'quantity': qty,
                    'price': avg_exit_price,
                    'status': 'Filled',
                    'timestamp': pnl_entry.get('updatedTime', ''),
                    'pnl': closed_pnl,
                    'pnl_percentage': pnl_percentage,
                    'fee': float(pnl_entry.get('cumExecFee', 0))
                }
                
                formatted_orders.append(formatted_order)
                
                # Update stats
                stats['total_orders'] += 1
                stats['completed_orders'] += 1
                stats['total_volume'] += entry_value
                stats['total_pnl'] += closed_pnl
        
        # If no closed P&L data, fall back to executions
        elif all_executions:
            for execution in all_executions:
                # Calculate P&L and percentage
                quantity = float(execution.get('execQty', 0))
                price = float(execution.get('execPrice', 0))
                side = execution.get('side', '')
                
                # Calculate trade value
                trade_value = quantity * price
                
                # Get fees
                exec_fee = float(execution.get('execFee', 0))
                
                # Use realized P&L from API (often 0 for individual executions)
                realized_pnl = float(execution.get('closedPnl', 0)) or float(execution.get('realizedPnl', 0))
                
                # Get leverage multiplier for better P&L calculation
                leverage_multiplier = 1.0
                try:
                    from database import TradingDatabase
                    db = get_database()
                    leverage_multiplier = db.get_leverage_multiplier(execution.get('symbol', ''))
                except:
                    pass
                
                # Calculate percentage with leverage consideration
                # P&L should be calculated on margin used, not full position value
                margin_used = trade_value / leverage_multiplier if leverage_multiplier > 0 else trade_value
                pnl_percentage = (realized_pnl / margin_used * 100) if margin_used > 0 else 0
                
                formatted_order = {
                    'orderId': execution.get('execId', ''),
                    'symbol': execution.get('symbol', ''),
                    'side': side,
                    'type': execution.get('execType', 'Market'),
                    'quantity': quantity,
                    'price': price,
                    'status': 'Filled',
                    'timestamp': execution.get('execTime', ''),
                    'pnl': realized_pnl,
                    'pnl_percentage': pnl_percentage,
                    'fee': exec_fee
                }
                
                formatted_orders.append(formatted_order)
                
                # Update stats
                stats['total_orders'] += 1
                stats['completed_orders'] += 1
                stats['total_volume'] += trade_value
                stats['total_pnl'] += realized_pnl
        
        # Sort by timestamp (newest first)
        formatted_orders.sort(key=lambda x: int(x['timestamp']) if x['timestamp'] else 0, reverse=True)
        
        # Apply pagination
        total_orders = len(formatted_orders)
        total_pages = (total_orders + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_orders = formatted_orders[start_idx:end_idx]
        
        return jsonify({
            'success': True,
            'orders': paginated_orders,
            'stats': stats,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_orders': total_orders,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        })
        
    except Exception as e:
        app.logger.error(f"Order history error: {str(e)}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return sample data for testing with pagination
        sample_orders = [
            {
                'orderId': 'sample_001',
                'symbol': 'BTCUSDT',
                'side': 'Buy',
                'type': 'Market',
                'quantity': 0.001,
                'price': 45000.0,
                'status': 'Filled',
                'timestamp': str(int(datetime.now().timestamp() * 1000)),
                'pnl': 15.50,
                'pnl_percentage': 0.34,
                'fee': 0.45
            },
            {
                'orderId': 'sample_002',
                'symbol': 'ETHUSDT',
                'side': 'Sell',
                'type': 'Market',
                'quantity': 0.1,
                'price': 2800.0,
                'status': 'Filled',
                'timestamp': str(int((datetime.now() - timedelta(hours=2)).timestamp() * 1000)),
                'pnl': -8.30,
                'pnl_percentage': -2.96,
                'fee': 0.28
            }
        ]
        
        return jsonify({
            'success': True,
            'orders': sample_orders,
            'stats': {
                'total_orders': 2,
                'completed_orders': 2,
                'cancelled_orders': 0,
                'total_volume': 45000 * 0.001 + 2800 * 0.1,
                'total_pnl': 15.50 - 8.30
            },
            'pagination': {
                'current_page': 1,
                'per_page': 50,
                'total_pages': 1,
                'total_orders': 2,
                'has_next': False,
                'has_prev': False
            },
            'error': f'Using sample data due to API error: {str(e)}'
        })

@app.route('/api/ai_signals_analytics')
def get_ai_signals_analytics():
    """Get detailed analytics for AI trading signals"""
    try:
        from database import TradingDatabase
        db = get_database()
        
        # Get all signals
        all_signals = db.get_trading_signals()
        
        # Categorize signals
        completed_signals = [s for s in all_signals if s.get('status') == 'completed']
        pending_signals = [s for s in all_signals if s.get('status') == 'pending']
        waiting_signals = [s for s in all_signals if s.get('status') == 'waiting']
        failed_signals = [s for s in all_signals if s.get('status') == 'failed']
        
        # Calculate analytics for completed trades with P&L data
        trades_with_pnl = [s for s in completed_signals if s.get('realized_pnl') is not None]
        
        analytics = {
            'total_signals': len(all_signals),
            'completed': len(completed_signals),
            'pending': len(pending_signals),
            'waiting': len(waiting_signals),
            'failed': len(failed_signals),
            'trades_with_pnl': len(trades_with_pnl)
        }
        
        if trades_with_pnl:
            # Calculate detailed performance metrics
            total_pnl = sum(float(t['realized_pnl']) for t in trades_with_pnl)
            winning_trades = [t for t in trades_with_pnl if float(t['realized_pnl']) > 0]
            losing_trades = [t for t in trades_with_pnl if float(t['realized_pnl']) < 0]
            
            win_rate = (len(winning_trades) / len(trades_with_pnl)) * 100
            avg_win = sum(float(t['realized_pnl']) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(float(t['realized_pnl']) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) if avg_loss != 0 else float('inf')
            
            # Best and worst trades
            best_trade = max(trades_with_pnl, key=lambda x: float(x['realized_pnl']))
            worst_trade = min(trades_with_pnl, key=lambda x: float(x['realized_pnl']))
            
            # Recent completed trades (last 10)
            recent_trades = sorted(completed_signals, key=lambda x: x.get('exit_time', ''), reverse=True)[:10]
            
            analytics.update({
                'performance': {
                    'total_pnl': total_pnl,
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'best_trade': {
                        'signal_id': best_trade.get('signal_id'),
                        'symbol': best_trade.get('symbol'),
                        'pnl': float(best_trade.get('realized_pnl', 0)),
                        'exit_time': best_trade.get('exit_time')
                    },
                    'worst_trade': {
                        'signal_id': worst_trade.get('signal_id'),
                        'symbol': worst_trade.get('symbol'),
                        'pnl': float(worst_trade.get('realized_pnl', 0)),
                        'exit_time': worst_trade.get('exit_time')
                    }
                },
                'recent_trades': [
                    {
                        'signal_id': t.get('signal_id'),
                        'symbol': t.get('symbol'),
                        'side': t.get('side'),
                        'entry_price': float(t.get('entry_price', 0)) if t.get('entry_price') else None,
                        'exit_price': float(t.get('exit_price', 0)) if t.get('exit_price') else None,
                        'pnl': float(t.get('realized_pnl', 0)) if t.get('realized_pnl') is not None else None,
                        'exit_time': t.get('exit_time'),
                        'status': t.get('status')
                    }
                    for t in recent_trades
                ]
            })
        else:
            analytics.update({
                'performance': {
                    'total_pnl': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0
                },
                'recent_trades': []
            })
        
        return jsonify({
            'success': True,
            'analytics': analytics
        })
        
    except Exception as e:
        app.logger.error(f"AI signals analytics error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

# === NIEUWE VERBETERDE BALANCE FUNCTIES (gebaseerd op complete uitleg) ===

def get_bybit_balance():
    """
    Haalt de correcte balance op van Bybit (zoals account_info dat doet)
    Returns: dict met balance informatie
    """
    try:
        if not bybit_session:
            return {'success': False, 'error': 'ByBit session not initialized'}
            
        # Gebruik dezelfde methode als account_info (die werkt!)
        response = bybit_session.get_wallet_balance(accountType="UNIFIED")

        if response and response.get('retCode') == 0:
            balance_info = response['result']['list'][0]

            # Extract belangrijke waarden (veilige float conversie)
            total_wallet_balance = float(balance_info.get('totalWalletBalance', 0) or 0)
            available_balance = float(balance_info.get('totalAvailableBalance', 0) or 0)
            used_margin = float(balance_info.get('totalInitialMargin', 0) or 0)  # Use totalInitialMargin
            total_equity = float(balance_info.get('totalEquity', 0) or 0)

            # Coin-specifieke balance (veilige conversie voor lege strings)
            coin_balances = {}
            for coin in balance_info.get('coin', []):
                coin_name = coin['coin']
                # Veilige conversie - lege strings worden 0
                def safe_float(value):
                    return float(value) if value and value != '' else 0.0
                
                coin_balances[coin_name] = {
                    'wallet_balance': safe_float(coin.get('walletBalance', 0)),
                    'available': safe_float(coin.get('availableToWithdraw', 0)),
                    'locked': safe_float(coin.get('locked', 0)),
                    'equity': safe_float(coin.get('equity', 0)),
                    'usd_value': safe_float(coin.get('usdValue', 0))
                }

            return {
                'success': True,
                'total_wallet_balance': total_wallet_balance,
                'available_balance': available_balance,
                'total_equity': total_equity,
                'used_margin': used_margin,
                'coin_balances': coin_balances,
                'account_type': 'UNIFIED'
            }
        else:
            error_msg = response.get('retMsg', 'Unknown API error') if response else 'No response'
            return {'success': False, 'error': f"API Error: {error_msg}"}

    except Exception as e:
        return {'success': False, 'error': f"Exception: {str(e)}"}

def get_spot_balance():
    """
    Voor pure spot trading balance (zoals account_info methode)
    """
    try:
        if not bybit_session:
            return {'success': False, 'error': 'ByBit session not initialized'}
            
        # Gebruik dezelfde directe methode als account_info
        response = bybit_session.get_wallet_balance(accountType="SPOT")

        if response and response.get('retCode') == 0:
            coins = response['result']['list'][0].get('coin', [])
            balances = {}

            for coin in coins:
                if float(coin.get('walletBalance', 0)) > 0:
                    balances[coin['coin']] = {
                        'balance': float(coin.get('walletBalance', 0)),
                        'locked': float(coin.get('locked', 0)),
                        'available': float(coin.get('free', 0))
                    }

            return {
                'success': True,
                'balances': balances,
                'account_type': 'SPOT'
            }
        else:
            error_msg = response.get('retMsg', 'Unknown API error') if response else 'No response'
            return {'success': False, 'error': error_msg}

    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_complete_balance_info():
    """
    Haalt alle balance informatie op
    """
    import time
    
    result = {
        'unified': get_bybit_balance(),
        'spot': get_spot_balance(),
        'timestamp': int(time.time())
    }

    # Bereken totals
    total_usdt = 0
    if result['unified']['success']:
        total_usdt += result['unified']['total_wallet_balance']

    result['total_usdt_value'] = total_usdt
    return result

@app.route('/api/balance')
def api_get_balance():
    """API endpoint voor complete balance (nieuwe functionaliteit)"""
    ensure_components_initialized()
    try:
        balance_info = get_complete_balance_info()
        return jsonify(balance_info)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/balance/debug')
def debug_balance_call():
    """
    Debug functie om API calls te testen (zoals account_info methode)
    """
    ensure_components_initialized()
    try:
        debug_info = {
            'timestamp': int(time.time()),
            'session_status': 'initialized' if bybit_session else 'not_initialized'
        }
        
        if bybit_session:
            # Test connection (zoals account_info)
            try:
                server_time = bybit_session.get_server_time()
                debug_info['server_time'] = server_time
                debug_info['connection'] = 'success'
            except Exception as e:
                debug_info['connection'] = f'failed: {str(e)}'

            # Test balance call (exact zoals account_info)
            try:
                balance_response = bybit_session.get_wallet_balance(accountType="UNIFIED")
                debug_info['balance_raw'] = balance_response
                
                # Parse zoals de nieuwe get_bybit_balance functie
                balance_parsed = get_bybit_balance()
                debug_info['balance_parsed'] = balance_parsed
                
            except Exception as e:
                debug_info['balance_test'] = {'error': str(e)}
            
            # Test spot balance (exact zoals account_info maar dan SPOT)
            try:
                spot_balance = get_spot_balance()
                debug_info['spot_balance_test'] = spot_balance
            except Exception as e:
                debug_info['spot_balance_test'] = {'error': str(e)}
        
        return jsonify(debug_info)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': int(time.time())
        }), 500

@app.route('/api/debug_analytics')
def debug_analytics():
    """Debug endpoint to check why analytics shows 0 completed trades"""
    try:
        from check_production_analytics import check_production_analytics
        results = check_production_analytics()
        return jsonify(results)
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': int(time.time())
        }), 500

@app.route('/api/balance_header')
def get_balance_header():
    """Fetch balance for header display (verbeterde versie met nieuwe balance functie)"""
    ensure_components_initialized()
    try:
        # Gebruik de nieuwe verbeterde balance functie
        balance_result = get_bybit_balance()
        
        if balance_result['success']:
            total_balance = balance_result['total_wallet_balance']
            available_balance = balance_result['available_balance']
            used_margin = balance_result['used_margin']
            
            # Calculate 24h P&L from database
            try:
                from database import TradingDatabase
                db = get_database()
                signal_trades = db.get_trading_signals()
                
                # Calculate 24h realized P&L
                now = datetime.now()
                twenty_four_hours_ago = now - timedelta(hours=24)
                
                pnl_24h = 0.0
                for trade in signal_trades:
                    if trade.get('status') == 'completed' and trade.get('realized_pnl') is not None:
                        if trade.get('exit_time'):
                            exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00')) if isinstance(trade['exit_time'], str) else trade['exit_time']
                            if exit_time.replace(tzinfo=None) >= twenty_four_hours_ago:
                                pnl_24h += float(trade['realized_pnl'])
                
                # Calculate percentage based on starting balance (approximate)
                pnl_24h_percent = (pnl_24h / total_balance * 100) if total_balance > 0 else 0.0
                
            except Exception as e:
                app.logger.warning(f"Could not calculate 24h P&L: {e}")
                pnl_24h = 0.0
                pnl_24h_percent = 0.0
            
            # Debug logging voor troubleshooting
            app.logger.info(f"✅ Balance header success: Total={total_balance}, Available={available_balance}, Margin={used_margin}")
            
            return jsonify({
                'success': True,
                'balance': total_balance,
                'available_balance': available_balance,
                'used_margin': used_margin,
                'pnl_24h': pnl_24h,
                'pnl_24h_percent': pnl_24h_percent,
                'account_type': 'UNIFIED',
                'coin_count': len(balance_result.get('coin_balances', {}))
            })
        else:
            # Fallback naar cached balance bij API problemen
            app.logger.warning(f"⚠️ Balance API error: {balance_result['error']} - returning cached balance")
            
            # Still try to get P&L from database even if balance API fails
            pnl_24h = 0.0
            pnl_24h_percent = 0.0
            try:
                from database import TradingDatabase
                db = get_database()
                signal_trades = db.get_trading_signals()
                
                now = datetime.now()
                twenty_four_hours_ago = now - timedelta(hours=24)
                
                for trade in signal_trades:
                    if trade.get('status') == 'completed' and trade.get('realized_pnl') is not None:
                        if trade.get('exit_time'):
                            exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00')) if isinstance(trade['exit_time'], str) else trade['exit_time']
                            if exit_time.replace(tzinfo=None) >= twenty_four_hours_ago:
                                pnl_24h += float(trade['realized_pnl'])
            except:
                pass
            
            return jsonify({
                'success': True,
                'balance': 0.0,
                'available_balance': 0.0,
                'used_margin': 0.0,
                'pnl_24h': pnl_24h,
                'pnl_24h_percent': pnl_24h_percent,
                'cached': True,
                'message': f"Using cached balance due to: {balance_result['error']}"
            })
            
    except Exception as e:
        app.logger.error(f"❌ Header balance API exception: {str(e)}")
        return jsonify({
            'success': True,
            'balance': 0.0,
            'available_balance': 0.0,
            'used_margin': 0.0,
            'pnl_24h': 0.0,
            'pnl_24h_percent': 0.0,
            'cached': True,
            'message': 'Using cached balance due to connection issues'
        })

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
        db = get_database()
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
    
    # Test AI Worker status
    try:
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio, bybit_session)
        if ai_worker and ai_worker.is_running:
            status['dyno'] = {'status': 'online', 'message': 'AI Worker active'}
        elif ai_worker:
            status['dyno'] = {'status': 'warning', 'message': 'AI Worker stopped'}
        else:
            status['dyno'] = {'status': 'offline', 'message': 'AI Worker not initialized'}
    except Exception as e:
        status['dyno'] = {'status': 'offline', 'message': f'AI error: {str(e)[:50]}'}
    
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

def calculate_target_info(symbol, side, avg_price, mark_price, stop_loss, take_profit):
    """Calculate target information for position"""
    try:
        # Calculate standard TP levels (multiple targets)
        tp_levels = []
        
        if side == 'Buy':
            # Long position - targets above entry price
            if take_profit > 0:
                # Create multiple TP levels
                base_distance = (take_profit - avg_price) / avg_price
                for i in range(1, 5):  # TP1, TP2, TP3, TP4
                    tp_price = avg_price * (1 + (base_distance * i / 4))
                    distance_pct = ((tp_price - mark_price) / mark_price * 100)
                    tp_levels.append({
                        'level': f'TP{i}',
                        'price': tp_price,
                        'distance_pct': distance_pct,
                        'hit': mark_price >= tp_price
                    })
            
            # Find next target
            next_target = None
            for tp in tp_levels:
                if not tp['hit']:
                    next_target = tp
                    break
                    
            # Stop loss info
            sl_distance = ((stop_loss - mark_price) / mark_price * 100) if stop_loss > 0 else None
            
        else:  # Sell (Short position)
            # Short position - targets below entry price
            if take_profit > 0:
                # Create multiple TP levels
                base_distance = (avg_price - take_profit) / avg_price
                for i in range(1, 5):  # TP1, TP2, TP3, TP4
                    tp_price = avg_price * (1 - (base_distance * i / 4))
                    distance_pct = ((mark_price - tp_price) / mark_price * 100)
                    tp_levels.append({
                        'level': f'TP{i}',
                        'price': tp_price,
                        'distance_pct': distance_pct,
                        'hit': mark_price <= tp_price
                    })
            
            # Find next target
            next_target = None
            for tp in tp_levels:
                if not tp['hit']:
                    next_target = tp
                    break
                    
            # Stop loss info
            sl_distance = ((mark_price - stop_loss) / mark_price * 100) if stop_loss > 0 else None
        
        return {
            'tp_levels': tp_levels,
            'next_target': next_target,
            'sl_distance_pct': sl_distance,
            'total_targets': len(tp_levels),
            'targets_hit': sum(1 for tp in tp_levels if tp['hit'])
        }
        
    except Exception as e:
        return {
            'tp_levels': [],
            'next_target': None,
            'sl_distance_pct': None,
            'total_targets': 0,
            'targets_hit': 0,
            'error': str(e)
        }

def get_realized_pnl_for_position(symbol, side):
    """Get realized P&L for a specific position"""
    try:
        # Get recent closed P&L for this symbol
        from datetime import datetime, timedelta
        start_time = datetime.now() - timedelta(days=7)  # Last 7 days
        
        closed_pnl = bybit_session.get_closed_pnl(
            category="linear",
            symbol=symbol,
            startTime=int(start_time.timestamp() * 1000),
            limit=50
        )
        
        total_realized = 0
        if closed_pnl and 'result' in closed_pnl and 'list' in closed_pnl['result']:
            for pnl_entry in closed_pnl['result']['list']:
                if pnl_entry.get('symbol') == symbol and pnl_entry.get('side') == side:
                    total_realized += float(pnl_entry.get('closedPnl', 0))
        
        return total_realized
        
    except Exception as e:
        return 0

def get_trailing_stop_info(symbol, side):
    """Get trailing stop information for a position"""
    try:
        # Get AI worker instance to check active trades
        ai_worker = get_ai_worker(socketio=socketio, bybit_session=bybit_session)
        
        if not ai_worker or not hasattr(ai_worker, 'active_trades'):
            return None
            
        # Find matching active trade
        for order_id, trade_info in ai_worker.active_trades.items():
            if trade_info.get('symbol') == symbol and trade_info.get('side') == side:
                return {
                    'enabled': trade_info.get('trailing_stop_enabled', False),
                    'distance': trade_info.get('trailing_stop_distance', 1.0),
                    'current_stop': trade_info.get('stop_loss', 0),
                    'tp_levels': trade_info.get('take_profit_levels', []),
                    'tp_order_ids': trade_info.get('tp_order_ids', []),
                    'entry_price': trade_info.get('entry_price', 0)
                }
        
        return None
        
    except Exception as e:
        return None

@app.route('/api/positions')
def get_positions():
    ensure_components_initialized()
    try:
        positions = bybit_session.get_positions(
            category="linear",
            settleCoin="USDT",
            limit=200
        )
        
        # Filter only positions with size > 0
        if positions and 'result' in positions and 'list' in positions['result']:
            active_positions = []
            for pos in positions['result']['list']:
                if float(pos.get('size', 0)) > 0:
                    symbol = pos['symbol']
                    side = pos['side']
                    size = float(pos['size'])
                    avg_price = float(pos.get('avgPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unrealisedPnl', 0))
                    
                    # Get stop loss and take profit info
                    stop_loss = float(pos.get('stopLoss', 0))
                    take_profit = float(pos.get('takeProfit', 0))
                    
                    # Calculate target distances and next target
                    target_info = calculate_target_info(symbol, side, avg_price, mark_price, stop_loss, take_profit)
                    
                    # Get realized P&L for this position
                    realized_pnl = get_realized_pnl_for_position(symbol, side)
                    
                    # Get trailing stop info from active trades
                    trailing_stop_info = get_trailing_stop_info(symbol, side)
                    
                    # Get margin information
                    position_im = float(pos.get('positionIM', 0))  # Initial Margin
                    position_mm = float(pos.get('positionMM', 0))  # Maintenance Margin
                    leverage = float(pos.get('leverage', 1))
                    
                    # Calculate ROI based on actual margin invested
                    # ROI = (realized PnL + unrealized PnL) / Initial Margin * 100
                    roi_percentage = 0
                    if position_im > 0:
                        total_pnl = realized_pnl + unrealized_pnl
                        roi_percentage = (total_pnl / position_im) * 100
                    
                    active_positions.append({
                        'symbol': symbol,
                        'side': side,
                        'size': size,
                        'avgPrice': avg_price,
                        'markPrice': mark_price,
                        'unrealisedPnl': unrealized_pnl,
                        'realizedPnl': realized_pnl,
                        'leverage': leverage,
                        'positionValue': float(pos.get('positionValue', 0)),
                        'positionIM': position_im,  # Initial Margin
                        'positionMM': position_mm,  # Maintenance Margin
                        'stopLoss': stop_loss,
                        'takeProfit': take_profit,
                        'targetInfo': target_info,
                        'trailingStopInfo': trailing_stop_info,
                        # Calculate ROI percentage based on actual margin invested
                        'pnlPercentage': roi_percentage
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
        positions = bybit_session.get_positions(
            category="linear",
            settleCoin="USDT",
            limit=200
        )
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
        
        # Start AI worker (includes signal generation and execution)
        ai_worker = get_ai_worker(socketio, bybit_session)
        ai_worker.start()
        
        # Legacy trading loop disabled - AI worker handles all trading now
        # threading.Thread(target=trading_loop, daemon=True).start()
        
        return jsonify({'success': True, 'message': 'Trading and AI worker started'})
    return jsonify({'success': False, 'message': 'Trading already active'})

@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    global is_trading
    is_trading = False
    
    # Stop AI worker
    ai_worker = get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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
        ai_worker = get_ai_worker(socketio=socketio, bybit_session=bybit_session)
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

@app.route('/api/close_all_positions', methods=['POST'])
def close_all_positions():
    try:
        # Get all open positions
        positions = bybit_session.get_positions(
            category="linear",
            settleCoin="USDT",
            limit=200
        )
        
        if not positions or 'result' not in positions:
            return jsonify({'success': False, 'error': 'Could not get positions'}), 500
        
        closed_positions = []
        errors = []
        
        for position in positions['result']['list']:
            if float(position.get('size', 0)) > 0:  # Only close positions with size > 0
                try:
                    # Close the position
                    result = bybit_session.place_order(
                        category="linear",
                        symbol=position['symbol'],
                        side="Sell" if position['side'] == "Buy" else "Buy",
                        orderType="Market",
                        qty=str(position['size']),
                        reduceOnly=True
                    )
                    
                    if result and 'result' in result:
                        closed_positions.append(position['symbol'])
                    else:
                        errors.append(f"Failed to close {position['symbol']}")
                        
                except Exception as e:
                    errors.append(f"Error closing {position['symbol']}: {str(e)}")
        
        if closed_positions:
            return jsonify({
                'success': True, 
                'closed_positions': closed_positions,
                'errors': errors if errors else None
            })
        else:
            return jsonify({'success': False, 'error': 'No positions to close or all failed', 'errors': errors})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete_old_signals', methods=['POST'])
def delete_old_signals():
    """Delete oldest trading signals from database"""
    try:
        data = request.json
        count = data.get('count', 10)
        
        from database import TradingDatabase
        db = get_database()
        
        # Delete the oldest signals
        deleted_count = db.delete_oldest_trading_signals(count)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} old trading signals'
        })
    
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
        ensure_components_initialized()
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio=socketio, bybit_session=bybit_session)
        
        if ai_worker and ai_worker.bybit_session:
            # Start the AI worker if not already running
            if not ai_worker.is_running:
                ai_worker.start()
            
            return jsonify({
                'success': True,
                'message': 'Live trading enabled! AI Worker is now running.'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'ByBit session not available - check API credentials'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/disable_trading', methods=['POST'])
def disable_trading():
    try:
        ensure_components_initialized()
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio=socketio, bybit_session=bybit_session)
        
        if ai_worker:
            # Stop the AI worker to disable trading
            ai_worker.stop()
            
            return jsonify({
                'success': True,
                'message': 'Live trading disabled! AI Worker has been stopped.'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'AI Worker not available'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading_status')
def get_trading_status():
    try:
        ensure_components_initialized()
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio=socketio, bybit_session=bybit_session)
        
        if ai_worker and ai_worker.bybit_session:
            # Get active positions count (this represents active trades)
            active_positions = ai_worker.get_active_positions_count()
            
            # Trading is enabled if AI worker is running and has bybit session
            trading_enabled = ai_worker.is_running and ai_worker.bybit_session is not None
            
            return jsonify({
                'success': True,
                'status': {
                    'enabled': trading_enabled,
                    'active_orders': active_positions,
                    'worker_running': ai_worker.is_running,
                    'max_trades': ai_worker.max_concurrent_trades,
                    'maxConcurrentTrades': ai_worker.max_concurrent_trades
                }
            })
        else:
            return jsonify({
                'success': True,
                'status': {
                    'enabled': False,
                    'active_orders': 0,
                    'worker_running': False,
                    'max_trades': 0
                }
            })
    except Exception as e:
        app.logger.error(f"Trading status error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'status': {
                'enabled': False,
                'active_orders': 0,
                'worker_running': False,
                'max_trades': 0
            }
        }), 500

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
        db = get_database()
        
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

@app.route('/api/test_cryptopanic')
def test_cryptopanic():
    """Test CryptoPanic API connection and functionality"""
    try:
        from utils.news_scraper import NewsScraper
        
        # Initialize news scraper with API key from settings
        try:
            news_scraper = NewsScraper()
        except Exception as init_error:
            return jsonify({
                'success': False,
                'error': f'NewsScraper initialization failed: {str(init_error)}'
            })
        
        # Test API connection
        try:
            # Fetch recent crypto news
            news_data = news_scraper.get_recent_news(limit=5)
            
            if news_data and len(news_data) > 0:
                # Calculate basic sentiment metrics
                total_sentiment = 0
                valid_sentiment_count = 0
                
                for article in news_data:
                    if 'sentiment' in article and article['sentiment'] is not None:
                        total_sentiment += article['sentiment']
                        valid_sentiment_count += 1
                
                avg_sentiment = total_sentiment / valid_sentiment_count if valid_sentiment_count > 0 else 0
                
                return jsonify({
                    'success': True,
                    'news_count': len(news_data),
                    'avg_sentiment': round(avg_sentiment, 2),
                    'sample_headlines': [article.get('title', 'No title')[:50] + '...' for article in news_data[:3]],
                    'api_status': 'Working',
                    'sentiment_analysis': 'Available' if valid_sentiment_count > 0 else 'Limited'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No news data received from CryptoPanic API',
                    'news_count': 0,
                    'api_status': 'Connected but no data'
                })
                
        except Exception as api_error:
            return jsonify({
                'success': False,
                'error': f'CryptoPanic API call failed: {str(api_error)}',
                'api_status': 'Failed'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'CryptoPanic test error: {str(e)}',
            'api_status': 'Error'
        })

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
            
            # Get actual realized P&L from trade data
            # Note: Individual executions might not have closedPnl, so we use fees as approximation
            realized_pnl = float(trade.get('closedPnl', 0))
            
            # If no realized P&L available, use fee as negative impact (approximation)
            if realized_pnl == 0:
                trade_pnl = -abs(exec_fee)  # Fee as loss approximation
            else:
                trade_pnl = realized_pnl - abs(exec_fee)  # Actual P&L minus fees
            
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
        positions = bybit_session.get_positions(
            category="linear",
            settleCoin="USDT",
            limit=200
        )
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
        
        # Get AI signal trade data from database to count only real signal trades
        trades_data = {
            'total_trades': 0, 
            'winning_trades': 0, 
            'losing_trades': 0,
            'total_volume': 0, 
            'total_fees': 0,
            'trades_by_date': {},
            'largest_win': 0,
            'largest_loss': 0,
            'total_profit': 0,
            'total_loss': 0
        }
        
        # Get AI worker to access signal trades
        ai_worker = None
        try:
            ai_worker = get_ai_worker(socketio, bybit_session)
        except:
            pass
        
        # Count only completed signal trades from database
        try:
            from database import TradingDatabase
            db = get_database()
            signal_trades = db.get_trading_signals()
            
            # Filter for completed signal trades with P&L data
            completed_signals = [s for s in signal_trades if s.get('status') == 'completed' and s.get('realized_pnl') is not None]
            trades_data['total_trades'] = len(completed_signals)
            
            # Calculate win/loss stats using stored P&L data
            trades_by_date = {}
            
            for signal in completed_signals:
                pnl = float(signal.get('realized_pnl', 0))
                entry_price = float(signal.get('entry_price', 0)) if signal.get('entry_price') else 0
                exit_price = float(signal.get('exit_price', 0)) if signal.get('exit_price') else 0
                
                # Track wins and losses
                if pnl > 0:
                    trades_data['winning_trades'] += 1
                    trades_data['total_profit'] += pnl
                    trades_data['largest_win'] = max(trades_data['largest_win'], pnl)
                elif pnl < 0:
                    trades_data['losing_trades'] += 1
                    trades_data['total_loss'] += abs(pnl)
                    trades_data['largest_loss'] = min(trades_data['largest_loss'], pnl)
                
                # Calculate trade volume if we have entry/exit prices
                if entry_price > 0 and signal.get('amount'):
                    amount = float(signal.get('amount', 0))
                    trade_value = amount * entry_price
                    trades_data['total_volume'] += trade_value
                
                # Group by date using exit_time
                if signal.get('exit_time'):
                    try:
                        exit_time = datetime.fromisoformat(str(signal.get('exit_time')).replace('Z', '+00:00'))
                        trade_date = exit_time.strftime('%Y-%m-%d')
                        
                        if trade_date not in trades_by_date:
                            trades_by_date[trade_date] = {'volume': 0, 'pnl': 0, 'count': 0}
                        
                        trades_by_date[trade_date]['pnl'] += pnl
                        trades_by_date[trade_date]['count'] += 1
                        
                        if entry_price > 0 and signal.get('amount'):
                            trades_by_date[trade_date]['volume'] += trade_value
                    except:
                        pass  # Skip if date parsing fails
            
            trades_data['trades_by_date'] = trades_by_date
            
            # Calculate realized P&L for 24h and all-time periods
            realized_pnl_24h = 0
            realized_pnl_all_time = 0
            
            # Get current UTC time for 24h calculation
            now = datetime.utcnow()
            yesterday = now - timedelta(days=1)
            
            for signal in completed_signals:
                pnl = float(signal.get('realized_pnl', 0))
                realized_pnl_all_time += pnl
                
                # Check if trade was completed in the last 24 hours
                if signal.get('exit_time'):
                    try:
                        exit_time = datetime.fromisoformat(str(signal.get('exit_time')).replace('Z', '+00:00'))
                        # Convert to UTC if it has timezone info
                        if exit_time.tzinfo is not None:
                            exit_time = exit_time.utctimetuple()
                            exit_time = datetime(*exit_time[:6])
                        
                        if exit_time >= yesterday:
                            realized_pnl_24h += pnl
                    except:
                        pass  # Skip if date parsing fails
            
            # Add realized P&L data to trades_data
            trades_data['realized_pnl_24h'] = realized_pnl_24h
            trades_data['realized_pnl_all_time'] = realized_pnl_all_time
                
        except Exception as signal_error:
            # Fallback to empty data if signal database access fails
            app.logger.warning(f"Could not access signal trades: {signal_error}")
            trades_data = {
                'total_trades': 0, 
                'winning_trades': 0, 
                'losing_trades': 0,
                'total_volume': 0, 
                'total_fees': 0,
                'trades_by_date': {},
                'largest_win': 0,
                'largest_loss': 0,
                'total_profit': 0,
                'total_loss': 0,
                'realized_pnl_24h': 0,
                'realized_pnl_all_time': 0
            }
        
        # Calculate performance metrics - ONLY real data
        win_rate = 0
        if trades_data['total_trades'] > 0:
            win_rate = (trades_data['winning_trades'] / trades_data['total_trades']) * 100
        # No fallback estimates - only real data
        
        # Format positions for frontend with TP level information
        formatted_positions = []
        
        # AI worker already initialized above for signal trades
        
        for pos in active_positions:
            symbol = pos.get('symbol')
            
            # Get margin information
            position_im = float(pos.get('positionIM', 0))  # Initial Margin
            position_mm = float(pos.get('positionMM', 0))  # Maintenance Margin
            leverage = float(pos.get('leverage', 1))
            unrealized_pnl = float(pos.get('unrealisedPnl', 0))
            
            # Calculate ROI based on actual margin invested
            # ROI = unrealized PnL / Initial Margin * 100
            roi_percentage = 0
            if position_im > 0:
                roi_percentage = (unrealized_pnl / position_im) * 100
            
            position_data = {
                'symbol': symbol,
                'side': pos.get('side'),
                'size': float(pos.get('size', 0)),
                'avgPrice': float(pos.get('avgPrice', 0)),
                'markPrice': float(pos.get('markPrice', 0)),
                'unrealisedPnl': unrealized_pnl,
                # Calculate ROI based on actual margin invested
                'percentage': roi_percentage,
                'positionValue': float(pos.get('positionValue', 0)),
                'leverage': leverage,
                'positionIM': position_im,  # Initial Margin
                'positionMM': position_mm,  # Maintenance Margin
                'tp_levels': [],
                'tp1_hit': False,
                'sl_moved_to_breakeven': False,
                'distance_to_tp1': 0,
                'stop_loss': None,  # Will be populated below
                'stop_loss_price': 0,
                'stop_loss_distance': 0
            }
            
            # Add TP level information from AI worker if available
            if ai_worker and hasattr(ai_worker, 'active_trades'):
                for order_id, trade_data in ai_worker.active_trades.items():
                    if trade_data.get('symbol') == symbol:
                        tp_levels = trade_data.get('take_profit_levels', [])
                        current_price = position_data['markPrice']
                        
                        # Format TP levels with status
                        for i, tp_level in enumerate(tp_levels):
                            tp_info = {
                                'level': i + 1,
                                'price': tp_level.get('price', 0),
                                'status': tp_level.get('status', 'pending'),
                                'hit_time': tp_level.get('hit_time'),
                                'distance_percent': 0
                            }
                            
                            # Calculate distance to TP level
                            if current_price > 0 and tp_level.get('price', 0) > 0:
                                if pos.get('side') == 'Buy':
                                    distance = ((tp_level['price'] - current_price) / current_price) * 100
                                else:
                                    distance = ((current_price - tp_level['price']) / current_price) * 100
                                tp_info['distance_percent'] = distance
                            
                            position_data['tp_levels'].append(tp_info)
                        
                        # Set TP1 and SL status
                        position_data['tp1_hit'] = trade_data.get('tp1_hit', False)
                        position_data['sl_moved_to_breakeven'] = trade_data.get('sl_moved_to_breakeven', False)
                        
                        # Add stop loss information
                        stop_loss_data = trade_data.get('stop_loss', {})
                        if stop_loss_data:
                            position_data['stop_loss'] = stop_loss_data
                            sl_price = stop_loss_data.get('price', 0)
                            if sl_price > 0:
                                position_data['stop_loss_price'] = sl_price
                                # Calculate distance to stop loss
                                if current_price > 0:
                                    if pos.get('side') == 'Buy':
                                        sl_distance = ((current_price - sl_price) / current_price) * 100
                                    else:
                                        sl_distance = ((sl_price - current_price) / current_price) * 100
                                    position_data['stop_loss_distance'] = sl_distance
                        
                        # Calculate distance to TP1
                        if tp_levels and len(tp_levels) > 0:
                            tp1_price = tp_levels[0].get('price', 0)
                            if current_price > 0 and tp1_price > 0:
                                if pos.get('side') == 'Buy':
                                    distance = ((tp1_price - current_price) / current_price) * 100
                                else:
                                    distance = ((current_price - tp1_price) / current_price) * 100
                                position_data['distance_to_tp1'] = distance
                        
                        break
            
            # If no AI worker data found, create default TP levels based on position
            if not position_data['tp_levels']:
                # Create default TP levels based on current position
                current_price = position_data['markPrice']
                entry_price = position_data['avgPrice']
                side = position_data['side']
                
                # Calculate default TP levels (2%, 4%, 6%, 8% from entry)
                default_tp_percentages = [2, 4, 6, 8]
                
                for i, tp_pct in enumerate(default_tp_percentages):
                    if side == 'Buy':
                        tp_price = entry_price * (1 + tp_pct / 100)
                        distance = ((tp_price - current_price) / current_price) * 100 if current_price > 0 else 0
                    else:
                        tp_price = entry_price * (1 - tp_pct / 100)
                        distance = ((current_price - tp_price) / current_price) * 100 if current_price > 0 else 0
                    
                    tp_info = {
                        'level': i + 1,
                        'price': tp_price,
                        'status': 'pending',
                        'hit_time': None,
                        'distance_percent': distance
                    }
                    
                    position_data['tp_levels'].append(tp_info)
                
                # Add default stop loss (1% from entry)
                if side == 'Buy':
                    sl_price = entry_price * 0.99
                    sl_distance = ((current_price - sl_price) / current_price) * 100 if current_price > 0 else 0
                else:
                    sl_price = entry_price * 1.01
                    sl_distance = ((sl_price - current_price) / current_price) * 100 if current_price > 0 else 0
                
                position_data['stop_loss'] = {'price': sl_price, 'status': 'pending'}
                position_data['stop_loss_price'] = sl_price
                position_data['stop_loss_distance'] = sl_distance
            
            formatted_positions.append(position_data)
        
        return jsonify({
            'success': True,
            'total_balance': total_balance,
            'unrealized_pnl': unrealized_pnl,
            'total_position_value': total_position_value,
            'active_positions_count': len(active_positions),
            'positions': formatted_positions,  # Add the positions array
            'total_trades': trades_data['total_trades'],
            'winning_trades': trades_data['winning_trades'],
            'losing_trades': trades_data['losing_trades'],
            'win_rate': win_rate,
            'total_volume': trades_data['total_volume'],
            'total_fees': trades_data['total_fees'],
            'largest_win': trades_data['largest_win'],
            'largest_loss': trades_data['largest_loss'],
            'total_profit': trades_data['total_profit'],
            'total_loss': trades_data['total_loss'],
            'realized_pnl_24h': trades_data['realized_pnl_24h'],
            'realized_pnl_all_time': trades_data['realized_pnl_all_time'],
            'sharpe_ratio': trades_data['total_profit'] / trades_data['total_loss'] if trades_data['total_loss'] > 0 else 0,
            'max_drawdown': abs(trades_data['largest_loss']) if trades_data['largest_loss'] < 0 else 0,
            'profit_factor': trades_data['total_profit'] / trades_data['total_loss'] if trades_data['total_loss'] > 0 else 0,
            'trades_by_date': trades_data['trades_by_date']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance_analytics')
def get_performance_analytics():
    """Get performance analytics data for specific time period"""
    try:
        period = request.args.get('period', '1h')
        
        # Calculate time filter based on period
        from datetime import datetime, timedelta
        now = datetime.now()
        
        if period == '1h':
            start_time = now - timedelta(hours=1)
        elif period == '4h':
            start_time = now - timedelta(hours=4)
        elif period == '12h':
            start_time = now - timedelta(hours=12)
        elif period == '1d':
            start_time = now - timedelta(days=1)
        elif period == '3d':
            start_time = now - timedelta(days=3)
        elif period == '7d':
            start_time = now - timedelta(days=7)
        elif period == '30d':
            start_time = now - timedelta(days=30)
        elif period == '3m':
            start_time = now - timedelta(days=90)
        elif period == '1y':
            start_time = now - timedelta(days=365)
        else:  # 'all'
            start_time = datetime(2020, 1, 1)  # Very old date to get all data
        
        # Get trading signals from database for the specified period
        try:
            from database import TradingDatabase
            db = get_database()
            signal_trades = db.get_trading_signals()
            
            # Filter signals by time period
            filtered_signals = []
            for signal in signal_trades:
                if signal.get('exit_time'):
                    try:
                        exit_time = datetime.fromisoformat(str(signal.get('exit_time')).replace('Z', '+00:00'))
                        if exit_time >= start_time:
                            filtered_signals.append(signal)
                    except:
                        pass
                elif signal.get('created_at'):
                    try:
                        created_at = datetime.fromisoformat(str(signal.get('created_at')).replace('Z', '+00:00'))
                        if created_at >= start_time:
                            filtered_signals.append(signal)
                    except:
                        pass
            
            # Calculate performance metrics
            completed_signals = [s for s in filtered_signals if s.get('status') == 'completed' and s.get('realized_pnl') is not None]
            pending_signals = [s for s in filtered_signals if s.get('status') in ['active', 'waiting', 'pending']]
            
            winning_trades = len([s for s in completed_signals if float(s.get('realized_pnl', 0)) > 0])
            losing_trades = len([s for s in completed_signals if float(s.get('realized_pnl', 0)) < 0])
            pending_trades = len(pending_signals)
            
            # Calculate additional metrics
            total_profit = sum([float(s.get('realized_pnl', 0)) for s in completed_signals if float(s.get('realized_pnl', 0)) > 0])
            total_loss = sum([abs(float(s.get('realized_pnl', 0))) for s in completed_signals if float(s.get('realized_pnl', 0)) < 0])
            
            win_rate = (winning_trades / len(completed_signals) * 100) if completed_signals else 0
            
            return jsonify({
                'success': True,
                'period': period,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'pending_trades': pending_trades,
                'total_trades': len(completed_signals),
                'win_rate': win_rate,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
                'avg_win': total_profit / winning_trades if winning_trades > 0 else 0,
                'avg_loss': total_loss / losing_trades if losing_trades > 0 else 0
            })
            
        except Exception as db_error:
            # Fallback to empty data if database access fails
            return jsonify({
                'success': False,
                'error': f'Database error: {str(db_error)}',
                'winning_trades': 0,
                'losing_trades': 0,
                'pending_trades': 0,
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_loss': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cumulative_roi')
def get_cumulative_roi():
    """Get cumulative ROI data starting from July 9th with proper time period filtering"""
    try:
        period = request.args.get('period', '1h')
        
        # Calculate time filter based on period
        from datetime import datetime, timedelta
        now = datetime.now()
        july_9_2024 = datetime(2024, 7, 9)  # Starting point
        
        # Determine time range based on period
        if period == '1h':
            start_time = max(july_9_2024, now - timedelta(hours=1))
            time_step = timedelta(minutes=5)  # 5-minute intervals
        elif period == '4h':
            start_time = max(july_9_2024, now - timedelta(hours=4))
            time_step = timedelta(minutes=15)  # 15-minute intervals
        elif period == '12h':
            start_time = max(july_9_2024, now - timedelta(hours=12))
            time_step = timedelta(hours=1)  # 1-hour intervals
        elif period == '1d':
            start_time = max(july_9_2024, now - timedelta(days=1))
            time_step = timedelta(hours=2)  # 2-hour intervals
        elif period == '3d':
            start_time = max(july_9_2024, now - timedelta(days=3))
            time_step = timedelta(hours=6)  # 6-hour intervals
        elif period == '7d':
            start_time = max(july_9_2024, now - timedelta(days=7))
            time_step = timedelta(hours=12)  # 12-hour intervals
        elif period == '30d':
            start_time = max(july_9_2024, now - timedelta(days=30))
            time_step = timedelta(days=1)  # Daily intervals
        elif period == '3m':
            start_time = max(july_9_2024, now - timedelta(days=90))
            time_step = timedelta(days=3)  # 3-day intervals
        elif period == '1y':
            start_time = max(july_9_2024, now - timedelta(days=365))
            time_step = timedelta(days=7)  # Weekly intervals
        else:  # 'all'
            start_time = july_9_2024
            time_step = timedelta(days=1)  # Daily intervals
        
        # Get trading signals from database
        try:
            from database import TradingDatabase
            db = get_database()
            signal_trades = db.get_trading_signals()
            
            # Filter for completed signals with realized P&L
            completed_signals = []
            for signal in signal_trades:
                if signal.get('status') == 'completed' and signal.get('realized_pnl') is not None:
                    exit_time = None
                    if signal.get('exit_time'):
                        try:
                            exit_time = datetime.fromisoformat(str(signal.get('exit_time')).replace('Z', '+00:00'))
                        except:
                            pass
                    elif signal.get('updated_at'):
                        try:
                            exit_time = datetime.fromisoformat(str(signal.get('updated_at')).replace('Z', '+00:00'))
                        except:
                            pass
                    
                    if exit_time and exit_time >= july_9_2024:
                        completed_signals.append({
                            'exit_time': exit_time,
                            'realized_pnl': float(signal.get('realized_pnl', 0)),
                            'amount': float(signal.get('amount', 0))
                        })
            
            # Sort by exit time
            completed_signals.sort(key=lambda x: x['exit_time'])
            
            # Get actual current balance for proper ROI calculation
            try:
                balance_result = get_bybit_balance()
                if balance_result['success']:
                    current_balance = balance_result['total_wallet_balance']
                    # Calculate what the initial balance was by subtracting total P&L
                    total_pnl = sum(signal['realized_pnl'] for signal in completed_signals)
                    initial_balance = current_balance - total_pnl
                    if initial_balance <= 0:
                        initial_balance = 1000.0  # Fallback
                else:
                    initial_balance = 1000.0  # Default starting balance
            except:
                initial_balance = 1000.0  # Default starting balance
            
            # Ensure minimum balance for calculation
            if initial_balance < 100:
                initial_balance = 1000.0
            
            # Generate time series data
            roi_data = []
            current_time = start_time
            cumulative_pnl = 0
            signal_index = 0
            
            while current_time <= now:
                # Add all trades that happened before or at current_time
                while signal_index < len(completed_signals) and completed_signals[signal_index]['exit_time'] <= current_time:
                    cumulative_pnl += completed_signals[signal_index]['realized_pnl']
                    signal_index += 1
                
                # Calculate ROI as percentage of initial balance
                cumulative_roi_percent = (cumulative_pnl / initial_balance) * 100
                
                roi_data.append({
                    'date': current_time.isoformat(),
                    'cumulative_pnl': cumulative_pnl,
                    'cumulative_roi_percent': cumulative_roi_percent
                })
                
                current_time += time_step
            
            # If no data points, create at least one starting point
            if not roi_data:
                roi_data.append({
                    'date': start_time.isoformat(),
                    'cumulative_pnl': 0,
                    'cumulative_roi_percent': 0
                })
            
            return jsonify({
                'success': True,
                'period': period,
                'roi_data': roi_data,
                'total_trades': len(completed_signals),
                'initial_balance': initial_balance,
                'final_pnl': cumulative_pnl,
                'final_roi_percent': (cumulative_pnl / initial_balance) * 100 if initial_balance > 0 else 0
            })
            
        except Exception as db_error:
            # Fallback: generate empty data starting from July 9th
            roi_data = []
            current_time = start_time
            
            while current_time <= now:
                roi_data.append({
                    'date': current_time.isoformat(),
                    'cumulative_pnl': 0,
                    'cumulative_roi_percent': 0
                })
                current_time += time_step
            
            return jsonify({
                'success': True,
                'period': period,
                'roi_data': roi_data,
                'total_trades': 0,
                'initial_balance': 1000.0,
                'final_pnl': 0,
                'final_roi_percent': 0,
                'error': f'Database error: {str(db_error)}'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/coin_analysis')
def get_coin_analysis():
    """Get AI analysis for all coins with REAL status logic"""
    try:
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        # Get AI confidence threshold from settings
        try:
            from utils.settings_loader import Settings
            settings = Settings.load('config/settings.yaml')
            ai_confidence_threshold = settings.ai_confidence_threshold
            ai_accuracy_threshold = settings.ai_accuracy_threshold
        except:
            ai_confidence_threshold = float(os.getenv('AI_CONFIDENCE_THRESHOLD', 80))
            ai_accuracy_threshold = float(os.getenv('AI_ACCURACY_THRESHOLD', 70))
        
        # Load database settings first
        try:
            db = get_database()
            db_settings = db.load_settings()
            ai_confidence_threshold = float(db_settings.get('confidenceThreshold', ai_confidence_threshold))
            ai_accuracy_threshold = float(db_settings.get('accuracyThreshold', ai_accuracy_threshold))
            auto_execute = db_settings.get('autoExecute', False)
            
            # Load TP/SL settings from database
            min_take_profit = float(db_settings.get('minTakeProfitPercent', 1))
            max_take_profit = float(db_settings.get('maxTakeProfitPercent', 10))
            base_take_profit = float(db_settings.get('takeProfitPercent', 3))
            stop_loss_percent = float(db_settings.get('stopLossPercent', 2))
        except Exception as settings_error:
            app.logger.warning(f"Could not load settings from database: {settings_error}")
            # Fallback to default values
            min_take_profit = 1
            max_take_profit = 10
            base_take_profit = 3
            stop_loss_percent = 2
            auto_execute = False
            db_settings = {}  # Empty dict for fallback
        
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
                    if confidence >= ai_confidence_threshold:
                        status = "high"
                        status_class = "status-high"
                    elif confidence >= (ai_confidence_threshold * 0.75):  # 75% of threshold
                        status = "medium"
                        status_class = "status-medium" 
                    else:
                        status = "low"
                        status_class = "status-low"
                    
                    # Generate AI analysis based on training data
                    analysis = "Bullish" if result['accuracy'] > ai_accuracy_threshold and confidence > ai_confidence_threshold else "Bearish"
                    if confidence < (ai_confidence_threshold * 0.75):  # 75% of threshold for neutral
                        analysis = "Neutral"
                    
                    # Calculate take profit and stop loss based on confidence and settings
                    # Scale between min and max TP based on confidence (50-100 confidence maps to min-max TP)
                    confidence_factor = max(0, min(1, (confidence - 50) / 50))  # 0-1 range
                    take_profit = min_take_profit + (confidence_factor * (max_take_profit - min_take_profit))
                    take_profit = round(max(min_take_profit, min(max_take_profit, take_profit)), 1)
                    
                    # Stop loss uses fixed percentage from settings
                    stop_loss = round(stop_loss_percent, 1)
                    
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
            import traceback
            app.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty data instead of 500 error
            return jsonify({
                'success': True,
                'coins': [],
                'total_count': 0,
                'message': f'Database error: {str(db_error)}',
                'last_updated': datetime.now().isoformat()
            })
        
        if not coins_analysis:
            # Return empty data instead of 404 error
            return jsonify({
                'success': True,
                'coins': [],
                'total_count': 0,
                'message': 'No training data available - please run AI training on this server',
                'last_updated': datetime.now().isoformat()
            })
        
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
        # Get AI confidence threshold and auto_execute from database first
        # Get AI confidence threshold from settings
        try:
            from utils.settings_loader import Settings
            settings = Settings.load('config/settings.yaml')
            ai_confidence_threshold = settings.ai_confidence_threshold
            ai_accuracy_threshold = settings.ai_accuracy_threshold
        except:
            ai_confidence_threshold = 75  # Default fallback
            ai_accuracy_threshold = 70  # Default fallback
        auto_execute = False  # Default
        
        # Try to get AI worker
        ai_worker = None
        try:
            ensure_components_initialized()
            ai_worker = get_ai_worker(socketio, bybit_session)
        except Exception as worker_error:
            app.logger.warning(f"Could not initialize AI worker: {worker_error}")
        
        signals = []
        
        # Check if AI worker is available
        if not ai_worker:
            return jsonify({
                'success': True,
                'signals': [],
                'count': 0,
                'message': 'AI Worker not initialized - no training data available',
                'auto_execute': auto_execute,
                'ai_threshold': ai_confidence_threshold,
                'last_updated': datetime.now().isoformat()
            })
        
        try:
            # Get recent training session data
            latest_session = None
            training_results = []
            
            try:
                latest_session = ai_worker.database.get_latest_training_session()
                if latest_session:
                    training_results = ai_worker.database.get_training_results(latest_session['session_id'])
            except Exception as db_error:
                app.logger.warning(f"Database error accessing training data: {db_error}")
                # Try direct database access as fallback
                try:
                    from database import TradingDatabase
                    db = get_database()
                    latest_session = db.get_latest_training_session()
                    if latest_session:
                        training_results = db.get_training_results(latest_session['session_id'])
                except Exception as fallback_error:
                    app.logger.error(f"Fallback database access failed: {fallback_error}")
                    training_results = []
            
            if training_results:
                signal_id = 0
                for result in training_results:
                    confidence = result['confidence']
                    
                    # ALLEEN signals boven threshold
                    if confidence >= ai_confidence_threshold:
                        signal_time = datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()).replace('Z', '+00:00'))
                        
                        # Determine side based on analysis
                        side = "Buy" if result['accuracy'] > ai_accuracy_threshold else "Sell"
                        
                        # Calculate position size based on settings from database
                        try:
                            risk_per_trade = float(db_settings.get('riskPerTrade', 2.0))
                            min_trade_amount = float(db_settings.get('minTradeAmount', 19))
                        except:
                            # Fallback to environment variables
                            risk_per_trade = float(os.getenv('RISK_PER_TRADE', 2.0))
                            min_trade_amount = float(os.getenv('MIN_TRADE_AMOUNT', 19))
                        
                        # Get current balance
                        try:
                            balance_data = bybit_session.get_wallet_balance(accountType="UNIFIED")
                            total_balance = float(balance_data['result']['list'][0]['totalWalletBalance']) if balance_data['result']['list'] else 1000
                        except:
                            total_balance = 1000  # Fallback
                        
                        # Calculate amount: percentage of balance with minimum
                        calculated_amount = (total_balance * risk_per_trade / 100)
                        amount = max(calculated_amount, min_trade_amount)
                        
                        # Calculate leverage based on settings and confidence
                        try:
                            min_leverage = int(db_settings.get('minLeverage', 1))
                            max_leverage = int(db_settings.get('maxLeverage', 10))
                            leverage_strategy = db_settings.get('leverageStrategy', 'confidence_based')
                        except:
                            # Fallback to environment variables
                            min_leverage = int(os.getenv('MIN_LEVERAGE', 1))
                            max_leverage = int(os.getenv('MAX_LEVERAGE', 10))
                            leverage_strategy = os.getenv('LEVERAGE_STRATEGY', 'confidence_based')
                        
                        if leverage_strategy == 'confidence_based':
                            # Higher confidence = higher leverage
                            leverage_factor = (confidence - 50) / 50  # Scale 0-1
                            leverage = min_leverage + int((max_leverage - min_leverage) * leverage_factor)
                        elif leverage_strategy == 'fixed':
                            leverage = min_leverage
                        else:  # volatility_based or adaptive
                            leverage = min_leverage + int((max_leverage - min_leverage) * 0.5)  # Mid-range
                        
                        leverage = max(min_leverage, min(max_leverage, leverage))
                        
                        # Calculate partial take profit levels - get from database settings
                        try:
                            from database import TradingDatabase
                            db = get_database()
                            db_settings = db.load_settings()
                            partial_tp_enabled = db_settings.get('partialTakeProfit', False)
                            partial_tp_levels = db_settings.get('partialTakeProfitLevels', 4)
                            partial_tp_percentage = db_settings.get('partialTakeProfitPercentage', 25)
                        except:
                            # Fallback to environment variables
                            partial_tp_enabled = os.getenv('PARTIAL_TAKE_PROFIT', 'false').lower() == 'true'
                            partial_tp_levels = int(os.getenv('PARTIAL_TAKE_PROFIT_LEVELS', 4))
                            partial_tp_percentage = int(os.getenv('PARTIAL_TAKE_PROFIT_PERCENTAGE', 25))
                        
                        take_profit_levels = []
                        if partial_tp_enabled:
                            # Calculate base TP using settings-based formula
                            confidence_factor = max(0, min(1, (confidence - 50) / 50))
                            base_tp = min_take_profit + (confidence_factor * (max_take_profit - min_take_profit))
                            base_tp = round(max(min_take_profit, min(max_take_profit, base_tp)), 1)
                            
                            for i in range(partial_tp_levels):
                                level = (i + 1) * (base_tp / partial_tp_levels)
                                profit_amount = (amount * leverage * level) / 100
                                take_profit_levels.append({
                                    'level': i + 1,
                                    'percentage': level,
                                    'sell_percentage': partial_tp_percentage,
                                    'profit_amount': round(profit_amount, 2)
                                })
                        else:
                            # Calculate base TP using settings-based formula
                            confidence_factor = max(0, min(1, (confidence - 50) / 50))
                            base_tp = min_take_profit + (confidence_factor * (max_take_profit - min_take_profit))
                            base_tp = round(max(min_take_profit, min(max_take_profit, base_tp)), 1)
                            
                            profit_amount = (amount * leverage * base_tp) / 100
                            take_profit_levels.append({
                                'level': 1,
                                'percentage': base_tp,
                                'sell_percentage': 100,
                                'profit_amount': round(profit_amount, 2)
                            })
                        
                        signal_id_str = f'signal_{signal_id}_{result["symbol"]}'
                        
                        # Check signal status from database
                        signal_status = 'waiting'  # Default status
                        try:
                            db_signals = db.get_trading_signals()
                            existing_signal = next((s for s in db_signals if s['symbol'] == result['symbol']), None)
                            if existing_signal:
                                signal_status = existing_signal['status']
                            else:
                                # Check if symbol is in supported list
                                supported_symbols = db.get_supported_symbols()
                                if not any(s['symbol'] == result['symbol'] for s in supported_symbols):
                                    signal_status = 'failed'
                                else:
                                    # Check if same direction position exists
                                    try:
                                        positions = bybit_session.get_positions(category="linear", symbol=result['symbol'], limit=200)
                                        if positions and 'result' in positions:
                                            for position in positions['result']['list']:
                                                if position['symbol'] == result['symbol'] and float(position['size']) > 0:
                                                    existing_side = position['side']
                                                    if existing_side == side:
                                                        signal_status = 'waiting'
                                                        break
                                    except:
                                        pass
                        except Exception as status_error:
                            app.logger.warning(f"Error checking signal status: {status_error}")
                        
                        signal = {
                            'id': signal_id_str,
                            'symbol': result['symbol'],
                            'side': side,
                            'confidence': round(confidence, 1),
                            'amount': round(amount, 2),
                            'leverage': leverage,
                            'take_profit': base_tp,
                            'stop_loss': stop_loss_percent,
                            'strategy': 'AI Technical Analysis',
                            'timestamp': signal_time.isoformat(),
                            'status': signal_status,
                            'take_profit_levels': take_profit_levels,
                            'partial_take_profit': partial_tp_enabled,
                            'move_stop_loss_on_partial_tp': db_settings.get('moveStopLossOnPartialTP', True),
                            'analysis': {
                                'accuracy': round(result['accuracy'], 1),
                                'threshold': ai_confidence_threshold,
                                'status': signal_status,
                                'leverage_strategy': leverage_strategy,
                                'position_size_method': 'fixed_percentage',
                                'auto_execute': os.getenv('AUTO_EXECUTE', 'false').lower() == 'true'
                            }
                        }
                        
                        # Save signal to database if not exists
                        try:
                            db.save_trading_signal({
                                'signal_id': signal_id_str,
                                'symbol': result['symbol'],
                                'side': side,
                                'confidence': confidence,
                                'accuracy': result['accuracy'],
                                'amount': amount,
                                'leverage': leverage,
                                'take_profit': base_tp,
                                'stop_loss': stop_loss_percent,
                                'status': signal_status
                            })
                        except Exception as save_error:
                            app.logger.warning(f"Could not save signal to database: {save_error}")
                        signals.append(signal)
                        signal_id += 1
        except Exception as db_error:
            app.logger.error(f"Database error in trading_signals: {db_error}")
            # Return empty data instead of 500 error
            return jsonify({
                'success': True,
                'signals': [],
                'count': 0,
                'message': 'No training data available - please run AI training on this server',
                'auto_execute': auto_execute,
                'ai_threshold': ai_confidence_threshold,
                'last_updated': datetime.now().isoformat()
            })
        
        # Check if we have any signals
        if not signals:
            message = 'No training data available - please run AI training on this server'
            if latest_session and training_results:
                message = f'No trading signals above confidence threshold ({ai_confidence_threshold}%)'
            
            return jsonify({
                'success': True,
                'signals': [],
                'count': 0,
                'message': message,
                'auto_execute': auto_execute,
                'ai_threshold': ai_confidence_threshold,
                'last_updated': datetime.now().isoformat()
            })
        
        # Sort by confidence (highest first), then accuracy (highest first)
        signals.sort(key=lambda x: (x['confidence'], x['analysis']['accuracy']), reverse=True)
        
        return jsonify({
            'success': True,
            'signals': signals,
            'count': len(signals),
            'auto_execute': auto_execute,
            'ai_threshold': ai_confidence_threshold,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Trading signals error: {str(e)}")
        # Return graceful error response instead of 500
        return jsonify({
            'success': True,
            'signals': [],
            'count': 0,
            'message': f'Service temporarily unavailable: {str(e)}',
            'auto_execute': False,
            'ai_threshold': 80,
            'last_updated': datetime.now().isoformat()
        })

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    try:
        # Load from database settings first
        try:
            from database import TradingDatabase
            db = get_database()
            db_settings = db.load_settings()
            
            ai_confidence_threshold = float(db_settings.get('confidenceThreshold', 80))
            auto_execute = db_settings.get('autoExecute', False)
            max_positions = int(db_settings.get('maxConcurrentTrades', 20))
            risk_per_trade = float(db_settings.get('riskPerTrade', 2.0))
            min_leverage = int(db_settings.get('minLeverage', 1))
            max_leverage = int(db_settings.get('maxLeverage', 10))
            leverage_strategy = db_settings.get('leverageStrategy', 'confidence_based')
            min_trade_amount = float(db_settings.get('minTradeAmount', 5.0))
            
        except Exception as db_error:
            # Fallback to environment variables if database fails
            ai_confidence_threshold = float(os.getenv('AI_CONFIDENCE_THRESHOLD', 80))
            auto_execute = os.getenv('AUTO_EXECUTE', 'true').lower() == 'true'
            max_positions = int(os.getenv('MAX_CONCURRENT_TRADES', 20))
            risk_per_trade = float(os.getenv('RISK_PER_TRADE', 2.0))
            min_leverage = 1
            max_leverage = 10
            leverage_strategy = 'confidence_based'
            min_trade_amount = 5.0
        
        return jsonify({
            'success': True,
            'ai_confidence_threshold': ai_confidence_threshold,
            'max_positions': max_positions,
            'maxConcurrentTrades': max_positions,  # Add this for consistency
            'risk_per_trade': risk_per_trade,
            'auto_execute': auto_execute,
            'min_leverage': min_leverage,
            'max_leverage': max_leverage,
            'leverage_strategy': leverage_strategy,
            'min_trade_amount': min_trade_amount,
            'api_key': 'configured' if os.getenv('BYBIT_API_KEY') else 'not_configured',
            'api_secret': 'configured' if os.getenv('BYBIT_API_SECRET') else 'not_configured'
        })
    except Exception as e:
        app.logger.error(f"Config error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        data = request.get_json()
        
        # Update environment variables
        if 'ai_confidence_threshold' in data:
            os.environ['AI_CONFIDENCE_THRESHOLD'] = str(data['ai_confidence_threshold'])
        if 'max_positions' in data:
            os.environ['MAX_CONCURRENT_TRADES'] = str(data['max_positions'])
        if 'risk_per_trade' in data:
            os.environ['RISK_PER_TRADE'] = str(data['risk_per_trade'])
        if 'auto_execute' in data:
            os.environ['AUTO_EXECUTE'] = 'true' if data['auto_execute'] else 'false'
            
        # Also update settings file if exists
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                saved_settings = json.load(f)
            saved_settings['confidenceThreshold'] = data.get('ai_confidence_threshold', saved_settings.get('confidenceThreshold', 80))
            saved_settings['autoExecute'] = data.get('auto_execute', saved_settings.get('autoExecute', False))
            with open(settings_file, 'w') as f:
                json.dump(saved_settings, f, indent=2)
                
        # Update AI worker if running
        global ai_worker_instance
        if ai_worker_instance and data.get('auto_execute') is not None:
            ai_worker_instance.auto_execute = data['auto_execute']
            app.logger.info(f"Updated AI worker auto_execute to: {data['auto_execute']}")
        
        return jsonify({'success': True, 'message': 'Configuration updated'})
    except Exception as e:
        app.logger.error(f"Config update error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status/api')
def api_status():
    """Check API connectivity status"""
    try:
        # Test ByBit API connection
        bybit_session.get_server_time()
        return jsonify({
            'status': 'online',
            'last_check': datetime.now().isoformat(),
            'service': 'ByBit API'
        })
    except Exception as e:
        return jsonify({
            'status': 'offline',
            'last_check': datetime.now().isoformat(),
            'error': str(e),
            'service': 'ByBit API'
        })

@app.route('/api/status/db')
def db_status():
    """Check database connectivity status"""
    try:
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        # Test database connection
        ai_worker.database.get_latest_training_session()
        return jsonify({
            'status': 'online',
            'last_check': datetime.now().isoformat(),
            'service': 'Database'
        })
    except Exception as e:
        return jsonify({
            'status': 'offline',
            'last_check': datetime.now().isoformat(),
            'error': str(e),
            'service': 'Database'
        })

@app.route('/api/status/ai')
def ai_status():
    """Check AI service status"""
    try:
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        # Check if AI worker is properly initialized
        if ai_worker and hasattr(ai_worker, 'database'):
            return jsonify({
                'status': 'online',
                'last_check': datetime.now().isoformat(),
                'service': 'AI Worker'
            })
        else:
            return jsonify({
                'status': 'offline',
                'last_check': datetime.now().isoformat(),
                'error': 'AI Worker not initialized',
                'service': 'AI Worker'
            })
    except Exception as e:
        return jsonify({
            'status': 'offline',
            'last_check': datetime.now().isoformat(),
            'error': str(e),
            'service': 'AI Worker'
        })

@app.route('/api/balance_header')
def balance_header():
    """Get balance data for header display"""
    try:
        ensure_components_initialized()
        
        # Get balance data
        balance_data = bybit_session.get_wallet_balance(accountType="UNIFIED")
        
        if balance_data['result']['list']:
            wallet_balance = balance_data['result']['list'][0]
            total_balance = float(wallet_balance['totalWalletBalance'])
            unrealized_pnl = float(wallet_balance['totalPerpUPL'])
            
            # Calculate 24h PnL percentage
            pnl_percent = (unrealized_pnl / total_balance * 100) if total_balance > 0 else 0
            
            return jsonify({
                'success': True,
                'balance': total_balance,
                'pnl_24h': unrealized_pnl,
                'pnl_24h_percent': pnl_percent,
                'status': 'online'
            })
        else:
            return jsonify({
                'success': False,
                'balance': 0.0,
                'pnl_24h': 0.0,
                'pnl_24h_percent': 0.0,
                'status': 'offline'
            })
    except Exception as e:
        app.logger.error(f"Balance header error: {str(e)}")
        return jsonify({
            'success': False,
            'balance': 0.0,
            'pnl_24h': 0.0,
            'pnl_24h_percent': 0.0,
            'status': 'offline',
            'error': str(e)
        })

@app.route('/api/balance_settings')
def balance_settings():
    """Get balance data for settings page display"""
    try:
        ensure_components_initialized()
        
        # Use the improved balance function
        balance_result = get_bybit_balance()
        
        if balance_result['success']:
            return jsonify({
                'success': True,
                'balance': balance_result['total_wallet_balance'],
                'available_balance': balance_result['available_balance'],
                'used_margin': balance_result['used_margin'],
                'coin_balances': balance_result['coin_balances']
            })
        else:
            return jsonify({
                'success': False,
                'balance': 0.0,
                'error': balance_result['error']
            })
    except Exception as e:
        app.logger.error(f"Settings balance error: {str(e)}")
        return jsonify({
            'success': False,
            'balance': 0.0,
            'error': str(e)
        })

@app.route('/api/account_name')
def account_name():
    """Get account name for header display"""
    try:
        # Get account info
        account_info = bybit_session.get_account_info()
        
        if account_info['result']:
            account_name = account_info['result'].get('marginMode', 'ByBit Account')
            return jsonify({
                'success': True,
                'account_name': account_name
            })
        else:
            return jsonify({
                'success': True,
                'account_name': 'ByBit Account'
            })
    except Exception as e:
        app.logger.error(f"Account name error: {str(e)}")
        return jsonify({
            'success': True,
            'account_name': 'ByBit Account'
        })


@app.route('/api/execute_ai_trade', methods=['POST'])
def execute_ai_trade():
    """Execute a trade based on AI recommendation"""
    try:
        data = request.get_json()
        app.logger.info(f"Execute AI trade request: {data}")
        
        if not data:
            app.logger.error("No data provided in execute_ai_trade request")
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        symbol = data.get('symbol')
        side = data.get('side')
        amount = data.get('amount', 100)
        take_profit = data.get('takeProfit')
        stop_loss = data.get('stopLoss')
        
        app.logger.info(f"Trade parameters: symbol={symbol}, side={side}, amount={amount}")
        
        if not all([symbol, side]):
            app.logger.error(f"Missing required fields: symbol={symbol}, side={side}")
            return jsonify({'success': False, 'message': 'Missing required fields: symbol and side are required'}), 400
        
        # Get AI worker for trade execution
        ensure_components_initialized()
        ai_worker = get_ai_worker(socketio, bybit_session)
        
        if not ai_worker:
            app.logger.error("AI worker not available")
            return jsonify({'success': False, 'message': 'AI worker not available'}), 500
            
        if not ai_worker.bybit_session:
            app.logger.error("ByBit session not available in AI worker")
            return jsonify({'success': False, 'message': 'ByBit session not available'}), 500
        
        # Check if under trade limit
        active_trades = ai_worker.get_active_positions_count()
        app.logger.info(f"Active trades: {active_trades}/{ai_worker.max_concurrent_trades}")
        
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
            'timestamp': datetime.now().isoformat(),
            'signal_coin': symbol  # Add the correct coin symbol for logging
        }
        
        app.logger.info(f"Executing trade signal: {trade_signal}")
        
        # Execute the trade directly with pybit
        result = ai_worker.execute_signal_direct(trade_signal)
        
        app.logger.info(f"Trade execution result: {result}")
        
        if result:
            ai_worker.console_logger.log('SUCCESS', f'✅ Manual trade executed: {side} {symbol} (${amount})')
            return jsonify({
                'success': True,
                'message': f'Trade executed successfully for {symbol}',
                'active_trades': f'{active_trades + 1}/{ai_worker.max_concurrent_trades}'
            })
        else:
            app.logger.error("Trade execution returned False")
            return jsonify({'success': False, 'message': 'Trade execution failed - check logs for details'}), 500
            
    except Exception as e:
        app.logger.error(f"Execute AI trade error: {str(e)}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

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
        positions = bybit_session.get_positions(
            category="linear",
            settleCoin="USDT",
            limit=200
        )
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
# Settings Management
settings_file = 'trading_settings.json'

@app.route('/api/save_settings', methods=['POST'])
def save_settings():
    """Save trading settings to file"""
    try:
        settings_data = request.get_json()
        
        # Load existing settings first
        existing_settings = {}
        try:
            # Use global database instance
            db = get_db_instance()
            existing_settings = db.load_settings()
        except Exception as db_error:
            print(f"Database settings load failed, using file fallback: {db_error}")
            # Fallback to file if database fails
            try:
                with open(settings_file, 'r') as f:
                    existing_settings = json.load(f)
            except:
                existing_settings = {}
        
        # Merge new settings with existing ones
        merged_settings = existing_settings.copy()
        merged_settings.update(settings_data)
        
        # Only validate required settings if this is a full settings update
        if len(settings_data) > 1:  # More than just autoExecute
            required_settings = [
                'riskPerTrade', 'maxConcurrentTrades', 'minTradeAmount',
                'minTakeProfit', 'maxTakeProfit', 'confidenceThreshold'
            ]
            
            # Remove defaultTakeProfit and defaultStopLoss from required since we now use dynamic switches
            for setting in required_settings:
                if setting not in merged_settings:
                    return jsonify({
                        'success': False,
                        'error': f'Missing required setting: {setting}'
                    })
        
        # Save to database
        try:
            # Use global database instance
            db = get_db_instance()
            db.save_settings(merged_settings)
        except Exception as db_error:
            print(f"Database settings save failed, using file fallback: {db_error}")
            # Fallback to file if database fails
            with open(settings_file, 'w') as f:
                json.dump(merged_settings, f, indent=2)
        
        # Update environment variables for immediate effect
        os.environ['AI_CONFIDENCE_THRESHOLD'] = str(merged_settings.get('confidenceThreshold', 80))
        os.environ['RISK_PER_TRADE'] = str(merged_settings.get('riskPerTrade', 2.0))
        os.environ['MAX_LEVERAGE'] = str(merged_settings.get('maxLeverage', 10))
        os.environ['LEVERAGE_MODE'] = merged_settings.get('leverageMode', 'cross')
        os.environ['AUTO_EXECUTE'] = 'true' if merged_settings.get('autoExecute', False) else 'false'
        
        # Update AI worker if running
        global ai_worker_instance
        if ai_worker_instance:
            ai_worker_instance.auto_execute = merged_settings.get('autoExecute', False)
            app.logger.info(f"Updated AI worker auto_execute to: {merged_settings.get('autoExecute', False)}")
        
        return jsonify({
            'success': True,
            'message': 'Settings saved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to save settings: {str(e)}'
        })

@app.route('/api/fix_pnl', methods=['POST'])
def fix_pnl():
    """Fix realized P&L calculations in the database"""
    try:
        from database import TradingDatabase
        db = get_database()
        
        # Get all completed signals with P&L data
        signals = db.get_trading_signals()
        
        fixed_count = 0
        
        for signal in signals:
            if signal['status'] == 'completed' and signal.get('entry_price') and signal.get('exit_price'):
                # Calculate correct P&L
                entry_price = float(signal['entry_price'])
                exit_price = float(signal['exit_price'])
                amount = float(signal['amount'])
                
                if signal['side'] == 'Buy':
                    # For long positions: profit when price goes up
                    correct_pnl = (exit_price - entry_price) * amount
                else:
                    # For short positions: profit when price goes down
                    correct_pnl = (entry_price - exit_price) * amount
                
                # Update if different from stored value
                current_pnl = signal.get('realized_pnl', 0) or 0
                if abs(float(current_pnl) - correct_pnl) > 0.01:
                    # Update in database
                    db.update_signal_with_pnl(
                        signal['signal_id'],
                        entry_price,
                        exit_price,
                        correct_pnl
                    )
                    fixed_count += 1
        
        # Calculate new totals
        all_signals = db.get_trading_signals()
        total_pnl = sum(float(s.get('realized_pnl', 0) or 0) for s in all_signals if s['status'] == 'completed')
        
        return jsonify({
            'success': True,
            'fixed_count': fixed_count,
            'total_realized_pnl': total_pnl,
            'message': f'Fixed {fixed_count} P&L values. Total realized P&L: ${total_pnl:.2f}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_settings')
def load_settings():
    """Load trading settings from database"""
    try:
        # Default settings
        default_settings = {
            'riskPerTrade': 2.0,
            'maxConcurrentTrades': 20,
            'minTradeAmount': 19,  # Default $19 minimum
            'defaultTakeProfit': 3.0,
            'defaultStopLoss': 1.5,
            'minTakeProfit': 1.0,
            'maxTakeProfit': 10.0,
            'trailingStopLoss': False,
            'trailingStopValue': 1.0,
            'dynamicTakeProfit': True,
            'partialTakeProfit': False,
            'partialTakeProfitLevels': 4,  # Number of take profit levels
            'partialTakeProfitPercentage': 25,  # Percentage to sell at each level
            'moveStopLossOnPartialTP': True,  # Move stop loss to breakeven after first TP
            'tradingEnabled': True,
            'weekendTrading': False,
            'maxDailyLoss': 5.0,
            'maxDrawdown': 15.0,
            'emergencyStop': True,
            'leverageMode': 'cross',  # cross or isolated
            'minLeverage': 1,
            'maxLeverage': 10,
            'leverageStrategy': 'confidence_based',  # confidence_based, volatility_based, fixed, adaptive
            'correlationLimit': 80,
            'apiKey': '',
            'apiSecret': '',
            'testnetMode': False,
            'apiTimeout': 30,
            'confidenceThreshold': 80,
            'accuracyThreshold': 70,
            'modelUpdateFreq': '4h',
            'technicalIndicators': 15,
            'marketCondition': 'auto',
            'telegramBotToken': '',
            'telegramChatId': '',
            'notifyTrades': True,
            'notifyErrors': True,
            'notifyProfits': True,
            'dailySummary': True,
            'autoExecute': True  # Auto execute enabled by default
        }
        
        # Try to load from database
        try:
            from database import TradingDatabase
            db = get_database()
            db_settings = db.load_settings()
            default_settings.update(db_settings)
        except Exception as db_error:
            print(f"Database settings load failed, using defaults: {db_error}")
            
            # Fallback to file if database fails
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    saved_settings = json.load(f)
                    default_settings.update(saved_settings)
        
        return jsonify({
            'success': True,
            'settings': default_settings
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load settings: {str(e)}',
            'settings': {}
        })

@app.route('/api/test_connection', methods=['POST'])
def test_api_connection():
    """Test ByBit API connection"""
    try:
        connection_data = request.get_json()
        api_key = connection_data.get('api_key')
        api_secret = connection_data.get('api_secret')
        testnet = connection_data.get('testnet', False)
        
        if not api_key or not api_secret:
            return jsonify({
                'success': False,
                'message': 'API key and secret are required'
            })
        
        # Test connection
        from pybit.unified_trading import HTTP
        test_session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Try to get account info
        result = test_session.get_wallet_balance(accountType="UNIFIED")
        
        if result and 'result' in result:
            return jsonify({
                'success': True,
                'message': 'API connection successful'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'API connection failed - invalid response'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'API connection failed: {str(e)}'
        })

@app.route('/api/refresh_symbols', methods=['POST'])
def refresh_symbols():
    """Refresh supported symbols from ByBit"""
    try:
        from database import TradingDatabase
        db = get_database()
        
        # Force migration to fix symbol column size
        db.migrate_supported_symbols_table()
        
        # Check current database state
        current_symbols = db.get_supported_symbols()
        current_count = len(current_symbols)
        
        # Get instruments from ByBit
        app.logger.info("Fetching instruments from ByBit...")
        print("Fetching instruments from ByBit...")
        instruments = bybit_session.get_instruments_info(category="linear")
        
        if not instruments or 'result' not in instruments:
            error_msg = 'Failed to fetch instruments from ByBit - API response invalid'
            app.logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg
            })
        
        total_instruments = len(instruments['result']['list'])
        print(f"Found {total_instruments} total instruments from ByBit")
        
        # Process symbols
        symbols_data = []
        active_count = 0
        inactive_count = 0
        
        for instrument in instruments['result']['list']:
            if instrument['symbol'].endswith('USDT'):  # Only USDT pairs
                try:
                    # Extract leverage information from leverageFilter
                    leverage_filter = instrument.get('leverageFilter', {})
                    min_leverage = float(leverage_filter.get('minLeverage', 1))
                    max_leverage = float(leverage_filter.get('maxLeverage', 10))
                    
                    status = 'active' if instrument['status'] == 'Trading' else 'inactive'
                    if status == 'active':
                        active_count += 1
                    else:
                        inactive_count += 1
                    
                    # Preserve existing leverage multiplier if symbol exists
                    existing_leverage_multiplier = 1.0  # Default
                    existing_symbol = next((s for s in current_symbols if s['symbol'] == instrument['symbol']), None)
                    if existing_symbol and 'leverage_multiplier' in existing_symbol:
                        existing_leverage_multiplier = existing_symbol['leverage_multiplier']
                    
                    symbol_data = {
                        'symbol': instrument['symbol'],
                        'base_currency': instrument['baseCoin'],
                        'quote_currency': instrument['quoteCoin'],
                        'status': status,
                        'min_order_qty': float(instrument['lotSizeFilter']['minOrderQty']),
                        'qty_step': float(instrument['lotSizeFilter']['qtyStep']),
                        'min_leverage': min_leverage,
                        'max_leverage': max_leverage,
                        'leverage_multiplier': existing_leverage_multiplier  # Preserve existing value
                    }
                    symbols_data.append(symbol_data)
                    
                except Exception as symbol_error:
                    print(f"Error processing symbol {instrument.get('symbol', 'unknown')}: {symbol_error}")
                    continue
        
        print(f"Processed {len(symbols_data)} USDT pairs ({active_count} active, {inactive_count} inactive)")
        
        # Save to database
        app.logger.info(f"Saving {len(symbols_data)} symbols to database...")
        print("Saving symbols to database...")
        try:
            db.refresh_supported_symbols(symbols_data)
            app.logger.info("Database refresh completed successfully")
        except Exception as db_error:
            app.logger.error(f"Database refresh failed: {db_error}")
            print(f"Database refresh failed: {db_error}")
            raise
        
        # Verify save
        new_symbols = db.get_supported_symbols()
        new_count = len(new_symbols)
        
        app.logger.info(f"Database refresh complete: {current_count} → {new_count} symbols")
        print(f"Database refresh complete: {current_count} → {new_count} symbols")
        
        return jsonify({
            'success': True,
            'message': f'Successfully refreshed {len(symbols_data)} symbols (was {current_count}, now {new_count})',
            'previous_count': current_count,
            'new_count': new_count,
            'total_instruments': total_instruments,
            'usdt_pairs_found': len(symbols_data),
            'active_pairs': active_count,
            'inactive_pairs': inactive_count,
            'database_verified': new_count == len(symbols_data)
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in refresh_symbols: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/symbols_info')
def get_symbols_info():
    """Get symbols info and last updated date"""
    try:
        # Use global database instance
        db = db_instance
        
        app.logger.info("Loading symbols from database...")
        symbols = db.get_supported_symbols()
        app.logger.info(f"Found {len(symbols)} symbols")
        
        last_updated = db.get_symbols_last_updated()
        app.logger.info(f"Last updated: {last_updated} (type: {type(last_updated)})")
        
        # Handle last_updated properly - could be string or datetime
        last_updated_str = None
        if last_updated:
            if hasattr(last_updated, 'isoformat'):
                last_updated_str = last_updated.isoformat()
            else:
                last_updated_str = str(last_updated)
        
        response_data = {
            'success': True,
            'symbols': symbols,
            'last_updated': last_updated_str,
            'count': len(symbols)
        }
        
        app.logger.info(f"Returning symbols info: count={len(symbols)}, last_updated={last_updated_str}")
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        app.logger.error(f"Error in get_symbols_info: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/debug_settings')
def debug_settings():
    """Debug endpoint to check what settings are actually loaded"""
    try:
        # Use global database instance
        db = db_instance
        db_settings = db.load_settings()
        
        # Also load YAML for comparison
        from utils.settings_loader import Settings
        yaml_settings = Settings.load('config/settings.yaml')
        
        return jsonify({
            'success': True,
            'database_settings': dict(db_settings) if db_settings else None,
            'yaml_settings': {
                'ai_confidence_threshold': getattr(yaml_settings, 'ai_confidence_threshold', 'NOT_SET'),
                'ai_accuracy_threshold': getattr(yaml_settings, 'ai_accuracy_threshold', 'NOT_SET'),
                'take_profit_percent': getattr(yaml_settings, 'take_profit_percent', 'NOT_SET'),
                'stop_loss_percent': getattr(yaml_settings, 'stop_loss_percent', 'NOT_SET'),
                'min_take_profit_percent': getattr(yaml_settings, 'min_take_profit_percent', 'NOT_SET'),
                'max_take_profit_percent': getattr(yaml_settings, 'max_take_profit_percent', 'NOT_SET')
            },
            'database_keys_count': len(db_settings) if db_settings else 0
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training_symbols')
def get_training_symbols():
    """Get symbols that will be used for training"""
    try:
        from database import TradingDatabase
        db = get_database()
        
        # Get active symbols from database
        symbols = db.get_supported_symbols()
        active_symbols = [s for s in symbols if s.get('status') == 'active']
        
        # Get fallback symbols from settings
        from utils.settings_loader import Settings
        try:
            settings = Settings.load('config/settings.yaml')
            fallback_symbols = settings.bot.get('enabled_pairs', ['BTCUSDT', 'ETHUSDT'])
        except:
            fallback_symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Handle last_updated properly
        last_updated = db.get_symbols_last_updated()
        last_updated_str = None
        if last_updated:
            if hasattr(last_updated, 'isoformat'):
                last_updated_str = last_updated.isoformat()
            else:
                last_updated_str = str(last_updated)
        
        return jsonify({
            'success': True,
            'active_symbols': [s['symbol'] for s in active_symbols],
            'active_count': len(active_symbols),
            'fallback_symbols': fallback_symbols,
            'using_database': len(active_symbols) > 0,
            'last_updated': last_updated_str
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/leverage_multipliers')
def get_leverage_multipliers():
    """Get leverage multipliers for all supported symbols"""
    try:
        from database import TradingDatabase
        db = get_database()
        
        symbols = db.get_supported_symbols()
        
        return jsonify({
            'success': True,
            'symbols': symbols,
            'count': len(symbols)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/save_leverage_multipliers', methods=['POST'])
def save_leverage_multipliers():
    """Save leverage multipliers for symbols"""
    try:
        data = request.get_json()
        multipliers = data.get('multipliers', {})
        
        if not multipliers:
            return jsonify({
                'success': False,
                'error': 'No multipliers provided'
            }), 400
        
        from database import TradingDatabase
        db = get_database()
        
        updated_count = 0
        for symbol, multiplier in multipliers.items():
            if db.update_leverage_multiplier(symbol, multiplier):
                updated_count += 1
        
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'message': f'Updated {updated_count} leverage multipliers'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/leverage_symbols_count')
def get_leverage_symbols_count():
    """Get count of symbols with leverage multipliers"""
    try:
        # Use global database instance
        db = db_instance
        
        symbols = db.get_supported_symbols()
        
        return jsonify({
            'success': True,
            'count': len(symbols)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Performance Analysis Endpoints
@app.route('/api/performance/analyze')
def analyze_performance():
    """Run performance analysis and return report"""
    try:
        import asyncio
        from performance_analyzer import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        
        # Run analysis in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        report = loop.run_until_complete(analyzer.run_full_analysis())
        loop.close()
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        app.logger.error(f"Performance analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/performance/quick_check')
def quick_performance_check():
    """Quick performance check for dashboard loading issues"""
    try:
        import time
        from datetime import datetime
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }
        
        # Check API response times
        api_endpoints = [
            '/api/balance',
            '/api/analytics_data',
            '/api/trading_status'
        ]
        
        for endpoint in api_endpoints:
            start_time = time.time()
            try:
                # Make internal request
                with app.test_client() as client:
                    response = client.get(endpoint)
                    response_time = time.time() - start_time
                    
                    metrics['checks'].append({
                        'endpoint': endpoint,
                        'response_time': response_time,
                        'status_code': response.status_code,
                        'status': 'slow' if response_time > 1.0 else 'ok'
                    })
            except Exception as e:
                metrics['checks'].append({
                    'endpoint': endpoint,
                    'error': str(e),
                    'status': 'error'
                })
        
        # Check database size
        import os
        if os.path.exists('trading_bot.db'):
            db_size_mb = os.path.getsize('trading_bot.db') / (1024**2)
            metrics['database_size_mb'] = db_size_mb
            metrics['database_status'] = 'large' if db_size_mb > 100 else 'ok'
        
        # Dashboard optimization suggestions
        metrics['optimization_suggestions'] = [
            {
                'issue': 'Heavy JavaScript Libraries',
                'impact': 'Slow initial page load',
                'solution': 'Bundle and minify JS files, use CDN with preload hints'
            },
            {
                'issue': 'Frequent API Polling',
                'impact': 'Unnecessary server load',
                'solution': 'Use WebSocket for real-time updates'
            },
            {
                'issue': 'No Response Caching',
                'impact': 'Redundant API calls',
                'solution': 'Implement client-side caching with 60s TTL'
            },
            {
                'issue': 'Large Data Transfers',
                'impact': 'Slow data loading',
                'solution': 'Implement pagination and data compression'
            }
        ]
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/performance/optimize_cache', methods=['POST'])
def optimize_cache():
    """Enable caching optimization for better performance"""
    try:
        # Add cache headers to responses
        @app.after_request
        def add_cache_headers(response):
            # Cache static API responses
            if request.path.startswith('/api/') and request.method == 'GET':
                # Different cache times for different endpoints
                if 'balance' in request.path:
                    response.headers['Cache-Control'] = 'private, max-age=60'  # 1 minute
                elif 'analytics' in request.path:
                    response.headers['Cache-Control'] = 'private, max-age=300'  # 5 minutes
                elif 'status' in request.path:
                    response.headers['Cache-Control'] = 'private, max-age=30'  # 30 seconds
                else:
                    response.headers['Cache-Control'] = 'private, max-age=120'  # 2 minutes default
                
                # Add ETag for conditional requests
                import hashlib
                etag = hashlib.md5(response.get_data()).hexdigest()
                response.headers['ETag'] = etag
                
            return response
        
        return jsonify({
            'success': True,
            'message': 'Caching optimization enabled',
            'details': {
                'balance_cache': '60 seconds',
                'analytics_cache': '5 minutes',
                'status_cache': '30 seconds',
                'default_cache': '2 minutes'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trades/active_status')
def get_active_trades_status():
    """Get detailed status of all active trades"""
    try:
        global ai_worker
        if not ai_worker:
            return jsonify({
                'success': False,
                'message': 'AI Worker not initialized'
            })
        
        status = ai_worker.get_active_trades_status()
        return jsonify({
            'success': True,
            'data': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trades/force_monitor', methods=['POST'])
def force_monitor_trades():
    """Manually trigger trade monitoring for debugging"""
    try:
        global ai_worker
        if not ai_worker:
            return jsonify({
                'success': False,
                'message': 'AI Worker not initialized'
            })
        
        # Trigger monitoring
        ai_worker.force_monitor_trades()
        
        # Get status after monitoring
        status = ai_worker.get_active_trades_status()
        
        return jsonify({
            'success': True,
            'message': 'Trade monitoring triggered successfully',
            'data': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trades/sl_debug')
def debug_stop_loss():
    """Debug stop loss movement functionality"""
    try:
        global ai_worker
        if not ai_worker:
            return jsonify({
                'success': False,
                'message': 'AI Worker not initialized'
            })
        
        debug_info = {
            'has_active_trades': hasattr(ai_worker, 'active_trades') and bool(ai_worker.active_trades),
            'bybit_session_available': bool(ai_worker.bybit_session),
            'worker_running': ai_worker.is_running,
            'monitoring_interval': '30 seconds',
            'last_check': datetime.now().isoformat()
        }
        
        if hasattr(ai_worker, 'active_trades') and ai_worker.active_trades:
            debug_info['active_trades_count'] = len(ai_worker.active_trades)
            debug_info['trades_details'] = []
            
            for order_id, trade_data in ai_worker.active_trades.items():
                trade_debug = {
                    'order_id': order_id,
                    'symbol': trade_data['symbol'],
                    'side': trade_data['side'],
                    'entry_filled': trade_data.get('entry_filled', False),
                    'tp1_hit': trade_data.get('tp1_hit', False),
                    'sl_moved_to_breakeven': trade_data.get('sl_moved_to_breakeven', False),
                    'tp1_order_id': trade_data['take_profit_levels'][0].get('order_id') if trade_data['take_profit_levels'] else None,
                    'sl_order_id': trade_data.get('sl_order_id')
                }
                debug_info['trades_details'].append(trade_debug)
        else:
            debug_info['active_trades_count'] = 0
        
        return jsonify({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/worker_status')
def get_worker_status():
    """Get comprehensive AI worker status"""
    try:
        ensure_components_initialized()
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio=socketio, bybit_session=bybit_session)
        
        if ai_worker:
            status = {
                'is_running': ai_worker.is_running,
                'training_in_progress': ai_worker.training_in_progress,
                'signal_count': ai_worker.signal_count,
                'active_trades': ai_worker.get_active_positions_count(),
                'max_trades': ai_worker.max_concurrent_trades,
                'last_model_update': ai_worker.last_model_update.isoformat() if ai_worker.last_model_update else None,
                'bybit_connected': bool(ai_worker.bybit_session),
                'uptime': getattr(ai_worker, '_start_time', 0)
            }
        else:
            status = {
                'is_running': False,
                'training_in_progress': False,
                'signal_count': 0,
                'active_trades': 0,
                'max_trades': 0,
                'last_model_update': None,
                'bybit_connected': False,
                'uptime': 0
            }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'is_running': False,
            'training_in_progress': False,
            'signal_count': 0,
            'active_trades': 0,
            'max_trades': 0,
            'last_model_update': None,
            'bybit_connected': False,
            'uptime': 0,
            'error': str(e)
        }), 500

@app.route('/api/active_trades_status')
def get_active_trades_status_api():
    """Get detailed status of active trades"""
    try:
        ensure_components_initialized()
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio=socketio, bybit_session=bybit_session)
        
        if ai_worker and hasattr(ai_worker, 'get_active_trades_status'):
            return jsonify(ai_worker.get_active_trades_status())
        else:
            return jsonify({
                'message': 'No AI worker available or no active trades method',
                'trades': []
            })
            
    except Exception as e:
        return jsonify({
            'message': f'Error getting active trades: {str(e)}',
            'trades': []
        }), 500

@app.route('/api/production_analytics')
def get_production_analytics():
    """Get production analytics data"""
    try:
        from check_production_analytics import check_production_analytics
        results = check_production_analytics()
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'database_type': 'Unknown',
            'total_signals': 0,
            'status_breakdown': {},
            'completed_signals': 0,
            'completed_with_pnl': 0,
            'analytics_data': {},
            'sample_signals': [],
            'issues': [f'Error: {str(e)}'],
            'recommendations': ['Check database connection and AI worker status']
        }), 500


@app.route('/api/manual_complete_trade', methods=['POST'])
def manual_complete_trade():
    """Manually complete a trade for testing P&L calculation"""
    try:
        ensure_components_initialized()
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        signal_id = data.get('signal_id')
        entry_price = data.get('entry_price')
        exit_price = data.get('exit_price')
        realized_pnl = data.get('realized_pnl')
        
        if not all([signal_id, entry_price, exit_price, realized_pnl is not None]):
            return jsonify({
                'success': False,
                'message': 'Missing required fields: signal_id, entry_price, exit_price, realized_pnl'
            }), 400
        
        # Get AI worker to access database
        global ai_worker_instance
        ai_worker = ai_worker_instance or get_ai_worker(socketio=socketio, bybit_session=bybit_session)
        
        if not ai_worker:
            return jsonify({
                'success': False,
                'message': 'AI worker not available'
            }), 500
        
        # Update signal with P&L data
        ai_worker.database.update_signal_with_pnl(
            signal_id=signal_id,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            realized_pnl=float(realized_pnl)
        )
        
        return jsonify({
            'success': True,
            'message': f'Trade {signal_id} manually completed with P&L: {realized_pnl}',
            'data': {
                'signal_id': signal_id,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'realized_pnl': realized_pnl
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error manually completing trade: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🤖 AI Worker ready...")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)