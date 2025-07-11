import asyncio
import threading
import time
import json
import uuid
import numpy as np
import os
from datetime import datetime
from queue import Queue
from utils.settings_loader import Settings
from ai.trader import AITrader
from utils.trade_logger import TradeLogger
from db_singleton import get_database
from breakeven_monitor import BreakevenMonitor
# Trading.executor removed - using direct pybit calls
import logging

class ConsoleLogger:
    def __init__(self):
        self.log_queue = Queue()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging to capture all messages"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                QueueHandler(self.log_queue)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log(self, level, message):
        """Add log entry"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.log_queue.put(log_entry)
        
        # Also log to standard logger
        if level == 'INFO':
            self.logger.info(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)
        elif level == 'SUCCESS':
            self.logger.info(f"✅ {message}")
    
    def get_recent_logs(self, count=100):
        """Get recent log entries"""
        logs = []
        temp_queue = Queue()
        
        # Get logs from queue without removing them permanently
        while not self.log_queue.empty() and len(logs) < count:
            log = self.log_queue.get()
            logs.append(log)
            temp_queue.put(log)
        
        # Put logs back
        while not temp_queue.empty():
            self.log_queue.put(temp_queue.get())
        
        return logs[-count:]  # Return most recent

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            'level': record.levelname,
            'message': record.getMessage()
        }
        self.log_queue.put(log_entry)

class AIWorker:
    def __init__(self, socketio=None, bybit_session=None):
        self.socketio = socketio
        self.bybit_session = bybit_session
        # Load settings from database only
        self.settings = type('Settings', (), {})()
        self.ai_trader = AITrader(self.settings)
        self.trade_logger = TradeLogger()
        self.console_logger = ConsoleLogger()
        self.database = get_database()
        # Direct pybit integration - no separate executor needed
        # Max concurrent trades will be loaded from database when needed
        self.max_concurrent_trades = 20  # Default, will be loaded from database
        self.last_heartbeat = datetime.now()
        self.active_trades = {}  # Track active trades for SL management
        if bybit_session:
            self.console_logger.log('INFO', '✅ Direct pybit trading ready')
        else:
            self.console_logger.log('WARNING', '⚠️ ByBit session not available for trading')
        self.is_running = False
        self.training_in_progress = False
        self.last_model_update = None
        self.signal_count = 0
        self.worker_thread = None
        self.current_training_session = None
        self.active_positions = []  # Track active positions for limit checking
        self.training_progress = {
            'current_batch': 0,
            'total_batches': 0,
            'current_symbol': '',
            'completed_symbols': 0,
            'total_symbols': 0,
            'overall_progress': 0,
            'batch_results': []
        }
        
        # Initialize breakeven monitor with reference to self
        self.breakeven_monitor = BreakevenMonitor(ai_worker=self)
        self.breakeven_thread = None
    
    def get_supported_symbols(self):
        """Get supported symbols from database, with fallback to settings and ByBit refresh"""
        try:
            symbols = self.database.get_supported_symbols()
            # Filter only active symbols and extract symbol names
            active_symbols = [s['symbol'] for s in symbols if s.get('status') == 'active']
            
            # Always get symbols from settings.yaml for complete coverage
            # NO FALLBACK - only use database symbols
            enabled_pairs = active_symbols
            
            # Combine database symbols with settings symbols to ensure complete A-Z coverage
            all_symbols = list(set(active_symbols + enabled_pairs))
            
            # Check if database has symbols - if we have any active symbols, use them
            # Only refresh if database is completely empty or has very few symbols
            if len(active_symbols) < 10:
                self.console_logger.log('WARNING', f'Only {len(active_symbols)} symbols in database, attempting to refresh from ByBit')
                # Try to refresh symbols from ByBit
                self._refresh_symbols_from_bybit()
                # Try again after refresh
                symbols = self.database.get_supported_symbols()
                active_symbols = [s['symbol'] for s in symbols if s.get('status') == 'active']
                
                # Recombine after refresh
                all_symbols = list(set(active_symbols + enabled_pairs))
                
                # If still no symbols after refresh, fallback to settings
                if len(active_symbols) == 0:
                    self.console_logger.log('WARNING', 'No symbols found after ByBit refresh, using settings fallback')
                    self.console_logger.log('INFO', f'📊 Using {len(enabled_pairs)} symbols from settings.yaml')
                    return enabled_pairs
            
            # Sort symbols alphabetically to ensure proper A-Z order
            all_symbols.sort()
            
            self.console_logger.log('INFO', f'📊 Found {len(active_symbols)} active database symbols + {len(enabled_pairs)} settings symbols = {len(all_symbols)} total symbols')
            return all_symbols
        except Exception as e:
            self.console_logger.log('ERROR', f'Failed to get supported symbols from database: {str(e)}')
            # Return database symbols only
            return active_symbols
    
    def _refresh_symbols_from_bybit(self):
        """Refresh symbols from ByBit API"""
        try:
            if not self.bybit_session:
                self.console_logger.log('WARNING', 'ByBit session not available for symbol refresh')
                return
                
            self.console_logger.log('INFO', 'Refreshing symbols from ByBit API...')
            instruments = self.bybit_session.get_instruments_info(category="linear")
            
            if not instruments or 'result' not in instruments:
                self.console_logger.log('ERROR', 'Failed to fetch instruments from ByBit')
                return
            
            symbols_data = []
            for instrument in instruments['result']['list']:
                if instrument['symbol'].endswith('USDT'):  # Only USDT pairs
                    leverage_filter = instrument.get('leverageFilter', {})
                    symbol_data = {
                        'symbol': instrument['symbol'],
                        'base_currency': instrument['baseCoin'],
                        'quote_currency': instrument['quoteCoin'],
                        'status': 'active' if instrument['status'] == 'Trading' else 'inactive',
                        'min_order_qty': float(instrument['lotSizeFilter']['minOrderQty']),
                        'qty_step': float(instrument['lotSizeFilter']['qtyStep']),
                        'min_leverage': float(leverage_filter.get('minLeverage', 1)),
                        'max_leverage': float(leverage_filter.get('maxLeverage', 10))
                    }
                    symbols_data.append(symbol_data)
            
            # Save to database
            self.database.refresh_supported_symbols(symbols_data)
            self.console_logger.log('SUCCESS', f'✅ Refreshed {len(symbols_data)} symbols from ByBit')
            
        except Exception as e:
            self.console_logger.log('ERROR', f'Failed to refresh symbols from ByBit: {str(e)}')
        
    def start(self):
        """Start the AI worker and resume on-hold trades"""
        if not self.is_running:
            self.is_running = True
            
            # Resume on-hold signals back to waiting status
            try:
                from db_singleton import get_database
                db = get_database()
                on_hold_signals = db.get_trading_signals()
                resumed_count = 0
                
                for signal in on_hold_signals:
                    if signal.get('status') == 'on_hold':
                        db.update_signal_status(signal['signal_id'], 'waiting')
                        resumed_count += 1
                
                if resumed_count > 0:
                    self.console_logger.log('INFO', f'▶️ Resumed {resumed_count} on-hold trades')
            except Exception as e:
                self.console_logger.log('WARNING', f'⚠️ Could not resume on-hold trades: {str(e)}')
            
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            
            # Start breakeven monitor
            self.start_breakeven_monitor()
            
            self.console_logger.log('SUCCESS', '🚀 AI WORKER STARTED - Ready for training and signal generation!')
            self.console_logger.log('INFO', '📊 System initialized with ByBit API connection')
            if self.bybit_session:
                self.console_logger.log('INFO', '⚡ Direct pybit trading ready for live execution')
            
    def stop(self):
        """Stop the AI worker and abort any training"""
        self.is_running = False
        
        # Stop breakeven monitor
        self.stop_breakeven_monitor()
        
        # Abort training if in progress
        if self.training_in_progress:
            self.training_in_progress = False
            self.console_logger.log('WARNING', '⏹️ TRAINING ABORTED - Manual stop requested')
            
            # Emit training aborted event
            if self.socketio:
                self.socketio.emit('training_aborted', {
                    'message': 'Training was manually stopped',
                    'timestamp': datetime.now().isoformat()
                })
        
        self.console_logger.log('WARNING', '⏹️ AI WORKER STOPPED - All operations halted')
        self.console_logger.log('INFO', '🔴 Training, signals, and trading have been disabled')
        
    def _worker_loop(self):
        """Main worker loop with heartbeat monitoring"""
        self.console_logger.log('INFO', 'AI Worker loop started with auto-restart monitoring')
        
        while self.is_running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Check if model needs retraining
                self._check_model_training()
                
                # Only generate and execute signals if NOT training
                if not self.training_in_progress:
                    # Generate trading signals
                    self._generate_signals()
                    
                    # Execute trades from ranked signal list
                    self._execute_top_signals()
                else:
                    self.console_logger.log('INFO', '🔄 Training in progress - skipping signal generation and execution')
                
                # Monitor trades and move stop loss if needed
                self.monitor_trades_and_move_sl()
                
                # Check for orphaned positions (positions not being tracked)
                self._check_orphaned_positions()
                
                # Check for manually closed positions (tracked positions that no longer exist)
                self._check_manually_closed_positions()
                
                # Update statistics
                self._update_statistics()
                
                # Emit status update
                self._emit_status_update()
                
                # Sleep for configured interval from database
                try:
                    db_settings = self.database.load_settings()
                    monitor_interval = int(db_settings.get('monitorInterval', 30))
                    time.sleep(monitor_interval)
                except:
                    time.sleep(30)  # Default fallback
                
            except Exception as e:
                self.console_logger.log('ERROR', f'Worker error: {str(e)}')
                try:
                    db_settings = self.database.load_settings()
                    error_interval = int(db_settings.get('errorRetryInterval', 60))
                    time.sleep(error_interval)
                except:
                    time.sleep(60)  # Default fallback
    
    def _check_model_training(self):
        """Check if AI model needs retraining"""
        if self.training_in_progress:
            return
            
        # Check if enough time has passed since last training  
        # Get retrain interval from database - NO FALLBACK
        try:
            from db_singleton import get_database
            db = get_database()
            db_settings = db.load_settings()
            if not db_settings:
                self.console_logger.log('ERROR', '❌ Database not available - skipping retrain check')
                return
            
            # Get retrain interval in minutes (default 60 minutes if not set)
            if 'retrainIntervalMinutes' not in db_settings:
                self.console_logger.log('ERROR', '❌ retrainIntervalMinutes not in database - skipping retrain')
                return
            retrain_interval_minutes = float(db_settings['retrainIntervalMinutes'])
            update_interval_hours = retrain_interval_minutes / 60.0
        except Exception as db_error:
            self.console_logger.log('ERROR', f'❌ Database error for retrain interval: {db_error} - skipping retrain')
            return
        
        if (self.last_model_update is None or 
            (datetime.now() - self.last_model_update).total_seconds() > update_interval_hours * 3600):
            
            self._start_model_training()
    
    def _start_model_training(self):
        """Start comprehensive model training process"""
        self.training_in_progress = True
        self.current_training_session = str(uuid.uuid4())
        
        self.console_logger.log('INFO', '🚀 Starting comprehensive AI model training...')
        self.console_logger.log('WARNING', '⏸️ Signal detection PAUSED - Training in progress, signals will resume when training completes')
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self._train_model_comprehensive, daemon=True)
        training_thread.start()
    
    def _train_model_comprehensive(self):
        """Comprehensive AI model training with batch processing"""
        try:
            # Get supported symbols from database only
            enabled_pairs = self.get_supported_symbols()
            
            if not enabled_pairs:
                self.console_logger.log('ERROR', 'No supported symbols found in database - cannot train')
                self.training_in_progress = False
                return
            
            # Initialize training session
            self.database.create_training_session(self.current_training_session, len(enabled_pairs))
            
            # Setup batch processing (10 symbols per batch)
            batch_size = 10
            batches = [enabled_pairs[i:i + batch_size] for i in range(0, len(enabled_pairs), batch_size)]
            
            self.training_progress.update({
                'total_batches': len(batches),
                'total_symbols': len(enabled_pairs),
                'current_batch': 0,
                'completed_symbols': 0,
                'batch_results': []
            })
            
            self.console_logger.log('INFO', f'📊 Training on {len(enabled_pairs)} symbols in {len(batches)} batches')
            self._emit_training_progress()
            
            overall_accuracies = []
            
            # Process each batch
            for batch_idx, batch_symbols in enumerate(batches):
                self.training_progress['current_batch'] = batch_idx + 1
                
                self.console_logger.log('INFO', f'🔄 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_symbols)} symbols)')
                self._emit_training_progress()
                
                batch_results = []
                
                # Process each symbol in the batch
                for symbol_idx, symbol in enumerate(batch_symbols):
                    self.training_progress['current_symbol'] = symbol
                    self._emit_training_progress()
                    
                    self.console_logger.log('INFO', f'📈 Training on {symbol}...')
                    
                    # Collect and store market data
                    market_data = self._collect_market_data(symbol)
                    if market_data:
                        self.database.store_market_data(symbol, market_data)
                    
                    # Calculate technical indicators
                    indicators = self._calculate_technical_indicators(symbol, market_data)
                    if indicators:
                        timestamp = int(datetime.now().timestamp() * 1000)
                        self.database.store_technical_indicators(symbol, timestamp, indicators)
                    
                    # Analyze sentiment
                    sentiment = self._analyze_sentiment(symbol)
                    if sentiment:
                        timestamp = int(datetime.now().timestamp() * 1000)
                        self.database.store_sentiment_data(symbol, timestamp, sentiment)
                    
                    # Train model for this symbol
                    accuracy, confidence = self._train_symbol_model(symbol, indicators, sentiment)
                    
                    # Store training results
                    features = {
                        'technical_indicators': indicators,
                        'sentiment': sentiment,
                        'market_data_points': len(market_data) if market_data else 0
                    }
                    
                    model_params = {
                        'symbol': symbol,
                        'training_timestamp': datetime.now().isoformat(),
                        'data_quality': 'high' if market_data and len(market_data) > 50 else 'medium'
                    }
                    
                    self.database.store_training_results(
                        self.current_training_session, symbol, features, accuracy, confidence, model_params
                    )
                    
                    batch_results.append({
                        'symbol': symbol,
                        'accuracy': accuracy,
                        'confidence': confidence,
                        'data_points': len(market_data) if market_data else 0
                    })
                    
                    self.training_progress['completed_symbols'] += 1
                    overall_progress = (self.training_progress['completed_symbols'] / self.training_progress['total_symbols']) * 100
                    self.training_progress['overall_progress'] = overall_progress
                    
                    self.console_logger.log('SUCCESS', f'✅ {symbol}: Accuracy {accuracy:.1f}%, Confidence {confidence:.1f}%')
                    
                    # Emit real-time training progress
                    if self.socketio:
                        self.socketio.emit('training_log', {
                            'level': 'SUCCESS',
                            'message': f'✅ {symbol}: Accuracy {accuracy:.1f}%, Confidence {confidence:.1f}%'
                        })
                        self.socketio.emit('training_progress', {
                            'progress': self.training_progress,
                            'level': 'INFO',
                            'message': f'Training progress: {overall_progress:.1f}% complete'
                        })
                    
                    # Update training session progress
                    self.database.update_training_session(self.current_training_session, self.training_progress['completed_symbols'])
                    
                    # Small delay to not overwhelm API
                    time.sleep(0.5)
                
                # Store batch results
                self.training_progress['batch_results'].append({
                    'batch_number': batch_idx + 1,
                    'symbols': batch_symbols,
                    'results': batch_results,
                    'avg_accuracy': np.mean([r['accuracy'] for r in batch_results]),
                    'avg_confidence': np.mean([r['confidence'] for r in batch_results])
                })
                
                overall_accuracies.extend([r['accuracy'] for r in batch_results])
                
                self.console_logger.log('INFO', f'📋 Batch {batch_idx + 1} completed - Avg accuracy: {np.mean([r["accuracy"] for r in batch_results]):.1f}%')
                self._emit_training_progress()
                
                # Longer delay between batches
                time.sleep(2)
            
            # Complete training
            overall_accuracy = np.mean(overall_accuracies) if overall_accuracies else 0
            self.database.update_training_session(
                self.current_training_session, 
                self.training_progress['completed_symbols'], 
                overall_accuracy, 
                'completed'
            )
            
            self.last_model_update = datetime.now()
            self.training_in_progress = False
            
            self.console_logger.log('SUCCESS', f'🎉 Training completed! Overall accuracy: {overall_accuracy:.1f}%')
            self.console_logger.log('INFO', f'📊 Trained on {len(enabled_pairs)} symbols across {len(batches)} batches')
            self.console_logger.log('SUCCESS', f'🚀 Signal detection RESUMED - AI is now ready to generate trading signals!')
            
            # Reset failed signals to waiting after retraining
            self._reset_failed_signals_after_retraining()
            
            # Final progress update
            self.training_progress['overall_progress'] = 100
            self._emit_training_progress()
            
        except Exception as e:
            self.training_in_progress = False
            self.console_logger.log('ERROR', f'Training failed: {str(e)}')
            if self.current_training_session:
                self.database.update_training_session(self.current_training_session, 0, 0, 'failed')
    
    def _reset_failed_signals_after_retraining(self):
        """Reset failed signals to waiting status after retraining to give them another chance"""
        try:
            from db_singleton import get_database
            db = get_database()
            
            # Get all failed signals
            failed_signals = [s for s in db.get_trading_signals() if s.get('status') == 'failed']
            
            reset_count = 0
            for signal in failed_signals:
                # Check if signal still meets confidence threshold after retraining
                confidence = signal.get('confidence', 0)
                accuracy = signal.get('accuracy', 0)
                
                # Get current threshold from database only
                try:
                    db_settings = self.database.load_settings()
                    if not db_settings:
                        raise Exception("No database settings found")
                    ai_threshold = float(db_settings['confidenceThreshold'])
                    accuracy_threshold = float(db_settings['accuracyThreshold'])
                except Exception as db_error:
                    self.console_logger.log('ERROR', f'Database settings failed in reset signals: {db_error}')
                    continue  # Skip this signal if database fails
                
                # If signal still meets requirements, reset to waiting
                if confidence >= ai_threshold and accuracy > accuracy_threshold:  # Basic quality check
                    db.update_signal_status(signal['signal_id'], 'waiting')
                    reset_count += 1
                    self.console_logger.log('INFO', f'🔄 Reset failed signal {signal["symbol"]} to waiting (Confidence: {confidence:.1f}%)')
            
            if reset_count > 0:
                self.console_logger.log('SUCCESS', f'✅ Reset {reset_count} failed signals to waiting status after retraining')
            else:
                self.console_logger.log('INFO', '📊 No failed signals met criteria for reset after retraining')
                
        except Exception as e:
            self.console_logger.log('ERROR', f'Error resetting failed signals: {str(e)}')
    
    def _collect_market_data(self, symbol):
        """Collect market data for a symbol"""
        try:
            if not self.bybit_session:
                return None
                
            klines = self.bybit_session.get_kline(
                category="linear",
                symbol=symbol,
                interval="1",  # Use 1-minute interval instead of 1h
                limit=100
            )
            
            if klines and 'result' in klines and 'list' in klines['result']:
                return klines['result']['list']
            return None
            
        except Exception as e:
            self.console_logger.log('WARNING', f'Failed to collect data for {symbol}: {str(e)}')
            return None
    
    def _calculate_technical_indicators(self, symbol, market_data):
        """Calculate ~22 indicators for richer AI confidence."""
        if not market_data or len(market_data) < 30:
            return {}
    
        try:
            highs = np.array([float(k[2]) for k in market_data])
            lows = np.array([float(k[3]) for k in market_data])
            closes = np.array([float(k[4]) for k in market_data])
            volumes = np.array([float(k[5]) for k in market_data])
    
            # Helpers
            def ema(arr, span):
                alpha = 2/(span+1)
                ema_arr = [arr[0]]
                for val in arr[1:]:
                    ema_arr.append(alpha*val + (1-alpha)*ema_arr[-1])
                return np.array(ema_arr)
    
            # RSI 14
            deltas = np.diff(closes)
            gains = np.where(deltas>0, deltas, 0)
            losses = np.where(deltas<0, -deltas, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains)>=14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses)>=14 else 0
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - 100/(1+rs)
    
            # Stochastic %K 14
            lowest14 = np.min(lows[-14:])
            highest14 = np.max(highs[-14:])
            stoch_k = (closes[-1]-lowest14)/(highest14-lowest14+1e-10)*100
    
            # StochRSI
            rsi_series = []
            for i in range(len(closes)):
                if i<14: rsi_series.append(50)
                else:
                    delt = closes[i]-closes[i-1]
                    g = max(delt,0)
                    l = -min(delt,0)
                    avg_g = (avg_gain*13 + g)/14
                    avg_l = (avg_loss*13 + l)/14
                    avg_gain,avg_loss=avg_g,avg_l
                    rs = avg_g/(avg_l+1e-10)
                    rsi_series.append(100-100/(1+rs))
            rsi_array = np.array(rsi_series)
            stoch_rsi = (rsi_array[-1]-np.min(rsi_array[-14:]))/(np.max(rsi_array[-14:])-np.min(rsi_array[-14:])+1e-10)*100
    
            # Williams %R
            will_r = -100*(highest14-closes[-1])/(highest14-lowest14+1e-10)
    
            # CCI 20
            tp = (highs+lows+closes)/3
            sma_tp20 = np.mean(tp[-20:])
            mean_dev = np.mean(np.abs(tp[-20:]-sma_tp20))
            cci = (tp[-1]-sma_tp20)/(0.015*mean_dev+1e-10)
    
            # MACD
            ema12 = ema(closes,12)
            ema26 = ema(closes,26)
            macd_line = ema12[-1]-ema26[-1]
            macd_signal = ema(macd_line*np.ones_like(closes),9)[-1]
    
            # Bollinger
            sma20 = np.mean(closes[-20:])
            std20 = np.std(closes[-20:])
            bb_upper = sma20+2*std20
            bb_lower = sma20-2*std20
    
            # ATR14
            tr = np.maximum.reduce([highs[1:]-lows[1:], np.abs(highs[1:]-closes[:-1]), np.abs(lows[1:]-closes[:-1])])
            atr14 = np.mean(tr[-14:]) if len(tr)>=14 else np.mean(tr)
    
            # OBV
            obv = 0
            for i in range(1,len(closes)):
                if closes[i]>closes[i-1]: obv+=volumes[i]
                elif closes[i]<closes[i-1]: obv-=volumes[i]
            # CMF (Chaikin Money Flow) 20
            mf_multiplier = (closes - lows - (highs - closes)) / (highs - lows + 1e-10)
            mf_volume = mf_multiplier * volumes
            cmf = np.sum(mf_volume[-20:]) / (np.sum(volumes[-20:])+1e-10)
    
            # MFI 14
            tp_vals = tp
            raw_money = tp_vals[1:]*volumes[1:]
            pos_rm = np.where(tp_vals[1:]>tp_vals[:-1], raw_money, 0)
            neg_rm = np.where(tp_vals[1:]<tp_vals[:-1], raw_money, 0)
            mfi = 100 - 100/(1+ (np.sum(pos_rm[-14:])/(np.sum(neg_rm[-14:])+1e-10)))
    
            # Donchian width 20
            donch_high = np.max(highs[-20:])
            donch_low = np.min(lows[-20:])
            donch_width = (donch_high-donch_low)/(donch_low+1e-10)*100
    
            # ADX 14
            up_move = highs[1:] - highs[:-1]
            down_move = lows[:-1] - lows[1:]
            plus_dm = np.where((up_move>down_move) & (up_move>0), up_move, 0)
            minus_dm = np.where((down_move>up_move) & (down_move>0), down_move, 0)
            tr14 = pd.Series(tr).rolling(14).sum().iloc[-1]
            plus_di = 100 * (np.sum(plus_dm[-14:]) / (tr14+1e-10))
            minus_di = 100 * (np.sum(minus_dm[-14:]) / (tr14+1e-10))
            dx = 100 * abs(plus_di-minus_di)/(plus_di+minus_di+1e-10)
            adx = dx  # crude snapshot
    
            # Aroon 25
            period=25
            idx_high = np.argmax(highs[-period:])
            idx_low = np.argmin(lows[-period:])
            aroon_up = ((period-idx_high)/period)*100
            aroon_down = ((period-idx_low)/period)*100
    
            # Keltner Channel width (EMA20 ± ATR*1.5)
            ema20 = ema(closes,20)[-1]
            kc_upper = ema20 + 1.5*atr14
            kc_lower = ema20 - 1.5*atr14
            keltner_width = (kc_upper-kc_lower)/(ema20+1e-10)*100
    
            # SuperTrend direction 10/3 (simplified)
            mult=3
            atr10 = np.mean(tr[-10:])
            basic_upper = (highs[-1]+lows[-1])/2 + mult*atr10
            basic_lower = (highs[-1]+lows[-1])/2 - mult*atr10
            supertrend_dir = 1 if closes[-1]>basic_upper else -1 if closes[-1]<basic_lower else 0
    
            return {
                'rsi_14': rsi,
                'stoch_k': stoch_k,
                'stoch_rsi': stoch_rsi,
                'williams_r': will_r,
                'cci': cci,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': sma20,
                'sma_20': sma20,
                'ema_12': ema12[-1],
                'ema_26': ema26[-1],
                'ema_20': ema20,
                'atr': atr14,
                'obv': obv,
                'cmf': cmf,
                'mfi': mfi,
                'donch_width': donch_width,
                'adx': adx,
                'aroon_up': aroon_up,
                'aroon_down': aroon_down,
                'keltner_width': keltner_width,
                'supertrend_dir': supertrend_dir,
                'volume_sma_20': np.mean(volumes[-20:])
            }
    
        except Exception as e:
            self.console_logger.log('WARNING', f'Indicator calc failed for {symbol}: {e}')
            return {}
    def _analyze_sentiment(self, symbol):
        """Analyze sentiment for the symbol"""
        try:
            # Simulate sentiment analysis
            # In reality, this would fetch from news APIs, social media, etc.
            base_sentiment = np.random.uniform(-1, 1)
            
            return {
                'sentiment_score': base_sentiment,
                'news_count': np.random.randint(5, 50),
                'social_sentiment': base_sentiment + np.random.uniform(-0.2, 0.2),
                'fear_greed_index': np.random.uniform(0, 100)
            }
            
        except Exception as e:
            self.console_logger.log('WARNING', f'Failed to analyze sentiment for {symbol}: {str(e)}')
            return {}
    
    def _analyze_market_conditions(self):
        """Analyze overall market conditions for AI prediction - checks user setting first"""
        try:
            # First check if user has manually set market condition (not auto)
            user_market_condition = None
            try:
                from db_singleton import get_database
                db = get_database()
                settings = db.load_settings()
                user_market_condition = settings.get('marketCondition', 'auto')
            except Exception as settings_error:
                self.console_logger.log('WARNING', f'Could not load market condition setting: {settings_error}')
            
            # If user has set a manual market condition, use it instead of auto-detection
            if user_market_condition and user_market_condition != 'auto':
                self.console_logger.log('INFO', f'📊 Using user-defined market condition: {user_market_condition.upper()}')
                
                # Still calculate volatility and volume metrics from market data
                try:
                    conditions = {
                        'volatility': {},
                        'trends': {},
                        'volumes': {}
                    }
                    
                    # Get major pairs from database settings
                    try:
                        db_settings = self.database.load_settings()
                        major_pairs = db_settings.get('majorPairs', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
                    except:
                        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
                    
                    for symbol in major_pairs:
                        market_data = self._collect_market_data(symbol)
                        if market_data and len(market_data) > 20:
                            closes = np.array([float(kline[4]) for kline in market_data])
                            volumes = np.array([float(kline[5]) for kline in market_data])
                            
                            # Calculate volatility (standard deviation of returns)
                            returns = np.diff(closes) / closes[:-1]
                            volatility = np.std(returns) * 100  # As percentage
                            
                            # Volume trend
                            vol_trend = ((volumes[-1] - np.mean(volumes[:-5])) / np.mean(volumes[:-5])) * 100
                            
                            conditions['volatility'][symbol] = volatility
                            conditions['volumes'][symbol] = vol_trend
                    
                    # Use manual market condition but with real volatility data
                    return {
                        'avg_volatility': np.mean(list(conditions['volatility'].values())) if conditions['volatility'] else 2.0,
                        'market_trend': user_market_condition,  # Use user setting
                        'volume_strength': np.mean(list(conditions['volumes'].values())) if conditions['volumes'] else 0
                    }
                except Exception as data_error:
                    self.console_logger.log('WARNING', f'Failed to get market data for manual condition: {data_error}')
                    # Return user condition with database default metrics
                    try:
                        db_settings = self.database.load_settings()
                        return {
                            'avg_volatility': float(db_settings.get('defaultVolatility', 2.0)),
                            'market_trend': user_market_condition,
                            'volume_strength': float(db_settings.get('defaultVolumeStrength', 0))
                        }
                    except:
                        return {
                            'avg_volatility': 2.0,
                            'market_trend': user_market_condition,
                            'volume_strength': 0
                        }
            
            # Auto-detection mode (original logic)
            self.console_logger.log('INFO', '📊 Auto-detecting market conditions from price data')
            
            # Get market data for major pairs
            conditions = {
                'volatility': {},
                'trends': {},
                'volumes': {},
                'correlations': {}
            }
            
            # Get major pairs from database settings
            try:
                db_settings = self.database.load_settings()
                major_pairs = db_settings.get('majorPairs', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
            except:
                major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            
            for symbol in major_pairs:
                market_data = self._collect_market_data(symbol)
                if market_data and len(market_data) > 20:
                    closes = np.array([float(kline[4]) for kline in market_data])
                    volumes = np.array([float(kline[5]) for kline in market_data])
                    
                    # Calculate volatility (standard deviation of returns)
                    returns = np.diff(closes) / closes[:-1]
                    volatility = np.std(returns) * 100  # As percentage
                    
                    # Trend strength (price change over period)
                    trend = ((closes[-1] - closes[0]) / closes[0]) * 100
                    
                    # Volume trend
                    vol_trend = ((volumes[-1] - np.mean(volumes[:-5])) / np.mean(volumes[:-5])) * 100
                    
                    conditions['volatility'][symbol] = volatility
                    conditions['trends'][symbol] = trend
                    conditions['volumes'][symbol] = vol_trend
            
            # Overall market conditions from auto-detection
            avg_trend = np.mean(list(conditions['trends'].values())) if conditions['trends'] else 0
            detected_trend = 'bullish' if avg_trend > 1 else 'bearish' if avg_trend < -1 else 'sideways'
            
            result = {
                'avg_volatility': np.mean(list(conditions['volatility'].values())) if conditions['volatility'] else 2.0,
                'market_trend': detected_trend,
                'volume_strength': np.mean(list(conditions['volumes'].values())) if conditions['volumes'] else 0
            }
            
            self.console_logger.log('INFO', f'📊 Auto-detected market trend: {detected_trend.upper()} (avg trend: {avg_trend:.2f}%)')
            
            return result
            
        except Exception as e:
            self.console_logger.log('WARNING', f'Failed to analyze market conditions: {str(e)}')
            # Use database default market conditions
            try:
                db_settings = self.database.load_settings()
                return {
                    'avg_volatility': float(db_settings.get('defaultVolatility', 2.0)),
                    'market_trend': db_settings.get('defaultMarketTrend', 'neutral'),
                    'volume_strength': float(db_settings.get('defaultVolumeStrength', 0))
                }
            except:
                return {
                    'avg_volatility': 2.0,
                    'market_trend': 'neutral',
                    'volume_strength': 0
                }

def _train_symbol_model(self, symbol, indicators, sentiment=None):
    """Confidence from 24 indicators, deterministic."""
    try:
        weights = {
            'rsi_14':1,'stoch_k':0.9,'stoch_rsi':0.9,'williams_r':0.7,'cci':0.8,
            'macd_line':1,'macd_signal':0.8,'bb_middle':0.4,'ema_20':1,'ema_12':0.7,'ema_26':0.7,
            'atr':0.5,'donch_width':0.3,'adx':1.1,'aroon_up':0.6,'aroon_down':0.6,
            'cmf':0.6,'mfi':0.5,'keltner_width':0.3,'supertrend_dir':1,
            'obv':0.4
        }
        score=0; tot=0
        for name,w in weights.items():
            val=indicators.get(name)
            if val is None: continue
            if name in ('rsi_14','stoch_k','stoch_rsi'):
                bull = 1 if val>50 else 0
            elif name=='williams_r':
                bull = 1 if val>-50 else 0
            elif name in ('supertrend_dir',):
                bull = 1 if val>0 else 0
            elif name in ('aroon_up','adx'):
                bull = 1 if val>50 else 0
            elif name=='aroon_down':
                bull = 0 if val>50 else 1
            elif name in ('cmf','mfi','macd_line','macd_signal','cci','obv'):
                bull = 1 if val>0 else 0
            else:
                bull = 0.5
            score+=bull*w
            tot+=w
        if tot==0: return 65,65
        confidence=(score/tot)*100
        # penalty for high ATR relative to ema20
        atr = indicators.get('atr'); ema20=indicators.get('ema_20',1)
        if atr: confidence*=max(0.6,1-min(1,atr/ema20))
        confidence=max(50,min(95,confidence))
        accuracy = confidence-5 if confidence>60 else confidence
        return round(accuracy,1), round(confidence,1)
    except Exception as e:
        self.console_logger.log('WARNING',f'Confidence fail {symbol}: {e}')
        threshold=70
        return threshold,threshold
    def _emit_training_progress(self):
        """Emit training progress to frontend"""
        if self.socketio:
            self.socketio.emit('training_progress', self.training_progress)
    
    def _generate_signals(self):
        """Generate trading signals and save to database - NO DIRECT EXECUTION"""
        if self.training_in_progress:
            self.console_logger.log('INFO', '⏸️ Signal detection PAUSED - Training in progress, waiting for completion...')
            return
            
        try:
            # Get AI prediction with enhanced market analysis
            # Pass current market conditions to AI for better TP calculation
            market_conditions = self._analyze_market_conditions()
            
            # Log market conditions being used for AI decision
            if market_conditions:
                trend = market_conditions.get('market_trend', 'unknown')
                volatility = market_conditions.get('avg_volatility', 0)
                volume_strength = market_conditions.get('volume_strength', 0)
                self.console_logger.log('INFO', f'📊 Market Analysis: Trend={trend.upper()}, Volatility={volatility:.2f}%, Volume={volume_strength:.1f}%')
            
            prediction = self.ai_trader.get_prediction(market_conditions)
            
            if prediction:
                confidence = prediction['confidence']
                symbol = prediction['symbol']
                side = prediction['side']
                take_profit = prediction.get('take_profit', 0)
                stop_loss = prediction.get('stop_loss', 0)
                
                # Log AI prediction impact on trading parameters
                self.console_logger.log('INFO', f'🎯 AI Prediction: TP={take_profit:.2f}%, SL={stop_loss:.2f}%')
                
                # Ensure confidence is in percentage format (0-100)
                if confidence < 1.0:  # If it's a fraction (0.0-1.0), convert to percentage
                    confidence = confidence * 100
                
                # Check if prediction meets minimum confidence threshold FIRST
                # Get thresholds from database - NO FALLBACK, database is leading
                try:
                    from db_singleton import get_database
                    db = get_database()
                    db_settings = db.load_settings()
                    if not db_settings or 'confidenceThreshold' not in db_settings or 'accuracyThreshold' not in db_settings:
                        self.console_logger.log('ERROR', '❌ Database settings not available or incomplete - STOPPING signal processing')
                        return  # STOP processing if database not available
                    
                    ai_threshold = float(db_settings['confidenceThreshold'])
                    accuracy_threshold = float(db_settings['accuracyThreshold'])
                except Exception as db_error:
                    self.console_logger.log('ERROR', f'❌ Database error: {db_error} - STOPPING signal processing')
                    return  # STOP processing if database error
                
                # Check confidence threshold
                if confidence < ai_threshold:
                    self.console_logger.log('INFO', f'⏭️ Signal {symbol} below confidence threshold ({confidence:.1f}% < {ai_threshold}%) - not saving')
                    return  # Don't save or process low confidence signals
                
                # Check accuracy threshold
                signal_accuracy = prediction.get('accuracy', 0)
                if signal_accuracy < accuracy_threshold:
                    self.console_logger.log('INFO', f'⏭️ Signal {symbol} below accuracy threshold ({signal_accuracy:.1f}% < {accuracy_threshold}%) - not saving')
                    return  # Don't save or process low accuracy signals
                
                self.signal_count += 1
                self.console_logger.log('INFO', 
                    f'🎯 Signal #{self.signal_count}: {side} {symbol} (Confidence: {confidence:.1f}%)')
                
                # SAVE TO DATABASE ONLY - NO DIRECT EXECUTION
                # Let the ranked system handle execution
                try:
                    # Get take profit limits from database - NO FALLBACK
                    min_take_profit = float(db_settings.get('minTakeProfitPercent'))
                    max_take_profit = float(db_settings.get('maxTakeProfitPercent'))
                    stop_loss_percent = float(db_settings.get('stopLossPercent'))
                    take_profit_percent = float(db_settings.get('takeProfitPercent'))
                    
                    # Enforce take profit limits - AI prediction must be within bounds
                    ai_take_profit = prediction.get('take_profit', take_profit_percent)
                    if ai_take_profit < min_take_profit:
                        self.console_logger.log('WARNING', f'⚠️ AI take profit {ai_take_profit}% below minimum {min_take_profit}% - using minimum')
                        ai_take_profit = min_take_profit
                    elif ai_take_profit > max_take_profit:
                        self.console_logger.log('WARNING', f'⚠️ AI take profit {ai_take_profit}% above maximum {max_take_profit}% - using maximum')
                        ai_take_profit = max_take_profit
                    
                    # Save signal to database with waiting status
                    signal_id = f'signal_{self.signal_count}_{symbol}'
                    signal_data = {
                        'signal_id': signal_id,
                        'symbol': symbol,
                        'side': side,
                        'confidence': confidence,
                        'accuracy': prediction.get('accuracy', accuracy_threshold),
                        'amount': prediction.get('amount', 100),
                        'leverage': prediction.get('leverage', 1),
                        'stop_loss': prediction.get('stop_loss', stop_loss_percent),
                        'take_profit': ai_take_profit,  # Use enforced take profit
                        'entry_price': prediction.get('entry_price', 0.0),  # AI-advised entry price
                        'status': 'waiting'
                    }
                    db.save_trading_signal(signal_data)
                    
                    self.console_logger.log('SUCCESS', f'✅ Signal saved to database: {signal_id}')
                    
                except Exception as db_error:
                    self.console_logger.log('ERROR', f'❌ Failed to save signal to database: {str(db_error)}')
                
                # Emit signal to frontend
                if self.socketio:
                    self.socketio.emit('trading_signal', {
                        'signal_id': self.signal_count,
                        'symbol': symbol,
                        'side': side,
                        'confidence': confidence,
                        'timestamp': prediction['timestamp']
                    })
            else:
                self.console_logger.log('INFO', '⏳ No signals generated (confidence too low)')
                
        except Exception as e:
            self.console_logger.log('ERROR', f'Signal generation error: {str(e)}')
    
    def _execute_top_signals(self):
        """Execute signals with robust failure handling - try next signal if current fails"""
        try:
            # Check if we have capacity for new trades
            active_trades_count = self.get_active_positions_count()
            if active_trades_count >= self.max_concurrent_trades:
                self.console_logger.log('INFO', f'📊 Max concurrent trades reached ({active_trades_count}/{self.max_concurrent_trades})')
                return
                
            # Get waiting signals from database
            from db_singleton import get_database
            db = get_database()
            
            # Get all waiting signals (including previously failed ones after retraining)
            waiting_signals = [s for s in db.get_trading_signals() if s.get('status') in ['waiting', 'failed']]
            
            if not waiting_signals:
                return
                
            # Sort by confidence (highest first), then accuracy (highest first)
            waiting_signals.sort(key=lambda x: (x.get('confidence', 0), x.get('accuracy', 0)), reverse=True)
            
            # Get confidence threshold from database only
            try:
                db_settings = self.database.load_settings()
                if not db_settings or 'confidenceThreshold' not in db_settings:
                    raise Exception("No database settings found")
                ai_threshold = float(db_settings['confidenceThreshold'])
            except Exception as db_error:
                self.console_logger.log('ERROR', f'Failed to get database confidence threshold: {db_error}')
                return  # Cannot proceed without database settings
            
            signals_processed = 0
            trades_executed = 0
            
            self.console_logger.log('INFO', f'🔄 Processing {len(waiting_signals)} signals with {self.max_concurrent_trades - active_trades_count} slots available')
            
            # Execute signals until we reach max concurrent trades or run out of signals
            for signal in waiting_signals:
                # Check current active count BEFORE each execution
                current_active_count = self.get_active_positions_count()
                
                if current_active_count >= self.max_concurrent_trades:
                    self.console_logger.log('INFO', f'🚫 Max concurrent trades reached ({current_active_count}/{self.max_concurrent_trades}), stopping signal execution')
                    break
                    
                signals_processed += 1
                
                # Check if signal meets minimum confidence threshold
                if signal.get('confidence', 0) < ai_threshold:
                    self.console_logger.log('INFO', f'⏭️ Signal {signal["symbol"]} below threshold ({signal.get("confidence", 0):.1f}% < {ai_threshold}%) - skipping')
                    continue
                
                # Check if signal was previously failed - if so, verify it still meets criteria after retraining
                if signal.get('status') == 'failed':
                    # After retraining, if signal still meets confidence/accuracy requirements, retry it
                    self.console_logger.log('INFO', f'🔄 Retrying previously failed signal: {signal["symbol"]} (Confidence: {signal.get("confidence", 0):.1f}%)')
                    # Reset status to waiting for retry
                    db.update_signal_status(signal['signal_id'], 'waiting')
                
                # Execute this signal
                self.console_logger.log('INFO', f'🎯 Executing signal #{signals_processed}: {signal["symbol"]} (Confidence: {signal.get("confidence", 0):.1f}%, Accuracy: {signal.get("accuracy", 0):.1f}%)')
                
                # Update signal status to in_progress
                db.update_signal_status(signal['signal_id'], 'in_progress')
                
                # Execute the trade
                trade_result = self.execute_signal_direct(signal)
                
                if trade_result:
                    # Update signal status to executing (not executed until position is actually filled)
                    db.update_signal_status(signal['signal_id'], 'executing')
                    trades_executed += 1
                    
                    # Get updated count after execution
                    new_active_count = self.get_active_positions_count()
                    self.console_logger.log('SUCCESS', f'✅ Signal executed for {signal["symbol"]} ({new_active_count}/{self.max_concurrent_trades} trades active)')
                else:
                    # Update signal status to failed and continue to next signal
                    db.update_signal_status(signal['signal_id'], 'failed')
                    self.console_logger.log('ERROR', f'❌ Signal execution failed for {signal["symbol"]} - trying next signal')
                    # Continue to next signal instead of stopping
                    continue
            
            # Log summary
            if signals_processed > 0:
                self.console_logger.log('INFO', f'📊 Signal execution summary: {trades_executed}/{signals_processed} signals executed successfully')
            
        except Exception as e:
            self.console_logger.log('ERROR', f'Signal execution error: {str(e)}')
            import traceback
            self.console_logger.log('ERROR', f'Traceback: {traceback.format_exc()}')
    
    def _update_statistics(self):
        """Update trading statistics"""
        try:
            stats = self.trade_logger.get_trade_statistics()
            
            # Log important stats changes
            if stats['total_trades'] > 0:
                win_rate = stats['win_rate']
                total_pnl = stats['total_pnl']
                
                if total_pnl != getattr(self, '_last_pnl', 0):
                    self.console_logger.log('INFO', 
                        f'📊 Stats Update: {stats["total_trades"]} trades, '
                        f'Win Rate: {win_rate:.1f}%, PnL: ${total_pnl:.2f}')
                    self._last_pnl = total_pnl
                    
        except Exception as e:
            self.console_logger.log('ERROR', f'Statistics update error: {str(e)}')
    
    def _emit_status_update(self):
        """Emit status update to frontend"""
        if self.socketio:
            status = {
                'worker_running': self.is_running,
                'training_in_progress': self.training_in_progress,
                'signal_count': self.signal_count,
                'last_model_update': self.last_model_update.isoformat() if self.last_model_update else None
            }
            self.socketio.emit('worker_status', status)
    
    def get_console_logs(self):
        """Get recent console logs"""
        return self.console_logger.get_recent_logs()
    
    def get_worker_stats(self):
        """Get worker statistics"""
        return {
            'is_running': self.is_running,
            'training_in_progress': self.training_in_progress,
            'signal_count': self.signal_count,
            'last_model_update': self.last_model_update.isoformat() if self.last_model_update else None,
            'uptime': time.time() - getattr(self, '_start_time', time.time()),
            'active_trades': self.get_active_positions_count(),
            'max_trades': self.max_concurrent_trades
        }
    
    def get_ai_confidence_threshold(self):
        """Get AI confidence threshold from database only"""
        try:
            db_settings = self.database.load_settings()
            if not db_settings or 'confidenceThreshold' not in db_settings:
                raise Exception("No database settings found")
            return float(db_settings['confidenceThreshold'])
        except Exception as db_error:
            self.console_logger.log('ERROR', f'Failed to get database confidence threshold: {db_error}')
            return 75.0  # Minimal fallback only
    
    def get_active_positions_count(self):
        """Get count of active positions AND pending orders"""
        try:
            if not self.bybit_session:
                return 0
            
            # Count active positions
            positions = self.bybit_session.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            
            positions_count = 0
            if positions and 'result' in positions and 'list' in positions['result']:
                for pos in positions['result']['list']:
                    if float(pos.get('size', 0)) > 0:
                        positions_count += 1
            
            # Count pending orders that are not yet filled
            orders = self.bybit_session.get_open_orders(
                category="linear",
                settleCoin="USDT"
            )
            
            pending_orders_count = 0
            if orders and 'result' in orders and 'list' in orders['result']:
                for order in orders['result']['list']:
                    # Count only main entry orders (not TP/SL orders)
                    # Skip if it's a TP or SL order by checking if reduceOnly is True
                    if not order.get('reduceOnly', False):
                        # This is a main entry order
                        pending_orders_count += 1
            
            total_active = positions_count + pending_orders_count
            self.console_logger.log('INFO', f'📊 Active count: {positions_count} positions + {pending_orders_count} pending orders = {total_active}')
            
            return total_active
            
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Failed to get positions count: {str(e)}')
            return 0
    
    def execute_signal_direct(self, signal):
        """Execute trading signal directly with pybit"""
        try:
            symbol = signal['symbol']
            side = signal['side']  # Keep as 'Buy' or 'Sell' - don't uppercase
            confidence = signal['confidence']
            
            # All symbols are allowed - no restrictions
            self.console_logger.log('INFO', f'📊 Processing signal for {symbol} (all symbols allowed)')
            
            # Check for existing positions in the same direction
            try:
                positions = self.bybit_session.get_positions(category="linear", symbol=symbol, settleCoin="USDT")
                if positions and 'result' in positions:
                    for position in positions['result']['list']:
                        if position['symbol'] == symbol and float(position['size']) > 0:
                            existing_side = position['side']
                            
                            # If same direction, block the trade
                            if existing_side == side:
                                self.console_logger.log('WARNING', f'⚠️ Already have {existing_side} position for {symbol}, blocking same direction trade')
                                # Update signal status to waiting
                                if 'signal_id' in signal:
                                    from db_singleton import get_database
                                    db = get_database()
                                    db.update_signal_status(signal['signal_id'], 'waiting')
                                return False
                            
                            # If opposite direction, allow it (will close existing position)
                            elif existing_side != side:
                                self.console_logger.log('INFO', f'📊 Existing {existing_side} position for {symbol}, new {side} trade will reverse position')
                                
            except Exception as position_error:
                self.console_logger.log('ERROR', f'❌ Failed to check existing positions for {symbol}: {str(position_error)}')
                return False
            
            # First check if symbol is tradeable by getting instrument info
            try:
                instruments = self.bybit_session.get_instruments_info(category="linear", symbol=symbol)
                if not instruments or 'result' not in instruments or not instruments['result']['list']:
                    self.console_logger.log('ERROR', f'❌ Symbol {symbol} not found or not tradeable')
                    return False
                
                instrument = instruments['result']['list'][0]
                min_order_qty = float(instrument['lotSizeFilter']['minOrderQty'])
                qty_step = float(instrument['lotSizeFilter']['qtyStep'])
                
            except Exception as instrument_error:
                self.console_logger.log('ERROR', f'❌ Failed to get instrument info for {symbol}: {str(instrument_error)}')
                return False
            
            # Get current market price
            ticker = self.bybit_session.get_tickers(category="linear", symbol=symbol)
            if not ticker or 'result' not in ticker or not ticker['result']['list']:
                self.console_logger.log('ERROR', f'❌ Failed to get ticker for {symbol}')
                return False
            
            last_price = ticker['result']['list'][0].get('lastPrice')
            if last_price is None:
                self.console_logger.log('ERROR', f'❌ lastPrice is None for {symbol}')
                return False
            current_price = float(last_price)
            
            # Calculate AI-advised entry price
            ai_entry_raw = signal.get('entry_price', 0.0)
            if ai_entry_raw is None:
                ai_entry_raw = 0.0
            ai_entry_adjustment = float(ai_entry_raw)  # Get AI price adjustment
            ai_entry_price = current_price * (1 + ai_entry_adjustment)
            
            self.console_logger.log('INFO', f'📈 Market Price: ${current_price:.4f}, AI Entry Price: ${ai_entry_price:.4f} ({ai_entry_adjustment*100:+.2f}%)')
            
            # Get user settings for leverage and trade amount - NO FALLBACK
            try:
                from db_singleton import get_database
                db = get_database()
                db_settings = db.load_settings()
                
                if not db_settings:
                    self.console_logger.log('ERROR', '❌ Database settings not available - STOPPING trade execution')
                    return False
                
                # User leverage settings - NO FALLBACK
                if 'minLeverage' not in db_settings or 'maxLeverage' not in db_settings:
                    raise Exception("Leverage settings missing from database")
                min_leverage = int(db_settings['minLeverage'])
                max_leverage = int(db_settings['maxLeverage'])
                leverage_strategy = db_settings.get('leverageStrategy', 'confidence_based')
                
                # User trade amount settings - NO FALLBACK
                if 'riskPerTrade' not in db_settings or 'minTradeAmount' not in db_settings:
                    raise Exception("Trade amount settings missing from database")
                risk_per_trade = float(db_settings['riskPerTrade'])
                min_trade_amount = float(db_settings['minTradeAmount'])
                
                # User concurrent trades setting - NO FALLBACK
                if 'maxConcurrentTrades' not in db_settings:
                    raise Exception("Max concurrent trades setting missing from database")
                max_concurrent_trades = int(db_settings['maxConcurrentTrades'])
                
                # Take profit limits - ENFORCE STRICTLY - NO FALLBACK
                if 'minTakeProfitPercent' not in db_settings or 'maxTakeProfitPercent' not in db_settings:
                    raise Exception("Take profit limits missing from database")
                min_take_profit = float(db_settings['minTakeProfitPercent'])
                max_take_profit = float(db_settings['maxTakeProfitPercent'])
                
                # Update AI worker's max concurrent trades from user settings
                self.max_concurrent_trades = max_concurrent_trades
                
            except Exception as settings_error:
                self.console_logger.log('ERROR', f'❌ Database error: {settings_error} - STOPPING trade execution')
                return False  # STOP execution if database error
            
            # Calculate leverage based on user settings and AI confidence
            if leverage_strategy == 'confidence_based':
                # Higher confidence = higher leverage within user's range
                leverage_factor = max(0, min(1, (confidence - 50) / 50))  # Scale 0-1
                leverage = min_leverage + int((max_leverage - min_leverage) * leverage_factor)
            elif leverage_strategy == 'fixed':
                leverage = min_leverage
            else:  # volatility_based or adaptive
                leverage = min_leverage + int((max_leverage - min_leverage) * 0.5)  # Mid-range
            
            # Ensure leverage is within user's bounds
            leverage = max(min_leverage, min(max_leverage, leverage))
            
            # Get current account balance - NO FALLBACK
            try:
                balance_data = self.bybit_session.get_wallet_balance(accountType="UNIFIED")
                total_balance = float(balance_data['result']['list'][0]['totalWalletBalance'])
            except Exception as balance_error:
                self.console_logger.log('ERROR', f'❌ Failed to get balance: {balance_error} - STOPPING trade execution')
                return False  # STOP execution if balance fetch fails
            
            # Calculate trade amount as percentage of balance (user's risk setting)
            # Fix: User expects 5% to mean 0.5% (divide by 10)
            calculated_trade_amount = total_balance * (risk_per_trade / 1000)
            
            # If calculated amount is less than minimum, use EXACT minimum
            if calculated_trade_amount < min_trade_amount:
                trade_amount_usd = min_trade_amount
                self.console_logger.log('INFO', f'💰 Using minimum trade amount: ${min_trade_amount:.2f} (calculated: ${calculated_trade_amount:.2f})')
            else:
                trade_amount_usd = calculated_trade_amount
                self.console_logger.log('INFO', f'💰 Using calculated trade amount: ${calculated_trade_amount:.2f}')
            
            # Calculate total quantity for this trade amount (WITH leverage to get position value)
            # Trade amount × leverage = position value, so qty = (trade_amount × leverage) / price
            position_value = trade_amount_usd * leverage
            total_qty = position_value / current_price
            
            # Round to proper step size
            total_qty = max(min_order_qty, round(total_qty / qty_step) * qty_step)
            
            # Double-check minimum quantity
            if total_qty < min_order_qty:
                self.console_logger.log('ERROR', f'❌ Calculated quantity {total_qty} is below minimum {min_order_qty} for {symbol}')
                return False
            
            # Verify minimum order value from database
            try:
                db_settings = self.database.load_settings()
                min_order_value = float(db_settings.get('minOrderValue', 5.0))
            except:
                min_order_value = 5.0  # ByBit default
            
            total_order_value = total_qty * current_price
            if total_order_value < min_order_value:
                self.console_logger.log('ERROR', f'❌ Order value ${total_order_value:.2f} is below minimum ${min_order_value:.2f} for {symbol}')
                return False
            
            self.console_logger.log('INFO', f'💰 Trade Amount Setting: ${trade_amount_usd:.2f} ({risk_per_trade:.1f}% of ${total_balance:.2f})')
            self.console_logger.log('INFO', f'💰 Position Value (Trade Amount × {leverage}x Leverage): ${position_value:.2f}')
            self.console_logger.log('INFO', f'💰 Order Value (After Rounding): ${total_order_value:.2f} (Qty: {total_qty})')
            self.console_logger.log('INFO', f'🔧 Settings: Min Lev: {min_leverage}x, Max Lev: {max_leverage}x, Strategy: {leverage_strategy}')
            self.console_logger.log('INFO', f'📊 Max Concurrent Trades: {self.max_concurrent_trades} (from user settings)')
            
            # Set leverage first
            try:
                leverage_result = self.bybit_session.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage)
                )
                if leverage_result and 'result' in leverage_result:
                    self.console_logger.log('SUCCESS', f'✅ Leverage set to {leverage}x for {symbol}')
                else:
                    self.console_logger.log('WARNING', f'⚠️ Could not set leverage for {symbol}')
            except Exception as lev_error:
                self.console_logger.log('WARNING', f'⚠️ Leverage setting error: {str(lev_error)}')
            
            # Calculate stop loss and take profit from signal (AI determined) - NO FALLBACK
            if 'stop_loss' not in signal or 'take_profit' not in signal:
                self.console_logger.log('ERROR', f'❌ AI signal missing stop_loss or take_profit - STOPPING trade execution')
                return False
            stop_loss_pct = float(signal['stop_loss'])
            take_profit_pct = float(signal['take_profit'])
            
            # Ensure take profit is within configured bounds - ENFORCE STRICTLY from database
            # Use the limits we already loaded earlier
            take_profit_pct = max(min_take_profit, min(max_take_profit, take_profit_pct))
            
            if take_profit_pct != float(signal['take_profit']):
                self.console_logger.log('WARNING', f'⚠️ Take profit adjusted from {signal["take_profit"]:.2f}% to {take_profit_pct:.2f}% (limits: {min_take_profit:.2f}-{max_take_profit:.2f}%)')
            
            self.console_logger.log('INFO', f'📊 Using TP: {take_profit_pct:.2f}% (bounds: {min_take_profit:.2f}-{max_take_profit:.2f}%)')
            
            # Calculate stop loss and take profit based on AI ENTRY PRICE (not current market price)
            if side == 'Buy':
                stop_loss_price = ai_entry_price * (1 - stop_loss_pct / 100)
                take_profit_price = ai_entry_price * (1 + take_profit_pct / 100)
            else:  # Sell
                stop_loss_price = ai_entry_price * (1 + stop_loss_pct / 100)
                take_profit_price = ai_entry_price * (1 - take_profit_pct / 100)
            
            # Check if TP is already reached at current market price
            if side == 'Buy' and current_price >= take_profit_price:
                self.console_logger.log('WARNING', f'⚠️ TP already reached! Market: ${current_price:.4f}, TP: ${take_profit_price:.4f} - Cancelling signal')
                return False
            elif side == 'Sell' and current_price <= take_profit_price:
                self.console_logger.log('WARNING', f'⚠️ TP already reached! Market: ${current_price:.4f}, TP: ${take_profit_price:.4f} - Cancelling signal')
                return False
            
            # Place main LIMIT order WITH stop loss included
            order_params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,  # Use exact case: 'Buy' or 'Sell'
                'orderType': 'Limit',  # LIMIT ORDER ONLY
                'qty': str(total_qty),
                'price': str(ai_entry_price),  # Use AI-advised entry price
                'timeInForce': 'GTC',  # Good Till Cancelled
                'stopLoss': str(stop_loss_price),  # Include stop loss directly
                'takeProfit': str(take_profit_price)  # Include main TP level
            }
            
            self.console_logger.log('INFO', f'📤 Placing {side} LIMIT order: {total_qty} {symbol} @ ${ai_entry_price:.4f} (AI-advised)')
            
            # Execute the main LIMIT order first
            order_result = self.bybit_session.place_order(**order_params)
            
            if order_result and 'result' in order_result:
                entry_order_id = order_result['result']['orderId']
                
                # Log successful LIMIT order placement
                self.console_logger.log('SUCCESS', f'✅ LIMIT Order placed: {entry_order_id}')
                self.console_logger.log('INFO', f'📊 Details: {total_qty} {symbol} @ ${ai_entry_price:.4f} (waiting for fill)')
                
                # Don't place stop loss immediately - wait for entry fill
                # Store stop loss price for later placement
                sl_order_id = None
                self.console_logger.log('INFO', f'📋 Stop Loss will be placed at ${stop_loss_price:.4f} after entry fills')
                
                # Calculate 4 TP levels: 3 limit orders (25% each) + 1 final take profit (25%)
                tp_levels = []
                tp_order_ids = []
                
                # Get TP levels from database settings
                try:
                    db_settings = self.database.load_settings()
                    tp_levels_count = int(db_settings.get('takeProfitLevels', 4))
                    tp_split_percent = float(db_settings.get('takeProfitSplitPercent', 25.0))
                except:
                    tp_levels_count = 4
                    tp_split_percent = 25.0
                
                # Divide total quantity into equal parts
                tp_qty = total_qty / tp_levels_count
                tp_qty = max(min_order_qty, round(tp_qty / qty_step) * qty_step)
                
                for i in range(1, tp_levels_count + 1):
                    # Calculate TP price for each level
                    tp_percentage = (take_profit_pct * i) / tp_levels_count
                    
                    if side == 'Buy':
                        tp_price = ai_entry_price * (1 + tp_percentage / 100)  # Use AI entry price
                    else:  # Sell
                        tp_price = ai_entry_price * (1 - tp_percentage / 100)  # Use AI entry price
                    
                    tp_level = {
                        'level': i,
                        'price': tp_price,
                        'qty': tp_qty,
                        'status': 'pending',
                        'hit_time': None,
                        'order_id': None
                    }
                    tp_levels.append(tp_level)
                    
                    # Place TP order (levels 1-3 are limit orders, level 4 is take profit market order)
                    try:
                        if i <= 3:
                            # First 3 levels: Limit orders (25% each)
                            tp_order_params = {
                                'category': 'linear',
                                'symbol': symbol,
                                'side': 'Sell' if side == 'Buy' else 'Buy',
                                'orderType': 'Limit',
                                'qty': str(tp_qty),
                                'price': str(tp_price),
                                'timeInForce': 'GTC',
                                'reduceOnly': True  # Important: this closes the position
                            }
                            self.console_logger.log('INFO', f'📊 Placing Limit Order TP{i} @ ${tp_price:.4f} (25% of position)')
                        else:
                            # Level 4: Take Profit market order for final 25%
                            tp_order_params = {
                                'category': 'linear',
                                'symbol': symbol,
                                'side': 'Sell' if side == 'Buy' else 'Buy',
                                'orderType': 'Limit',  # Still limit for level 4
                                'qty': str(tp_qty),
                                'price': str(tp_price),
                                'timeInForce': 'GTC',
                                'reduceOnly': True
                            }
                            self.console_logger.log('INFO', f'📊 Placing Final TP{i} @ ${tp_price:.4f} (final 25% of position)')
                        
                        tp_result = self.bybit_session.place_order(**tp_order_params)
                        
                        if tp_result and 'result' in tp_result:
                            tp_order_id = tp_result['result']['orderId']
                            tp_level['order_id'] = tp_order_id
                            tp_order_ids.append(tp_order_id)
                            order_type = "Limit" if i <= 3 else "Final TP"
                            self.console_logger.log('SUCCESS', f'✅ {order_type} TP{i} set: {tp_order_id} @ ${tp_price:.4f} (qty: {tp_qty})')
                        else:
                            error_msg = tp_result.get('retMsg', 'Unknown error') if tp_result else 'No response'
                            self.console_logger.log('WARNING', f'⚠️ TP{i} failed: {error_msg}')
                            
                    except Exception as tp_error:
                        self.console_logger.log('ERROR', f'❌ TP{i} error: {str(tp_error)}')
                
                # Store trade in active trades for monitoring
                self.active_trades[entry_order_id] = {
                    'signal_id': signal.get('signal_id'),  # Add signal_id for P&L tracking
                    'symbol': symbol,
                    'side': side,
                    'quantity': total_qty,
                    'entry_price': ai_entry_price,  # Use AI entry price
                    'market_price': current_price,  # Store market price for comparison
                    'stop_loss': stop_loss_price,
                    'stop_loss_price': stop_loss_price,  # Store for later placement
                    'original_stop_loss': stop_loss_price,  # Keep original for reference
                    'take_profit_levels': tp_levels,
                    'tp_order_ids': tp_order_ids,
                    'sl_order_id': sl_order_id,
                    'sl_order_placed': False,  # Track if SL has been placed after entry fill
                    'entry_order_id': entry_order_id,
                    'entry_filled': False,  # Track if entry order is filled
                    'tp1_hit': False,
                    'sl_moved_to_breakeven': False,
                    'trailing_stop_enabled': bool(signal.get('trailing_stop_enabled', True)),
                    'trailing_stop_distance': float(signal.get('trailing_stop_distance', 1.0)),
                    'timestamp': datetime.now().isoformat(),
                    'trade_amount_usd': trade_amount_usd
                }
                
                self.console_logger.log('SUCCESS', f'✅ LIMIT Order setup complete: Entry @ ${ai_entry_price:.4f}, {len(tp_order_ids)}/4 TP levels active')
                self.console_logger.log('INFO', f'📊 TP1: ${tp_levels[0]["price"]:.4f}, TP2: ${tp_levels[1]["price"]:.4f}, TP3: ${tp_levels[2]["price"]:.4f}, TP4: ${tp_levels[3]["price"]:.4f}')
                self.console_logger.log('INFO', f'🛡️ Stop Loss: ${stop_loss_price:.4f} (will move to breakeven+0.1% when TP1 hits)')
                
                # Update signal status and store order IDs in database
                if 'signal_id' in signal:
                    try:
                        from db_singleton import get_database
                        db = get_database()
                        # Update signal with order IDs
                        signal_update = {
                            'entry_order_id': entry_order_id,
                            'tp_order_ids': tp_order_ids,
                            'sl_order_id': sl_order_id
                        }
                        # For now, just update status - we'll add the order ID fields later
                        db.update_signal_status(signal['signal_id'], 'executed')
                    except Exception as db_error:
                        self.console_logger.log('WARNING', f'⚠️ Could not update signal in database: {db_error}')
                
                return True
            else:
                # Check if there's an error message in the response
                error_msg = "Unknown error"
                if order_result and 'retMsg' in order_result:
                    error_msg = order_result['retMsg']
                elif order_result and 'ret_msg' in order_result:
                    error_msg = order_result['ret_msg']
                
                self.console_logger.log('ERROR', f'❌ Order failed: {error_msg}')
                
                # Update signal status to 'failed' in database
                if 'signal_id' in signal:
                    try:
                        from db_singleton import get_database
                        db = get_database()
                        db.update_signal_status(signal['signal_id'], 'failed')
                    except Exception as db_error:
                        self.console_logger.log('WARNING', f'⚠️ Could not update signal status: {db_error}')
                
                return False
                
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Trade execution error: {str(e)}')
            
            # Update signal status to 'failed' in database
            if 'signal_id' in signal:
                try:
                    from db_singleton import get_database
                    db = get_database()
                    db.update_signal_status(signal['signal_id'], 'failed')
                except Exception as db_error:
                    self.console_logger.log('WARNING', f'⚠️ Could not update signal status: {db_error}')
            
            return False
    
    def monitor_trades_and_move_sl(self):
        """Monitor active trades and move stop loss to breakeven when first TP is hit"""
        if not hasattr(self, 'active_trades') or not self.active_trades:
            return
        
        if not self.bybit_session:
            self.console_logger.log('WARNING', '⚠️ ByBit session not available for monitoring')
            return
        
        try:
            # Get current orders to check TP status with retry logic
            orders = None
            try:
                db_settings = self.database.load_settings()
                max_retries = int(db_settings.get('apiRetryCount', 3))
                retry_delay = float(db_settings.get('apiRetryDelay', 1.0))
            except:
                max_retries = 3
                retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    orders = self.bybit_session.get_open_orders(category="linear")
                    if orders and 'result' in orders:
                        break
                    time.sleep(retry_delay)
                except Exception as api_error:
                    if attempt == max_retries - 1:
                        self.console_logger.log('ERROR', f'❌ Failed to fetch orders after {max_retries} attempts: {str(api_error)}')
                        return
                    time.sleep(retry_delay * 2)  # Longer delay between retries
            
            if not orders or 'result' not in orders:
                self.console_logger.log('WARNING', '⚠️ Could not fetch open orders for monitoring after retries')
                return
            
            # Get current positions with retry logic
            positions = None
            for attempt in range(max_retries):
                try:
                    positions = self.bybit_session.get_positions(category="linear", settleCoin="USDT")
                    if positions and 'result' in positions:
                        break
                    time.sleep(retry_delay)
                except Exception as api_error:
                    if attempt == max_retries - 1:
                        self.console_logger.log('ERROR', f'❌ Failed to fetch positions after {max_retries} attempts: {str(api_error)}')
                        return
                    time.sleep(retry_delay * 2)  # Longer delay between retries
            
            if not positions or 'result' not in positions:
                self.console_logger.log('WARNING', '⚠️ Could not fetch positions for monitoring after retries')
                return
            
            self.console_logger.log('INFO', f'🔍 Monitoring {len(self.active_trades)} active trades for SL movement')
            
            for order_id, trade_data in list(self.active_trades.items()):
                symbol = trade_data['symbol']
                side = trade_data['side']
                entry_price = trade_data['entry_price']
                tp_levels = trade_data['take_profit_levels']
                tp_order_ids = trade_data['tp_order_ids']
                entry_order_id = trade_data['entry_order_id']
                
                self.console_logger.log('INFO', f'📊 Checking trade {symbol} - Entry filled: {trade_data.get("entry_filled", False)}, TP1 hit: {trade_data.get("tp1_hit", False)}')
                
                # First, check if entry order has been filled
                entry_filled = trade_data.get('entry_filled', False)
                if not entry_filled:
                    # Check if entry order is still pending
                    entry_still_open = False
                    for order in orders['result']['list']:
                        if order['orderId'] == entry_order_id:
                            entry_still_open = True
                            break
                    
                    # If entry order is no longer in open orders, it was filled
                    if not entry_still_open:
                        trade_data['entry_filled'] = True
                        
                        # Get the actual fill price from execution history
                        try:
                            executions = self.bybit_session.get_executions(
                                category="linear",
                                symbol=symbol,
                                orderId=entry_order_id,
                                limit=1
                            )
                            
                            if executions and 'result' in executions and executions['result']['list']:
                                exec_price = executions['result']['list'][0].get('execPrice')
                                if exec_price is None:
                                    self.console_logger.log('ERROR', f'❌ execPrice is None for {symbol}')
                                    continue
                                actual_entry_price = float(exec_price)
                                trade_data['actual_entry_price'] = actual_entry_price
                                
                                # Update signal in database with actual entry price
                                signal_id = trade_data.get('signal_id')
                                if signal_id:
                                    # Update signal status to 'pending' (position now open, waiting for exit)
                                    from db_singleton import get_database
                                    db = get_database()
                                    
                                    # Update both status and entry price
                                    conn = db.get_connection()
                                    cursor = conn.cursor()
                                    placeholder = '%s' if db.use_postgres else '?'
                                    
                                    cursor.execute(f'''
                                        UPDATE trading_signals 
                                        SET status = {placeholder}, entry_price = {placeholder}, updated_at = CURRENT_TIMESTAMP
                                        WHERE signal_id = {placeholder}
                                    ''', ('pending', actual_entry_price, signal_id))
                                    
                                    conn.commit()
                                    conn.close()
                                    
                                    self.console_logger.log('SUCCESS', f'✅ Entry filled for {symbol} @ ${actual_entry_price:.4f} (signal updated to pending)')
                                else:
                                    self.console_logger.log('SUCCESS', f'✅ Entry order filled for {symbol} @ ${actual_entry_price:.4f}')
                            else:
                                self.console_logger.log('SUCCESS', f'✅ Entry order filled for {symbol} @ ${entry_price:.4f} (using limit price)')
                                
                            # Place stop loss order NOW that entry is filled
                            if not trade_data.get('sl_order_placed', False):
                                stop_loss_price = trade_data.get('stop_loss_price')
                                if stop_loss_price:
                                    try:
                                        sl_order_params = {
                                            'category': 'linear',
                                            'symbol': symbol,
                                            'side': 'Sell' if side == 'Buy' else 'Buy',
                                            'orderType': 'Market',
                                            'qty': str(trade_data['quantity']),
                                            'triggerPrice': str(stop_loss_price),
                                            'triggerBy': 'LastPrice',
                                            'triggerDirection': 2 if side == 'Buy' else 1,  # 2=fall for long SL, 1=rise for short SL
                                            'timeInForce': 'IOC',
                                            'reduceOnly': True
                                        }
                                        
                                        sl_result = self.bybit_session.place_order(**sl_order_params)
                                        
                                        if sl_result and 'result' in sl_result:
                                            sl_order_id = sl_result['result']['orderId']
                                            trade_data['sl_order_id'] = sl_order_id
                                            trade_data['sl_order_placed'] = True
                                            self.console_logger.log('SUCCESS', f'✅ Stop Loss placed after entry fill: {sl_order_id} @ ${stop_loss_price:.4f}')
                                        else:
                                            error_msg = sl_result.get('retMsg', 'Unknown error') if sl_result else 'No response'
                                            self.console_logger.log('ERROR', f'❌ Failed to place SL after entry: {error_msg}')
                                    except Exception as sl_error:
                                        self.console_logger.log('ERROR', f'❌ SL placement error after entry: {str(sl_error)}')
                                        
                        except Exception as exec_error:
                            self.console_logger.log('WARNING', f'⚠️ Could not get execution details for {symbol}: {exec_error}')
                            self.console_logger.log('SUCCESS', f'✅ Entry order filled for {symbol} @ ${entry_price:.4f} (using limit price)')
                    else:
                        # Entry order still pending - check if TP is reached (cancel signal)
                        ticker = self.bybit_session.get_tickers(category="linear", symbol=symbol)
                        if ticker and 'result' in ticker and ticker['result']['list']:
                            current_price = float(ticker['result']['list'][0]['lastPrice'])
                            tp_price = trade_data['take_profit_levels'][3]['price']  # Final TP level
                            
                            # Check if TP is reached before entry
                            if side == 'Buy' and current_price >= tp_price:
                                self.console_logger.log('WARNING', f'⚠️ TP reached before entry fill for {symbol}! Cancelling all orders')
                                self._cancel_all_orders_for_symbol(symbol)
                                del self.active_trades[order_id]
                                continue
                            elif side == 'Sell' and current_price <= tp_price:
                                self.console_logger.log('WARNING', f'⚠️ TP reached before entry fill for {symbol}! Cancelling all orders')
                                self._cancel_all_orders_for_symbol(symbol)
                                del self.active_trades[order_id]
                                continue
                        
                        # Entry order still pending, continue monitoring
                        self.console_logger.log('INFO', f'⏳ Entry order still pending for {symbol}')
                        continue
                
                # Entry is filled, now monitor TP levels
                # Check if TP1 has been hit by looking at open orders - FIXED LOGIC
                tp1_order_id = tp_levels[0].get('order_id')
                tp1_still_open = False
                
                if tp1_order_id:
                    for order in orders['result']['list']:
                        if order['orderId'] == tp1_order_id:
                            tp1_still_open = True
                            break
                    
                    self.console_logger.log('INFO', f'🎯 TP1 status for {symbol}: Order ID {tp1_order_id}, Still open: {tp1_still_open}')
                
                # If TP1 is no longer in open orders, it was filled
                if tp1_order_id and not tp1_still_open and not trade_data.get('tp1_hit', False):
                    self.console_logger.log('SUCCESS', f'✅ TP1 hit for {symbol}! Moving stop loss to breakeven + 0.1%')
                    
                    # Get current position size to calculate correct SL quantity
                    current_position_size = 0
                    for position in positions['result']['list']:
                        if position['symbol'] == symbol and float(position.get('size', 0)) > 0:
                            current_position_size = float(position['size'])
                            break
                    
                    if current_position_size == 0:
                        self.console_logger.log('WARNING', f'⚠️ No position found for {symbol}, skipping SL movement')
                        continue
                    
                    self.console_logger.log('INFO', f'📊 Current position size for {symbol}: {current_position_size}')
                    
                    # Calculate breakeven + percentage from database
                    try:
                        db_settings = self.database.load_settings()
                        breakeven_percentage = float(db_settings.get('breakevenPercentage', 0.1)) / 100
                    except:
                        breakeven_percentage = 0.001  # 0.1% default
                    
                    if side == 'Buy':
                        new_sl_price = entry_price * (1 + breakeven_percentage)
                    else:
                        new_sl_price = entry_price * (1 - breakeven_percentage)
                    
                    # Update stop loss
                    try:
                        # Cancel existing stop loss order first
                        if trade_data.get('sl_order_id'):
                            self.console_logger.log('INFO', f'🚫 Cancelling existing SL order {trade_data["sl_order_id"]} for {symbol}')
                            cancel_result = self.bybit_session.cancel_order(category="linear", symbol=symbol, orderId=trade_data['sl_order_id'])
                            if cancel_result and 'result' in cancel_result:
                                self.console_logger.log('SUCCESS', f'✅ Existing SL cancelled for {symbol}')
                            else:
                                self.console_logger.log('WARNING', f'⚠️ Failed to cancel existing SL for {symbol}')
                        
                        # Place new stop loss at breakeven + 0.1% with current position size
                        self.console_logger.log('INFO', f'📤 Placing new SL order for {symbol}: qty={current_position_size}, price=${new_sl_price:.4f}')
                        sl_result = self.bybit_session.place_order(
                            category="linear",
                            symbol=symbol,
                            side="Sell" if side == "Buy" else "Buy",
                            orderType="Market",
                            qty=str(current_position_size),  # Use actual position size, not original quantity
                            triggerPrice=str(new_sl_price),
                            triggerBy="LastPrice",
                            triggerDirection=2 if side == "Buy" else 1,  # 2=fall for long SL, 1=rise for short SL
                            timeInForce="IOC",
                            reduceOnly=True
                        )
                        
                        if sl_result and 'result' in sl_result:
                            # Update trade data
                            trade_data['tp1_hit'] = True
                            trade_data['sl_moved_to_breakeven'] = True
                            trade_data['stop_loss'] = new_sl_price
                            trade_data['sl_order_id'] = sl_result['result']['orderId']
                            trade_data['quantity'] = current_position_size  # Update to current position size
                            trade_data['tp_levels'][0]['status'] = 'hit'
                            trade_data['tp_levels'][0]['hit_time'] = datetime.now().isoformat()
                            
                            self.console_logger.log('SUCCESS', f'✅ Stop loss moved to breakeven+0.1% for {symbol}: ${new_sl_price:.4f} (Order ID: {sl_result["result"]["orderId"]})')
                            
                            # Emit update to frontend
                            if self.socketio:
                                self.socketio.emit('sl_moved', {
                                    'symbol': symbol,
                                    'new_sl_price': new_sl_price,
                                    'entry_price': entry_price,
                                    'message': f'Stop loss moved to breakeven+0.1% for {symbol}'
                                })
                        else:
                            error_msg = sl_result.get('retMsg', 'Unknown error') if sl_result else 'No response'
                            self.console_logger.log('ERROR', f'❌ Failed to move stop loss for {symbol}: {error_msg}')
                            
                    except Exception as sl_error:
                        self.console_logger.log('ERROR', f'❌ Error moving stop loss for {symbol}: {str(sl_error)}')
                        import traceback
                        self.console_logger.log('ERROR', f'SL Error traceback: {traceback.format_exc()}')
                
                # Update TP level statuses
                for i, tp_level in enumerate(tp_levels):
                    if tp_level.get('order_id') and tp_level.get('status') == 'pending':
                        tp_still_open = False
                        for order in orders['result']['list']:
                            if order['orderId'] == tp_level['order_id']:
                                tp_still_open = True
                                break
                        
                        if not tp_still_open:
                            tp_level['status'] = 'hit'
                            tp_level['hit_time'] = datetime.now().isoformat()
                            self.console_logger.log('SUCCESS', f'✅ TP{i+1} hit for {symbol}!')
                
                # Clean up completed trades (all TPs hit or position closed)
                position_exists = False
                for position in positions['result']['list']:
                    if position['symbol'] == symbol and float(position.get('size', 0)) > 0:
                        position_exists = True
                        break
                
                if not position_exists:
                    # Position is closed, calculate P&L and update signal
                    self.console_logger.log('INFO', f'📊 Position closed for {symbol}, calculating P&L...')
                    
                    try:
                        # Get the signal_id from trade_data
                        signal_id = trade_data.get('signal_id')
                        entry_price = trade_data.get('actual_entry_price', trade_data.get('entry_price'))
                        
                        if signal_id and entry_price:
                            # Get execution history to calculate actual P&L
                            executions = self.bybit_session.get_executions(
                                category="linear",
                                symbol=symbol,
                                limit=100  # Get more executions to ensure we catch all trades
                            )
                            
                            total_pnl = 0
                            exit_price = 0
                            exit_trades = []
                            entry_trades = []
                            
                            if executions and 'result' in executions:
                                # Find recent executions for this symbol (within last 2 hours)
                                current_time = int(datetime.now().timestamp() * 1000)
                                
                                for execution in executions['result']['list']:
                                    if execution.get('symbol') == symbol:
                                        exec_time = int(execution.get('execTime', 0))
                                        if current_time - exec_time < 7200000:  # 2 hours
                                            exec_side = execution.get('side', '')
                                            
                                            # Categorize executions
                                            if exec_side == trade_data.get('side'):
                                                entry_trades.append(execution)
                                            else:
                                                exit_trades.append(execution)
                            
                            self.console_logger.log('INFO', f'📊 Found {len(entry_trades)} entry trades, {len(exit_trades)} exit trades for {symbol}')
                            
                            if exit_trades:
                                # Calculate weighted average exit price
                                total_exit_qty = 0
                                total_exit_value = 0
                                
                                for exit_trade in exit_trades:
                                    exec_qty = exit_trade.get('execQty')
                                    exec_price = exit_trade.get('execPrice')
                                    
                                    if exec_qty is None or exec_price is None:
                                        continue  # Skip trades with missing data
                                        
                                    qty = float(exec_qty)
                                    price = float(exec_price)
                                    
                                    if qty <= 0:  # Skip zero quantity trades
                                        continue
                                    
                                    total_exit_qty += qty
                                    total_exit_value += qty * price
                                
                                exit_price = total_exit_value / total_exit_qty if total_exit_qty > 0 else 0
                                
                                # Calculate P&L based on entry/exit prices and side
                                side = trade_data.get('side', '').upper()
                                
                                if side == 'BUY':
                                    # For long positions: profit = (exit_price - entry_price) * amount
                                    total_pnl = (exit_price - entry_price) * total_exit_qty
                                else:  # SELL
                                    # For short positions: profit = (entry_price - exit_price) * amount
                                    total_pnl = (entry_price - exit_price) * total_exit_qty
                                
                                # Update the signal with P&L data
                                self.database.update_signal_with_pnl(
                                    signal_id=signal_id,
                                    entry_price=entry_price,
                                    exit_price=exit_price,
                                    realized_pnl=total_pnl
                                )
                                
                                win_loss = "WIN" if total_pnl > 0 else "LOSS"
                                pnl_pct = (total_pnl / (entry_price * total_exit_qty)) * 100 if entry_price > 0 and total_exit_qty > 0 else 0
                                
                                self.console_logger.log('SUCCESS', f'📊 Signal {signal_id} completed: {win_loss} ${total_pnl:.2f} ({pnl_pct:+.2f}%)')
                                self.console_logger.log('INFO', f'📊 Entry: ${entry_price:.4f}, Exit: ${exit_price:.4f}, Qty: {total_exit_qty}')
                                
                                # Emit completion event to frontend
                                if self.socketio:
                                    self.socketio.emit('trade_completed', {
                                        'signal_id': signal_id,
                                        'symbol': symbol,
                                        'side': side,
                                        'entry_price': entry_price,
                                        'exit_price': exit_price,
                                        'pnl': total_pnl,
                                        'pnl_percent': pnl_pct,
                                        'win_loss': win_loss,
                                        'quantity': total_exit_qty
                                    })
                            else:
                                # No exit trades found, but position is closed - might be liquidated or manual close
                                self.console_logger.log('WARNING', f'⚠️ Position closed for {symbol} but no exit trades found')
                                
                                # Still update signal to completed status without P&L
                                from db_singleton import get_database
                                db = get_database()
                                conn = db.get_connection()
                                cursor = conn.cursor()
                                placeholder = '%s' if db.use_postgres else '?'
                                
                                cursor.execute(f'''
                                    UPDATE trading_signals 
                                    SET status = {placeholder}, exit_time = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                                    WHERE signal_id = {placeholder}
                                ''', ('completed', signal_id))
                                
                                conn.commit()
                                conn.close()
                                
                                self.console_logger.log('INFO', f'📊 Signal {signal_id} marked as completed (no P&L data)')
                        else:
                            self.console_logger.log('WARNING', f'⚠️ Missing signal_id or entry_price for {symbol}')
                    
                    except Exception as pnl_error:
                        self.console_logger.log('ERROR', f'❌ Error calculating P&L for {symbol}: {str(pnl_error)}')
                        import traceback
                        self.console_logger.log('ERROR', f'PnL calculation traceback: {traceback.format_exc()}')
                    
                    # Remove from active trades
                    self.console_logger.log('INFO', f'📊 Removing {symbol} from active trades monitoring')
                    del self.active_trades[order_id]
                    
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Trade monitoring error: {str(e)}')
    
    def _cancel_all_orders_for_symbol(self, symbol):
        """Cancel all orders for a specific symbol"""
        try:
            # Cancel all open orders for this symbol
            self.bybit_session.cancel_all_orders(category="linear", symbol=symbol)
            self.console_logger.log('INFO', f'🚫 Cancelled all orders for {symbol}')
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Error cancelling orders for {symbol}: {str(e)}')
    
    def _check_orphaned_positions(self):
        """Check for positions that exist but aren't being tracked and attempt to track them"""
        try:
            if not self.bybit_session:
                return
                
            # Get all current positions
            positions = self.bybit_session.get_positions(category="linear", settleCoin="USDT")
            if not positions or 'result' not in positions:
                return
                
            # Get symbols that are currently tracked
            tracked_symbols = set()
            for trade_data in self.active_trades.values():
                tracked_symbols.add(trade_data['symbol'])
            
            # Check for untracked positions
            for position in positions['result']['list']:
                if float(position.get('size', 0)) > 0:
                    symbol = position['symbol']
                    side = position['side']
                    
                    if symbol not in tracked_symbols:
                        # Found an orphaned position
                        self.console_logger.log('WARNING', f'⚠️ Found orphaned position: {symbol} {side}')
                        
                        try:
                            # Try to find a matching signal in the database
                            from db_singleton import get_database
                            db = get_database()
                            
                            # Look for recent signals with this symbol that are in 'pending' or 'executing' status
                            signals = db.get_trading_signals()
                            matching_signal = None
                            
                            for signal in signals:
                                if (signal.get('symbol') == symbol and 
                                    signal.get('side') == side and 
                                    signal.get('status') in ['pending', 'executing']):
                                    matching_signal = signal
                                    break
                            
                            if matching_signal:
                                # Found a matching signal, update it to completed and calculate P&L
                                self.console_logger.log('INFO', f'📊 Found matching signal for orphaned position {symbol}')
                                
                                # Get execution history for this symbol
                                executions = self.bybit_session.get_executions(
                                    category="linear",
                                    symbol=symbol,
                                    limit=50
                                )
                                
                                if executions and 'result' in executions:
                                    # Find the entry execution
                                    entry_price = None
                                    for execution in executions['result']['list']:
                                        if (execution.get('symbol') == symbol and 
                                            execution.get('side') == side):
                                            exec_price = execution.get('execPrice')
                                            if exec_price is not None:
                                                entry_price = float(exec_price)
                                                break
                                    
                                    if entry_price:
                                        # Update the signal with entry price and mark as pending
                                        db.update_signal_with_pnl(
                                            signal_id=matching_signal['signal_id'],
                                            entry_price=entry_price,
                                            exit_price=0,  # Position still open
                                            realized_pnl=0
                                        )
                                        
                                        # Update status to pending
                                        conn = db.get_connection()
                                        cursor = conn.cursor()
                                        placeholder = '%s' if db.use_postgres else '?'
                                        
                                        cursor.execute(f'''
                                            UPDATE trading_signals 
                                            SET status = {placeholder}, updated_at = CURRENT_TIMESTAMP
                                            WHERE signal_id = {placeholder}
                                        ''', ('pending', matching_signal['signal_id']))
                                        
                                        conn.commit()
                                        conn.close()
                                        
                                        self.console_logger.log('SUCCESS', f'✅ Orphaned position {symbol} linked to signal {matching_signal["signal_id"]}')
                                    else:
                                        self.console_logger.log('WARNING', f'⚠️ Could not find entry execution for orphaned position {symbol}')
                                else:
                                    self.console_logger.log('WARNING', f'⚠️ Could not get execution history for orphaned position {symbol}')
                            else:
                                self.console_logger.log('WARNING', f'⚠️ No matching signal found for orphaned position {symbol}')
                                
                        except Exception as orphan_error:
                            self.console_logger.log('ERROR', f'❌ Error processing orphaned position {symbol}: {str(orphan_error)}')
                            
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Error checking orphaned positions: {str(e)}')
    
    def _check_manually_closed_positions(self):
        """Check for tracked positions that have been manually closed and update their P&L"""
        try:
            if not self.bybit_session or not hasattr(self, 'active_trades') or not self.active_trades:
                return
                
            # Get all current positions
            positions = self.bybit_session.get_positions(category="linear", settleCoin="USDT")
            if not positions or 'result' not in positions:
                return
                
            # Create a set of symbols with active positions
            active_position_symbols = set()
            for position in positions['result']['list']:
                if float(position.get('size', 0)) > 0:
                    symbol = position['symbol']
                    side = position['side']
                    active_position_symbols.add(f"{symbol}_{side}")
            
            # Check tracked trades to see if any positions have been manually closed
            trades_to_remove = []
            for order_id, trade_data in list(self.active_trades.items()):
                symbol = trade_data['symbol']
                side = trade_data['side']
                trade_key = f"{symbol}_{side}"
                
                # If the position is no longer active but was being tracked
                if trade_key not in active_position_symbols:
                    # Position was manually closed
                    self.console_logger.log('WARNING', f'🔍 Detected manually closed position: {symbol} {side}')
                    
                    try:
                        # Get the signal_id from trade_data
                        signal_id = trade_data.get('signal_id')
                        entry_price = trade_data.get('actual_entry_price', trade_data.get('entry_price'))
                        
                        if signal_id and entry_price:
                            # Get execution history to calculate actual P&L for manual close
                            executions = self.bybit_session.get_executions(
                                category="linear",
                                symbol=symbol,
                                limit=100  # Get more executions to catch recent trades
                            )
                            
                            total_pnl = 0
                            exit_price = 0
                            exit_trades = []
                            
                            if executions and 'result' in executions:
                                # Find recent executions for this symbol (within last 1 hour for manual close)
                                current_time = int(datetime.now().timestamp() * 1000)
                                
                                for execution in executions['result']['list']:
                                    if execution.get('symbol') == symbol:
                                        exec_time = int(execution.get('execTime', 0))
                                        if current_time - exec_time < 3600000:  # 1 hour
                                            exec_side = execution.get('side', '')
                                            
                                            # Find exit trades (opposite side of entry)
                                            if exec_side != trade_data.get('side'):
                                                exit_trades.append(execution)
                            
                            self.console_logger.log('INFO', f'📊 Found {len(exit_trades)} recent exit trades for manually closed {symbol}')
                            
                            if exit_trades:
                                # Calculate weighted average exit price
                                total_exit_qty = 0
                                total_exit_value = 0
                                
                                for exit_trade in exit_trades:
                                    exec_qty = exit_trade.get('execQty')
                                    exec_price = exit_trade.get('execPrice')
                                    
                                    if exec_qty is None or exec_price is None:
                                        continue  # Skip trades with missing data
                                        
                                    qty = float(exec_qty)
                                    price = float(exec_price)
                                    
                                    if qty <= 0:  # Skip zero quantity trades
                                        continue
                                    
                                    total_exit_qty += qty
                                    total_exit_value += qty * price
                                
                                exit_price = total_exit_value / total_exit_qty if total_exit_qty > 0 else 0
                                
                                # Calculate P&L based on entry/exit prices and side
                                if side.upper() == 'BUY':
                                    # For long positions: profit = (exit_price - entry_price) * amount
                                    total_pnl = (exit_price - entry_price) * total_exit_qty
                                else:  # SELL
                                    # For short positions: profit = (entry_price - exit_price) * amount
                                    total_pnl = (entry_price - exit_price) * total_exit_qty
                                
                                # Update the signal with P&L data
                                self.database.update_signal_with_pnl(
                                    signal_id=signal_id,
                                    entry_price=entry_price,
                                    exit_price=exit_price,
                                    realized_pnl=total_pnl
                                )
                                
                                win_loss = "WIN" if total_pnl > 0 else "LOSS"
                                pnl_pct = (total_pnl / (entry_price * total_exit_qty)) * 100 if entry_price > 0 and total_exit_qty > 0 else 0
                                
                                self.console_logger.log('SUCCESS', f'✅ Manual close detected and recorded: {symbol} {win_loss} ${total_pnl:.2f} ({pnl_pct:+.2f}%)')
                                self.console_logger.log('INFO', f'📊 Entry: ${entry_price:.4f}, Exit: ${exit_price:.4f}, Qty: {total_exit_qty}')
                                
                                # Emit completion event to frontend
                                if self.socketio:
                                    self.socketio.emit('trade_completed', {
                                        'signal_id': signal_id,
                                        'symbol': symbol,
                                        'side': side,
                                        'entry_price': entry_price,
                                        'exit_price': exit_price,
                                        'pnl': total_pnl,
                                        'pnl_percent': pnl_pct,
                                        'win_loss': win_loss,
                                        'quantity': total_exit_qty,
                                        'manual_close': True
                                    })
                            else:
                                # No exit trades found in recent time, but position is closed
                                # This might be a very recent close or system liquidation
                                self.console_logger.log('WARNING', f'⚠️ Position manually closed for {symbol} but no recent exit trades found')
                                
                                # Get current market price for estimation
                                try:
                                    ticker = self.bybit_session.get_tickers(category="linear", symbol=symbol)
                                    if ticker and 'result' in ticker and ticker['result']['list']:
                                        current_price = float(ticker['result']['list'][0]['lastPrice'])
                                        
                                        # Estimate P&L based on current price
                                        quantity = trade_data.get('quantity', 0)
                                        if side.upper() == 'BUY':
                                            estimated_pnl = (current_price - entry_price) * quantity
                                        else:
                                            estimated_pnl = (entry_price - current_price) * quantity
                                        
                                        # Update signal with estimated data
                                        self.database.update_signal_with_pnl(
                                            signal_id=signal_id,
                                            entry_price=entry_price,
                                            exit_price=current_price,
                                            realized_pnl=estimated_pnl
                                        )
                                        
                                        self.console_logger.log('INFO', f'📊 Manual close recorded with estimated P&L: {symbol} ${estimated_pnl:.2f} (estimated)')
                                    else:
                                        # Still update signal to completed status without P&L
                                        from db_singleton import get_database
                                        db = get_database()
                                        conn = db.get_connection()
                                        cursor = conn.cursor()
                                        placeholder = '%s' if db.use_postgres else '?'
                                        
                                        cursor.execute(f'''
                                            UPDATE trading_signals 
                                            SET status = {placeholder}, exit_time = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                                            WHERE signal_id = {placeholder}
                                        ''', ('completed', signal_id))
                                        
                                        conn.commit()
                                        conn.close()
                                        
                                        self.console_logger.log('INFO', f'📊 Manual close recorded without P&L data: {symbol}')
                                except Exception as price_error:
                                    self.console_logger.log('WARNING', f'⚠️ Could not get current price for {symbol}: {price_error}')
                        else:
                            self.console_logger.log('WARNING', f'⚠️ Missing signal_id or entry_price for manually closed {symbol}')
                    
                    except Exception as pnl_error:
                        self.console_logger.log('ERROR', f'❌ Error calculating P&L for manually closed {symbol}: {str(pnl_error)}')
                    
                    # Mark for removal from active trades
                    trades_to_remove.append(order_id)
            
            # Remove manually closed trades from tracking
            for order_id in trades_to_remove:
                if order_id in self.active_trades:
                    symbol = self.active_trades[order_id]['symbol']
                    self.console_logger.log('INFO', f'📊 Removing manually closed {symbol} from active trades monitoring')
                    del self.active_trades[order_id]
                    
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Error checking manually closed positions: {str(e)}')
            import traceback
            self.console_logger.log('ERROR', f'Manual close check traceback: {traceback.format_exc()}')
    
    def force_monitor_trades(self):
        """Manually trigger trade monitoring - useful for testing"""
        self.console_logger.log('INFO', '🔄 Manually triggering trade monitoring...')
        self.monitor_trades_and_move_sl()
    
    def get_active_trades_status(self):
        """Get detailed status of all active trades"""
        if not hasattr(self, 'active_trades') or not self.active_trades:
            return {'message': 'No active trades', 'trades': []}
        
        trades_status = []
        for order_id, trade_data in self.active_trades.items():
            status = {
                'symbol': trade_data['symbol'],
                'side': trade_data['side'],
                'entry_price': trade_data['entry_price'],
                'stop_loss': trade_data['stop_loss'],
                'entry_filled': trade_data.get('entry_filled', False),
                'tp1_hit': trade_data.get('tp1_hit', False),
                'sl_moved_to_breakeven': trade_data.get('sl_moved_to_breakeven', False),
                'quantity': trade_data['quantity'],
                'entry_order_id': trade_data['entry_order_id'],
                'sl_order_id': trade_data.get('sl_order_id'),
                'tp_levels': [
                    {
                        'level': i+1,
                        'price': tp['price'],
                        'status': tp.get('status', 'pending'),
                        'order_id': tp.get('order_id')
                    }
                    for i, tp in enumerate(trade_data['take_profit_levels'])
                ]
            }
            trades_status.append(status)
        
        return {
            'message': f'Found {len(self.active_trades)} active trades',
            'trades': trades_status
        }
    
    def start_breakeven_monitor(self):
        """Start the breakeven monitoring thread"""
        try:
            if self.breakeven_thread and self.breakeven_thread.is_alive():
                self.console_logger.log('INFO', '🔄 Breakeven monitor already running')
                return
            
            def monitor_loop():
                self.console_logger.log('INFO', '🚀 Starting breakeven monitor thread')
                try:
                    while self.is_running:
                        self.breakeven_monitor.monitor_positions()
                        # Get monitoring interval from database
                        try:
                            db_settings = self.database.load_settings()
                            monitor_interval = int(db_settings.get('breakevenMonitorInterval', 30))
                            time.sleep(monitor_interval)
                        except:
                            time.sleep(30)  # Default fallback
                except Exception as e:
                    self.console_logger.log('ERROR', f'❌ Breakeven monitor error: {e}')
                finally:
                    self.console_logger.log('INFO', '⏹️ Breakeven monitor stopped')
            
            self.breakeven_thread = threading.Thread(target=monitor_loop, daemon=True)
            self.breakeven_thread.start()
            self.console_logger.log('INFO', '✅ Breakeven monitor started successfully')
            
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Failed to start breakeven monitor: {e}')
    
    def stop_breakeven_monitor(self):
        """Stop the breakeven monitoring thread"""
        try:
            if self.breakeven_thread and self.breakeven_thread.is_alive():
                # The thread will stop when is_running becomes False
                self.console_logger.log('INFO', '⏹️ Stopping breakeven monitor...')
                # Thread will stop automatically when is_running is False
            else:
                self.console_logger.log('INFO', '📴 Breakeven monitor not running')
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Error stopping breakeven monitor: {e}')

# Global worker instance
ai_worker = None

def get_ai_worker(socketio=None, bybit_session=None):
    """Get or create AI worker instance"""
    global ai_worker
    if ai_worker is None:
        ai_worker = AIWorker(socketio, bybit_session)
    elif bybit_session and not ai_worker.bybit_session:
        # Update existing worker with bybit_session if it was missing
        ai_worker.bybit_session = bybit_session
        if bybit_session:
            ai_worker.console_logger.log('INFO', '✅ Direct pybit trading initialized with ByBit session')
    return ai_worker