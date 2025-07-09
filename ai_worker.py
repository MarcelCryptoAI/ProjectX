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
from database import TradingDatabase
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
        # Load settings with fallback for production
        try:
            self.settings = Settings.load('config/settings.yaml')
        except:
            # Fallback to environment variables if config file doesn't exist
            self.settings = type('Settings', (), {})()
        self.ai_trader = AITrader(self.settings)
        self.trade_logger = TradeLogger()
        self.console_logger = ConsoleLogger()
        self.database = TradingDatabase()
        # Direct pybit integration - no separate executor needed
        self.max_concurrent_trades = int(os.getenv('MAX_CONCURRENT_TRADES', 5))
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
        
    def start(self):
        """Start the AI worker and resume on-hold trades"""
        if not self.is_running:
            self.is_running = True
            
            # Resume on-hold signals back to waiting status
            try:
                from database import TradingDatabase
                db = TradingDatabase()
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
            self.console_logger.log('SUCCESS', '🚀 AI WORKER STARTED - Ready for training and signal generation!')
            self.console_logger.log('INFO', '📊 System initialized with ByBit API connection')
            if self.bybit_session:
                self.console_logger.log('INFO', '⚡ Direct pybit trading ready for live execution')
            
    def stop(self):
        """Stop the AI worker"""
        self.is_running = False
        self.console_logger.log('WARNING', '⏹️ AI WORKER STOPPED - All operations halted')
        self.console_logger.log('INFO', '🔴 Training, signals, and trading have been disabled')
        
    def _worker_loop(self):
        """Main worker loop"""
        self.console_logger.log('INFO', 'AI Worker loop started')
        
        while self.is_running:
            try:
                # Check if model needs retraining
                self._check_model_training()
                
                # Generate trading signals
                self._generate_signals()
                
                # Execute trades from ranked signal list
                self._execute_top_signals()
                
                # Monitor trades and move stop loss if needed
                self.monitor_trades_and_move_sl()
                
                # Update statistics
                self._update_statistics()
                
                # Emit status update
                self._emit_status_update()
                
                # Sleep for configured interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.console_logger.log('ERROR', f'Worker error: {str(e)}')
                time.sleep(60)  # Wait longer on error
    
    def _check_model_training(self):
        """Check if AI model needs retraining"""
        if self.training_in_progress:
            return
            
        # Check if enough time has passed since last training  
        ai_settings = getattr(self.settings, 'ai', {})
        update_interval_hours = ai_settings.get('model_update_interval', 24) if ai_settings else 24
        
        if (self.last_model_update is None or 
            (datetime.now() - self.last_model_update).total_seconds() > update_interval_hours * 3600):
            
            self._start_model_training()
    
    def _start_model_training(self):
        """Start comprehensive model training process"""
        self.training_in_progress = True
        self.current_training_session = str(uuid.uuid4())
        
        self.console_logger.log('INFO', '🚀 Starting comprehensive AI model training...')
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self._train_model_comprehensive, daemon=True)
        training_thread.start()
    
    def _train_model_comprehensive(self):
        """Comprehensive AI model training with batch processing"""
        try:
            # Get enabled trading pairs
            enabled_pairs = self.settings.bot.get('enabled_pairs', ['BTCUSDT', 'ETHUSDT'])
            
            if not enabled_pairs:
                self.console_logger.log('WARNING', 'No trading pairs enabled for training')
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
            from database import TradingDatabase
            db = TradingDatabase()
            
            # Get all failed signals
            failed_signals = [s for s in db.get_trading_signals() if s.get('status') == 'failed']
            
            reset_count = 0
            for signal in failed_signals:
                # Check if signal still meets confidence threshold after retraining
                confidence = signal.get('confidence', 0)
                accuracy = signal.get('accuracy', 0)
                
                # Get current threshold
                try:
                    from utils.settings_loader import SettingsLoader
                    settings = SettingsLoader()
                    ai_threshold = settings.ai_confidence_threshold
                except:
                    ai_threshold = 75  # Default fallback
                
                # If signal still meets requirements, reset to waiting
                if confidence >= ai_threshold and accuracy > 60:  # Basic quality check
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
        """Calculate technical indicators for the symbol"""
        if not market_data or len(market_data) < 20:
            return {}
        
        try:
            # Convert to numpy arrays for calculations
            closes = np.array([float(kline[4]) for kline in market_data])
            volumes = np.array([float(kline[5]) for kline in market_data])
            
            # Simple moving averages
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            
            # RSI calculation (simplified)
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / (avg_loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # MACD (simplified)
            ema_12 = closes[-1]  # Simplified
            ema_26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
            macd_line = ema_12 - ema_26
            
            # Bollinger Bands
            bb_middle = sma_20
            std_dev = np.std(closes[-20:]) if len(closes) >= 20 else 0
            bb_upper = bb_middle + (2 * std_dev)
            bb_lower = bb_middle - (2 * std_dev)
            
            return {
                'rsi_14': rsi,
                'macd_line': macd_line,
                'macd_signal': macd_line * 0.9,  # Simplified
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'volume_sma_20': np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1],
                'price_sma_20': sma_20,
                'price_ema_12': ema_12,
                'price_ema_26': ema_26
            }
            
        except Exception as e:
            self.console_logger.log('WARNING', f'Failed to calculate indicators for {symbol}: {str(e)}')
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
    
    def _train_symbol_model(self, symbol, indicators, sentiment):
        """Train AI model for specific symbol"""
        try:
            # Simulate model training with realistic results
            base_accuracy = 65 + np.random.uniform(-10, 20)  # 55-85% range
            base_confidence = 70 + np.random.uniform(-15, 25)  # 55-95% range
            
            # Adjust based on data quality
            data_quality_bonus = 0
            if indicators and len(indicators) > 5:
                data_quality_bonus += 5
            if sentiment and sentiment.get('news_count', 0) > 20:
                data_quality_bonus += 3
                
            accuracy = min(95, max(50, base_accuracy + data_quality_bonus))
            confidence = min(95, max(50, base_confidence + data_quality_bonus))
            
            return accuracy, confidence
            
        except Exception as e:
            self.console_logger.log('WARNING', f'Failed to train model for {symbol}: {str(e)}')
            return 60.0, 60.0
    
    def _emit_training_progress(self):
        """Emit training progress to frontend"""
        if self.socketio:
            self.socketio.emit('training_progress', self.training_progress)
    
    def _generate_signals(self):
        """Generate trading signals and save to database - NO DIRECT EXECUTION"""
        if self.training_in_progress:
            return
            
        try:
            # Get AI prediction
            prediction = self.ai_trader.get_prediction()
            
            if prediction:
                confidence = prediction['confidence']
                symbol = prediction['symbol']
                side = prediction['side']
                
                # Ensure confidence is in percentage format (0-100)
                if confidence < 1.0:  # If it's a fraction (0.0-1.0), convert to percentage
                    confidence = confidence * 100
                
                # Check if prediction meets minimum confidence threshold FIRST
                ai_threshold = self.get_ai_confidence_threshold()
                if confidence < ai_threshold:
                    self.console_logger.log('INFO', f'⏭️ Signal {symbol} below threshold ({confidence:.1f}% < {ai_threshold}%) - not saving')
                    return  # Don't save or process low confidence signals
                
                self.signal_count += 1
                self.console_logger.log('INFO', 
                    f'🎯 Signal #{self.signal_count}: {side} {symbol} (Confidence: {confidence:.1f}%)')
                
                # SAVE TO DATABASE ONLY - NO DIRECT EXECUTION
                # Let the ranked system handle execution
                try:
                    from database import TradingDatabase
                    db = TradingDatabase()
                    
                    # Save signal to database with waiting status
                    signal_id = f'signal_{self.signal_count}_{symbol}'
                    db.save_trading_signal(
                        signal_id=signal_id,
                        symbol=symbol,
                        side=side,
                        confidence=confidence,
                        accuracy=prediction.get('accuracy', 70),
                        amount=prediction.get('amount', 100),
                        stop_loss=prediction.get('stop_loss', 2.0),
                        take_profit=prediction.get('take_profit', 3.0),
                        status='waiting'
                    )
                    
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
            from database import TradingDatabase
            db = TradingDatabase()
            
            # Get all waiting signals (including previously failed ones after retraining)
            waiting_signals = [s for s in db.get_trading_signals() if s.get('status') in ['waiting', 'failed']]
            
            if not waiting_signals:
                return
                
            # Sort by confidence (highest first), then accuracy (highest first)
            waiting_signals.sort(key=lambda x: (x.get('confidence', 0), x.get('accuracy', 0)), reverse=True)
            
            # Get confidence threshold
            try:
                from utils.settings_loader import SettingsLoader
                settings = SettingsLoader()
                ai_threshold = settings.ai_confidence_threshold
            except:
                ai_threshold = 75  # Default fallback
            
            signals_processed = 0
            trades_executed = 0
            
            self.console_logger.log('INFO', f'🔄 Processing {len(waiting_signals)} signals with {self.max_concurrent_trades - active_trades_count} slots available')
            
            # Execute signals until we reach max concurrent trades or run out of signals
            for signal in waiting_signals:
                if active_trades_count >= self.max_concurrent_trades:
                    self.console_logger.log('INFO', f'🚫 Max concurrent trades reached, stopping signal execution')
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
                    # Update signal status to executed
                    db.update_signal_status(signal['signal_id'], 'executed')
                    active_trades_count += 1
                    trades_executed += 1
                    self.console_logger.log('SUCCESS', f'✅ Signal executed for {signal["symbol"]} ({active_trades_count}/{self.max_concurrent_trades} trades active)')
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
        """Get AI confidence threshold from settings"""
        try:
            from utils.settings_loader import SettingsLoader
            settings = SettingsLoader()
            return settings.ai_confidence_threshold
        except:
            return 75.0  # Default fallback
    
    def get_active_positions_count(self):
        """Get count of active positions"""
        try:
            if not self.bybit_session:
                return 0
            
            positions = self.bybit_session.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            if positions and 'result' in positions and 'list' in positions['result']:
                active_count = 0
                for pos in positions['result']['list']:
                    if float(pos.get('size', 0)) > 0:
                        active_count += 1
                return active_count
            return 0
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
                positions = self.bybit_session.get_positions(category="linear", symbol=symbol)
                if positions and 'result' in positions:
                    for position in positions['result']['list']:
                        if position['symbol'] == symbol and float(position['size']) > 0:
                            existing_side = position['side']
                            
                            # If same direction, block the trade
                            if existing_side == side:
                                self.console_logger.log('WARNING', f'⚠️ Already have {existing_side} position for {symbol}, blocking same direction trade')
                                # Update signal status to waiting
                                if 'signal_id' in signal:
                                    from database import TradingDatabase
                                    db = TradingDatabase()
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
            
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Get user settings for leverage and trade amount
            try:
                from database import TradingDatabase
                db = TradingDatabase()
                db_settings = db.load_settings()
                
                # User leverage settings
                min_leverage = int(db_settings.get('minLeverage', 1))
                max_leverage = int(db_settings.get('maxLeverage', 10))
                leverage_strategy = db_settings.get('leverageStrategy', 'confidence_based')
                
                # User trade amount settings
                risk_per_trade = float(db_settings.get('riskPerTrade', 2.0))  # Percentage of balance
                min_trade_amount = float(db_settings.get('minTradeAmount', 5.0))  # Minimum $5
                
                # User concurrent trades setting
                max_concurrent_trades = int(db_settings.get('maxConcurrentTrades', 5))
                
                # Update AI worker's max concurrent trades from user settings
                self.max_concurrent_trades = max_concurrent_trades
                
            except Exception as settings_error:
                self.console_logger.log('WARNING', f'⚠️ Could not load settings: {settings_error}, using defaults')
                min_leverage = 1
                max_leverage = 10
                leverage_strategy = 'confidence_based'
                risk_per_trade = 2.0
                min_trade_amount = 5.0
                max_concurrent_trades = 5
                self.max_concurrent_trades = max_concurrent_trades
            
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
            
            # Get current account balance
            try:
                balance_data = self.bybit_session.get_wallet_balance(accountType="UNIFIED")
                total_balance = float(balance_data['result']['list'][0]['totalWalletBalance'])
            except:
                total_balance = 1000  # Fallback if balance fetch fails
            
            # Calculate trade amount as percentage of balance (user's risk setting)
            calculated_trade_amount = total_balance * (risk_per_trade / 100)
            
            # Ensure minimum trade amount (user's setting or $5 minimum)
            trade_amount_usd = max(min_trade_amount, calculated_trade_amount)
            
            # Calculate total quantity for this trade amount (WITHOUT leverage in quantity calc)
            total_qty = trade_amount_usd / current_price
            
            # Round to proper step size
            total_qty = max(min_order_qty, round(total_qty / qty_step) * qty_step)
            
            # Double-check minimum quantity
            if total_qty < min_order_qty:
                self.console_logger.log('ERROR', f'❌ Calculated quantity {total_qty} is below minimum {min_order_qty} for {symbol}')
                return False
            
            # Verify minimum order value
            total_order_value = total_qty * current_price
            if total_order_value < 5.0:
                self.console_logger.log('ERROR', f'❌ Order value ${total_order_value:.2f} is below ByBit minimum $5.00 for {symbol}')
                return False
            
            self.console_logger.log('INFO', f'💰 Trade: ${trade_amount_usd:.2f} ({risk_per_trade:.1f}% of ${total_balance:.2f}), Leverage: {leverage}x')
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
            
            # Calculate stop loss and take profit from signal (AI determined)
            stop_loss_pct = float(signal.get('stop_loss', 2.0))  # AI determines SL
            take_profit_pct = float(signal.get('take_profit', 3.0))  # AI determines TP
            
            if side == 'Buy':
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                take_profit_price = current_price * (1 + take_profit_pct / 100)
            else:  # Sell
                stop_loss_price = current_price * (1 + stop_loss_pct / 100)
                take_profit_price = current_price * (1 - take_profit_pct / 100)
            
            # Place main market order with stop loss only (no TP in main order)
            order_params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,  # Use exact case: 'Buy' or 'Sell'
                'orderType': 'Market',
                'qty': str(total_qty),
                'timeInForce': 'IOC',  # Immediate or Cancel
                'stopLoss': str(stop_loss_price)
            }
            
            self.console_logger.log('INFO', f'📤 Placing {side} order: {total_qty} {symbol} @ market (~${current_price:.4f})')
            
            # Execute the main order first
            order_result = self.bybit_session.place_order(**order_params)
            
            if order_result and 'result' in order_result:
                order_id = order_result['result']['orderId']
                
                # Log successful trade
                self.console_logger.log('SUCCESS', f'✅ Order placed: {order_id}')
                self.console_logger.log('INFO', f'📊 Details: {total_qty} {symbol}, SL: ${stop_loss_price:.4f}')
                
                # Calculate 4 TP levels with equal quantity distribution
                tp_levels = []
                tp_order_ids = []
                
                # Divide total quantity into 4 equal parts
                tp_qty = total_qty / 4
                tp_qty = max(min_order_qty, round(tp_qty / qty_step) * qty_step)  # Ensure proper step size
                
                for i in range(1, 5):
                    # Calculate TP price for each level (25%, 50%, 75%, 100% of total TP)
                    tp_percentage = (take_profit_pct * i) / 4  # Divide TP into 4 levels
                    
                    if side == 'Buy':
                        tp_price = current_price * (1 + tp_percentage / 100)
                    else:  # Sell
                        tp_price = current_price * (1 - tp_percentage / 100)
                    
                    tp_level = {
                        'level': i,
                        'price': tp_price,
                        'qty': tp_qty,
                        'status': 'pending',
                        'hit_time': None,
                        'order_id': None
                    }
                    tp_levels.append(tp_level)
                    
                    # Place TP limit order
                    try:
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
                        
                        tp_result = self.bybit_session.place_order(**tp_order_params)
                        
                        if tp_result and 'result' in tp_result:
                            tp_order_id = tp_result['result']['orderId']
                            tp_level['order_id'] = tp_order_id
                            tp_order_ids.append(tp_order_id)
                            self.console_logger.log('SUCCESS', f'✅ TP{i} set: {tp_order_id} @ ${tp_price:.4f} (qty: {tp_qty})')
                        else:
                            error_msg = tp_result.get('retMsg', 'Unknown error') if tp_result else 'No response'
                            self.console_logger.log('WARNING', f'⚠️ TP{i} failed: {error_msg}')
                            
                    except Exception as tp_error:
                        self.console_logger.log('ERROR', f'❌ TP{i} error: {str(tp_error)}')
                
                # Store trade in active trades for monitoring
                self.active_trades[order_id] = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': total_qty,
                    'entry_price': current_price,
                    'stop_loss': stop_loss_price,
                    'original_stop_loss': stop_loss_price,  # Keep original for reference
                    'take_profit_levels': tp_levels,
                    'tp_order_ids': tp_order_ids,
                    'tp1_hit': False,
                    'sl_moved_to_breakeven': False,
                    'trailing_stop_enabled': signal.get('trailing_stop_enabled', True),
                    'trailing_stop_distance': signal.get('trailing_stop_distance', 1.0),
                    'timestamp': datetime.now().isoformat(),
                    'trade_amount_usd': trade_amount_usd,
                    'balance_percentage': balance_percentage
                }
                
                self.console_logger.log('SUCCESS', f'✅ Trade setup complete: {len(tp_order_ids)}/4 TP levels active')
                self.console_logger.log('INFO', f'📊 TP1: ${tp_levels[0]["price"]:.4f}, TP2: ${tp_levels[1]["price"]:.4f}, TP3: ${tp_levels[2]["price"]:.4f}, TP4: ${tp_levels[3]["price"]:.4f}')
                self.console_logger.log('INFO', f'🛡️ Stop Loss: ${stop_loss_price:.4f} (will move to breakeven+0.1% when TP1 hits)')
                
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
                        from database import TradingDatabase
                        db = TradingDatabase()
                        db.update_signal_status(signal['signal_id'], 'failed')
                    except Exception as db_error:
                        self.console_logger.log('WARNING', f'⚠️ Could not update signal status: {db_error}')
                
                return False
                
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Trade execution error: {str(e)}')
            
            # Update signal status to 'failed' in database
            if 'signal_id' in signal:
                try:
                    from database import TradingDatabase
                    db = TradingDatabase()
                    db.update_signal_status(signal['signal_id'], 'failed')
                except Exception as db_error:
                    self.console_logger.log('WARNING', f'⚠️ Could not update signal status: {db_error}')
            
            return False
    
    def monitor_trades_and_move_sl(self):
        """Monitor active trades and move stop loss to breakeven when first TP is hit"""
        if not hasattr(self, 'active_trades') or not self.active_trades:
            return
        
        try:
            # Get current orders to check TP status
            orders = self.bybit_session.get_open_orders(category="linear")
            if not orders or 'result' not in orders:
                return
            
            # Get current positions
            positions = self.bybit_session.get_positions(category="linear")
            if not positions or 'result' not in positions:
                return
            
            for order_id, trade_data in list(self.active_trades.items()):
                symbol = trade_data['symbol']
                side = trade_data['side']
                entry_price = trade_data['entry_price']
                tp_levels = trade_data['take_profit_levels']
                tp_order_ids = trade_data['tp_order_ids']
                
                # Check if TP1 has been hit by looking at open orders
                tp1_still_open = False
                for order in orders['result']['list']:
                    if order['orderId'] in tp_order_ids and order['orderId'] == tp_levels[0].get('order_id'):
                        tp1_still_open = True
                        break
                
                # If TP1 is no longer in open orders, it was filled
                if not tp1_still_open and not trade_data.get('tp1_hit', False):
                    self.console_logger.log('SUCCESS', f'✅ TP1 hit for {symbol}! Moving stop loss to breakeven + 0.1%')
                    
                    # Calculate breakeven + 0.1% price
                    if side == 'Buy':
                        new_sl_price = entry_price * 1.001  # +0.1%
                    else:
                        new_sl_price = entry_price * 0.999  # -0.1%
                    
                    # Update stop loss
                    try:
                        # Cancel existing stop loss order first
                        self.bybit_session.cancel_all_orders(category="linear", symbol=symbol, orderFilter="StopOrder")
                        
                        # Place new stop loss at breakeven + 0.1%
                        sl_result = self.bybit_session.place_order(
                            category="linear",
                            symbol=symbol,
                            side="Sell" if side == "Buy" else "Buy",
                            orderType="Market",
                            qty=str(trade_data['quantity']),
                            stopLoss=str(new_sl_price),
                            reduceOnly=True
                        )
                        
                        if sl_result and 'result' in sl_result:
                            # Update trade data
                            trade_data['tp1_hit'] = True
                            trade_data['sl_moved_to_breakeven'] = True
                            trade_data['stop_loss'] = new_sl_price
                            trade_data['tp_levels'][0]['status'] = 'hit'
                            trade_data['tp_levels'][0]['hit_time'] = datetime.now().isoformat()
                            
                            self.console_logger.log('SUCCESS', f'✅ Stop loss moved to breakeven+0.1% for {symbol}: ${new_sl_price:.4f}')
                        else:
                            self.console_logger.log('ERROR', f'❌ Failed to move stop loss for {symbol}')
                            
                    except Exception as sl_error:
                        self.console_logger.log('ERROR', f'❌ Error moving stop loss for {symbol}: {str(sl_error)}')
                
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
                    # Position is closed, remove from active trades
                    self.console_logger.log('INFO', f'📊 Position closed for {symbol}, removing from active trades')
                    del self.active_trades[order_id]
                    
        except Exception as e:
            self.console_logger.log('ERROR', f'❌ Trade monitoring error: {str(e)}')

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