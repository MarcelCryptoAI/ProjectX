import sqlite3
import json
import pandas as pd
from datetime import datetime
import os
from urllib.parse import urlparse

class TradingDatabase:
    def __init__(self, db_path='trading_data.db'):
        # Check if we're in production with PostgreSQL
        self.database_url = os.getenv('DATABASE_URL')
        self.use_postgres = bool(self.database_url)
        
        if self.use_postgres:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor
            # Parse the DATABASE_URL
            result = urlparse(self.database_url)
            self.db_config = {
                'host': result.hostname,
                'port': result.port,
                'database': result.path[1:],
                'user': result.username,
                'password': result.password
            }
        else:
            self.db_path = db_path
            
        self.init_database()
    
    def get_connection(self):
        """Get database connection based on environment"""
        if self.use_postgres:
            return self.psycopg2.connect(**self.db_config)
        else:
            return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Market data table for storing OHLCV data
        if self.use_postgres:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    open_price DECIMAL(20,8),
                    high_price DECIMAL(20,8),
                    low_price DECIMAL(20,8),
                    close_price DECIMAL(20,8),
                    volume DECIMAL(20,8),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
        
        # Technical indicators table
        if self.use_postgres:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    rsi_14 DECIMAL(10,4),
                    macd_line DECIMAL(20,8),
                    macd_signal DECIMAL(20,8),
                    bb_upper DECIMAL(20,8),
                    bb_lower DECIMAL(20,8),
                    bb_middle DECIMAL(20,8),
                    volume_sma_20 DECIMAL(20,8),
                    price_sma_20 DECIMAL(20,8),
                    price_ema_12 DECIMAL(20,8),
                    price_ema_26 DECIMAL(20,8),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    rsi_14 REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    bb_middle REAL,
                    volume_sma_20 REAL,
                    price_sma_20 REAL,
                    price_ema_12 REAL,
                    price_ema_26 REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
        
        # Sentiment data table
        if self.use_postgres:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    sentiment_score DECIMAL(10,4),
                    news_count INTEGER,
                    social_sentiment DECIMAL(10,4),
                    fear_greed_index DECIMAL(10,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    sentiment_score REAL,
                    news_count INTEGER,
                    social_sentiment REAL,
                    fear_greed_index REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
        
        # AI training results table
        if self.use_postgres:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_results (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    features TEXT,
                    accuracy DECIMAL(10,4),
                    confidence DECIMAL(10,4),
                    model_params TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    features TEXT,
                    accuracy REAL,
                    confidence REAL,
                    model_params TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # Training sessions table
        if self.use_postgres:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) UNIQUE NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_symbols INTEGER,
                    completed_symbols INTEGER,
                    overall_accuracy DECIMAL(10,4),
                    status VARCHAR(20) DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_symbols INTEGER,
                    completed_symbols INTEGER,
                    overall_accuracy REAL,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # Settings table for storing application settings
        if self.use_postgres:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    id SERIAL PRIMARY KEY,
                    key VARCHAR(100) UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # Trading signals table for tracking signal statuses
        if self.use_postgres:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id SERIAL PRIMARY KEY,
                    signal_id VARCHAR(100) UNIQUE NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    confidence DECIMAL(5,2),
                    accuracy DECIMAL(5,2),
                    amount DECIMAL(20,8),
                    leverage INTEGER,
                    take_profit DECIMAL(5,2),
                    stop_loss DECIMAL(5,2),
                    status VARCHAR(20) DEFAULT 'waiting',
                    order_id VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    confidence REAL,
                    accuracy REAL,
                    amount REAL,
                    leverage INTEGER,
                    take_profit REAL,
                    stop_loss REAL,
                    status TEXT DEFAULT 'waiting',
                    order_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # Supported symbols table for coin list management
        if self.use_postgres:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS supported_symbols (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) UNIQUE NOT NULL,
                    base_currency VARCHAR(10),
                    quote_currency VARCHAR(10),
                    status VARCHAR(20) DEFAULT 'active',
                    min_order_qty DECIMAL(20,8),
                    qty_step DECIMAL(20,8),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS supported_symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    base_currency TEXT,
                    quote_currency TEXT,
                    status TEXT DEFAULT 'active',
                    min_order_qty REAL,
                    qty_step REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        conn.commit()
        conn.close()
    
    def save_settings(self, settings_dict):
        """Save settings to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for key, value in settings_dict.items():
            try:
                if self.use_postgres:
                    cursor.execute('''
                        INSERT INTO settings (key, value, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    ''', (key, json.dumps(value)))
                else:
                    cursor.execute('''
                        INSERT OR REPLACE INTO settings (key, value, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    ''', (key, json.dumps(value)))
            except Exception as e:
                print(f"Error saving setting {key}: {e}")
        
        conn.commit()
        conn.close()
    
    def load_settings(self):
        """Load settings from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgres:
                cursor.execute('SELECT key, value FROM settings')
                results = cursor.fetchall()
            else:
                cursor.execute('SELECT key, value FROM settings')
                results = cursor.fetchall()
            
            settings = {}
            for row in results:
                key = row[0]
                value = json.loads(row[1])
                settings[key] = value
            
            conn.close()
            return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            conn.close()
            return {}
    
    def store_market_data(self, symbol, kline_data):
        """Store market data for a symbol"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for kline in kline_data:
            try:
                if self.use_postgres:
                    cursor.execute('''
                        INSERT INTO market_data 
                        (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timestamp) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume
                    ''', (
                        symbol,
                        int(kline[0]),  # timestamp
                        float(kline[1]),  # open
                        float(kline[2]),  # high
                        float(kline[3]),  # low
                        float(kline[4]),  # close
                        float(kline[5])   # volume
                    ))
                else:
                    cursor.execute('''
                        INSERT OR REPLACE INTO market_data 
                        (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        int(kline[0]),  # timestamp
                        float(kline[1]),  # open
                        float(kline[2]),  # high
                        float(kline[3]),  # low
                        float(kline[4]),  # close
                        float(kline[5])   # volume
                    ))
            except Exception as e:
                print(f"Error storing market data for {symbol}: {e}")
        
        conn.commit()
        conn.close()
    
    def store_technical_indicators(self, symbol, timestamp, indicators):
        """Store calculated technical indicators"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgres:
                cursor.execute('''
                    INSERT INTO technical_indicators 
                    (symbol, timestamp, rsi_14, macd_line, macd_signal, bb_upper, bb_lower, bb_middle,
                     volume_sma_20, price_sma_20, price_ema_12, price_ema_26)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    rsi_14 = EXCLUDED.rsi_14,
                    macd_line = EXCLUDED.macd_line,
                    macd_signal = EXCLUDED.macd_signal,
                    bb_upper = EXCLUDED.bb_upper,
                    bb_lower = EXCLUDED.bb_lower,
                    bb_middle = EXCLUDED.bb_middle,
                    volume_sma_20 = EXCLUDED.volume_sma_20,
                    price_sma_20 = EXCLUDED.price_sma_20,
                    price_ema_12 = EXCLUDED.price_ema_12,
                    price_ema_26 = EXCLUDED.price_ema_26
                ''', (
                symbol, timestamp,
                indicators.get('rsi_14'),
                indicators.get('macd_line'),
                indicators.get('macd_signal'),
                indicators.get('bb_upper'),
                indicators.get('bb_lower'),
                indicators.get('bb_middle'),
                indicators.get('volume_sma_20'),
                indicators.get('price_sma_20'),
                indicators.get('price_ema_12'),
                indicators.get('price_ema_26')
            ))
            else:
                cursor.execute('''
                    INSERT OR REPLACE INTO technical_indicators 
                    (symbol, timestamp, rsi_14, macd_line, macd_signal, bb_upper, bb_lower, bb_middle,
                     volume_sma_20, price_sma_20, price_ema_12, price_ema_26)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                symbol, timestamp,
                indicators.get('rsi_14'),
                indicators.get('macd_line'),
                indicators.get('macd_signal'),
                indicators.get('bb_upper'),
                indicators.get('bb_lower'),
                indicators.get('bb_middle'),
                indicators.get('volume_sma_20'),
                indicators.get('price_sma_20'),
                indicators.get('price_ema_12'),
                indicators.get('price_ema_26')
            ))
        except Exception as e:
            print(f"Error storing indicators for {symbol}: {e}")
        
        conn.commit()
        conn.close()
    
    def store_sentiment_data(self, symbol, timestamp, sentiment):
        """Store sentiment analysis data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.use_postgres:
                cursor.execute('''
                    INSERT INTO sentiment_data 
                    (symbol, timestamp, sentiment_score, news_count, social_sentiment, fear_greed_index)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    sentiment_score = EXCLUDED.sentiment_score,
                    news_count = EXCLUDED.news_count,
                    social_sentiment = EXCLUDED.social_sentiment,
                    fear_greed_index = EXCLUDED.fear_greed_index
                ''', (
                symbol, timestamp,
                sentiment.get('sentiment_score'),
                sentiment.get('news_count'),
                sentiment.get('social_sentiment'),
                sentiment.get('fear_greed_index')
            ))
            else:
                cursor.execute('''
                    INSERT OR REPLACE INTO sentiment_data 
                    (symbol, timestamp, sentiment_score, news_count, social_sentiment, fear_greed_index)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                symbol, timestamp,
                sentiment.get('sentiment_score'),
                sentiment.get('news_count'),
                sentiment.get('social_sentiment'),
                sentiment.get('fear_greed_index')
            ))
        except Exception as e:
            print(f"Error storing sentiment for {symbol}: {e}")
        
        conn.commit()
        conn.close()
    
    def store_training_results(self, session_id, symbol, features, accuracy, confidence, model_params):
        """Store AI training results"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            placeholder = '%s' if self.use_postgres else '?'
            cursor.execute(f'''
                INSERT INTO training_results 
                (session_id, symbol, features, accuracy, 
                 confidence, model_params)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            ''', (
                session_id, symbol,
                json.dumps(features),
                accuracy, confidence,
                json.dumps(model_params)
            ))
        except Exception as e:
            print(f"Error storing training results for {symbol}: {e}")
        
        conn.commit()
        conn.close()
    
    def create_training_session(self, session_id, total_symbols):
        """Create a new training session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if self.use_postgres:
            cursor.execute('''
                INSERT INTO training_sessions 
                (session_id, start_time, total_symbols, completed_symbols, status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                start_time = EXCLUDED.start_time,
                total_symbols = EXCLUDED.total_symbols,
                completed_symbols = EXCLUDED.completed_symbols,
                status = EXCLUDED.status
            ''', (session_id, datetime.now(), total_symbols, 0, 'running'))
        else:
            cursor.execute('''
                INSERT OR REPLACE INTO training_sessions 
                (session_id, start_time, total_symbols, completed_symbols, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, datetime.now(), total_symbols, 0, 'running'))
        
        conn.commit()
        conn.close()
    
    def update_training_session(self, session_id, completed_symbols, overall_accuracy=None, status=None):
        """Update training session progress"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        placeholder = '%s' if self.use_postgres else '?'
        if status == 'completed':
            cursor.execute(f'''
                UPDATE training_sessions 
                SET completed_symbols = {placeholder}, overall_accuracy = {placeholder}, status = {placeholder}, end_time = {placeholder}
                WHERE session_id = {placeholder}
            ''', (completed_symbols, overall_accuracy, status, datetime.now(), session_id))
        else:
            cursor.execute(f'''
                UPDATE training_sessions 
                SET completed_symbols = {placeholder}
                WHERE session_id = {placeholder}
            ''', (completed_symbols, session_id))
        
        conn.commit()
        conn.close()
    
    def get_training_history(self, limit=10):
        """Get recent training sessions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM training_sessions 
            ORDER BY created_at DESC 
            LIMIT %s
        ''' if self.use_postgres else '''
            SELECT * FROM training_sessions 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        sessions = cursor.fetchall()
        conn.close()
        
        return sessions
    
    def get_market_data(self, symbol, limit=100):
        """Get market data for a symbol"""
        conn = self.get_connection()
        
        placeholder = '%s' if self.use_postgres else '?'
        df = pd.read_sql_query(f'''
            SELECT * FROM market_data 
            WHERE symbol = {placeholder}
            ORDER BY timestamp DESC 
            LIMIT {placeholder}
        ''', conn, params=(symbol, limit))
        
        conn.close()
        return df
    
    def get_features_for_training(self, symbol, limit=1000):
        """Get combined features for AI training"""
        conn = self.get_connection()
        
        placeholder = '%s' if self.use_postgres else '?'
        query = f'''
            SELECT 
                m.symbol, m.timestamp, m.close_price, m.volume,
                t.rsi_14, t.macd_line, t.macd_signal, t.bb_upper, t.bb_lower,
                t.price_sma_20, t.price_ema_12, t.price_ema_26,
                s.sentiment_score, s.social_sentiment, s.fear_greed_index
            FROM market_data m
            LEFT JOIN technical_indicators t ON m.symbol = t.symbol AND m.timestamp = t.timestamp
            LEFT JOIN sentiment_data s ON m.symbol = s.symbol AND m.timestamp = s.timestamp
            WHERE m.symbol = {placeholder}
            ORDER BY m.timestamp DESC
            LIMIT {placeholder}
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()
        
        return df
    
    def get_latest_training_session(self):
        """Get the most recent training session"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=self.RealDictCursor) if self.use_postgres else conn.cursor()
        
        cursor.execute('''
            SELECT session_id, total_symbols, completed_symbols, overall_accuracy, status, created_at
            FROM training_sessions
            ORDER BY created_at DESC
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            if self.use_postgres:
                # PostgreSQL with RealDictCursor returns dict-like objects
                return {
                    'session_id': result['session_id'],
                    'total_symbols': result['total_symbols'],
                    'completed_symbols': result['completed_symbols'],
                    'overall_accuracy': float(result['overall_accuracy']) if result['overall_accuracy'] else None,
                    'status': result['status'],
                    'created_at': result['created_at'].isoformat() if hasattr(result['created_at'], 'isoformat') else str(result['created_at'])
                }
            else:
                # SQLite returns tuples
                return {
                    'session_id': result[0],
                    'total_symbols': result[1],
                    'completed_symbols': result[2],
                    'overall_accuracy': result[3],
                    'status': result[4],
                    'created_at': result[5]
                }
        return None
    
    def get_training_results(self, session_id):
        """Get training results for a specific session"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=self.RealDictCursor) if self.use_postgres else conn.cursor()
        
        placeholder = '%s' if self.use_postgres else '?'
        cursor.execute(f'''
            SELECT symbol, accuracy, confidence, features, model_params, timestamp
            FROM training_results
            WHERE session_id = {placeholder}
            ORDER BY confidence DESC, accuracy DESC
        ''', (session_id,))
        
        results = []
        for row in cursor.fetchall():
            if self.use_postgres:
                # PostgreSQL with RealDictCursor returns dict-like objects
                results.append({
                    'symbol': row['symbol'],
                    'accuracy': float(row['accuracy']) if row['accuracy'] else 0.0,
                    'confidence': float(row['confidence']) if row['confidence'] else 0.0,
                    'features': json.loads(row['features']) if row['features'] else {},
                    'model_params': json.loads(row['model_params']) if row['model_params'] else {},
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                })
            else:
                # SQLite returns tuples
                results.append({
                    'symbol': row[0],
                    'accuracy': row[1],
                    'confidence': row[2],
                    'features': json.loads(row[3]) if row[3] else {},
                    'model_params': json.loads(row[4]) if row[4] else {},
                    'timestamp': row[5]
                })
        
        conn.close()
        return results
    
    def save_trading_signal(self, signal_data):
        """Save trading signal to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            placeholder = '%s' if self.use_postgres else '?'
            cursor.execute(f'''
                INSERT INTO trading_signals 
                (signal_id, symbol, side, confidence, accuracy, amount, leverage, take_profit, stop_loss, status)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                ON CONFLICT (signal_id) DO UPDATE SET
                status = EXCLUDED.status,
                updated_at = CURRENT_TIMESTAMP
            ''' if self.use_postgres else f'''
                INSERT OR REPLACE INTO trading_signals 
                (signal_id, symbol, side, confidence, accuracy, amount, leverage, take_profit, stop_loss, status)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            ''', (
                signal_data['signal_id'],
                signal_data['symbol'],
                signal_data['side'],
                signal_data['confidence'],
                signal_data['accuracy'],
                signal_data['amount'],
                signal_data['leverage'],
                signal_data['take_profit'],
                signal_data['stop_loss'],
                signal_data.get('status', 'waiting')
            ))
            
            conn.commit()
            
        except Exception as e:
            print(f"Error saving trading signal: {e}")
            conn.rollback()
        
        conn.close()
    
    def update_signal_status(self, signal_id, status, order_id=None):
        """Update signal status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            placeholder = '%s' if self.use_postgres else '?'
            if order_id:
                cursor.execute(f'''
                    UPDATE trading_signals 
                    SET status = {placeholder}, order_id = {placeholder}, updated_at = CURRENT_TIMESTAMP
                    WHERE signal_id = {placeholder}
                ''', (status, order_id, signal_id))
            else:
                cursor.execute(f'''
                    UPDATE trading_signals 
                    SET status = {placeholder}, updated_at = CURRENT_TIMESTAMP
                    WHERE signal_id = {placeholder}
                ''', (status, signal_id))
            
            conn.commit()
            
        except Exception as e:
            print(f"Error updating signal status: {e}")
            conn.rollback()
        
        conn.close()
    
    def get_trading_signals(self):
        """Get all trading signals"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT signal_id, symbol, side, confidence, accuracy, amount, leverage, 
                       take_profit, stop_loss, status, order_id, created_at, updated_at
                FROM trading_signals
                ORDER BY created_at DESC
            ''')
            
            results = cursor.fetchall()
            signals = []
            
            for row in results:
                signal = {
                    'signal_id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'confidence': float(row[3]) if row[3] else 0,
                    'accuracy': float(row[4]) if row[4] else 0,
                    'amount': float(row[5]) if row[5] else 0,
                    'leverage': int(row[6]) if row[6] else 1,
                    'take_profit': float(row[7]) if row[7] else 0,
                    'stop_loss': float(row[8]) if row[8] else 0,
                    'status': row[9],
                    'order_id': row[10],
                    'created_at': row[11],
                    'updated_at': row[12]
                }
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            print(f"Error getting trading signals: {e}")
            return []
        
        finally:
            conn.close()
    
    def delete_oldest_trading_signals(self, count):
        """Delete the oldest trading signals"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            placeholder = '%s' if self.use_postgres else '?'
            
            # Get the signal_ids of the oldest signals
            cursor.execute(f'''
                SELECT signal_id FROM trading_signals
                ORDER BY created_at ASC
                LIMIT {placeholder}
            ''', (count,))
            
            old_signals = cursor.fetchall()
            
            if old_signals:
                # Delete these signals
                signal_ids = [signal[0] for signal in old_signals]
                placeholders = ','.join([placeholder] * len(signal_ids))
                
                cursor.execute(f'''
                    DELETE FROM trading_signals
                    WHERE signal_id IN ({placeholders})
                ''', signal_ids)
                
                conn.commit()
                return len(signal_ids)
            else:
                return 0
            
        except Exception as e:
            print(f"Error deleting old signals: {e}")
            conn.rollback()
            return 0
        
        finally:
            conn.close()
    
    def refresh_supported_symbols(self, symbols_data):
        """Refresh supported symbols list"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Clear existing symbols
            cursor.execute('DELETE FROM supported_symbols')
            
            # Insert new symbols
            placeholder = '%s' if self.use_postgres else '?'
            for symbol_data in symbols_data:
                cursor.execute(f'''
                    INSERT INTO supported_symbols 
                    (symbol, base_currency, quote_currency, status, min_order_qty, qty_step)
                    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                ''', (
                    symbol_data['symbol'],
                    symbol_data.get('base_currency', ''),
                    symbol_data.get('quote_currency', ''),
                    symbol_data.get('status', 'active'),
                    symbol_data.get('min_order_qty', 0),
                    symbol_data.get('qty_step', 0)
                ))
            
            conn.commit()
            
        except Exception as e:
            print(f"Error refreshing supported symbols: {e}")
            conn.rollback()
        
        conn.close()
    
    def get_supported_symbols(self):
        """Get supported symbols list"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT symbol, status, last_updated FROM supported_symbols ORDER BY symbol')
            results = cursor.fetchall()
            
            symbols = []
            for row in results:
                symbols.append({
                    'symbol': row[0],
                    'status': row[1],
                    'last_updated': row[2]
                })
            
            return symbols
            
        except Exception as e:
            print(f"Error getting supported symbols: {e}")
            return []
        
        finally:
            conn.close()
    
    def get_symbols_last_updated(self):
        """Get last updated date for symbols"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT MAX(last_updated) FROM supported_symbols')
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
            
        except Exception as e:
            print(f"Error getting symbols last updated: {e}")
            return None
        
        finally:
            conn.close()