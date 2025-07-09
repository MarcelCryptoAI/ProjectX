#!/usr/bin/env python3
import json
from datetime import datetime
from database import TradingDatabase

def check_database_status():
    """Check the current database settings and trading activity"""
    print("=== CHECKING DATABASE STATUS ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Initialize database
    db = TradingDatabase()
    
    # 1. Check current settings
    print("1. CURRENT SETTINGS:")
    print("-" * 50)
    settings = db.load_settings()
    
    if not settings:
        print("No settings found in database!")
        return
    
    # Check autoExecute setting
    auto_execute = settings.get('autoExecute', False)
    print(f"Auto Execute: {auto_execute}")
    
    # Check leverage settings
    min_leverage = settings.get('minLeverage', 1)
    max_leverage = settings.get('maxLeverage', 5)
    leverage_strategy = settings.get('leverageStrategy', 'fixed')
    
    print(f"Min Leverage: {min_leverage}")
    print(f"Max Leverage: {max_leverage}")
    print(f"Leverage Strategy: {leverage_strategy}")
    
    # Check other important settings
    for key, value in settings.items():
        if key not in ['autoExecute', 'minLeverage', 'maxLeverage', 'leverageStrategy']:
            print(f"{key}: {value}")
    
    print()
    
    # 2. Check trading signals
    print("2. TRADING SIGNALS:")
    print("-" * 50)
    signals = db.get_trading_signals()
    
    if not signals:
        print("No trading signals found!")
    else:
        print(f"Total signals: {len(signals)}")
        
        # Group by status
        status_counts = {}
        for signal in signals:
            status = signal['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("Signal status counts:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        print()
        
        # Show recent signals (last 10)
        print("Recent signals (last 10):")
        for signal in signals[:10]:
            print(f"  {signal['created_at']} | {signal['symbol']} | {signal['side']} | {signal['status']} | Conf: {signal['confidence']:.1f}% | Lev: {signal['leverage']}x")
        
        # Check for BTC signals specifically
        btc_signals = [s for s in signals if 'BTC' in s['symbol']]
        if btc_signals:
            print(f"\nBTC signals found: {len(btc_signals)}")
            for signal in btc_signals[:5]:  # Show last 5 BTC signals
                print(f"  {signal['created_at']} | {signal['symbol']} | {signal['side']} | {signal['status']} | Amount: {signal['amount']}")
    
    print()
    
    # 3. Check latest training session
    print("3. LATEST TRAINING SESSION:")
    print("-" * 50)
    latest_session = db.get_latest_training_session()
    
    if latest_session:
        print(f"Session ID: {latest_session['session_id']}")
        print(f"Status: {latest_session['status']}")
        print(f"Progress: {latest_session['completed_symbols']}/{latest_session['total_symbols']}")
        print(f"Accuracy: {latest_session['overall_accuracy']}")
        print(f"Created: {latest_session['created_at']}")
    else:
        print("No training sessions found!")
    
    print()
    
    # 4. Check supported symbols
    print("4. SUPPORTED SYMBOLS:")
    print("-" * 50)
    symbols = db.get_supported_symbols()
    
    if symbols:
        print(f"Total supported symbols: {len(symbols)}")
        last_updated = db.get_symbols_last_updated()
        print(f"Last updated: {last_updated}")
        
        # Show some symbols
        active_symbols = [s for s in symbols if s['status'] == 'active']
        print(f"Active symbols: {len(active_symbols)}")
        
        # Show first 10 symbols
        print("First 10 symbols:")
        for symbol in symbols[:10]:
            print(f"  {symbol['symbol']} ({symbol['status']})")
    else:
        print("No supported symbols found!")
    
    print()
    print("=== END DATABASE STATUS CHECK ===")

if __name__ == "__main__":
    check_database_status()