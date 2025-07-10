#!/usr/bin/env python3
"""
Debug production database issues - works in both local and production environments
"""
import os
import sys
from datetime import datetime

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def debug_production_database():
    """Debug database issues in production"""
    try:
        print("=" * 80)
        print("PRODUCTION DATABASE DEBUG")
        print("=" * 80)
        
        # Initialize database
        db = TradingDatabase()
        print(f"✓ Database initialized")
        print(f"  Using PostgreSQL: {db.use_postgres}")
        
        if db.use_postgres:
            print(f"  Database URL: {db.database_url[:50]}...")
        else:
            print("  Using SQLite locally")
            print("  NOTE: This is LOCAL database, not production!")
        
        # Get database connection for direct queries
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Check if trading_signals table exists
        print("\n" + "=" * 80)
        print("TABLE EXISTENCE CHECK")
        print("=" * 80)
        
        try:
            if db.use_postgres:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'trading_signals'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                print(f"✓ trading_signals table exists: {table_exists}")
                
                if table_exists:
                    # Get table structure
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable 
                        FROM information_schema.columns 
                        WHERE table_name = 'trading_signals' 
                        ORDER BY ordinal_position
                    """)
                    columns = cursor.fetchall()
                    print(f"✓ Table has {len(columns)} columns")
                    
                    # Check for required columns
                    column_names = [col[0] for col in columns]
                    required_columns = ['signal_id', 'status', 'realized_pnl', 'entry_price', 'exit_price']
                    missing_columns = [col for col in required_columns if col not in column_names]
                    
                    if missing_columns:
                        print(f"❌ Missing required columns: {missing_columns}")
                    else:
                        print("✓ All required columns present")
                
            else:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trading_signals'")
                table_exists = cursor.fetchone() is not None
                print(f"✓ trading_signals table exists: {table_exists}")
        
        except Exception as e:
            print(f"❌ Error checking table existence: {e}")
            table_exists = False
        
        if not table_exists:
            print("❌ CRITICAL: trading_signals table doesn't exist!")
            print("   This would cause 0 trades to be shown in analytics")
            conn.close()
            return False
        
        # Check total row count
        print("\n" + "=" * 80)
        print("ROW COUNT CHECK")
        print("=" * 80)
        
        try:
            cursor.execute("SELECT COUNT(*) FROM trading_signals")
            total_count = cursor.fetchone()[0]
            print(f"✓ Total signals in database: {total_count}")
            
            if total_count == 0:
                print("❌ CRITICAL: No signals in database!")
                print("   This explains why analytics shows 0 trades")
                conn.close()
                return False
            
            # Count by status
            cursor.execute("SELECT status, COUNT(*) FROM trading_signals GROUP BY status")
            status_counts = cursor.fetchall()
            
            print("\nStatus breakdown:")
            for status, count in status_counts:
                print(f"  {status}: {count} signals")
            
            # Count completed with P&L
            cursor.execute("""
                SELECT COUNT(*) FROM trading_signals 
                WHERE status = 'completed' AND realized_pnl IS NOT NULL
            """)
            completed_with_pnl = cursor.fetchone()[0]
            print(f"\n✓ Completed signals with P&L: {completed_with_pnl}")
            
            if completed_with_pnl == 0:
                print("❌ CRITICAL: No completed trades with P&L data!")
                print("   This is why analytics shows 0 trades and 0 realized P&L")
                
                # Check if there are any completed trades at all
                cursor.execute("SELECT COUNT(*) FROM trading_signals WHERE status = 'completed'")
                completed_total = cursor.fetchone()[0]
                print(f"   Completed trades (any): {completed_total}")
                
                if completed_total > 0:
                    print("   → Issue: Trades are completed but P&L data is missing")
                    # Show sample completed trades without P&L
                    cursor.execute("""
                        SELECT signal_id, symbol, side, status, entry_price, exit_price, realized_pnl, exit_time
                        FROM trading_signals 
                        WHERE status = 'completed' AND realized_pnl IS NULL
                        LIMIT 5
                    """)
                    samples = cursor.fetchall()
                    print("   Sample completed trades without P&L:")
                    for sample in samples:
                        print(f"     - {sample[0]} ({sample[1]}) - Entry: {sample[4]}, Exit: {sample[5]}, P&L: {sample[6]}")
                else:
                    print("   → Issue: No trades are being marked as completed")
                    # Check most recent signals
                    cursor.execute("""
                        SELECT signal_id, symbol, side, status, created_at, updated_at
                        FROM trading_signals 
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    samples = cursor.fetchall()
                    print("   Most recent signals:")
                    for sample in samples:
                        print(f"     - {sample[0]} ({sample[1]}) - Status: {sample[3]} - Created: {sample[4]}")
        
        except Exception as e:
            print(f"❌ Error checking row counts: {e}")
        
        # Test database ORM methods
        print("\n" + "=" * 80)
        print("ORM METHODS TEST")
        print("=" * 80)
        
        try:
            # Test the ORM method used by analytics
            signals = db.get_trading_signals()
            print(f"✓ db.get_trading_signals() returned {len(signals)} signals")
            
            if signals:
                # Count completed with P&L using ORM
                completed_with_pnl_orm = [s for s in signals if s.get('status') == 'completed' and s.get('realized_pnl') is not None]
                print(f"✓ ORM completed with P&L: {len(completed_with_pnl_orm)}")
                
                if completed_with_pnl_orm:
                    print("   Sample completed trades:")
                    for signal in completed_with_pnl_orm[:3]:
                        print(f"     - {signal['symbol']}: {signal['realized_pnl']} USDT")
            
        except Exception as e:
            print(f"❌ Error testing ORM methods: {e}")
        
        conn.close()
        
        # Final diagnosis
        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)
        
        if db.use_postgres:
            print("✓ Running in PRODUCTION mode (PostgreSQL)")
            if total_count == 0:
                print("❌ ISSUE: Production database is empty")
                print("   SOLUTION: The bot needs to start creating signals")
            elif completed_with_pnl == 0:
                print("❌ ISSUE: Production has signals but no completed trades with P&L")
                print("   SOLUTION: Check trade completion logic")
            else:
                print("✅ Production database has completed trades - analytics should work")
        else:
            print("⚠️  Running in LOCAL mode (SQLite)")
            print("   This is NOT the production database!")
            print("   Production uses PostgreSQL with DATABASE_URL environment variable")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in database debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_production_database()