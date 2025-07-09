#!/usr/bin/env python3
"""
Check database status and symbols table
"""
import os
import sys
from datetime import datetime

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def check_database_status():
    """Check the current status of the database"""
    try:
        print("=" * 60)
        print("DATABASE STATUS CHECK")
        print("=" * 60)
        
        # Initialize database
        db = TradingDatabase()
        print(f"✓ Database initialized")
        print(f"  Using PostgreSQL: {db.use_postgres}")
        if hasattr(db, 'database_url') and db.database_url:
            print(f"  Database URL: {db.database_url[:50]}...")
        
        print("\n" + "=" * 60)
        print("SUPPORTED SYMBOLS TABLE")
        print("=" * 60)
        
        # Check supported symbols
        symbols = db.get_supported_symbols()
        print(f"✓ Total symbols in database: {len(symbols)}")
        
        if symbols:
            # Show first few symbols
            print("\nSample symbols:")
            for i, symbol in enumerate(symbols[:10]):
                print(f"  {i+1}. {symbol['symbol']} - Status: {symbol['status']} - Leverage: {symbol.get('min_leverage', 'N/A')}-{symbol.get('max_leverage', 'N/A')}x - Multiplier: {symbol.get('leverage_multiplier', 1.0)}")
            
            if len(symbols) > 10:
                print(f"  ... and {len(symbols) - 10} more")
            
            # Check last updated
            last_updated = db.get_symbols_last_updated()
            if last_updated:
                print(f"\n✓ Last updated: {last_updated}")
            else:
                print("\n⚠️  Last updated: Never")
                
            # Count active vs inactive
            active_count = len([s for s in symbols if s.get('status') == 'active'])
            inactive_count = len(symbols) - active_count
            print(f"✓ Active symbols: {active_count}")
            print(f"✓ Inactive symbols: {inactive_count}")
            
        else:
            print("⚠️  No symbols found in database!")
            print("   This could mean:")
            print("   1. The table exists but is empty")
            print("   2. The table doesn't exist")
            print("   3. There's a database connection issue")
        
        print("\n" + "=" * 60)
        print("TESTING DATABASE OPERATIONS")
        print("=" * 60)
        
        # Test database connection and table structure
        conn = db.get_connection()
        cursor = conn.cursor()
        
        try:
            if db.use_postgres:
                # PostgreSQL
                cursor.execute("""
                    SELECT table_name, column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'supported_symbols' 
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()
                print("✓ PostgreSQL table structure:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
                    
                # Check if table has data
                cursor.execute("SELECT COUNT(*) FROM supported_symbols")
                count = cursor.fetchone()[0]
                print(f"✓ Row count from direct query: {count}")
                
            else:
                # SQLite
                cursor.execute("PRAGMA table_info(supported_symbols)")
                columns = cursor.fetchall()
                print("✓ SQLite table structure:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
                    
                # Check if table has data
                cursor.execute("SELECT COUNT(*) FROM supported_symbols")
                count = cursor.fetchone()[0]
                print(f"✓ Row count from direct query: {count}")
                
        except Exception as e:
            print(f"❌ Error checking table structure: {e}")
        
        conn.close()
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        if not symbols:
            print("📝 ISSUE: No symbols in database")
            print("   SOLUTION: Run the refresh symbols function from the web interface")
            print("   OR manually refresh using: /api/refresh_symbols")
        elif len(symbols) < 50:
            print("📝 ISSUE: Very few symbols in database")
            print("   SOLUTION: The symbols list might be incomplete - consider refreshing")
        else:
            print("✅ Database appears to be working correctly")
            if not last_updated:
                print("📝 RECOMMENDATION: Symbols exist but no last_updated timestamp")
                print("   SOLUTION: Run refresh to update timestamps")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_database_status()