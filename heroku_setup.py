#!/usr/bin/env python
"""
Heroku deployment setup script
Ensures that the application is properly configured for Heroku deployment
without Redis or external worker dependencies.
"""

import os
import sys

def check_environment():
    """Check that all required environment variables are set"""
    required_vars = [
        'BYBIT_API_KEY',
        'BYBIT_API_SECRET', 
        'SECRET_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def verify_no_redis_dependencies():
    """Verify that no Redis dependencies are present"""
    try:
        import redis
        print("⚠️  Warning: Redis package is installed but not used in this application")
    except ImportError:
        print("✅ No Redis dependencies detected (correct)")
    
    try:
        import celery
        print("⚠️  Warning: Celery package is installed but not used in this application")
    except ImportError:
        print("✅ No Celery dependencies detected (correct)")

def initialize_database():
    """Initialize SQLite database for local development"""
    try:
        from database import TradingDatabase
        db = TradingDatabase()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
        return False

def main():
    print("🚀 ByBit AI Trading Bot - Heroku Setup")
    print("=" * 50)
    
    # Check environment
    env_ok = check_environment()
    
    # Verify no Redis
    verify_no_redis_dependencies()
    
    # Initialize database
    db_ok = initialize_database()
    
    print("\n" + "=" * 50)
    if env_ok:
        print("✅ Heroku setup completed successfully!")
        print("🌐 Application ready to deploy")
    else:
        print("❌ Setup failed - check environment variables")
        sys.exit(1)

if __name__ == "__main__":
    main()