"""
Global database singleton to prevent connection pool exhaustion
"""
from database import TradingDatabase

# Global database instance
_db_instance = None

def get_database():
    """Get the global database instance (singleton pattern)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradingDatabase()
    return _db_instance

def reset_database():
    """Reset the global database instance (for testing)"""
    global _db_instance
    _db_instance = None