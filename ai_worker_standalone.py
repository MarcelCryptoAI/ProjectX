#!/usr/bin/env python3
"""
Standalone AI Worker voor Heroku Worker Dyno
Gescheiden van de web frontend voor betere resource management
"""

import os
import sys
import time
import signal
import logging
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from ai_worker import AIWorker
from utils.settings_loader import Settings

# Load environment variables
load_dotenv()

# Setup logging voor Heroku
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class StandaloneAIWorker:
    def __init__(self):
        self.ai_worker = None
        self.running = False
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
        
    def init_components(self):
        """Initialize ByBit session and settings"""
        try:
            # Load settings - use environment variables if available
            try:
                settings = Settings.load('config/settings.yaml')
            except:
                # Fallback to environment variables if config file doesn't exist
                settings = type('Settings', (), {})()
            
            # Get API keys from environment or settings
            api_key = os.getenv('BYBIT_API_KEY', getattr(settings, 'bybit_api_key', ''))
            api_secret = os.getenv('BYBIT_API_SECRET', getattr(settings, 'bybit_api_secret', ''))
            
            # Set attributes directly to avoid setter issues
            if hasattr(settings, '__dict__'):
                settings.__dict__['bybit_api_key'] = api_key
                settings.__dict__['bybit_api_secret'] = api_secret
            else:
                setattr(settings, 'bybit_api_key', api_key)
                setattr(settings, 'bybit_api_secret', api_secret)
            
            if not api_key or not api_secret:
                logger.error("BYBIT_API_KEY and BYBIT_API_SECRET environment variables are required!")
                return None, None
            
            # Initialize ByBit session
            bybit_session = HTTP(
                testnet=os.getenv('BYBIT_TESTNET', 'false').lower() == 'true',
                api_key=api_key,
                api_secret=api_secret,
            )
            
            logger.info("✅ ByBit session initialized successfully")
            return settings, bybit_session
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return None, None
    
    def start(self):
        """Start the standalone AI worker"""
        logger.info("🚀 Starting Standalone AI Worker...")
        
        # Initialize components
        settings, bybit_session = self.init_components()
        if not settings or not bybit_session:
            logger.error("❌ Failed to initialize, exiting...")
            return False
            
        # Create AI worker (without socketio for standalone mode)
        self.ai_worker = AIWorker(socketio=None, bybit_session=bybit_session)
        
        # Start the AI worker
        self.ai_worker.start()
        self.running = True
        
        logger.info("✅ AI Worker started successfully!")
        logger.info("📊 Worker will handle training, signal generation, and trade execution")
        
        # Check if trading should be enabled
        if os.getenv('TRADING_ENABLED', 'false').lower() == 'true':
            if self.ai_worker.trade_executor:
                self.ai_worker.trade_executor.enable_trading()
                logger.info("💰 Live trading ENABLED - Real money at risk!")
            else:
                logger.warning("⚠️ Trading enabled but no trade executor available")
        else:
            logger.info("🔒 Trading DISABLED - Safe mode active")
            
        # Start auto-training if enabled
        if os.getenv('ENABLE_AI_TRAINING', 'true').lower() == 'true':
            logger.info("🤖 AI Training enabled - Starting model training...")
            if not self.ai_worker.training_in_progress:
                self.ai_worker._start_model_training()
        
        return True
    
    def run(self):
        """Main worker loop"""
        if not self.start():
            return
            
        logger.info("🔄 Worker running... Press Ctrl+C to stop")
        
        # Keep the worker running
        try:
            while self.running:
                # Worker stats logging every 5 minutes
                time.sleep(300)
                if self.ai_worker:
                    stats = self.ai_worker.get_worker_stats()
                    logger.info(f"📊 Worker Status: Running={stats['is_running']}, "
                              f"Training={stats['training_in_progress']}, "
                              f"Signals={stats['signal_count']}")
                    
        except KeyboardInterrupt:
            logger.info("⌨️ Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Worker error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down AI Worker...")
        self.running = False
        
        if self.ai_worker:
            # Disable trading first
            if self.ai_worker.trade_executor:
                self.ai_worker.trade_executor.disable_trading()
                logger.info("🔒 Trading disabled")
            
            # Stop worker
            self.ai_worker.stop()
            logger.info("⏹️ AI Worker stopped")
            
        logger.info("✅ Shutdown complete")

def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("🤖 ByBit AI Trading Bot - Standalone Worker")
    logger.info("=" * 50)
    
    # Environment info
    logger.info(f"🌍 Environment: {os.getenv('FLASK_ENV', 'development')}")
    logger.info(f"🧪 Testnet Mode: {os.getenv('BYBIT_TESTNET', 'false')}")
    logger.info(f"💹 Trading Enabled: {os.getenv('TRADING_ENABLED', 'false')}")
    logger.info(f"🤖 AI Training: {os.getenv('ENABLE_AI_TRAINING', 'true')}")
    
    # Start worker
    worker = StandaloneAIWorker()
    worker.run()

if __name__ == "__main__":
    main()