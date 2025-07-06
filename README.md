# ByBit AI Trading Bot 🤖

Een AI-aangedreven cryptocurrency trading bot voor ByBit exchange met real-time web dashboard.

## ✨ Features

- 🤖 **AI-driven trading** - Machine learning voor market analyse
- 📊 **Real-time dashboard** - Live monitoring via web interface  
- 🔄 **WebSocket support** - Live updates zonder refresh
- 📈 **Technical indicators** - RSI, MACD, Bollinger Bands, etc.
- 🎯 **Risk management** - Position sizing en stop-loss automation
- 💼 **Portfolio tracking** - Real-time P&L en performance metrics
- 🔐 **Secure API integration** - ByBit API met rate limiting
- 🗄️ **Database storage** - Training data en trade history
- 📱 **Responsive design** - Werkt op desktop en mobile

## 🚀 Quick Start (Lokaal)

```bash
# Clone repository
git clone <repository-url>
cd bybit-ai-bot

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env met jouw ByBit API keys

# Start de applicatie
python web_app.py
```

Dashboard beschikbaar op: http://localhost:5000

## 🌐 Heroku Deployment

Zie [DEPLOYMENT.md](DEPLOYMENT.md) voor volledige deployment instructies.

### One-Click Deploy
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

### Manual Deploy
```bash
heroku create jouw-app-naam
heroku config:set BYBIT_API_KEY="jouw_key"
heroku config:set BYBIT_API_SECRET="jouw_secret"
heroku config:set BYBIT_TESTNET="true"
git push heroku main
```

## 🔧 Configuratie

### Environment Variables
| Variable | Beschrijving | Default |
|----------|--------------|---------|
| `BYBIT_API_KEY` | ByBit API key | Required |
| `BYBIT_API_SECRET` | ByBit API secret | Required |
| `BYBIT_TESTNET` | Testnet mode | `true` |
| `TRADING_ENABLED` | Live trading | `false` |
| `AI_CONFIDENCE_THRESHOLD` | Min confidence | `75` |
| `RISK_PER_TRADE_PERCENT` | Max risk per trade | `2` |

### Trading Safety
⚠️ **BELANGRIJK**: Start altijd met:
- `BYBIT_TESTNET=true` (testnet)
- `TRADING_ENABLED=false` (geen live trades)
- Lage risk percentages (1-2%)

## 📊 Dashboard Features

### Main Dashboard
- **System Status** - Trading/training status
- **Live Console** - Real-time logging
- **Control Panel** - START ALL / STOP ALL buttons
- **Performance Metrics** - P&L, win rate, positions

### Coin Status Page
- **AI Analysis** per coin - Bullish/Bearish/Neutral
- **Take Profit/Stop Loss** recommendations
- **Confidence Scores** - AI prediction confidence
- **Direct Trading** - Execute trades direct vanuit interface

### Settings Page
- **API Configuration** - ByBit connection settings
- **Risk Management** - Position sizing, leverage
- **AI Parameters** - Confidence thresholds, training intervals

### Analytics Page
- **Performance Charts** - P&L over time
- **Trade Statistics** - Win/loss ratios, best performing pairs
- **Risk Metrics** - Drawdown, Sharpe ratio

## 🏗️ Architecture

```
├── web_app.py              # Main Flask application
├── ai_worker.py            # AI training en signal generation
├── database.py             # PostgreSQL/SQLite database layer
├── trading/
│   └── executor.py         # Trade execution engine
├── ai/
│   └── trader.py           # AI model en predictions
├── utils/
│   ├── settings_loader.py  # Configuration management
│   └── trade_logger.py     # Trade logging
├── bot/
│   └── risk_manager.py     # Risk management
└── templates/              # HTML templates
```

## 🗄️ Database Schema

- **market_data** - OHLCV price data
- **technical_indicators** - RSI, MACD, BB values
- **sentiment_data** - Market sentiment scores
- **training_sessions** - AI training logs
- **training_results** - Model performance data

## 🔌 API Endpoints

### System Control
- `POST /api/start_all` - Start training, signals & trading
- `POST /api/stop_all` - Stop all operations

### Data Endpoints
- `GET /api/coin_analysis` - AI analysis per coin
- `GET /api/worker_stats` - System status
- `GET /api/database_stats` - Database statistics
- `GET /api/console_logs` - Live console logs

### Trading
- `POST /api/execute_ai_trade` - Execute recommended trade
- `GET /api/positions` - Current positions
- `GET /api/trading_history` - Trade history

## 🛡️ Security Features

- **API Key Encryption** - Secure storage via environment variables
- **Rate Limiting** - Respect ByBit API limits
- **Risk Controls** - Max position sizes, stop losses
- **Testnet Support** - Safe testing environment
- **Manual Overrides** - Emergency stop functionality

## 📈 Trading Strategy

1. **Data Collection** - Market data, technical indicators, sentiment
2. **AI Analysis** - Machine learning model predictions
3. **Signal Generation** - Confidence-based trade signals
4. **Risk Assessment** - Position sizing, stop loss calculation
5. **Trade Execution** - Automated order placement
6. **Monitoring** - Real-time position tracking
7. **Learning** - Continuous model improvement

## 🔧 Development

### Local Setup
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Unix
venv\\Scripts\\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run in development mode
export FLASK_ENV=development
python web_app.py
```

### Testing
- Start met testnet (`BYBIT_TESTNET=true`)
- Disable live trading (`TRADING_ENABLED=false`)
- Monitor logs voor errors
- Test met kleine amounts

## 🆘 Troubleshooting

### Common Issues
1. **API Connection Failed**
   - Check API keys in environment variables
   - Verify ByBit API status
   - Check network connectivity

2. **Database Errors**
   - PostgreSQL connection issues (production)
   - Check DATABASE_URL environment variable
   - Verify database permissions

3. **Trading Disabled**
   - Check `TRADING_ENABLED` setting
   - Verify `ENABLE_LIVE_TRADING` flag
   - Check confidence thresholds

4. **WebSocket Issues**
   - Browser compatibility
   - Network firewall blocking
   - Check real-time console logs

### Emergency Stops
```bash
# Stop all trading immediately
heroku config:set TRADING_ENABLED="false"

# Restart application  
heroku ps:restart

# Check status
heroku logs --tail
```

## 📚 Resources

- [ByBit API Documentation](https://bybit-exchange.github.io/docs/)
- [Heroku Deployment Guide](DEPLOYMENT.md)
- [Trading Strategy Guide](docs/strategy.md)

## ⚖️ Disclaimer

Dit is experimentele software voor educational purposes. 
- **Trade op eigen risiko**
- **Start altijd met testnet**
- **Monitor altijd je positions**
- **Gebruik alleen geld dat je kunt missen**

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - zie [LICENSE](LICENSE) file.

---

**Happy Trading! 🚀📈**