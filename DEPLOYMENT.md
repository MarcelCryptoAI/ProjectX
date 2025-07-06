# ByBit AI Trading Bot - Heroku Deployment Guide

## 🚀 Quick Deploy naar Heroku

### Vereisten
- Heroku account
- Heroku CLI geïnstalleerd
- Git geïnstalleerd
- ByBit API keys (live of testnet)

### 1. Heroku App Setup

```bash
# Login naar Heroku
heroku login

# Maak nieuwe app
heroku create jouw-bybit-bot-naam

# Voeg PostgreSQL toe
heroku addons:create heroku-postgresql:essential-0
```

### 2. Environment Variables Configureren

```bash
# ByBit API Configuratie
heroku config:set BYBIT_API_KEY="jouw_api_key"
heroku config:set BYBIT_API_SECRET="jouw_api_secret"
heroku config:set BYBIT_TESTNET="true"  # false voor live trading

# Flask Configuratie
heroku config:set SECRET_KEY="jouw_super_geheime_sleutel_hier"
heroku config:set FLASK_ENV="production"

# Bot Configuratie
heroku config:set BOT_ENABLED="true"
heroku config:set TRADING_ENABLED="false"  # start veilig
heroku config:set AI_CONFIDENCE_THRESHOLD="75"
heroku config:set RISK_PER_TRADE_PERCENT="2"

# Feature Flags
heroku config:set ENABLE_LIVE_TRADING="false"  # veiligheid eerst
heroku config:set ENABLE_AI_TRAINING="true"
```

### 3. Deploy Code

```bash
# In project directory
git add .
git commit -m "Production ready voor Heroku"
git push heroku main

# Start de app
heroku ps:scale web=1
```

### 4. Database Setup

```bash
# Database wordt automatisch geconfigureerd via DATABASE_URL
# Check database status
heroku pg:info
```

### 5. Logs Monitoring

```bash
# Live logs bekijken
heroku logs --tail

# Specifieke logs
heroku logs --app jouw-app-naam --tail
```

## 🔧 Team Handover Instructies

### Voor het Development Team

#### Toegang Setup
1. **Heroku Toegang**:
   ```bash
   heroku access:add team-member@email.com --app jouw-app-naam
   ```

2. **Repository Access**: Geef team toegang tot Git repository

#### Belangrijke Files
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku start commando
- `runtime.txt` - Python versie
- `.env.example` - Template voor environment variables
- `DEPLOYMENT.md` - Deze guide

#### Environment Variables Uitleg
| Variable | Beschrijving | Voorbeeld |
|----------|--------------|-----------|
| `BYBIT_API_KEY` | ByBit API key | `xxxxxxxxxxxx` |
| `BYBIT_API_SECRET` | ByBit API secret | `xxxxxxxxxxxx` |
| `BYBIT_TESTNET` | Testnet mode | `true`/`false` |
| `SECRET_KEY` | Flask session key | `random-string` |
| `TRADING_ENABLED` | Live trading schakelaar | `true`/`false` |
| `AI_CONFIDENCE_THRESHOLD` | Minimum confidence voor trades | `75` |
| `RISK_PER_TRADE_PERCENT` | Max risk per trade | `2` |

#### Database Schema
- **PostgreSQL** gebruikt in productie
- **SQLite** voor lokale development
- Auto-migratie bij startup
- Tabellen:
  - `market_data` - OHLCV data
  - `technical_indicators` - RSI, MACD, etc.
  - `sentiment_data` - Market sentiment
  - `training_sessions` - AI training logs
  - `training_results` - AI model resultaten

#### Applicatie Endpoints
- `/` - Main dashboard
- `/coin_status` - AI analysis per coin
- `/config` - Settings
- `/analytics` - Trading analytics
- `/api/*` - REST API endpoints

#### Safety Features
- **Trading disabled by default** - Veiligheid eerst
- **Testnet support** - Test zonder echt geld
- **Confidence thresholds** - Alleen high-confidence trades
- **Risk management** - Max % per trade
- **Real-time logging** - Alle acties gelogd

### Productie Monitoring

#### Health Checks
```bash
# App status
heroku ps:status

# Resource usage
heroku ps:exec
top

# Database performance
heroku pg:diagnose
```

#### Error Tracking
- Logs via `heroku logs --tail`
- Database errors in console
- API failures logged
- WebSocket connection status

#### Scaling
```bash
# Scale up voor meer traffic
heroku ps:scale web=2

# Scale down om kosten te besparen
heroku ps:scale web=1
```

## 🔒 Beveiliging & Best Practices

### API Keys
- **NOOIT** commit API keys naar Git
- Gebruik altijd environment variables
- Roteer keys regelmatig
- Test eerst op testnet

### Trading Safety
- Start met `TRADING_ENABLED=false`
- Begin met lage risk percentages (1-2%)
- Monitor 24/7 wanneer live trading actief
- Zet stop-losses en position limits

### Database
- PostgreSQL backups via Heroku
- Geen gevoelige data in logs
- Regular data cleanup voor performance

### Monitoring
- Check daily logs
- Monitor API rate limits
- Track P&L en performance metrics
- Set up alerts voor errors

## 🆘 Troubleshooting

### Common Issues

1. **App won't start**:
   ```bash
   heroku logs --tail
   # Check for missing environment variables
   ```

2. **Database errors**:
   ```bash
   heroku pg:diagnose
   heroku pg:reset DATABASE_URL  # only if needed
   ```

3. **API connection issues**:
   - Check API keys
   - Verify ByBit API status
   - Check rate limits

4. **Memory issues**:
   ```bash
   heroku ps:restart
   heroku ps:scale web=1
   ```

### Emergency Stops
```bash
# Stop all trading immediately
heroku config:set TRADING_ENABLED="false"

# Restart app
heroku ps:restart

# Scale down
heroku ps:scale web=0
```

## 📞 Team Contact

Voor vragen over deployment:
1. Check deze documentatie eerst
2. Check Heroku logs: `heroku logs --tail`
3. Check database status: `heroku pg:info`
4. Contact originele developer indien nodig

**Succes met de deployment! 🚀**