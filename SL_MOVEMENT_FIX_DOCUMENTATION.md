# Stop Loss Movement Fix Documentation

## Problem Description
The stop loss was not automatically moving to entry price + 0.1% after Take Profit 1 (TP1) was hit. This is a critical risk management feature that protects profits by ensuring the position becomes risk-free after the first profit target is reached.

## Root Causes Identified

### 1. TP1 Detection Logic Issue
**Problem:** The original code had redundant conditions for detecting TP1 hits:
```python
# OLD (PROBLEMATIC) CODE
if order['orderId'] in tp_order_ids and order['orderId'] == tp_levels[0].get('order_id'):
```

**Issue:** This double-check was unnecessary and potentially causing the condition to fail.

### 2. Quantity Mismatch for New Stop Loss
**Problem:** After TP1 was hit, the position size was reduced, but the new stop loss order still used the original full quantity.

**Issue:** This would cause the stop loss order to fail due to insufficient position size.

### 3. Insufficient Error Handling
**Problem:** API failures or network issues could silently break the monitoring without proper logging.

### 4. Missing Retry Logic
**Problem:** Temporary API failures would cause the entire monitoring cycle to fail.

## Solutions Implemented

### 1. Fixed TP1 Detection Logic ✅
**New approach:** Simplified and more reliable TP1 detection:
```python
# NEW (FIXED) CODE
tp1_order_id = tp_levels[0].get('order_id')
tp1_still_open = False

if tp1_order_id:
    for order in orders['result']['list']:
        if order['orderId'] == tp1_order_id:
            tp1_still_open = True
            break
    
    self.console_logger.log('INFO', f'🎯 TP1 status for {symbol}: Order ID {tp1_order_id}, Still open: {tp1_still_open}')

# If TP1 is no longer in open orders, it was filled
if tp1_order_id and not tp1_still_open and not trade_data.get('tp1_hit', False):
    # Move stop loss logic here
```

### 2. Fixed Stop Loss Quantity ✅
**New approach:** Get actual position size before placing new SL:
```python
# Get current position size to calculate correct SL quantity
current_position_size = 0
for position in positions['result']['list']:
    if position['symbol'] == symbol and float(position.get('size', 0)) > 0:
        current_position_size = float(position['size'])
        break

# Use actual position size for new SL order
sl_result = self.bybit_session.place_order(
    category="linear",
    symbol=symbol,
    side="Sell" if side == "Buy" else "Buy",
    orderType="StopMarket",
    qty=str(current_position_size),  # Use actual position size
    stopPrice=str(new_sl_price),
    timeInForce="GTC",
    reduceOnly=True
)
```

### 3. Enhanced Error Handling and Logging ✅
**Added comprehensive logging:**
```python
self.console_logger.log('INFO', f'📊 Checking trade {symbol} - Entry filled: {trade_data.get("entry_filled", False)}, TP1 hit: {trade_data.get("tp1_hit", False)}')
self.console_logger.log('INFO', f'🎯 TP1 status for {symbol}: Order ID {tp1_order_id}, Still open: {tp1_still_open}')
self.console_logger.log('INFO', f'📊 Current position size for {symbol}: {current_position_size}')
self.console_logger.log('INFO', f'📤 Placing new SL order for {symbol}: qty={current_position_size}, price=${new_sl_price:.4f}')
```

### 4. Added Retry Logic for API Calls ✅
**Robust API call handling:**
```python
# Get current orders with retry logic
orders = None
for attempt in range(3):  # Retry up to 3 times
    try:
        orders = self.bybit_session.get_open_orders(category="linear")
        if orders and 'result' in orders:
            break
        time.sleep(1)  # Wait 1 second between retries
    except Exception as api_error:
        if attempt == 2:  # Last attempt
            self.console_logger.log('ERROR', f'❌ Failed to fetch orders after 3 attempts: {str(api_error)}')
            return
        time.sleep(2)  # Wait longer between retries
```

### 5. Added Debugging and Monitoring Tools ✅

**New API Endpoints:**
- `GET /api/trades/active_status` - Get detailed status of all active trades
- `POST /api/trades/force_monitor` - Manually trigger trade monitoring
- `GET /api/trades/sl_debug` - Debug stop loss movement functionality

**New Methods in AIWorker:**
- `force_monitor_trades()` - Manual monitoring trigger
- `get_active_trades_status()` - Detailed trade status

### 6. Enhanced System Checks ✅
**Added validation for:**
- ByBit session availability
- Active trades existence
- Position existence before SL movement
- Proper order cancellation before placing new SL

## Fixed Calculation Logic

### For Buy Positions:
- **Entry Price:** $50,000
- **Breakeven + 0.1%:** $50,000 × 1.001 = $50,050
- **New SL Price:** $50,050

### For Sell Positions:
- **Entry Price:** $50,000
- **Breakeven - 0.1%:** $50,000 × 0.999 = $49,950
- **New SL Price:** $49,950

## Monitoring and Verification

### Log Messages to Watch For:
```
✅ TP1 hit for BTCUSDT! Moving stop loss to breakeven + 0.1%
📊 Current position size for BTCUSDT: 0.001
🚫 Cancelling existing SL order ABC123 for BTCUSDT
✅ Existing SL cancelled for BTCUSDT
📤 Placing new SL order for BTCUSDT: qty=0.001, price=$50050.0000
✅ Stop loss moved to breakeven+0.1% for BTCUSDT: $50050.0000 (Order ID: XYZ789)
```

### API Testing:
```bash
# Check debug status
curl http://localhost:5000/api/trades/sl_debug

# Get active trades
curl http://localhost:5000/api/trades/active_status

# Force monitoring
curl -X POST http://localhost:5000/api/trades/force_monitor
```

## Test Script Usage

Run the comprehensive test script:
```bash
python test_sl_movement.py
```

This will:
1. Check system status
2. Test SL calculation logic
3. Test all debugging endpoints
4. Verify the fixes are working

## Files Modified

1. **ai_worker.py**
   - `monitor_trades_and_move_sl()` - Complete rewrite with fixes
   - `force_monitor_trades()` - New debugging method
   - `get_active_trades_status()` - New status method

2. **web_app.py**
   - Added 3 new API endpoints for debugging and monitoring

3. **test_sl_movement.py** (New)
   - Comprehensive test suite for verification

## Deployment Instructions

1. **Deploy to Heroku:**
   ```bash
   git add .
   git commit -m "Fix stop loss movement to breakeven+0.1% after TP1 hit"
   git push heroku master
   ```

2. **Monitor in Production:**
   - Check logs for SL movement messages
   - Use debug endpoints to verify system status
   - Monitor trade performance to ensure SL is moving correctly

3. **Verification:**
   - Open a test trade
   - Wait for TP1 to hit
   - Verify SL moves to breakeven + 0.1%
   - Check logs for confirmation messages

## Risk Management Impact

✅ **Before Fix:** Stop loss remained at original level, risking full loss even after profit was taken  
✅ **After Fix:** Stop loss automatically moves to breakeven + 0.1%, guaranteeing profit protection

This fix ensures that once TP1 is hit, the worst-case scenario is a small profit (0.1%), eliminating the risk of turning a winning trade into a losing one.