// Enhanced coin details modal with leverage, amounts, and profit/loss calculations

function showCoinDetails(symbol) {
    const coin = allCoins.find(c => c.symbol === symbol);
    if (!coin) return;

    // Calculate trading details
    const leverage = Math.max(1, Math.min(10, Math.floor(1 + (coin.confidence - 50) / 10)));
    const leverageType = leverage > 1 ? 'Cross' : 'Isolated';
    const riskPerTrade = 2.0; // 2% van balance
    const baseAmount = Math.max(100, (currentBalance * riskPerTrade / 100));
    const tradeAmount = baseAmount * leverage;
    
    // Calculate potential profit and loss
    const potentialProfitPercent = coin.takeProfit;
    const potentialLossPercent = coin.stopLoss;
    const potentialProfitDollar = (baseAmount * potentialProfitPercent / 100) * leverage;
    const potentialLossDollar = (baseAmount * potentialLossPercent / 100) * leverage;

    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black/50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="glass-card rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-bold text-white">${coin.symbol} Analysis</h3>
                <button onclick="this.closest('.fixed').remove()" class="text-gray-400 hover:text-white">
                    <i class="fas fa-times"></i>
                </button>
            </div>

            <div class="space-y-4">
                <!-- Analysis Overview -->
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <div class="text-sm text-gray-400">Analysis</div>
                        <div class="font-bold text-${coin.analysis.toLowerCase() === 'bullish' ? 'green' : coin.analysis.toLowerCase() === 'bearish' ? 'red' : 'yellow'}-400">
                            ${coin.analysis}
                        </div>
                    </div>
                    <div>
                        <div class="text-sm text-gray-400">Confidence</div>
                        <div class="font-bold text-neon-cyan">${coin.confidence}%</div>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <div class="text-sm text-gray-400">Accuracy</div>
                        <div class="font-bold text-neon-green">${coin.accuracy}%</div>
                    </div>
                    <div>
                        <div class="text-sm text-gray-400">Status</div>
                        <div class="font-bold text-white">${coin.status}</div>
                    </div>
                </div>

                <!-- Leverage & Position Details -->
                <div class="border-t border-gray-700 pt-4">
                    <div class="text-sm text-gray-400 mb-2">Leverage & Position</div>
                    <div class="grid grid-cols-3 gap-4">
                        <div>
                            <div class="text-xs text-gray-500">Leverage</div>
                            <div class="font-bold text-neon-yellow">${leverage}x</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-500">Type</div>
                            <div class="font-bold text-neon-cyan">${leverageType}</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-500">Direction</div>
                            <div class="font-bold text-${coin.direction === 'Buy' ? 'green' : 'red'}-400">${coin.direction}</div>
                        </div>
                    </div>
                </div>

                <!-- Trade Amount Details -->
                <div class="border-t border-gray-700 pt-4">
                    <div class="text-sm text-gray-400 mb-2">Trade Amount</div>
                    <div class="grid grid-cols-3 gap-4">
                        <div>
                            <div class="text-xs text-gray-500">Base Amount</div>
                            <div class="font-bold text-white">$${baseAmount.toFixed(2)}</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-500">With Leverage</div>
                            <div class="font-bold text-neon-yellow">$${tradeAmount.toFixed(2)}</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-500">Risk %</div>
                            <div class="font-bold text-neon-pink">${riskPerTrade}%</div>
                        </div>
                    </div>
                </div>

                <!-- Profit/Loss Projections -->
                <div class="border-t border-gray-700 pt-4">
                    <div class="text-sm text-gray-400 mb-2">Profit/Loss Projections</div>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-green-900/20 border border-green-500/30 rounded-lg p-3">
                            <div class="text-xs text-green-400">Take Profit</div>
                            <div class="font-bold text-neon-green">${coin.takeProfit}%</div>
                            <div class="text-sm text-green-300">+$${potentialProfitDollar.toFixed(2)}</div>
                        </div>
                        <div class="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
                            <div class="text-xs text-red-400">Stop Loss</div>
                            <div class="font-bold text-neon-pink">${coin.stopLoss}%</div>
                            <div class="text-sm text-red-300">-$${potentialLossDollar.toFixed(2)}</div>
                        </div>
                    </div>
                </div>

                <!-- Risk Metrics -->
                <div class="border-t border-gray-700 pt-4">
                    <div class="text-sm text-gray-400 mb-2">Risk Metrics</div>
                    <div class="grid grid-cols-3 gap-4">
                        <div>
                            <div class="text-xs text-gray-500">Risk/Reward</div>
                            <div class="font-bold text-neon-cyan">${(potentialProfitPercent / potentialLossPercent).toFixed(2)}</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-500">Max Loss</div>
                            <div class="font-bold text-red-400">${((potentialLossDollar / currentBalance) * 100).toFixed(1)}%</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-500">Win Rate Needed</div>
                            <div class="font-bold text-yellow-400">${(potentialLossPercent / (potentialProfitPercent + potentialLossPercent) * 100).toFixed(0)}%</div>
                        </div>
                    </div>
                </div>

                <!-- AI Reasoning -->
                <div class="border-t border-gray-700 pt-4">
                    <div class="text-sm text-gray-400 mb-2">AI Reasoning</div>
                    <div class="text-sm text-gray-300 bg-dark-bg/50 p-3 rounded">
                        Based on technical analysis with ${coin.accuracy}% accuracy.
                        Confidence score of ${coin.confidence}% indicates ${coin.confidence > 75 ? 'strong' : coin.confidence > 50 ? 'moderate' : 'weak'} signal strength.
                        ${coin.analysis === 'Bullish' ? 'Upward momentum detected.' : coin.analysis === 'Bearish' ? 'Downward momentum detected.' : 'Mixed signals detected.'}
                        <br><br>
                        <strong>Leverage Calculation:</strong> ${leverage}x leverage selected based on ${coin.confidence}% confidence level.
                        <strong>Position Mode:</strong> ${leverageType} margin provides ${leverageType === 'Cross' ? 'shared risk across account' : 'isolated risk per position'}.
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="flex gap-2 pt-4">
                    <button class="flex-1 cyber-button py-2 px-4 rounded-lg text-sm"
                            onclick="executeTrade('${coin.symbol}', '${coin.direction}', ${coin.takeProfit}, ${coin.stopLoss}); this.closest('.fixed').remove();">
                        <i class="fas fa-rocket mr-2"></i>EXECUTE TRADE
                    </button>
                    <button class="flex-1 bg-gray-700 text-white py-2 px-4 rounded-lg text-sm hover:bg-gray-600 transition-colors"
                            onclick="this.closest('.fixed').remove()">
                        CLOSE
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
}