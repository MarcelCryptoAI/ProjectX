/**
 * Optimized Dashboard JavaScript
 * Implements performance improvements:
 * - Request caching with TTL
 * - Debounced updates
 * - Lazy loading of charts
 * - Smart polling with backoff
 */

class DashboardOptimized {
    constructor() {
        this.cache = new Map();
        this.cacheTTL = {
            '/api/balance': 60000,           // 1 minute
            '/api/analytics_data': 300000,   // 5 minutes
            '/api/trading_status': 30000,    // 30 seconds
            '/api/cumulative_roi': 300000,   // 5 minutes
            '/api/trading_signals': 60000    // 1 minute
        };
        
        this.updateIntervals = {
            dashboard: 30000,  // 30 seconds (was 10)
            status: 60000      // 60 seconds (was 15)
        };
        
        this.backoffMultiplier = 1;
        this.maxBackoff = 5;
        this.charts = {};
        this.isVisible = true;
        
        this.initVisibilityHandling();
        this.initLazyLoading();
    }
    
    /**
     * Cached fetch with TTL
     */
    async cachedFetch(url, options = {}) {
        const cacheKey = url;
        const cached = this.cache.get(cacheKey);
        const ttl = this.cacheTTL[url] || 120000; // Default 2 minutes
        
        // Check if cached data is still valid
        if (cached && (Date.now() - cached.timestamp < ttl)) {
            console.log(`Cache hit for ${url}`);
            return cached.data;
        }
        
        // Fetch fresh data
        try {
            const response = await fetch(url, options);
            const data = await response.json();
            
            // Cache successful responses
            if (response.ok) {
                this.cache.set(cacheKey, {
                    data: data,
                    timestamp: Date.now()
                });
                
                // Reset backoff on successful request
                this.backoffMultiplier = 1;
            }
            
            return data;
        } catch (error) {
            console.error(`Fetch error for ${url}:`, error);
            
            // Return cached data if available, even if expired
            if (cached) {
                console.log(`Returning stale cache for ${url} due to error`);
                return cached.data;
            }
            
            // Increase backoff on error
            this.backoffMultiplier = Math.min(this.backoffMultiplier * 1.5, this.maxBackoff);
            
            throw error;
        }
    }
    
    /**
     * Debounced update function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    /**
     * Smart polling with exponential backoff
     */
    async smartPoll(func, baseInterval) {
        const execute = async () => {
            if (!this.isVisible) {
                // Don't poll when tab is not visible
                setTimeout(execute, baseInterval);
                return;
            }
            
            try {
                await func();
                const nextInterval = baseInterval * this.backoffMultiplier;
                setTimeout(execute, nextInterval);
            } catch (error) {
                console.error('Polling error:', error);
                const nextInterval = baseInterval * this.backoffMultiplier;
                setTimeout(execute, Math.min(nextInterval, baseInterval * this.maxBackoff));
            }
        };
        
        execute();
    }
    
    /**
     * Handle visibility changes to pause updates when tab is hidden
     */
    initVisibilityHandling() {
        document.addEventListener('visibilitychange', () => {
            this.isVisible = !document.hidden;
            console.log(`Tab visibility changed: ${this.isVisible ? 'visible' : 'hidden'}`);
        });
    }
    
    /**
     * Lazy load charts only when visible
     */
    initLazyLoading() {
        const observerOptions = {
            root: null,
            rootMargin: '50px',
            threshold: 0.01
        };
        
        this.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const chartId = entry.target.id;
                    if (!this.charts[chartId]) {
                        this.loadChart(chartId);
                    }
                }
            });
        }, observerOptions);
        
        // Observe chart containers
        const chartContainers = document.querySelectorAll('[data-chart]');
        chartContainers.forEach(container => {
            this.observer.observe(container);
        });
    }
    
    /**
     * Load chart on demand
     */
    async loadChart(chartId) {
        console.log(`Lazy loading chart: ${chartId}`);
        
        switch(chartId) {
            case 'roiChart':
                await this.initROIChart();
                break;
            case 'performanceChart':
                await this.initPerformanceChart();
                break;
        }
    }
    
    /**
     * Optimized dashboard update
     */
    async updateDashboard() {
        try {
            // Batch API calls with Promise.all for parallel execution
            const [analyticsData, tradingStatus, signals] = await Promise.all([
                this.cachedFetch('/api/analytics_data'),
                this.cachedFetch('/api/trading_status'),
                this.cachedFetch('/api/trading_signals')
            ]);
            
            // Update UI with fetched data
            if (analyticsData.success) {
                this.updateAnalytics(analyticsData);
            }
            
            if (tradingStatus.success) {
                this.updateStatus(tradingStatus);
            }
            
            if (signals.success) {
                this.updateSignals(signals.signals);
            }
            
        } catch (error) {
            console.error('Dashboard update error:', error);
        }
    }
    
    /**
     * Batch DOM updates for better performance
     */
    updateAnalytics(data) {
        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(() => {
            // Batch all DOM updates
            const updates = {
                'totalBalance': `$${data.total_balance.toFixed(2)}`,
                'activePositions': data.active_positions_count,
                'winRate': `${data.win_rate.toFixed(1)}%`,
                'totalTrades': `${data.total_trades} TRADES`
            };
            
            // Apply all updates at once
            Object.entries(updates).forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element && element.textContent !== value) {
                    element.textContent = value;
                }
            });
            
            // Update P&L with color classes
            this.updatePnL(data.unrealized_pnl, data.total_balance);
            
            // Update positions table only if data changed
            this.updatePositionsTable(data.positions || []);
        });
    }
    
    /**
     * Efficient P&L update
     */
    updatePnL(pnl, totalBalance) {
        const pnlElement = document.getElementById('dailyPnL');
        const pnlPercentElement = document.getElementById('dailyPnLPercent');
        
        if (!pnlElement || !pnlPercentElement) return;
        
        const pnlPercent = totalBalance > 0 ? (pnl / totalBalance * 100) : 0;
        const isProfit = pnl >= 0;
        
        // Only update if values changed
        const newPnlText = `${isProfit ? '+' : '-'}$${Math.abs(pnl).toFixed(2)}`;
        const newPercentText = `${isProfit ? '+' : '-'}${Math.abs(pnlPercent).toFixed(2)}%`;
        
        if (pnlElement.textContent !== newPnlText) {
            pnlElement.innerHTML = `<span class="${isProfit ? 'profit-text' : 'loss-text'}">${newPnlText}</span>`;
        }
        
        if (pnlPercentElement.textContent !== newPercentText) {
            pnlPercentElement.innerHTML = `<span class="${isProfit ? 'profit-text' : 'loss-text'}">${newPercentText}</span>`;
        }
    }
    
    /**
     * Virtual DOM-like table update
     */
    updatePositionsTable(positions) {
        const tbody = document.getElementById('positionsTable');
        if (!tbody) return;
        
        // Generate new table content
        const newContent = positions.length === 0 
            ? '<tr><td colspan="11" class="text-center py-8 text-gray-500">No active positions</td></tr>'
            : positions.map(pos => this.generatePositionRow(pos)).join('');
        
        // Only update if content changed
        if (tbody.innerHTML !== newContent) {
            tbody.innerHTML = newContent;
        }
    }
    
    /**
     * Generate position row HTML
     */
    generatePositionRow(pos) {
        const pnl = parseFloat(pos.unrealisedPnl || 0);
        const pnlPercent = parseFloat(pos.percentage || 0);
        const pnlClass = pnl >= 0 ? 'profit-text' : 'loss-text';
        
        return `
            <tr class="border-b border-dark-border hover:bg-dark-hover transition-colors">
                <td class="py-3 font-medium">${pos.symbol}</td>
                <td class="py-3">
                    <span class="px-2 py-1 rounded text-xs font-bold ${pos.side === 'Buy' ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}">
                        ${pos.side}
                    </span>
                </td>
                <td class="py-3">${pos.size}</td>
                <td class="py-3">$${parseFloat(pos.avgPrice).toFixed(4)}</td>
                <td class="py-3">$${parseFloat(pos.markPrice).toFixed(4)}</td>
                <td class="py-3 ${pnlClass}">
                    ${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}
                </td>
                <td class="py-3 ${pnlClass}">
                    ${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%
                </td>
                <td class="py-3 text-neon-yellow">
                    ${parseFloat(pos.leverage || 1).toFixed(1)}x
                </td>
                <td class="py-3 text-gray-300">
                    $${parseFloat(pos.positionIM || 0).toFixed(2)}
                </td>
                <td class="py-3 text-gray-300">
                    $${parseFloat(pos.positionMM || 0).toFixed(2)}
                </td>
                <td class="py-3">
                    <button class="cyber-button py-1 px-3 rounded text-xs" onclick="closePosition('${pos.symbol}', '${pos.side}', '${pos.size}')">
                        Close
                    </button>
                </td>
            </tr>
        `;
    }
    
    /**
     * Initialize optimized dashboard
     */
    init() {
        console.log('Initializing optimized dashboard...');
        
        // Initial update
        this.updateDashboard();
        
        // Set up smart polling
        this.smartPoll(
            this.debounce(() => this.updateDashboard(), 500),
            this.updateIntervals.dashboard
        );
        
        // Preload critical resources
        this.preloadResources();
        
        // Set up WebSocket for real-time updates (if available)
        this.initWebSocket();
    }
    
    /**
     * Preload critical resources
     */
    preloadResources() {
        // Preload critical API data
        const criticalEndpoints = ['/api/balance', '/api/trading_status'];
        criticalEndpoints.forEach(endpoint => {
            this.cachedFetch(endpoint).catch(console.error);
        });
    }
    
    /**
     * Initialize WebSocket for real-time updates
     */
    initWebSocket() {
        if (typeof io !== 'undefined') {
            const socket = io();
            
            socket.on('position_update', () => {
                // Invalidate cache and update
                this.cache.delete('/api/analytics_data');
                this.updateDashboard();
            });
            
            socket.on('trade_update', () => {
                // Invalidate relevant caches
                this.cache.delete('/api/analytics_data');
                this.cache.delete('/api/balance');
                this.updateDashboard();
            });
        }
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardOptimized = new DashboardOptimized();
    window.dashboardOptimized.init();
});