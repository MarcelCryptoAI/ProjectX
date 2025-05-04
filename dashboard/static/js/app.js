// Example: Fetch balance and update UI
async function updateBalance() {
    const response = await fetch('/api/balance');
    const data = await response.json();
    document.getElementById('balance').textContent = `${data.balance} USDT`;
}

// Update chart data (real-time)
async function updateChart() {
    const response = await fetch('/api/market_data');
    const data = await response.json();
    const chart = new Chart(document.getElementById('priceChart'), {
        type: 'line',
        data: {
            labels: data.time,
            datasets: [{
                label: 'Price (USDT)',
                data: data.prices,
                borderColor: '#42A5F5',
                fill: false
            }]
        }
    });
}

setInterval(updateBalance, 5000);  // Update balance every 5 seconds
setInterval(updateChart, 5000);  // Update chart every 5 seconds
