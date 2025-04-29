document.addEventListener("DOMContentLoaded", () => {
    const ctx = document.getElementById("trendChart").getContext("2d");
  
    const chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: ["Current", "Forecast"],
        datasets: [{
          label: "AI Predicted Price (BTCUSDT)",
          data: [],
          borderColor: "#00b894",
          fill: false,
        }]
      },
      options: {
        scales: {
          y: { beginAtZero: false }
        }
      }
    });
  
    async function fetchForecast() {
      try {
        const response = await fetch("/api/forecast");
        const data = await response.json();
  
        chart.data.datasets[0].data = [data.from_price, data.to_price];
        chart.update();
      } catch (error) {
        console.error("Failed to fetch forecast:", error);
      }
    }
  
    fetchForecast();
    setInterval(fetchForecast, 60000); // Refresh every 60 seconds
  });
  