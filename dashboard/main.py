from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os

from ai.trend_predictor import TrendPredictor
from utils.trade_logger import TradeLogger

app = FastAPI()

app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")

USERNAME = "admin"
PASSWORD = "changeme123"

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    try:
        if username == USERNAME and password == PASSWORD:
            response = RedirectResponse(url="/dashboard", status_code=302)
            response.set_cookie("authenticated", "yes")
            return response
        else:
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": "Invalid credentials"
            })
    except Exception as e:
        print("❌ Login error:", e)
        return HTMLResponse(f"Login error: {e}", status_code=500)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if request.cookies.get("authenticated") != "yes":
        return RedirectResponse("/")

    trades = []
    stats = {}
    last_trade = {}

    if os.path.exists("trades.json"):
        with open("trades.json", "r") as f:
            trades = json.load(f)
            if trades:
                last_trade = trades[-1]

    if trades:
        logger = TradeLogger()
        stats = logger.calculate_stats()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "last_trade": last_trade,
        "trades": trades,
        "stats": stats
    })

@app.get("/api/trades")
async def get_trades():
    with open("trades.json", "r") as f:
        data = json.load(f)
    return JSONResponse(data)

@app.get("/api/forecast")
async def get_forecast():
    try:
        predictor = TrendPredictor()
        forecast = predictor.predict_trend(symbol="BTCUSDT")
        return JSONResponse({
            "symbol": "BTCUSDT",
            "confidence": forecast["confidence"],
            "direction": forecast["direction"],
            "from_price": forecast["from"],
            "to_price": forecast["to"]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})
