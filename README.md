# 📡 StockSense India — AI Stock Direction Predictor

> **Predict whether any Indian stock (NSE/BSE) will move UP or DOWN tomorrow using Live Market Data and a 5-Model Stacking Ensemble.**

---

## 🌟 Key Features

- **5-Model Stacking Ensemble**: Combines **Random Forest**, **Gradient Boosting**, **XGBoost**, **LightGBM**, and **Logistic Regression** into an out-of-fold meta-learner for higher directional accuracy.
- **55+ Technical Indicators**: Calculates real-time RSI, MACD, Bollinger Bands, ATR, ADX (Trend Strength), Aroon Oscillator, Chaikin Money Flow (CMF), Money Flow Index (MFI), VWAP deviation, candlestick pattern metrics, and streak counters.
- **Price Target & Range Estimation**: Uses Ridge regression to predict the exact percentage price movement and target price, alongside ATR-based upper/lower volatility bounds.
- **Support & Resistance Detection**: Automatically identifies 20-day key support and resistance levels.
- **Sub-Second Performance**: Vectorized indicator computations (100x faster Aroon via NumPy sliding windows) and 2-pass validation splitting ensure sub-second response times.
- **Instant Search & Autocomplete**: Search across 2,300+ NSE equities and major indices (`Nifty 50`, `Sensex`, `Nifty Bank`, `Nifty IT`).
- **Interactive UI**: Modern dark glassmorphism dashboard built with Vanilla CSS and Chart.js featuring dynamic price charts and key technical signals.

---

## 🛠️ Machine Learning Architecture

```
[ Live Data (yfinance) ] ➔ [ 55+ Technical Feature Engineering ]
                                       │
                         [ ExtraTrees Feature Selection ]
                                       │
            ┌──────────────┬───────────┴───┬──────────────┬──────────────┐
            ▼              ▼               ▼              ▼              ▼
     [RandomForest] [GradientBoost]   [XGBoost]      [LightGBM]   [LogisticReg]
            │              │               │              │              │
            └──────────────┴───────────┬───┴──────────────┴──────────────┘
                                       ▼
                         [ Stacking Meta-Learner (LR) ]
                                       │
               ┌───────────────────────┴───────────────────────┐
               ▼                                               ▼
   [ Tomorrow's Direction & % Conf ]              [ Price Target & ATR Bands ]
```

---

## 📁 Repository Structure

```
├── app.py                   # Main Flask web application & fast prediction engine
├── fetch_stocks.py          # Script to fetch/update list of all listed NSE stocks
├── nse_stocks.json          # Cached listing of 2,300+ NSE equity symbols & names
├── stock_predictor.py       # Baseline offline model training & evaluation script
├── stock_predictor_v2.py    # Enhanced model backtesting & Optuna hyperparameter tuning
├── predict_tomorrow.py      # CLI predictor tool (v1 baseline)
├── predict_tomorrow_v2.py   # CLI predictor tool (v2 with Optuna stack)
├── templates/
│   └── index.html           # Modern glassmorphism web interface with Chart.js
├── requirements.txt         # Production dependencies
└── README.md                # Documentation
```

---

## 🚀 Local Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/Arham21-pixel/stock-prediction.git
cd stock-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```

### 4. Open in Browser
Navigate to `http://localhost:5000`

---

## ☁️ Deployment Guide

### Deploying to Render.com (Recommended)

1. Create a free account on [Render.com](https://render.com).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository `Arham21-pixel/stock-prediction`.
4. Configure setting:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Create Web Service**!

---

## ⚠️ Disclaimer

*This application is built for educational and analytical purposes only. Stock predictions are based solely on quantitative historical technical indicators and do not account for breaking news, earnings reports, RBI policy changes, or macroeconomic events. **It is not financial advice. Always perform your own research before trading.***

---

## 📄 License

MIT License © 2026 StockSense India
