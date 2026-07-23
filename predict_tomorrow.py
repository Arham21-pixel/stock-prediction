"""
predict_tomorrow.py
===================
Uses the SAME pipeline as stock_predictor.py but trains on ALL available
historical data and then predicts the direction (UP / DOWN) for the
NEXT trading day from today.

No lookahead: the "today" row is the last row of downloaded data.
Its label (tomorrow's close) is unknown — that's what we predict.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
TICKER      = "HDFCBANK.NS"   # change to any yfinance ticker
START_DATE  = "2019-01-01"    # training history start
TODAY       = str(date.today())   # dynamically uses today's date
RANDOM_SEED = 42

print("=" * 65)
print(f"  Tomorrow's Direction Predictor")
print(f"  Ticker : {TICKER}")
print(f"  As of  : {TODAY}")
print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD DATA up to today
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[1] Downloading data from {START_DATE} to {TODAY} ...")

raw = yf.download(TICKER, start=START_DATE, end=TODAY, auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
df.ffill(inplace=True)
df.dropna(inplace=True)

print(f"    Downloaded {len(df)} trading days.")
print(f"    Latest data point : {df.index[-1].date()}  "
      f"Close = {df['Close'].iloc[-1]:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING (identical to stock_predictor.py)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2] Engineering features ...")

df["Return_1d"]  = df["Close"].pct_change()
for n in [3, 5, 10]:
    df[f"Return_{n}d"] = df["Close"].pct_change(n)

for w in [5, 10, 20]:
    df[f"MA_{w}"]          = df["Close"].rolling(w).mean()
    df[f"Price_vs_MA{w}"]  = df["Close"] / df[f"MA_{w}"] - 1

for w in [5, 10]:
    df[f"Vol_{w}d"] = df["Return_1d"].rolling(w).std()

df["Volume_chg"] = df["Volume"].pct_change()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["RSI_14"] = compute_rsi(df["Close"])

FEATURE_COLS = [
    "Return_1d",
    "Return_3d", "Return_5d", "Return_10d",
    "Price_vs_MA5", "Price_vs_MA10", "Price_vs_MA20",
    "Vol_5d", "Vol_10d",
    "Volume_chg",
    "RSI_14",
]

df.dropna(subset=FEATURE_COLS, inplace=True)

# Replace any infinity values (e.g. from pct_change on zero volume) with NaN, then drop
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df.dropna(subset=FEATURE_COLS, inplace=True)

print(f"    Features ready. {len(df)} rows after dropping NaNs/Infs.")



# ──────────────────────────────────────────────────────────────────────────────
# 3. SEPARATE "TODAY" FROM TRAINING DATA
#
# The LAST row = today. We compute its features but its label (tomorrow's
# close) is unknown. Everything before it is used for training.
# ──────────────────────────────────────────────────────────────────────────────
today_row   = df.iloc[[-1]]          # last row = today (features known)
train_df    = df.iloc[:-1].copy()    # all rows before today = training data

# Create labels for the training set: 1 if next day's close > today's close
train_df["Target"] = (train_df["Close"].shift(-1) > train_df["Close"]).astype(int)
train_df.dropna(subset=["Target"], inplace=True)

X_train = train_df[FEATURE_COLS].values
y_train = train_df["Target"].values
X_today = today_row[FEATURE_COLS].values   # what we'll predict on

print(f"\n[3] Training set: {train_df.index[0].date()} -> {train_df.index[-1].date()} "
      f"({len(train_df)} rows)")
print(f"    Predicting for: {today_row.index[0].date()} (today's data -> tomorrow's move)")


# ------------------------------------------------------------------------------
# 4. SCALE + TRAIN
#    Use TimeSeriesSplit grid-search, fit on all historical data.
# ------------------------------------------------------------------------------
print("\n[4] Scaling features and training models ...")

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_today_sc   = scaler.transform(X_today)

tscv = TimeSeriesSplit(n_splits=5)

# Random Forest
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
    param_grid={
        "n_estimators"    : [100, 200],
        "max_depth"       : [3, 5, 7],
        "min_samples_leaf": [5, 10],
    },
    cv=tscv, scoring="accuracy", n_jobs=-1, verbose=0,
)
rf_gs.fit(X_train_sc, y_train)
best_rf = rf_gs.best_estimator_
print(f"    RF  best params : {rf_gs.best_params_}  | CV acc: {rf_gs.best_score_:.4f}")

# Logistic Regression
lr_gs = GridSearchCV(
    LogisticRegression(random_state=RANDOM_SEED, class_weight="balanced", max_iter=1000),
    param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
    cv=tscv, scoring="accuracy", n_jobs=-1, verbose=0,
)
lr_gs.fit(X_train_sc, y_train)
best_lr = lr_gs.best_estimator_
print(f"    LR  best params : {lr_gs.best_params_}  | CV acc: {lr_gs.best_score_:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. PREDICT TOMORROW
# ──────────────────────────────────────────────────────────────────────────────
rf_pred    = best_rf.predict(X_today_sc)[0]
rf_proba   = best_rf.predict_proba(X_today_sc)[0]   # [P(DOWN), P(UP)]

lr_pred    = best_lr.predict(X_today_sc)[0]
lr_proba   = best_lr.predict_proba(X_today_sc)[0]

# Ensemble: average probabilities from both models
ensemble_up_prob = (rf_proba[1] + lr_proba[1]) / 2
ensemble_pred    = 1 if ensemble_up_prob >= 0.5 else 0

direction_map = {1: "UP   [^]", 0: "DOWN [v]"}
emoji_map     = {1: "[UP] ", 0: "[DN] "}

today_close = today_row["Close"].iloc[0]
today_date  = today_row.index[0].date()

print("\n" + "=" * 65)
print(f"  PREDICTION FOR NEXT TRADING DAY AFTER {today_date}")
print(f"  {TICKER}  |  Today's Close: Rs. {today_close:.2f}")
print("=" * 65)

print(f"\n  Random Forest      : {emoji_map[rf_pred]}  {direction_map[rf_pred]}"
      f"  (P(UP) = {rf_proba[1]*100:.1f}%,  P(DOWN) = {rf_proba[0]*100:.1f}%)")

print(f"  Logistic Regression: {emoji_map[lr_pred]}  {direction_map[lr_pred]}"
      f"  (P(UP) = {lr_proba[1]*100:.1f}%,  P(DOWN) = {lr_proba[0]*100:.1f}%)")

print(f"\n  -- ENSEMBLE (avg of both) --")
print(f"  {emoji_map[ensemble_pred]}  Predicted direction : {direction_map[ensemble_pred]}")
print(f"     Confidence UP   : {ensemble_up_prob*100:.1f}%")
print(f"     Confidence DOWN : {(1-ensemble_up_prob)*100:.1f}%")

print("\n" + "─" * 65)
print("  Today's feature snapshot:")
feat_vals = pd.Series(X_today[0], index=FEATURE_COLS)
for name, val in feat_vals.items():
    print(f"    {name:<22} : {val:+.4f}")

print("\n" + "─" * 65)
print("""
  ⚠  IMPORTANT DISCLAIMER
  This is a machine learning model trained on historical price
  patterns. It is NOT financial advice. Stock markets are
  influenced by news, macro events, and sentiment that this
  model does not capture.

  Model historical accuracy on test data: ~49-54%
  (barely above a coin flip — trade at your own risk)
""")
