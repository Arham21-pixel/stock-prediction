"""
Stock Movement Predictor
========================
Predicts whether a stock's closing price will go UP (1) or DOWN (0)
the next trading day — binary classification, not price regression.

Steps:
  1. Data download & cleaning
  2. Label creation (no lookahead leakage)
  3. Feature engineering
  4. Chronological train/test split
  5. Baselines (always-up, random)
  6. Model training (Random Forest + Logistic Regression) with TimeSeriesSplit CV
  7. Honest evaluation vs baselines
  8. Backtest (cumulative return vs buy-and-hold vs random)
  9. Honest summary
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — change the ticker or date range here
# ──────────────────────────────────────────────────────────────────────────────
TICKER      = "HDFCBANK.NS"
START_DATE  = "2019-01-01"
END_DATE    = "2024-12-31"
TRAIN_RATIO = 0.80          # first 80 % of dates → train
RANDOM_SEED = 42

print("=" * 70)
print(f"  Stock Movement Predictor  |  Ticker: {TICKER}  |  {START_DATE} -> {END_DATE}")
print("=" * 70)


# ======================================================================
# STEP 1 — DATA
# Download 5+ years of daily OHLCV data and do basic hygiene
# ======================================================================
print("\n--- STEP 1 : DATA ---------------------------------------------------")

raw = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

# Flatten MultiIndex columns if yfinance returns them (v0.2+ behaviour)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

# Keep only the OHLCV columns we need
df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

print(f"\nRaw data shape  : {df.shape}")
print(f"Date range      : {df.index[0].date()} -> {df.index[-1].date()}")
print(f"\nHead of raw data:\n{df.head()}")

# Drop rows where ALL OHLCV values are NaN
before = len(df)
df.dropna(how="all", inplace=True)

# Forward-fill any remaining isolated NaNs (e.g. a missing volume entry)
df.ffill(inplace=True)
df.dropna(inplace=True)

print(f"\nDropped {before - len(df)} all-NaN rows. Working shape: {df.shape}")


# ======================================================================
# STEP 2 — LABEL
# Binary target: 1 if NEXT day's close > today's close, else 0
# shift(-1) moves tomorrow's close into today's row — NO lookahead
# because we never use tomorrow's close as a feature.
# ======================================================================
print("\n--- STEP 2 : LABEL --------------------------------------------------")

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# The very last row has no "next day" -> drop it
df.dropna(subset=["Target"], inplace=True)
df["Target"] = df["Target"].astype(int)

up_pct   = df["Target"].mean() * 100
down_pct = 100 - up_pct
print(f"\nClass balance  ->  UP (1): {up_pct:.1f}%  |  DOWN (0): {down_pct:.1f}%")
print("(A perfectly balanced dataset would be 50/50; slight imbalance is normal)")


# ======================================================================
# STEP 3 — FEATURE ENGINEERING
# All features are derived ONLY from today-and-before data, no leakage.
# ======================================================================
print("\n--- STEP 3 : FEATURES -----------------------------------------------")

# Daily return (today's % change from yesterday)
df["Return_1d"] = df["Close"].pct_change()

# Multi-day rolling returns
for n in [3, 5, 10]:
    df[f"Return_{n}d"] = df["Close"].pct_change(n)

# Moving averages & price-relative-to-MA
for w in [5, 10, 20]:
    ma_col = f"MA_{w}"
    df[ma_col] = df["Close"].rolling(w).mean()
    df[f"Price_vs_MA{w}"] = df["Close"] / df[ma_col] - 1   # 0 = at the MA

# Rolling volatility (std of daily returns)
for w in [5, 10]:
    df[f"Vol_{w}d"] = df["Return_1d"].rolling(w).std()

# Volume % change vs yesterday
df["Volume_chg"] = df["Volume"].pct_change()

# RSI (14-day) — measures momentum
# Above 70 = overbought, below 30 = oversold
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["RSI_14"] = compute_rsi(df["Close"])

# Feature column list (no raw OHLCV, no target)
FEATURE_COLS = [
    "Return_1d",
    "Return_3d", "Return_5d", "Return_10d",
    "Price_vs_MA5", "Price_vs_MA10", "Price_vs_MA20",
    "Vol_5d", "Vol_10d",
    "Volume_chg",
    "RSI_14",
]

# Drop rows with NaNs introduced by rolling windows (early rows lack history)
before = len(df)
df.dropna(subset=FEATURE_COLS + ["Target"], inplace=True)
print(f"\nDropped {before - len(df)} rows with NaN from rolling windows.")
print(f"Final dataset shape: {df.shape}")
print(f"\nFeatures used ({len(FEATURE_COLS)} total):")
for f in FEATURE_COLS:
    print(f"  * {f}")


# ======================================================================
# STEP 4 — CHRONOLOGICAL SPLIT (NO SHUFFLING)
# Stock data is time-series: future rows must NEVER appear in training.
# Split by position: first 80% of dates -> train, last 20% -> test.
# ======================================================================
print("\n--- STEP 4 : TRAIN / TEST SPLIT -------------------------------------")

split_idx = int(len(df) * TRAIN_RATIO)

train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

X_train = train_df[FEATURE_COLS].values
y_train = train_df["Target"].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["Target"].values

print(f"\nTrain : {train_df.index[0].date()} -> {train_df.index[-1].date()}  "
      f"({len(train_df)} rows)")
print(f"Test  : {test_df.index[0].date()} -> {test_df.index[-1].date()}  "
      f"({len(test_df)} rows)")
print("\n[OK] No overlap -- train ends before test begins.")


# ======================================================================
# STEP 5 — BASELINES
# These set the bar. Any model that can't beat them is worthless.
# ======================================================================
print("\n--- STEP 5 : BASELINES ----------------------------------------------")

np.random.seed(RANDOM_SEED)

# Baseline A: always predict "up" (1) — exploits class imbalance
baseline_always_up  = np.ones(len(y_test), dtype=int)
acc_always_up       = accuracy_score(y_test, baseline_always_up)

# Baseline B: random coin flip (50/50)
baseline_random     = np.random.randint(0, 2, size=len(y_test))
acc_random          = accuracy_score(y_test, baseline_random)

print(f"\nBaseline A -- Always predict UP : {acc_always_up:.3f}  ({acc_always_up*100:.1f}%)")
print(f"Baseline B -- Random coin flip  : {acc_random:.3f}  ({acc_random*100:.1f}%)")
print("\nMy model must beat BOTH of these to be considered meaningful.")


# ======================================================================
# STEP 6 — MODEL TRAINING
# Scale features (fit on train only -> no leakage into test).
# Use TimeSeriesSplit for CV so folds respect time order.
# Grid-search hyper-parameters for each model.
# ======================================================================
print("\n--- STEP 6 : MODEL TRAINING -----------------------------------------")

# Scale: fit ONLY on train set, then transform both
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

# TimeSeriesSplit: 5 folds respecting temporal order
tscv = TimeSeriesSplit(n_splits=5)

# -- 6a. Random Forest ---------------------------------------------------
print("\n[6a] Random Forest -- grid search with TimeSeriesSplit ...")

rf_param_grid = {
    "n_estimators"    : [100, 200, 300],
    "max_depth"       : [3, 5, 7, None],
    "min_samples_leaf": [5, 10, 20],
}

rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
    param_grid = rf_param_grid,
    cv         = tscv,
    scoring    = "accuracy",
    n_jobs     = -1,
    verbose    = 0,
)
rf_gs.fit(X_train_sc, y_train)
best_rf = rf_gs.best_estimator_

print(f"  Best RF params : {rf_gs.best_params_}")
print(f"  Best CV score  : {rf_gs.best_score_:.4f}")

# -- 6b. Logistic Regression ---------------------------------------------
print("\n[6b] Logistic Regression -- grid search with TimeSeriesSplit ...")

lr_param_grid = {
    "C"       : [0.001, 0.01, 0.1, 1.0, 10.0],
    "penalty" : ["l2"],
    "solver"  : ["lbfgs"],
    "max_iter": [1000],
}

lr_gs = GridSearchCV(
    LogisticRegression(random_state=RANDOM_SEED, class_weight="balanced"),
    param_grid = lr_param_grid,
    cv         = tscv,
    scoring    = "accuracy",
    n_jobs     = -1,
    verbose    = 0,
)
lr_gs.fit(X_train_sc, y_train)
best_lr = lr_gs.best_estimator_

print(f"  Best LR params : {lr_gs.best_params_}")
print(f"  Best CV score  : {lr_gs.best_score_:.4f}")


# ======================================================================
# STEP 7 — HONEST EVALUATION
# Report full metrics on the held-out test set and compare to baselines.
# ======================================================================
print("\n--- STEP 7 : EVALUATION ---------------------------------------------")

def evaluate(name, y_true, y_pred):
    return {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
    }

y_pred_rf = best_rf.predict(X_test_sc)
y_pred_lr = best_lr.predict(X_test_sc)

results = pd.DataFrame([
    evaluate("Always UP (baseline A)",   y_test, baseline_always_up),
    evaluate("Random flip (baseline B)", y_test, baseline_random),
    evaluate("Random Forest",            y_test, y_pred_rf),
    evaluate("Logistic Regression",      y_test, y_pred_lr),
])
results = results.set_index("Model")

print("\n-- Metrics comparison table --")
print(results.to_string(float_format="{:.4f}".format))

# Confusion matrices side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f"Confusion Matrices -- {TICKER}", fontsize=14, fontweight="bold")
for ax, (name, y_pred) in zip(axes, [
    ("Random Forest",       y_pred_rf),
    ("Logistic Regression", y_pred_lr),
]):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN (0)", "UP (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()
print("\n[OK] Saved confusion_matrices.png")

# Feature importance (Random Forest)
importances = pd.Series(best_rf.feature_importances_, index=FEATURE_COLS)
importances.sort_values(ascending=True, inplace=True)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="steelblue", edgecolor="white")
plt.title(f"Random Forest -- Feature Importances ({TICKER})", fontweight="bold")
plt.xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("[OK] Saved feature_importance.png")

print("\nTop 5 features:")
for feat, imp in importances.sort_values(ascending=False).head(5).items():
    print(f"  {feat:<25}  {imp:.4f}")


# ======================================================================
# STEP 8 — BACKTEST
# Simulate trading on the TEST set only (no peeking at training period).
# Strategy: if model predicts UP -> buy (hold) that day; else stay cash.
# Daily return for a "hold" day = actual_close[t+1] / actual_close[t] - 1
# ======================================================================
print("\n--- STEP 8 : BACKTEST -----------------------------------------------")

# Next-day return for each row in the test window
next_day_return = test_df["Return_1d"].shift(-1).fillna(0).values

np.random.seed(RANDOM_SEED)
random_signal = np.random.randint(0, 2, size=len(y_test))

# Compute daily P&L for each strategy (0 return on cash days)
strat_rf    = np.where(y_pred_rf   == 1, next_day_return, 0)
strat_lr    = np.where(y_pred_lr   == 1, next_day_return, 0)
strat_rnd   = np.where(random_signal == 1, next_day_return, 0)
strat_bnh   = next_day_return                       # buy and hold every day

# Cumulative returns (start at 1.0 = $1 invested)
cum_rf    = (1 + strat_rf).cumprod()
cum_lr    = (1 + strat_lr).cumprod()
cum_rnd   = (1 + strat_rnd).cumprod()
cum_bnh   = (1 + strat_bnh).cumprod()

dates   = test_df.index[:-1]   # drop last row (no next-day return)
cum_rf  = cum_rf[:-1]
cum_lr  = cum_lr[:-1]
cum_rnd = cum_rnd[:-1]
cum_bnh = cum_bnh[:-1]

plt.figure(figsize=(12, 6))
plt.plot(dates, cum_bnh, label="Buy & Hold",           color="#2196F3", linewidth=2)
plt.plot(dates, cum_rf,  label="Random Forest",        color="#4CAF50", linewidth=2)
plt.plot(dates, cum_lr,  label="Logistic Regression",  color="#FF9800", linewidth=1.5, linestyle="--")
plt.plot(dates, cum_rnd, label="Random Strategy",      color="#9E9E9E", linewidth=1,   linestyle=":")
plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Starting capital")
plt.title(f"{TICKER} -- Backtest: Cumulative Return on Test Set", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (start = 1.0)")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("backtest.png", dpi=150)
plt.close()
print("[OK] Saved backtest.png")

final_rf  = cum_rf[-1]
final_lr  = cum_lr[-1]
final_rnd = cum_rnd[-1]
final_bnh = cum_bnh[-1]

print(f"\nFinal cumulative return on test set:")
print(f"  Buy & Hold          : {final_bnh:.4f}  ({(final_bnh-1)*100:+.1f}%)")
print(f"  Random Forest       : {final_rf:.4f}  ({(final_rf-1)*100:+.1f}%)")
print(f"  Logistic Regression : {final_lr:.4f}  ({(final_lr-1)*100:+.1f}%)")
print(f"  Random Strategy     : {final_rnd:.4f}  ({(final_rnd-1)*100:+.1f}%)")


# ======================================================================
# STEP 9 — HONEST SUMMARY
# State plainly what worked, what didn't, and what it means.
# ======================================================================
print("\n--- STEP 9 : HONEST SUMMARY -----------------------------------------\n")

acc_rf = results.loc["Random Forest", "Accuracy"]
acc_lr = results.loc["Logistic Regression", "Accuracy"]

best_model_acc  = max(acc_rf, acc_lr)
best_model_name = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"
best_model_ret  = final_rf if acc_rf >= acc_lr else final_lr

beat_baseline_A = best_model_acc > acc_always_up
beat_baseline_B = best_model_acc > acc_random
beat_bnh        = best_model_ret > final_bnh

print(f"  Ticker examined   : {TICKER}")
print(f"  Test period       : {test_df.index[0].date()} -> {test_df.index[-1].date()}")
print(f"  Best model        : {best_model_name}  (accuracy {best_model_acc:.4f})")
print()
print(f"  Did the model beat Baseline A (always UP, {acc_always_up:.4f})?")
print(f"    {'YES' if beat_baseline_A else 'NO'}  -- model accuracy: {best_model_acc:.4f}")
print()
print(f"  Did the model beat Baseline B (random flip, {acc_random:.4f})?")
print(f"    {'YES' if beat_baseline_B else 'NO'}  -- model accuracy: {best_model_acc:.4f}")
print()
print(f"  Did the model beat Buy-and-Hold in the backtest?")
print(f"    {'YES' if beat_bnh else 'NO'}  -- model return: {(best_model_ret-1)*100:+.1f}%  "
      f"vs B&H: {(final_bnh-1)*100:+.1f}%")

print("""
  INTERPRETATION
  --------------
  Beating both baselines on ACCURACY is the minimum bar for
  "the model learned something." Beating buy-and-hold on
  RETURNS is a much harder bar -- most academic literature
  shows that transaction-cost-free, data-mined ML strategies
  rarely beat buy-and-hold consistently out-of-sample.

  Key caveats:
  * This simulation has ZERO transaction costs / slippage.
  * It uses close-to-close returns; real execution uses open.
  * A single ticker / one test period is not robust evidence.
  * Past outperformance does NOT guarantee future results.
""")

print("Output files generated:")
print("  confusion_matrices.png")
print("  feature_importance.png")
print("  backtest.png")
print("\nDone.\n")
