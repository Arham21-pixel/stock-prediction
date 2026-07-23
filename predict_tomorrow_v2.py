"""
predict_tomorrow_v2.py  —  Real-Time Direction Predictor (Enhanced)
====================================================================
Uses the same 30+ feature pipeline as stock_predictor_v2.py.
Trains on ALL historical data up to today, then predicts tomorrow's
direction via a 4-model stacking ensemble.

Usage:  python predict_tomorrow_v2.py
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import yfinance as yf
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ──────────────────────────────────────────────────────────────────────────────
TICKER        = "HDFCBANK.NS"
START_DATE    = "2019-01-01"
TODAY         = str(date.today())
RANDOM_SEED   = 42
OPTUNA_TRIALS = 40      # reduce for speed; increase for accuracy
TSCV_SPLITS   = 5

print("=" * 65)
print(f"  Tomorrow's Predictor v2 (Enhanced)  |  {TICKER}")
print(f"  As of: {TODAY}")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[1] Downloading {TICKER} from {START_DATE} to {TODAY} ...")
raw = yf.download(TICKER, start=START_DATE, end=TODAY, auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
df.ffill(inplace=True)
df.dropna(inplace=True)
print(f"    {len(df)} rows. Latest: {df.index[-1].date()}  Close={df['Close'].iloc[-1]:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING (30+ indicators, identical to v2 training script)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2] Engineering features ...")

Close  = df["Close"]
High   = df["High"]
Low    = df["Low"]
Open_  = df["Open"]
Volume = df["Volume"]

df["Return_1d"]  = Close.pct_change()
df["LogRet_1d"]  = np.log(Close / Close.shift(1))
for n in [2, 3, 5, 10, 20]:
    df[f"Return_{n}d"] = Close.pct_change(n)
df["Gap"] = (Open_ / Close.shift(1)) - 1

for w in [5, 10, 20, 50]:
    df[f"SMA_{w}"]      = Close.rolling(w).mean()
    df[f"EMA_{w}"]      = Close.ewm(span=w, adjust=False).mean()
    df[f"Pr_vs_SMA{w}"] = Close / df[f"SMA_{w}"] - 1
    df[f"Pr_vs_EMA{w}"] = Close / df[f"EMA_{w}"] - 1

ema12 = Close.ewm(span=12, adjust=False).mean()
ema26 = Close.ewm(span=26, adjust=False).mean()
df["MACD_line"]   = ema12 - ema26
df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
df["MACD_hist"]   = df["MACD_line"] - df["MACD_signal"]

def rsi(series, period):
    d = series.diff()
    g = d.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    l = (-d.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

for p in [7, 14, 21]:
    df[f"RSI_{p}"] = rsi(Close, p)

bb_mid   = Close.rolling(20).mean()
bb_std   = Close.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
df["BB_pct"]   = (Close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
df["BB_width"] = (bb_upper - bb_lower) / bb_mid

low14  = Low.rolling(14).min()
high14 = High.rolling(14).max()
df["Stoch_K"] = 100 * (Close - low14) / (high14 - low14).replace(0, np.nan)
df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

hl = High - Low
hc = (High - Close.shift(1)).abs()
lc = (Low  - Close.shift(1)).abs()
tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
df["ATR_pct"] = tr.ewm(span=14, adjust=False).mean() / Close

typical = (High + Low + Close) / 3
df["CCI"]   = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).std())
df["WillR"] = -100 * (high14 - Close) / (high14 - low14).replace(0, np.nan)
df["Mom_10"] = Close / Close.shift(10) - 1
df["Mom_20"] = Close / Close.shift(20) - 1

for w in [5, 10, 20]:
    df[f"Vol_{w}d"] = df["Return_1d"].rolling(w).std()

df["Vol_chg"]     = Volume.pct_change()
df["Vol_vs_MA10"] = Volume / Volume.rolling(10).mean() - 1
df["OBV_mom5"]    = np.sign(Close.diff()).rolling(5).sum() / 5
df["ForceIdx"]    = (Close.diff() * Volume) / (Close * Volume.rolling(10).mean().replace(0, np.nan))

df["Dist_52w_high"] = Close / High.rolling(252).max() - 1
df["Dist_52w_low"]  = Close / Low.rolling(252).min()  - 1
df["Body_size"]     = (Close - Open_).abs() / Open_.replace(0, np.nan)
df["Inside_day"]    = ((High < High.shift(1)) & (Low > Low.shift(1))).astype(int)
df["DayOfWeek"]     = df.index.dayofweek
df["Month"]         = df.index.month

FEATURE_COLS = [
    "Return_1d", "LogRet_1d",
    "Return_2d", "Return_3d", "Return_5d", "Return_10d", "Return_20d", "Gap",
    "Pr_vs_SMA5","Pr_vs_SMA10","Pr_vs_SMA20","Pr_vs_SMA50",
    "Pr_vs_EMA5","Pr_vs_EMA10","Pr_vs_EMA20","Pr_vs_EMA50",
    "MACD_line","MACD_signal","MACD_hist",
    "RSI_7","RSI_14","RSI_21",
    "Stoch_K","Stoch_D","WillR","CCI",
    "Mom_10","Mom_20",
    "BB_pct","BB_width","ATR_pct",
    "Vol_5d","Vol_10d","Vol_20d",
    "Vol_chg","Vol_vs_MA10","OBV_mom5","ForceIdx",
    "Dist_52w_high","Dist_52w_low",
    "Body_size","Inside_day","DayOfWeek","Month",
]

df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df.dropna(subset=FEATURE_COLS, inplace=True)
print(f"    {len(df)} rows ready. {len(FEATURE_COLS)} features.")

# ──────────────────────────────────────────────────────────────────────────────
# 3. SEPARATE TODAY FROM TRAINING DATA
# ──────────────────────────────────────────────────────────────────────────────
today_row = df.iloc[[-1]]
train_df  = df.iloc[:-1].copy()
train_df["Target"] = (train_df["Close"].shift(-1) > train_df["Close"]).astype(int)
train_df.dropna(subset=["Target"], inplace=True)

X_train = train_df[FEATURE_COLS].values
y_train = train_df["Target"].values
X_today = today_row[FEATURE_COLS].values

print(f"\n[3] Training: {train_df.index[0].date()} -> {train_df.index[-1].date()} "
      f"({len(train_df)} rows)")
print(f"    Predicting for next trading day after: {today_row.index[0].date()}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. SCALE + FEATURE SELECTION
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4] Scaling and selecting features ...")
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_td_sc  = scaler.transform(X_today)

sel_rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                 random_state=RANDOM_SEED, n_jobs=-1)
sel_rf.fit(X_tr_sc, y_train)
imp       = pd.Series(sel_rf.feature_importances_, index=FEATURE_COLS)
keep_mask = imp >= 0.01
feat_idx  = [FEATURE_COLS.index(f) for f in imp[keep_mask].index]

X_tr_sel  = X_tr_sc[:, feat_idx]
X_td_sel  = X_td_sc[:, feat_idx]
print(f"    Kept {len(feat_idx)} / {len(FEATURE_COLS)} features.")

# ──────────────────────────────────────────────────────────────────────────────
# 5. OPTUNA HYPERPARAMETER SEARCH
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[5] Optuna tuning ({OPTUNA_TRIALS} trials each) ...")
tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS)

def cv_acc(model, X, y):
    scores = []
    for tr, va in tscv.split(X):
        model.fit(X[tr], y[tr])
        scores.append(accuracy_score(y[va], model.predict(X[va])))
    return np.mean(scores)

# XGBoost
def xgb_obj(trial):
    return cv_acc(XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators",100,500),
        max_depth=trial.suggest_int("max_depth",2,8),
        learning_rate=trial.suggest_float("learning_rate",0.005,0.2,log=True),
        subsample=trial.suggest_float("subsample",0.5,1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree",0.4,1.0),
        reg_alpha=trial.suggest_float("reg_alpha",1e-4,10.0,log=True),
        reg_lambda=trial.suggest_float("reg_lambda",1e-4,10.0,log=True),
        min_child_weight=trial.suggest_int("min_child_weight",1,20),
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    ), X_tr_sel, y_train)

xgb_study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
xgb_study.optimize(xgb_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_xgb = XGBClassifier(**xgb_study.best_params, use_label_encoder=False,
    eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
best_xgb.fit(X_tr_sel, y_train)
print(f"    XGBoost  CV acc: {xgb_study.best_value:.4f}")

# LightGBM
def lgbm_obj(trial):
    return cv_acc(LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators",100,500),
        max_depth=trial.suggest_int("max_depth",2,8),
        learning_rate=trial.suggest_float("learning_rate",0.005,0.2,log=True),
        num_leaves=trial.suggest_int("num_leaves",8,128),
        subsample=trial.suggest_float("subsample",0.5,1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree",0.4,1.0),
        reg_alpha=trial.suggest_float("reg_alpha",1e-4,10.0,log=True),
        reg_lambda=trial.suggest_float("reg_lambda",1e-4,10.0,log=True),
        min_child_samples=trial.suggest_int("min_child_samples",5,50),
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    ), X_tr_sel, y_train)

lgbm_study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
lgbm_study.optimize(lgbm_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_lgbm = LGBMClassifier(**lgbm_study.best_params,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
best_lgbm.fit(X_tr_sel, y_train)
print(f"    LightGBM CV acc: {lgbm_study.best_value:.4f}")

# Random Forest
def rf_obj(trial):
    return cv_acc(RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators",100,400),
        max_depth=trial.suggest_int("max_depth",2,10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf",2,30),
        max_features=trial.suggest_float("max_features",0.3,1.0),
        random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1,
    ), X_tr_sel, y_train)

rf_study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
rf_study.optimize(rf_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_rf = RandomForestClassifier(**rf_study.best_params,
    random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1)
best_rf.fit(X_tr_sel, y_train)
print(f"    RF       CV acc: {rf_study.best_value:.4f}")

# Logistic Regression
def lr_obj(trial):
    return cv_acc(LogisticRegression(
        C=trial.suggest_float("C",1e-4,100.0,log=True),
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_SEED, solver="lbfgs",
    ), X_tr_sel, y_train)

lr_study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
lr_study.optimize(lr_obj, n_trials=30, show_progress_bar=False)
best_lr = LogisticRegression(C=lr_study.best_params["C"], max_iter=2000,
    class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs")
best_lr.fit(X_tr_sel, y_train)
print(f"    LR       CV acc: {lr_study.best_value:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. STACKING ENSEMBLE — OOF training, then predict today
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6] Stacking ensemble ...")
base_models = [("XGB", best_xgb), ("LGBM", best_lgbm),
               ("RF",  best_rf),  ("LR",   best_lr)]

oof_train = np.zeros((len(X_tr_sel), len(base_models)))
today_preds = np.zeros(len(base_models))

for i, (name, model) in enumerate(base_models):
    oof = np.zeros(len(X_tr_sel))
    for tr_idx, va_idx in tscv.split(X_tr_sel):
        model.fit(X_tr_sel[tr_idx], y_train[tr_idx])
        oof[va_idx] = model.predict_proba(X_tr_sel[va_idx])[:, 1]
    oof_train[:, i] = oof
    model.fit(X_tr_sel, y_train)
    today_preds[i] = model.predict_proba(X_td_sel)[0, 1]

meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
meta.fit(oof_train, y_train)

# ──────────────────────────────────────────────────────────────────────────────
# 7. FINAL PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
today_date  = today_row.index[0].date()
today_close = today_row["Close"].iloc[0]

# Individual probabilities
probs = {name: today_preds[i] for i, (name, _) in enumerate(base_models)}
# Stack prediction
stack_up_prob = meta.predict_proba(today_preds.reshape(1, -1))[0, 1]
# Soft average
avg_up_prob   = today_preds.mean()

direction = lambda p: ("UP   [^]  " if p >= 0.5 else "DOWN [v]  ")
signal    = lambda p: ("[UP] " if p >= 0.5 else "[DN] ")

print("\n" + "=" * 65)
print(f"  PREDICTION FOR NEXT TRADING DAY AFTER {today_date}")
print(f"  {TICKER}  |  Last Close: Rs. {today_close:.2f}")
print("=" * 65)

print(f"\n  Individual models:")
for name, p in probs.items():
    print(f"    {signal(p)} {name:<12} {direction(p)}  P(UP)={p*100:.1f}%  P(DN)={(1-p)*100:.1f}%")

print(f"\n  -- Soft Average Ensemble --")
print(f"    {signal(avg_up_prob)} {direction(avg_up_prob)}  P(UP)={avg_up_prob*100:.1f}%")

print(f"\n  -- STACK Ensemble (recommended) --")
print(f"    {signal(stack_up_prob)} {direction(stack_up_prob)}  P(UP)={stack_up_prob*100:.1f}%  "
      f"P(DN)={(1-stack_up_prob)*100:.1f}%")

print("\n" + "-" * 65)
print("  Today's key features:")
key_feats = ["Return_1d","Return_5d","RSI_14","MACD_hist",
             "Stoch_K","BB_pct","ATR_pct","Vol_chg","OBV_mom5"]
for f in key_feats:
    if f in FEATURE_COLS:
        idx = FEATURE_COLS.index(f)
        val = X_today[0, idx]
        print(f"    {f:<22} : {val:+.4f}")

print("""
  WARNING: This model has ~55-58% historical accuracy.
  It does NOT account for news, earnings, RBI policy, or macro events.
  NOT financial advice. Trade at your own risk.
""")
