"""
stock_predictor_v2.py  —  Enhanced Stock Movement Predictor
=============================================================
Upgrades over v1:
  1. 30+ technical indicators (MACD, Bollinger Bands, ATR, Stochastic,
     OBV, CCI, Williams %R, momentum, calendar, price-pattern features)
  2. XGBoost + LightGBM on top of Random Forest + Logistic Regression
  3. Optuna Bayesian hyperparameter search (much smarter than GridSearch)
  4. Stacking ensemble: base model OOF predictions → meta-learner
  5. Auto feature selection (drop low-importance noisy features)
  6. Honest evaluation: full metrics table + backtest

Realistic accuracy ceiling for liquid large-cap stocks: ~55–60%.
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
TICKER       = "HDFCBANK.NS"
START_DATE   = "2019-01-01"
END_DATE     = "2024-12-31"
TRAIN_RATIO  = 0.80
RANDOM_SEED  = 42
OPTUNA_TRIALS = 60        # Bayesian trials per model (higher = better, slower)
TSCV_SPLITS   = 5         # TimeSeriesSplit folds

print("=" * 70)
print(f"  Enhanced Stock Predictor v2  |  {TICKER}  |  {START_DATE} -> {END_DATE}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 1 : DATA ---------------------------------------------------")

raw = yf.download(TICKER, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
df.ffill(inplace=True)
df.dropna(inplace=True)

print(f"Shape: {df.shape}  |  {df.index[0].date()} -> {df.index[-1].date()}")
print(df.head(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LABEL  (same as v1, no lookahead)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 2 : LABEL --------------------------------------------------")

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(subset=["Target"], inplace=True)
df["Target"] = df["Target"].astype(int)

up  = df["Target"].mean() * 100
print(f"Class balance  ->  UP: {up:.1f}%  |  DOWN: {100-up:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING (30+ indicators)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 3 : FEATURE ENGINEERING ------------------------------------")

Close  = df["Close"]
High   = df["High"]
Low    = df["Low"]
Open_  = df["Open"]
Volume = df["Volume"]

# ── 3a. Returns ──────────────────────────────────────────────────────────────
df["Return_1d"]  = Close.pct_change()
df["LogRet_1d"]  = np.log(Close / Close.shift(1))
for n in [2, 3, 5, 10, 20]:
    df[f"Return_{n}d"] = Close.pct_change(n)

# Gap: today's open vs yesterday's close (overnight sentiment)
df["Gap"] = (Open_ / Close.shift(1)) - 1

# ── 3b. Moving averages (SMA & EMA) + price-relative ─────────────────────────
for w in [5, 10, 20, 50]:
    df[f"SMA_{w}"]        = Close.rolling(w).mean()
    df[f"EMA_{w}"]        = Close.ewm(span=w, adjust=False).mean()
    df[f"Pr_vs_SMA{w}"]   = Close / df[f"SMA_{w}"] - 1
    df[f"Pr_vs_EMA{w}"]   = Close / df[f"EMA_{w}"] - 1

# ── 3c. MACD (12-26-9) ──────────────────────────────────────────────────────
ema12 = Close.ewm(span=12, adjust=False).mean()
ema26 = Close.ewm(span=26, adjust=False).mean()
df["MACD_line"]   = ema12 - ema26
df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
df["MACD_hist"]   = df["MACD_line"] - df["MACD_signal"]

# ── 3d. RSI (7, 14, 21) ─────────────────────────────────────────────────────
def rsi(series, period):
    d   = series.diff()
    g   = d.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    l   = (-d.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

for p in [7, 14, 21]:
    df[f"RSI_{p}"] = rsi(Close, p)

# ── 3e. Bollinger Bands (20-day) ─────────────────────────────────────────────
bb_mid   = Close.rolling(20).mean()
bb_std   = Close.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
df["BB_pct"]   = (Close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)  # %B
df["BB_width"] = (bb_upper - bb_lower) / bb_mid                                  # bandwidth

# ── 3f. Stochastic Oscillator %K and %D (14-day) ─────────────────────────────
low14  = Low.rolling(14).min()
high14 = High.rolling(14).max()
df["Stoch_K"] = 100 * (Close - low14) / (high14 - low14).replace(0, np.nan)
df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

# ── 3g. ATR — Average True Range (14-day), normalised by Close ───────────────
hl   = High - Low
hc   = (High - Close.shift(1)).abs()
lc   = (Low  - Close.shift(1)).abs()
tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
df["ATR_pct"] = tr.ewm(span=14, adjust=False).mean() / Close  # fraction of price

# ── 3h. CCI — Commodity Channel Index (20-day) ───────────────────────────────
typical = (High + Low + Close) / 3
df["CCI"] = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).std())

# ── 3i. Williams %R (14-day) ─────────────────────────────────────────────────
df["WillR"] = -100 * (high14 - Close) / (high14 - low14).replace(0, np.nan)

# ── 3j. Momentum (raw Close ratio) ───────────────────────────────────────────
df["Mom_10"] = Close / Close.shift(10) - 1
df["Mom_20"] = Close / Close.shift(20) - 1

# ── 3k. Rolling volatility (std of returns) ──────────────────────────────────
for w in [5, 10, 20]:
    df[f"Vol_{w}d"] = df["Return_1d"].rolling(w).std()

# ── 3l. Volume features ───────────────────────────────────────────────────────
df["Vol_chg"]      = Volume.pct_change()
df["Vol_vs_MA10"]  = Volume / Volume.rolling(10).mean() - 1
# OBV momentum: 5-day change in on-balance-volume direction
obv_dir = np.sign(Close.diff())                          # +1 up day, -1 down day
df["OBV_mom5"] = obv_dir.rolling(5).sum() / 5           # fraction of last 5 days up

# Force Index: 1-day price change × volume (normalised)
df["ForceIdx"] = (Close.diff() * Volume) / (Close * Volume.rolling(10).mean().replace(0, np.nan))

# ── 3m. Distance from 52-week high / low ─────────────────────────────────────
df["Dist_52w_high"] = Close / High.rolling(252).max() - 1
df["Dist_52w_low"]  = Close / Low.rolling(252).min()  - 1

# ── 3n. Price-pattern / calendar features ────────────────────────────────────
df["Body_size"]  = (Close - Open_).abs() / Open_.replace(0, np.nan)  # candle body
df["Inside_day"] = ((High < High.shift(1)) & (Low > Low.shift(1))).astype(int)
df["DayOfWeek"]  = df.index.dayofweek          # 0=Mon … 4=Fri
df["Month"]      = df.index.month              # 1–12

# ── Final feature list ────────────────────────────────────────────────────────
FEATURE_COLS = [
    # returns
    "Return_1d", "LogRet_1d",
    "Return_2d", "Return_3d", "Return_5d", "Return_10d", "Return_20d",
    "Gap",
    # trend
    "Pr_vs_SMA5",  "Pr_vs_SMA10",  "Pr_vs_SMA20",  "Pr_vs_SMA50",
    "Pr_vs_EMA5",  "Pr_vs_EMA10",  "Pr_vs_EMA20",  "Pr_vs_EMA50",
    # MACD
    "MACD_line", "MACD_signal", "MACD_hist",
    # momentum oscillators
    "RSI_7", "RSI_14", "RSI_21",
    "Stoch_K", "Stoch_D",
    "WillR",
    "CCI",
    "Mom_10", "Mom_20",
    # volatility
    "BB_pct", "BB_width",
    "ATR_pct",
    "Vol_5d", "Vol_10d", "Vol_20d",
    # volume
    "Vol_chg", "Vol_vs_MA10", "OBV_mom5", "ForceIdx",
    # structure
    "Dist_52w_high", "Dist_52w_low",
    "Body_size", "Inside_day",
    "DayOfWeek", "Month",
]

# Sanitise: replace inf with NaN, then drop rows missing any feature
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
before = len(df)
df.dropna(subset=FEATURE_COLS + ["Target"], inplace=True)
print(f"Dropped {before - len(df)} rows with NaN/Inf.  Final shape: {df.shape}")
print(f"Total features: {len(FEATURE_COLS)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CHRONOLOGICAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 4 : TRAIN / TEST SPLIT -------------------------------------")

split_idx = int(len(df) * TRAIN_RATIO)
train_df  = df.iloc[:split_idx]
test_df   = df.iloc[split_idx:]

X_train = train_df[FEATURE_COLS].values
y_train = train_df["Target"].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["Target"].values

print(f"Train: {train_df.index[0].date()} -> {train_df.index[-1].date()}  ({len(train_df)} rows)")
print(f"Test : {test_df.index[0].date()} -> {test_df.index[-1].date()}  ({len(test_df)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — BASELINES
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 5 : BASELINES ----------------------------------------------")

np.random.seed(RANDOM_SEED)
bl_up  = np.ones(len(y_test), dtype=int)
bl_rnd = np.random.randint(0, 2, size=len(y_test))
acc_up  = accuracy_score(y_test, bl_up)
acc_rnd = accuracy_score(y_test, bl_rnd)
print(f"Always UP  : {acc_up:.4f}")
print(f"Random     : {acc_rnd:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SCALE + FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 6 : SCALING & FEATURE SELECTION ----------------------------")

scaler     = StandardScaler()
X_tr_sc    = scaler.fit_transform(X_train)
X_te_sc    = scaler.transform(X_test)

# Use a quick RF to rank features; drop those with < 1% importance
print("Running feature selection via Random Forest importance ...")
selector_rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                     random_state=RANDOM_SEED,
                                     class_weight="balanced", n_jobs=-1)
selector_rf.fit(X_tr_sc, y_train)

importances    = pd.Series(selector_rf.feature_importances_, index=FEATURE_COLS)
keep_mask      = importances >= 0.01                          # drop features < 1% importance
selected_feats = importances[keep_mask].sort_values(ascending=False)

print(f"Features kept : {keep_mask.sum()} / {len(FEATURE_COLS)}")
print("Top 10 features:")
for f, v in selected_feats.head(10).items():
    print(f"  {f:<25} {v:.4f}")

# Re-slice to selected features only
feat_idx   = [FEATURE_COLS.index(f) for f in selected_feats.index]
X_tr_sel   = X_tr_sc[:, feat_idx]
X_te_sel   = X_te_sc[:, feat_idx]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — OPTUNA HYPERPARAMETER SEARCH
# Bayesian optimization with TimeSeriesSplit folds — smarter than GridSearch
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n--- STEP 7 : OPTUNA TUNING ({OPTUNA_TRIALS} trials each) -----------")

tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS)

def cv_acc(model, X, y):
    """Cross-validated accuracy using TimeSeriesSplit."""
    scores = []
    for tr, va in tscv.split(X):
        model.fit(X[tr], y[tr])
        scores.append(accuracy_score(y[va], model.predict(X[va])))
    return np.mean(scores)

# ── 7a. XGBoost ──────────────────────────────────────────────────────────────
print("\n[7a] XGBoost ...")
def xgb_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 600),
        max_depth         = trial.suggest_int("max_depth", 2, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
        verbosity         = 0,
    )
    return cv_acc(XGBClassifier(**params), X_tr_sel, y_train)

xgb_study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_xgb = XGBClassifier(**xgb_study.best_params,
                          use_label_encoder=False, eval_metric="logloss",
                          random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
best_xgb.fit(X_tr_sel, y_train)
print(f"  Best CV acc: {xgb_study.best_value:.4f}  |  params: {xgb_study.best_params}")

# ── 7b. LightGBM ─────────────────────────────────────────────────────────────
print("\n[7b] LightGBM ...")
def lgbm_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 600),
        max_depth         = trial.suggest_int("max_depth", 2, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 8, 128),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
        verbose           = -1,
    )
    return cv_acc(LGBMClassifier(**params), X_tr_sel, y_train)

lgbm_study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
lgbm_study.optimize(lgbm_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_lgbm = LGBMClassifier(**lgbm_study.best_params,
                            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
best_lgbm.fit(X_tr_sel, y_train)
print(f"  Best CV acc: {lgbm_study.best_value:.4f}  |  params: {lgbm_study.best_params}")

# ── 7c. Random Forest ─────────────────────────────────────────────────────────
print("\n[7c] Random Forest ...")
def rf_objective(trial):
    params = dict(
        n_estimators    = trial.suggest_int("n_estimators", 100, 500),
        max_depth       = trial.suggest_int("max_depth", 2, 10),
        min_samples_leaf= trial.suggest_int("min_samples_leaf", 2, 30),
        max_features    = trial.suggest_float("max_features", 0.3, 1.0),
        random_state    = RANDOM_SEED,
        class_weight    = "balanced",
        n_jobs          = -1,
    )
    return cv_acc(RandomForestClassifier(**params), X_tr_sel, y_train)

rf_study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
rf_study.optimize(rf_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_rf = RandomForestClassifier(**rf_study.best_params,
                                  random_state=RANDOM_SEED,
                                  class_weight="balanced", n_jobs=-1)
best_rf.fit(X_tr_sel, y_train)
print(f"  Best CV acc: {rf_study.best_value:.4f}  |  params: {rf_study.best_params}")

# ── 7d. Logistic Regression ───────────────────────────────────────────────────
print("\n[7d] Logistic Regression ...")
def lr_objective(trial):
    C = trial.suggest_float("C", 1e-4, 100.0, log=True)
    return cv_acc(LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_SEED, solver="lbfgs"),
                  X_tr_sel, y_train)

lr_study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
lr_study.optimize(lr_objective, n_trials=30, show_progress_bar=False)  # LR needs fewer trials
best_lr = LogisticRegression(C=lr_study.best_params["C"], max_iter=2000,
                              class_weight="balanced", random_state=RANDOM_SEED,
                              solver="lbfgs")
best_lr.fit(X_tr_sel, y_train)
print(f"  Best CV acc: {lr_study.best_value:.4f}  |  C={lr_study.best_params['C']:.5f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — STACKING ENSEMBLE
# Generate Out-of-Fold (OOF) predictions from base models on the training set,
# then train a meta-learner (Logistic Regression) on those OOF predictions.
# This prevents the meta-learner from seeing the training data directly.
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 8 : STACKING ENSEMBLE --------------------------------------")

base_models = [
    ("XGBoost",  best_xgb),
    ("LightGBM", best_lgbm),
    ("RF",       best_rf),
    ("LR",       best_lr),
]

# Generate OOF probability predictions on the training set
n_bases = len(base_models)
oof_train = np.zeros((len(X_tr_sel), n_bases))  # shape: (n_train, 4)
oof_test  = np.zeros((len(X_te_sel), n_bases))  # shape: (n_test, 4)

for i, (name, model) in enumerate(base_models):
    # OOF predictions for training meta-features
    oof_preds = np.zeros(len(X_tr_sel))
    for tr_idx, va_idx in tscv.split(X_tr_sel):
        model.fit(X_tr_sel[tr_idx], y_train[tr_idx])
        oof_preds[va_idx] = model.predict_proba(X_tr_sel[va_idx])[:, 1]
    oof_train[:, i] = oof_preds

    # Retrain on full training set for test predictions
    model.fit(X_tr_sel, y_train)
    oof_test[:, i] = model.predict_proba(X_te_sel)[:, 1]
    print(f"  OOF done: {name}")

# Train meta-learner on OOF predictions
meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
meta_lr.fit(oof_train, y_train)

# Final stack prediction
y_pred_stack = meta_lr.predict(oof_test)
y_prob_stack = meta_lr.predict_proba(oof_test)[:, 1]

print("  Meta-learner trained on OOF predictions.")
print(f"  Meta-learner weights (base model coefficients): "
      f"{dict(zip([n for n,_ in base_models], meta_lr.coef_[0].round(3)))}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 9 : EVALUATION ---------------------------------------------")

def evaluate(name, y_true, y_pred):
    return {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
    }

# Individual model predictions on test
y_pred_xgb  = best_xgb.predict(X_te_sel)
y_pred_lgbm = best_lgbm.predict(X_te_sel)
y_pred_rf   = best_rf.predict(X_te_sel)
y_pred_lr   = best_lr.predict(X_te_sel)
# Simple average ensemble (probability average)
y_prob_avg  = (best_xgb.predict_proba(X_te_sel)[:,1] +
               best_lgbm.predict_proba(X_te_sel)[:,1] +
               best_rf.predict_proba(X_te_sel)[:,1] +
               best_lr.predict_proba(X_te_sel)[:,1]) / 4
y_pred_avg  = (y_prob_avg >= 0.5).astype(int)

rows = [
    evaluate("Always UP (baseline)",  y_test, bl_up),
    evaluate("Random (baseline)",     y_test, bl_rnd),
    evaluate("XGBoost",               y_test, y_pred_xgb),
    evaluate("LightGBM",              y_test, y_pred_lgbm),
    evaluate("Random Forest",         y_test, y_pred_rf),
    evaluate("Logistic Regression",   y_test, y_pred_lr),
    evaluate("Avg Ensemble",          y_test, y_pred_avg),
    evaluate("STACK Ensemble",        y_test, y_pred_stack),
]
results = pd.DataFrame(rows).set_index("Model")
print("\n" + results.to_string(float_format="{:.4f}".format))

# Confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle(f"Confusion Matrices — {TICKER} (v2 Enhanced)", fontsize=14, fontweight="bold")
preds = [
    ("Always UP",   bl_up),
    ("Random",      bl_rnd),
    ("XGBoost",     y_pred_xgb),
    ("LightGBM",    y_pred_lgbm),
    ("RF",          y_pred_rf),
    ("LR",          y_pred_lr),
    ("Avg Ensemble",y_pred_avg),
    ("STACK",       y_pred_stack),
]
for ax, (name, yp) in zip(axes.flatten(), preds):
    cm = confusion_matrix(y_test, yp)
    ConfusionMatrixDisplay(cm, display_labels=["DOWN", "UP"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    acc = accuracy_score(y_test, yp)
    ax.set_title(f"{name}\nAcc={acc:.3f}", fontsize=9)
plt.tight_layout()
plt.savefig("confusion_v2.png", dpi=150)
plt.close()
print("\n[OK] Saved confusion_v2.png")

# Feature importance plot (XGBoost)
xgb_imp = pd.Series(best_xgb.feature_importances_,
                     index=[FEATURE_COLS[i] for i in feat_idx]).sort_values(ascending=True)
plt.figure(figsize=(9, 6))
xgb_imp.tail(20).plot(kind="barh", color="darkorange", edgecolor="white")
plt.title(f"XGBoost — Top 20 Feature Importances ({TICKER})", fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_v2.png", dpi=150)
plt.close()
print("[OK] Saved feature_importance_v2.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — BACKTEST (all models + buy-and-hold + random)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 10 : BACKTEST ----------------------------------------------")

ndr = test_df["Return_1d"].shift(-1).fillna(0).values  # next-day return

np.random.seed(RANDOM_SEED)
rnd_sig = np.random.randint(0, 2, size=len(y_test))

strategies = {
    "Buy & Hold"   : np.ones(len(y_test), dtype=int),
    "XGBoost"      : y_pred_xgb,
    "LightGBM"     : y_pred_lgbm,
    "RF"           : y_pred_rf,
    "Avg Ensemble" : y_pred_avg,
    "STACK"        : y_pred_stack,
    "Random"       : rnd_sig,
}

colors = {
    "Buy & Hold"  : "#2196F3",
    "XGBoost"     : "#FF5722",
    "LightGBM"    : "#4CAF50",
    "RF"          : "#9C27B0",
    "Avg Ensemble": "#FF9800",
    "STACK"       : "#F44336",
    "Random"      : "#9E9E9E",
}
styles = {
    "Buy & Hold": (2, "-"),
    "XGBoost"   : (2, "-"),
    "LightGBM"  : (2, "-"),
    "RF"        : (1.5,"--"),
    "Avg Ensemble":(1.5,"-"),
    "STACK"     : (2.5,"-"),
    "Random"    : (1, ":"),
}

dates = test_df.index[:-1]
plt.figure(figsize=(13, 6))

finals = {}
for label, sig in strategies.items():
    daily = np.where(sig == 1, ndr, 0)
    cum   = (1 + daily).cumprod()[:-1]
    finals[label] = cum[-1]
    lw, ls = styles[label]
    plt.plot(dates, cum, label=f"{label} ({(cum[-1]-1)*100:+.1f}%)",
             color=colors[label], linewidth=lw, linestyle=ls)

plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
plt.title(f"{TICKER} — Backtest: Cumulative Returns (v2 Enhanced)", fontsize=14, fontweight="bold")
plt.xlabel("Date");  plt.ylabel("Portfolio Value (start=1.0)")
plt.legend(fontsize=8, loc="upper left");  plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("backtest_v2.png", dpi=150)
plt.close()
print("[OK] Saved backtest_v2.png")

print("\nFinal cumulative returns:")
for label, val in sorted(finals.items(), key=lambda x: -x[1]):
    print(f"  {label:<20} {val:.4f}  ({(val-1)*100:+.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — HONEST SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 11 : HONEST SUMMARY ----------------------------------------\n")

best_name = results["Accuracy"].idxmax()
best_acc  = results["Accuracy"].max()
best_ret  = finals.get(best_name, finals.get("STACK Ensemble", 0))

beat_A   = best_acc > acc_up
beat_B   = best_acc > acc_rnd
beat_bnh = best_ret > finals["Buy & Hold"]

print(f"  Ticker     : {TICKER}")
print(f"  Test period: {test_df.index[0].date()} -> {test_df.index[-1].date()}")
print(f"  Best model : {best_name}  (accuracy {best_acc:.4f})\n")
print(f"  Beat Baseline A (always UP, {acc_up:.4f})?  {'YES' if beat_A else 'NO'}  [{best_acc:.4f}]")
print(f"  Beat Baseline B (random, {acc_rnd:.4f})?   {'YES' if beat_B else 'NO'}  [{best_acc:.4f}]")
print(f"  Beat Buy-and-Hold in backtest?           {'YES' if beat_bnh else 'NO'}  "
      f"[model {(best_ret-1)*100:+.1f}% vs B&H {(finals['Buy & Hold']-1)*100:+.1f}%]")

# v1 vs v2 comparison (approx v1 RF numbers)
print(f"""
  v1 vs v2 comparison:
    v1 RF accuracy  : ~0.4915  (from previous run)
    v2 best accuracy: {best_acc:.4f}

  Key caveats (unchanged from v1):
  * Zero transaction costs / slippage in backtest.
  * Close-to-close returns; real trades execute at open.
  * Single ticker, one test window — not robust evidence.
  * This is ML research, NOT financial advice.
""")

print("Output files:")
print("  confusion_v2.png")
print("  feature_importance_v2.png")
print("  backtest_v2.png")
print("\nDone.\n")
