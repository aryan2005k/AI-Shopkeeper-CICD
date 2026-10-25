"""Demand-forecasting pipeline.

Models compared per product:
  * Moving Average (7d)   - statistical baseline
  * Seasonal Naive (7d)   - same weekday last week (stronger baseline)
  * Random Forest         - bagged decision trees
  * LightGBM              - gradient-boosted decision trees

Evaluation
  * chronological 80/20 hold-out (never shuffled)
  * expanding-window time-series cross-validation on the training part
    (used for model selection and optional hyper-parameter tuning)
  * 7-day *recursive* back-test on the hold-out period, so the reported error
    matches how multi-day forecasts are actually produced.

The selected model is re-fitted on 100 % of the history and cached; it is only
re-trained when the sales data change (or when the user asks for it).
"""
from pathlib import Path
from datetime import datetime, timedelta
import json
import math
import re
import sqlite3
import threading

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "shop.db"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_VERSION = 2          # bump to invalidate every cached model
MIN_DAYS = 35
BACKTEST_HORIZON = 7

FEATURES = [
    "day_of_week", "week_of_year", "month", "quarter", "weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std",
    "unit_price", "price_change", "promotion"
]

MA, SN, RF, LGB = "Moving Average (7d)", "Seasonal Naive (7d)", "Random Forest", "LightGBM"
KIND = {MA: "ma", SN: "sn", RF: "rf", LGB: "lgb"}
TREE_KINDS = ("rf", "lgb")


def _make_rf(**p):
    params = dict(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1)
    params.update(p)
    return RandomForestRegressor(**params)


def _make_lgb(**p):
    # min_child_samples is lowered from LightGBM's default of 20: with only a few
    # dozen training rows the default stops the trees from splitting at all.
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                  min_child_samples=5, random_state=42, verbosity=-1)
    params.update(p)
    return LGBMRegressor(**params)


FACTORIES = {RF: _make_rf, LGB: _make_lgb}
GRIDS = {
    RF: [{}, {"max_depth": 8, "min_samples_leaf": 3}, {"max_depth": 12, "min_samples_leaf": 5},
         {"max_features": 0.6}],
    LGB: [{}, {"num_leaves": 15}, {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 500},
          {"num_leaves": 15, "min_child_samples": 10, "colsample_bytree": 0.8}],
}

_locks = {}
_locks_guard = threading.Lock()


def _lock_for(product_id):
    with _locks_guard:
        return _locks.setdefault(product_id, threading.Lock())


def safe_id(product_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(product_id))


def _meta_path(product_id):
    return MODEL_DIR / f"{safe_id(product_id)}_meta.json"


def _model_path(product_id):
    return MODEL_DIR / f"{safe_id(product_id)}_model.joblib"


# --------------------------------------------------------------------------- data
def conn():
    return sqlite3.connect(DB_PATH)


def load_sales(product_id=None):
    c = conn()
    q = "SELECT date, product_id, product_name, category, quantity_sold, unit_price, promotion FROM sales"
    params = []
    if product_id:
        q += " WHERE product_id=?"
        params.append(product_id)
    q += " ORDER BY date"
    df = pd.read_sql_query(q, c, params=params, parse_dates=["date"])
    c.close()
    return df


def data_signature(product_id):
    """Cheap fingerprint of a product's sales; changes whenever the data change."""
    c = conn()
    r = c.execute("""SELECT COUNT(*), MIN(date), MAX(date), COALESCE(SUM(quantity_sold),0),
                            COALESCE(SUM(unit_price),0), COALESCE(SUM(promotion),0)
                     FROM sales WHERE product_id=?""", (product_id,)).fetchone()
    c.close()
    if not r or r[0] == 0:
        raise ValueError(f"No sales data found for product '{product_id}'.")
    return "|".join(str(round(x, 4) if isinstance(x, float) else x) for x in r)


def prepare_daily(df):
    """One row per calendar day. Missing days are treated as zero demand, prices are
    carried forward, so shift-based lag features always refer to real calendar days."""
    if df.empty:
        raise ValueError("No sales found for this product.")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    for col in ("quantity_sold", "unit_price", "promotion"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    g = (d.sort_values("date").groupby("date")
          .agg(product_id=("product_id", "last"), product_name=("product_name", "last"),
               category=("category", "last"), quantity_sold=("quantity_sold", "sum"),
               unit_price=("unit_price", "mean"), promotion=("promotion", "max")))
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    g["quantity_sold"] = g["quantity_sold"].fillna(0.0)
    g["unit_price"] = g["unit_price"].ffill().bfill().fillna(0.0)
    g["promotion"] = g["promotion"].fillna(0.0)
    for col in ("product_id", "product_name", "category"):
        g[col] = g[col].ffill().bfill()
    return g.rename_axis("date").reset_index()


def add_features(df):
    df = df.sort_values("date").copy()
    df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce").fillna(0)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
    df["promotion"] = pd.to_numeric(df["promotion"], errors="coerce").fillna(0)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["quantity_sold"].shift(lag)
    for w in [7, 14, 28]:
        df[f"rolling_mean_{w}"] = df["quantity_sold"].shift(1).rolling(w).mean()
    df["rolling_std"] = df["quantity_sold"].shift(1).rolling(28).std().fillna(0)
    df["price_change"] = df["unit_price"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    return df


def holdout_days(product_id):
    """Length of the hold-out window used by the tree models (lets the LSTM use the same one)."""
    daily = prepare_daily(load_sales(product_id))
    f = add_features(daily).dropna()
    return int(len(f) - max(int(len(f) * 0.8), 1))


# ------------------------------------------------------------------------ metrics
def wape(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / denom) if denom else 0.0


def _metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(math.sqrt(mean_squared_error(y, p))),
            "wape": wape(y, p)}


def _cv_splitter(n_rows):
    if n_rows >= 90:
        return TimeSeriesSplit(n_splits=3)
    if n_rows >= 45:
        return TimeSeriesSplit(n_splits=2)
    return None


def _cv_wape_model(factory, params, X, y, splitter):
    err = act = 0.0
    for tr, va in splitter.split(X):
        m = factory(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = np.maximum(0, m.predict(X.iloc[va]))
        yy = y.iloc[va].to_numpy(dtype=float)
        err += float(np.abs(yy - p).sum())
        act += float(np.abs(yy).sum())
    return err / act if act else 0.0


def _cv_wape_baseline(col, X, y, splitter):
    err = act = 0.0
    for _, va in splitter.split(X):
        p = np.maximum(0, X[col].iloc[va].to_numpy(dtype=float))
        yy = y.iloc[va].to_numpy(dtype=float)
        err += float(np.abs(yy - p).sum())
        act += float(np.abs(yy).sum())
    return err / act if act else 0.0


# ---------------------------------------------------------------------------- SHAP
def _shap_explanation(model, X, limit=8):
    """Mean(|SHAP value|) per feature for a tree model, with a safe fallback."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        scores = np.mean(np.abs(values), axis=0)
        order = np.argsort(scores)[::-1][:limit]
        return [{"feature": FEATURES[i], "importance": float(scores[i])} for i in order], "SHAP TreeExplainer"
    except Exception as exc:
        importance = getattr(model, "feature_importances_", np.zeros(len(FEATURES)))
        order = np.argsort(importance)[::-1][:limit]
        return ([{"feature": FEATURES[i], "importance": float(importance[i])} for i in order],
                f"Feature importance fallback: {type(exc).__name__}")


# ------------------------------------------------------------- recursive forecasting
def _recursive(kind, model, history, horizon):
    """Forecast `horizon` days ahead, feeding each prediction back as history."""
    work = history.tail(60).reset_index(drop=True).copy()
    preds = []
    for _ in range(horizon):
        next_date = pd.Timestamp(work["date"].iloc[-1]) + timedelta(days=1)
        if kind == "ma":
            p = float(work["quantity_sold"].tail(7).mean())
        elif kind == "sn":
            p = float(work["quantity_sold"].iloc[-7])
        else:
            last = work.iloc[-1]
            temp = pd.DataFrame([{
                "date": next_date, "product_id": last["product_id"],
                "product_name": last["product_name"], "category": last["category"],
                "quantity_sold": np.nan, "unit_price": float(last["unit_price"]), "promotion": 0.0}])
            row = add_features(pd.concat([work, temp], ignore_index=True)).iloc[-1]
            X = pd.DataFrame([{k: float(row[k]) for k in FEATURES}]).fillna(0.0)
            p = float(model.predict(X)[0])
        p = max(0.0, p)
        preds.append(p)
        new = work.iloc[[-1]].copy()
        new["date"] = next_date
        new["quantity_sold"] = p
        new["promotion"] = 0.0
        work = pd.concat([work, new], ignore_index=True)
    return preds


def _backtest(kind, model, daily, start_idx, horizon=BACKTEST_HORIZON):
    """Pooled error of consecutive `horizon`-day recursive forecasts over the hold-out period."""
    origins = list(range(start_idx, len(daily) - horizon + 1, horizon))
    if not origins:
        return None
    abs_err = sq_err = act = 0.0
    n = 0
    qty = daily["quantity_sold"].to_numpy(dtype=float)
    for o in origins:
        pred = np.asarray(_recursive(kind, model, daily.iloc[:o], horizon))
        actual = qty[o:o + horizon]
        abs_err += float(np.abs(actual - pred).sum())
        sq_err += float(((actual - pred) ** 2).sum())
        act += float(np.abs(actual).sum())
        n += len(actual)
    return {"wape_7d": abs_err / act if act else 0.0, "mae_7d": abs_err / n,
            "rmse_7d": math.sqrt(sq_err / n), "windows": len(origins)}


# ---------------------------------------------------------------------- training
def _train_and_evaluate(product_id, tune, signature):
    daily = prepare_daily(load_sales(product_id))
    if len(daily) < MIN_DAYS:
        raise ValueError(f"At least {MIN_DAYS} days of sales history are needed (found {len(daily)}).")

    f = add_features(daily).dropna().reset_index(drop=True)
    if len(f) < 10:
        raise ValueError("Not enough usable rows after feature engineering.")
    offset = len(daily) - len(f)
    split = max(int(len(f) * 0.8), 1)
    train, test = f.iloc[:split], f.iloc[split:]
    if len(test) == 0:
        raise ValueError("Not enough data for a time-based test split.")

    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["quantity_sold"], test["quantity_sold"]

    warnings = []
    if len(test) < 14:
        warnings.append(f"Only {len(test)} hold-out days: error metrics are unreliable. "
                        "Import more history (6+ months recommended).")
    splitter = _cv_splitter(len(train))
    if splitter is None:
        warnings.append("Training window too short for time-series cross-validation; "
                        "the model was selected on the hold-out set instead.")
        if tune:
            warnings.append("Hyper-parameter tuning was skipped (it needs cross-validation).")
    tuned = bool(tune and splitter is not None)

    # candidates: name -> dict(kind, pred, cv, params, model)
    cands = {
        MA: {"kind": "ma", "pred": np.maximum(0, X_test["rolling_mean_7"].to_numpy(dtype=float)),
             "cv": _cv_wape_baseline("rolling_mean_7", X_train, y_train, splitter) if splitter else None,
             "params": {}, "model": None},
        SN: {"kind": "sn", "pred": np.maximum(0, X_test["lag_7"].to_numpy(dtype=float)),
             "cv": _cv_wape_baseline("lag_7", X_train, y_train, splitter) if splitter else None,
             "params": {}, "model": None},
    }
    for name, factory in FACTORIES.items():
        best_params, best_cv = {}, None
        if splitter is not None:
            grid = GRIDS[name] if tuned else [{}]
            scored = [(_cv_wape_model(factory, p, X_train, y_train, splitter), p) for p in grid]
            best_cv, best_params = min(scored, key=lambda t: t[0])
        model = factory(**best_params)
        model.fit(X_train, y_train)
        if name == RF:
            model.set_params(n_jobs=1)   # single-row predictions are faster without a thread pool
        cands[name] = {"kind": KIND[name], "pred": np.maximum(0, model.predict(X_test)),
                       "cv": best_cv, "params": best_params, "model": model}

    # 7-day recursive back-test over the hold-out period
    start_idx = offset + split
    for name, c in cands.items():
        c["bt"] = _backtest(c["kind"], c["model"], daily, start_idx)

    results = []
    for name, c in cands.items():
        m = _metrics(y_test, c["pred"])
        bt = c["bt"] or {}
        results.append({"model": name, **m, "cv_wape": c["cv"],
                        "wape_7d": bt.get("wape_7d"), "mae_7d": bt.get("mae_7d"),
                        "rmse_7d": bt.get("rmse_7d"), "params": c["params"], "available": True})

    use_cv = all(r["cv_wape"] is not None for r in results)
    key = (lambda r: (r["cv_wape"], r["mae"])) if use_cv else (lambda r: (r["wape"], r["mae"]))
    selection = ("lowest cross-validated WAPE (expanding-window CV on the training period)" if use_cv
                 else "lowest hold-out WAPE")
    best = min(results, key=key)
    best_name = best["model"]
    best_kind = KIND[best_name]
    results.sort(key=key)

    # explanation (SHAP) from the hold-out model
    if best_kind in TREE_KINDS:
        explain, method = _shap_explanation(cands[best_name]["model"], X_test)
    else:
        explain, method = [], "Baseline model selected - no feature explanation available"

    # re-fit the selected tree model on 100 % of the data for production forecasts
    mp = _model_path(product_id)
    if best_kind in TREE_KINDS:
        final = FACTORIES[best_name](**best["params"])
        final.fit(f[FEATURES], f["quantity_sold"])
        if best_name == RF:
            final.set_params(n_jobs=1)
        joblib.dump({"model": final, "features": FEATURES, "product_id": product_id,
                     "best_model": best_name}, mp)
    elif mp.exists():
        mp.unlink()

    sigma = best.get("rmse_7d") or best["rmse"]
    meta = {
        "version": MODEL_VERSION, "product_id": product_id, "signature": signature,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best_name, "selection_method": selection,
        "results": results, "explanation": explain, "explanation_method": method,
        "trained_rows": int(len(train)), "test_rows": int(len(test)), "total_days": int(len(daily)),
        "tuned": tuned, "sigma": float(sigma), "backtest_horizon": BACKTEST_HORIZON,
        "warnings": warnings,
    }
    _meta_path(product_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _load_cache(product_id, signature, need_tuned):
    p = _meta_path(product_id)
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("version") != MODEL_VERSION or meta.get("signature") != signature:
        return None
    if need_tuned and not meta.get("tuned"):
        return None
    if KIND.get(meta.get("best_model")) in TREE_KINDS and not _model_path(product_id).exists():
        return None
    return meta


def train_models(product_id, tune=False, force=False):
    """Train (or load from cache) the models for one product. Returns the metadata dict."""
    with _lock_for(product_id):
        sig = data_signature(product_id)
        if not force:
            cached = _load_cache(product_id, sig, tune)
            if cached:
                cached["cached"] = True
                return cached
        meta = _train_and_evaluate(product_id, tune, sig)
        _log_results(meta)
        meta["cached"] = False
        return meta


# ------------------------------------------------------------------- results log
def log_model_result(product_id, row, is_best=False, trained_rows=None, test_rows=None, tuned=False):
    """Append one model's metrics to the model_results table (used for the report)."""
    try:
        c = conn()
        c.execute("""INSERT INTO model_results(model_name, mae, rmse, wape, trained_at, product_id,
                                               cv_wape, wape_7d, is_best, trained_rows, test_rows, tuned)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (row.get("model"), row.get("mae"), row.get("rmse"), row.get("wape"),
                   datetime.now().isoformat(timespec="seconds"), product_id,
                   row.get("cv_wape"), row.get("wape_7d"), int(is_best), trained_rows, test_rows, int(tuned)))
        c.commit()
        c.close()
    except Exception:
        pass  # logging must never break training


def _log_results(meta):
    for r in meta["results"]:
        log_model_result(meta["product_id"], r, is_best=(r["model"] == meta["best_model"]),
                         trained_rows=meta["trained_rows"], test_rows=meta["test_rows"], tuned=meta["tuned"])


# ---------------------------------------------------------------------- forecast
def forecast(product_id, horizon=7, retrain=False, tune=False, lead_time_days=2, service_level=0.95):
    meta = train_models(product_id, tune=tune, force=retrain)
    daily = prepare_daily(load_sales(product_id))
    best = meta["best_model"]
    kind = KIND[best]

    model = None
    if kind in TREE_KINDS:
        model = joblib.load(_model_path(product_id))["model"]
    predictions = _recursive(kind, model, daily, horizon)

    last_date = pd.Timestamp(daily["date"].max())
    dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, horizon + 1)]
    total = float(sum(predictions))

    # approximate 80 % interval from the 7-day back-test error of the selected model
    sigma = float(meta.get("sigma") or 0.0)
    z80 = float(norm.ppf(0.90))
    fc = [{"date": d, "predicted_demand": round(float(p), 2),
           "lower": round(max(0.0, float(p) - z80 * sigma), 2),
           "upper": round(float(p) + z80 * sigma, 2)} for d, p in zip(dates, predictions)]

    c = conn()
    inv = pd.read_sql_query("SELECT current_stock, incoming_stock, safety_stock FROM inventory WHERE product_id=?",
                            c, params=[product_id])
    c.close()
    if inv.empty:
        current_stock = incoming_stock = safety_stock = 0.0
    else:
        current_stock = float(inv.iloc[0]["current_stock"])
        incoming_stock = float(inv.iloc[0]["incoming_stock"])
        safety_stock = float(inv.iloc[0]["safety_stock"])

    # statistically derived safety stock: z * sigma_daily_demand * sqrt(lead time)
    recent = daily["quantity_sold"].tail(90).to_numpy(dtype=float)
    sigma_d = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
    z = float(norm.ppf(service_level))
    suggested_safety = round(z * sigma_d * math.sqrt(max(lead_time_days, 0)), 2)

    reorder = max(0.0, total + safety_stock - current_stock - incoming_stock)
    projected = current_stock + incoming_stock - total
    if projected < 0:
        status = "Stockout Risk"
    elif current_stock > total + safety_stock * 2 and total > 0:
        status = "Overstock Risk"
    elif reorder > 0:
        status = "Reorder"
    elif current_stock <= total + safety_stock:
        status = "Monitor"
    else:
        status = "Healthy Stock"

    history = [{"date": pd.Timestamp(r.date).strftime("%Y-%m-%d"), "quantity": float(r.quantity_sold)}
               for r in daily.tail(30).itertuples(index=False)]

    return {
        "product_id": product_id, "best_model": best, "selection_method": meta["selection_method"],
        "forecast": fc, "history": history,
        "total_predicted_demand": round(total, 2),
        "interval_note": "Shaded range = approximate 80% interval from the 7-day back-test error.",
        "inventory": {"current_stock": current_stock, "incoming_stock": incoming_stock,
                      "safety_stock": safety_stock, "projected_stock_after_forecast": round(projected, 2),
                      "suggested_safety_stock": suggested_safety, "lead_time_days": lead_time_days,
                      "service_level": service_level},
        "recommendation": {"status": status, "reorder_quantity": round(reorder, 2)},
        "model_results": meta["results"], "explanation": meta["explanation"],
        "explanation_method": meta["explanation_method"],
        "trained_rows": meta["trained_rows"], "test_rows": meta["test_rows"], "total_days": meta["total_days"],
        "trained_at": meta["trained_at"], "cached": bool(meta.get("cached")), "tuned": meta["tuned"],
        "warnings": meta.get("warnings", []),
        "evaluation_note": ("MAE/RMSE/WAPE: one-step-ahead on the chronological hold-out. "
                            "7-day WAPE: recursive 7-day forecasts over the same period."),
    }
