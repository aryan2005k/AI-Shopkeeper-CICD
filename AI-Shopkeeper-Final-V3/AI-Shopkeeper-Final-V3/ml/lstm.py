"""Optional LSTM demand-forecasting experiment (needs TensorFlow).

TensorFlow is intentionally optional: install requirements-lstm.txt to use it.
The LSTM is evaluated on the same hold-out window as the tree models so the
numbers can be compared directly in the results table.
"""
from pathlib import Path
from datetime import datetime
import json
import math

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
LOOKBACK = 14
LSTM_VERSION = 2


def _tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise RuntimeError(
            "LSTM requires TensorFlow. Install the optional dependencies with "
            "'pip install -r requirements-lstm.txt' and run the LSTM experiment again."
        ) from exc


def _paths(product_id):
    from .forecast import safe_id
    s = safe_id(product_id)
    return (MODEL_DIR / f"{s}_lstm.keras", MODEL_DIR / f"{s}_lstm_scaler.joblib",
            MODEL_DIR / f"{s}_lstm_meta.json")


def _daily_values(product_id):
    from .forecast import load_sales, prepare_daily
    return prepare_daily(load_sales(product_id))


def _make_sequences(values, lookback):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_lstm(product_id, epochs=60, lookback=LOOKBACK, signature=None):
    from .forecast import data_signature, holdout_days, log_model_result
    tf = _tf()
    daily = _daily_values(product_id)
    if len(daily) < max(lookback + 20, 35):
        raise ValueError("At least 35 days of sales history are needed for the LSTM experiment.")

    values = daily["quantity_sold"].to_numpy(dtype=float).reshape(-1, 1)
    try:
        test_days = holdout_days(product_id)          # same hold-out window as the tree models
    except Exception:
        test_days = max(int(len(values) * 0.2), 1)
    split = max(len(values) - test_days, lookback + 1)

    scaler = MinMaxScaler()
    scaler.fit(values[:split])                        # fitted on training data only
    scaled = scaler.transform(values)

    X, y = _make_sequences(scaled, lookback)
    train_n = max(split - lookback, 1)
    X_train, y_train = X[:train_n], y[:train_n]
    X_test, y_test = X[train_n:], y[train_n:]
    if len(X_test) == 0:
        raise ValueError("Not enough observations for an LSTM time-based test split.")

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    use_val = len(X_train) >= 40                      # keras takes the LAST 15 % -> chronological
    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss" if use_val else "loss", patience=6, restore_best_weights=True)
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=min(16, len(X_train)),
                     validation_split=0.15 if use_val else 0.0, callbacks=[stopper], verbose=0)

    pred = np.maximum(0, scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel())
    actual = np.maximum(0, scaler.inverse_transform(y_test.reshape(-1, 1)).ravel())
    denom = float(np.sum(np.abs(actual)))
    metrics = {
        "model": "LSTM",
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
        "wape": float(np.sum(np.abs(actual - pred)) / denom) if denom else 0.0,
    }

    model_path, scaler_path, meta_path = _paths(product_id)
    model.save(model_path)
    joblib.dump({"scaler": scaler, "lookback": lookback, "product_id": product_id}, scaler_path)

    result = {**metrics, "trained_rows": int(len(X_train)), "test_rows": int(len(X_test)),
              "epochs": int(epochs), "epochs_run": int(len(hist.history["loss"])),
              "lookback": int(lookback), "available": True}
    meta_path.write_text(json.dumps({
        "version": LSTM_VERSION, "signature": signature or data_signature(product_id),
        "trained_at": datetime.now().isoformat(timespec="seconds"), "metrics": result}, indent=2),
        encoding="utf-8")
    log_model_result(product_id, metrics, is_best=False, trained_rows=result["trained_rows"],
                     test_rows=result["test_rows"])
    return result


def get_or_train_lstm(product_id, retrain=False):
    """Return LSTM metrics, re-training only if the data changed or retrain=True."""
    from .forecast import data_signature
    _tf()
    sig = data_signature(product_id)
    model_path, scaler_path, meta_path = _paths(product_id)
    if not retrain and model_path.exists() and scaler_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("version") == LSTM_VERSION and meta.get("signature") == sig:
                return {**meta["metrics"], "cached": True, "trained_at": meta.get("trained_at")}
        except Exception:
            pass
    result = train_lstm(product_id, signature=sig)
    return {**result, "cached": False}


def forecast_lstm(product_id, horizon=7):
    tf = _tf()
    daily = _daily_values(product_id)
    if len(daily) < LOOKBACK:
        raise ValueError("Not enough history for an LSTM forecast.")

    model_path, scaler_path, _ = _paths(product_id)
    if not model_path.exists() or not scaler_path.exists():
        train_lstm(product_id)

    model = tf.keras.models.load_model(model_path)
    bundle = joblib.load(scaler_path)
    scaler, lookback = bundle["scaler"], bundle["lookback"]
    recent = daily["quantity_sold"].to_numpy(dtype=float)[-lookback:].reshape(-1, 1)
    window = scaler.transform(recent).reshape(1, lookback, 1)
    predictions = []
    for _ in range(horizon):
        p_scaled = float(model.predict(window, verbose=0)[0, 0])
        p = max(0.0, float(scaler.inverse_transform([[p_scaled]])[0, 0]))
        predictions.append(p)
        window = np.concatenate([window[:, 1:, :], np.array([[[p_scaled]]], dtype=np.float32)], axis=1)
    return predictions
