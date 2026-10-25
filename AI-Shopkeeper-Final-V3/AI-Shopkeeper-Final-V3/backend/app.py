from pathlib import Path
import io

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .database import init_db, get_conn
from .importer import import_sales
from .models import InventoryUpdate, ForecastRequest
from ml.forecast import train_models, forecast

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(title="AI Shopkeeper", version="3.0")

init_db()
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


def _run(fn, *args, **kwargs):
    """Run ML code and translate errors into clean HTTP responses."""
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:          # e.g. optional TensorFlow not installed
        raise HTTPException(501, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/products")
def products():
    c = get_conn()
    rows = c.execute("""
        SELECT p.*, COALESCE(i.current_stock,0) current_stock,
               COALESCE(i.incoming_stock,0) incoming_stock,
               COALESCE(i.safety_stock,0) safety_stock
        FROM products p LEFT JOIN inventory i ON p.product_id=i.product_id
        ORDER BY p.product_name
    """).fetchall()
    c.close()
    return [dict(r) for r in rows]


@app.get("/api/analytics")
def analytics():
    c = get_conn()
    total_sales = c.execute("SELECT COALESCE(SUM(quantity_sold),0) FROM sales").fetchone()[0]
    revenue = c.execute("SELECT COALESCE(SUM(quantity_sold*unit_price),0) FROM sales").fetchone()[0]
    products_n = c.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    rows = c.execute("""
        SELECT product_id, product_name, SUM(quantity_sold) qty
        FROM sales GROUP BY product_id, product_name
        ORDER BY qty DESC LIMIT 10
    """).fetchall()
    trend = c.execute("""
        SELECT date, SUM(quantity_sold) qty
        FROM sales GROUP BY date ORDER BY date
    """).fetchall()
    c.close()
    return {
        "total_units": float(total_sales),
        "revenue": float(revenue),
        "product_count": int(products_n),
        "top_products": [dict(r) for r in rows],
        "trend": [dict(r) for r in trend]
    }


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (limit 50 MB).")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    try:
        summary = import_sales(df)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"message": "CSV imported successfully (existing dates were updated, not duplicated).", **summary}


@app.put("/api/inventory")
def update_inventory(item: InventoryUpdate):
    c = get_conn()
    c.execute("""
        INSERT INTO inventory(product_id,current_stock,incoming_stock,safety_stock)
        VALUES(?,?,?,?)
        ON CONFLICT(product_id) DO UPDATE SET
        current_stock=excluded.current_stock,
        incoming_stock=excluded.incoming_stock,
        safety_stock=excluded.safety_stock
    """, (item.product_id, item.current_stock, item.incoming_stock, item.safety_stock))
    c.commit()
    c.close()
    return {"message": "Inventory updated."}


@app.post("/api/train/{product_id}")
def train(product_id: str, tune: bool = False):
    """Force a fresh training run (optionally with hyper-parameter tuning)."""
    return _run(train_models, product_id, tune=tune, force=True)


@app.post("/api/forecast")
def make_forecast(req: ForecastRequest):
    return _run(forecast, req.product_id, req.horizon, retrain=req.retrain, tune=req.tune,
                lead_time_days=req.lead_time_days, service_level=req.service_level)


@app.post("/api/train-lstm/{product_id}")
def train_lstm_experiment(product_id: str):
    def go():
        from ml.lstm import get_or_train_lstm
        return get_or_train_lstm(product_id, retrain=True)
    return _run(go)


@app.post("/api/forecast-lstm")
def make_lstm_forecast(req: ForecastRequest):
    def go():
        from ml.lstm import get_or_train_lstm, forecast_lstm
        metrics = get_or_train_lstm(req.product_id, retrain=req.retrain)
        predictions = forecast_lstm(req.product_id, req.horizon)
        c = get_conn()
        row = c.execute("SELECT MAX(date) FROM sales WHERE product_id=?", (req.product_id,)).fetchone()
        c.close()
        start = pd.to_datetime(row[0])
        # side-by-side comparison with the standard models (cached unless the data changed)
        meta = train_models(req.product_id)
        comparison = [dict(r) for r in meta["results"]]
        comparison.append({k: metrics.get(k) for k in ("model", "mae", "rmse", "wape")})
        comparison.sort(key=lambda r: (r["wape"] if r.get("wape") is not None else 9e9))
        return {
            "product_id": req.product_id, "model": "LSTM", "metrics": metrics,
            "forecast": [{"date": (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                          "predicted_demand": round(float(p), 2)} for i, p in enumerate(predictions, 1)],
            "comparison": comparison,
            "note": "Hold-out WAPE is compared on the same chronological hold-out window for all models.",
        }
    return _run(go)


def _results_frame(product_id=None, history=False):
    c = get_conn()
    q = """SELECT product_id, model_name AS model, mae, rmse, wape, cv_wape, wape_7d, is_best,
                  trained_rows, test_rows, tuned, trained_at
           FROM model_results WHERE product_id IS NOT NULL"""
    params = []
    if product_id:
        q += " AND product_id=?"
        params.append(product_id)
    if not history:
        q += " AND id IN (SELECT MAX(id) FROM model_results WHERE product_id IS NOT NULL GROUP BY product_id, model_name)"
    q += " ORDER BY product_id, wape"
    df = pd.read_sql_query(q, c, params=params)
    c.close()
    return df


@app.get("/api/results")
def results(product_id: str = None, history: bool = False):
    """Logged model comparison (latest run per product/model unless history=true)."""
    return _results_frame(product_id, history).to_dict(orient="records")


@app.get("/api/results.csv")
def results_csv(product_id: str = None, history: bool = False):
    csv_text = _results_frame(product_id, history).to_csv(index=False)
    return Response(content=csv_text, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="model_results.csv"'})
