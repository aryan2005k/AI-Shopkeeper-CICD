"""End-to-end smoke test. Uses a temporary database and model folder, so it never
touches your real data.   Run:  python tests/smoke_test.py
"""
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.database as db          # noqa: E402
import ml.forecast as fc               # noqa: E402
from backend.importer import import_sales, clean_sales_df  # noqa: E402

passed = []


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    passed.append(bool(cond))


def main():
    tmp = Path(tempfile.mkdtemp())
    db.DB_PATH = tmp / "shop.db"
    fc.DB_PATH = tmp / "shop.db"
    fc.MODEL_DIR = tmp / "models"
    fc.MODEL_DIR.mkdir()
    try:
        # 1. migration of an OLD database that already contains duplicate sales rows
        con = sqlite3.connect(db.DB_PATH)
        con.executescript("""
            CREATE TABLE sales (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL, product_id TEXT NOT NULL,
                product_name TEXT, category TEXT, quantity_sold REAL NOT NULL, unit_price REAL DEFAULT 0, promotion INTEGER DEFAULT 0);
            INSERT INTO sales(date,product_id,quantity_sold) VALUES ('2026-01-01','A',5),('2026-01-01','A',5);
            CREATE TABLE model_results (id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT, mae REAL, rmse REAL, wape REAL, trained_at TEXT);
        """)
        con.commit(); con.close()
        db.init_db()
        con = sqlite3.connect(db.DB_PATH)
        check("old duplicate sales rows removed on startup", con.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 1)
        cols = {r[1] for r in con.execute("PRAGMA table_info(model_results)")}
        check("model_results migrated with new columns", {"product_id", "cv_wape", "wape_7d"} <= cols)
        con.close()
        db.reset_data()

        # 2. import: re-uploading does not duplicate, bad rows are skipped
        data = ROOT / "data" / "sample_sales_2yr_synthetic.csv"
        df = pd.read_csv(data)
        a = import_sales(df)
        b = import_sales(df)
        con = sqlite3.connect(db.DB_PATH)
        n = con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        con.close()
        check("re-uploading the same CSV does not duplicate sales", n == a["rows_imported"] == b["rows_imported"])
        bad = df.head(5).copy()
        bad.loc[0, "quantity_sold"] = -3
        bad.loc[1, "date"] = "not a date"
        _, skipped = clean_sales_df(bad)
        check("negative quantities and bad dates are skipped", skipped == 2)
        try:
            clean_sales_df(df.drop(columns=["promotion"]))
            check("missing column is rejected", False)
        except ValueError:
            check("missing column is rejected", True)

        # 3. training, caching, forecasting
        meta = fc.train_models("P001")
        check("training returns a best model", meta["best_model"] in fc.KIND)
        check("four models were compared", len(meta["results"]) == 4)
        check("hold-out is chronological and non-trivial", meta["test_rows"] >= 14)
        check("second call is served from cache", fc.train_models("P001")["cached"] is True)
        out = fc.forecast("P001", 7)
        check("7-day forecast produced", len(out["forecast"]) == 7)
        check("forecast values are non-negative", all(x["predicted_demand"] >= 0 for x in out["forecast"]))
        check("intervals bracket the forecast", all(x["lower"] <= x["predicted_demand"] <= x["upper"] for x in out["forecast"]))
        check("suggested safety stock is computed", out["inventory"]["suggested_safety_stock"] > 0)

        # 4. cache is invalidated when the data change
        new = pd.DataFrame([{"date": "2026-10-01", "product_id": "P001", "product_name": "Rice 5kg",
                             "category": "Grocery", "quantity_sold": 40, "unit_price": 42.0, "promotion": 0}])
        import_sales(new)
        check("new data invalidates the cached model", fc.train_models("P001")["cached"] is False)

        # 5. short history still works but warns
        short = df[(df.product_id == "P002")].head(60)
        short = short.assign(product_id="SHORT")
        import_sales(short)
        m = fc.train_models("SHORT")
        check("short history trains and returns a warning", len(m["warnings"]) > 0)
        check("unknown product raises a clear error", _raises(lambda: fc.train_models("NOPE")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    sys.exit(0 if all(passed) else 1)


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    main()
