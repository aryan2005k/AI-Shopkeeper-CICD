import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "shop.db"

# Columns added to model_results after the first version of the project.
_MODEL_RESULTS_EXTRA = {
    "product_id": "TEXT",
    "cv_wape": "REAL",
    "wape_7d": "REAL",
    "is_best": "INTEGER DEFAULT 0",
    "trained_rows": "INTEGER",
    "test_rows": "INTEGER",
    "tuned": "INTEGER DEFAULT 0",
}


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT,
        unit_price REAL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        product_id TEXT NOT NULL,
        product_name TEXT,
        category TEXT,
        quantity_sold REAL NOT NULL,
        unit_price REAL DEFAULT 0,
        promotion INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS inventory (
        product_id TEXT PRIMARY KEY,
        current_stock REAL DEFAULT 0,
        incoming_stock REAL DEFAULT 0,
        safety_stock REAL DEFAULT 0,
        FOREIGN KEY(product_id) REFERENCES products(product_id)
    );

    CREATE TABLE IF NOT EXISTS model_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        mae REAL,
        rmse REAL,
        wape REAL,
        trained_at TEXT
    );
    """)

    # --- Fix: one sales row per (date, product). Older databases may contain
    # duplicates created by uploading the same CSV twice; keep the newest row.
    existing = {r[1] for r in cur.execute("PRAGMA index_list(sales)").fetchall()}
    if "idx_sales_date_product" not in existing:
        cur.execute("""DELETE FROM sales WHERE id NOT IN
                       (SELECT MAX(id) FROM sales GROUP BY date, product_id)""")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sales_date_product ON sales(date, product_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_product_date ON sales(product_id, date)")

    # --- model_results: add the columns used by the results log.
    cols = {r[1] for r in cur.execute("PRAGMA table_info(model_results)").fetchall()}
    for name, ddl in _MODEL_RESULTS_EXTRA.items():
        if name not in cols:
            cur.execute(f"ALTER TABLE model_results ADD COLUMN {name} {ddl}")

    conn.commit()
    conn.close()


def reset_data():
    """Delete all sales, products, inventory and logged results."""
    conn = get_conn()
    for table in ("sales", "inventory", "products", "model_results"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
