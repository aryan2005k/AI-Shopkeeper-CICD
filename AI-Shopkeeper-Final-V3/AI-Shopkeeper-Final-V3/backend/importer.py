"""CSV validation and import (shared by the API and scripts/import_csv.py)."""
import pandas as pd

from .database import get_conn

REQUIRED = ["date", "product_id", "product_name", "category",
            "quantity_sold", "unit_price", "promotion"]


def clean_sales_df(df: pd.DataFrame):
    """Validate a raw sales DataFrame. Returns (clean_df, rows_skipped).

    One output row per (date, product_id): same-day duplicates are aggregated
    (quantity summed, price averaged, promotion = any).
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    d = df[REQUIRED].copy()
    n_in = len(d)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["quantity_sold"] = pd.to_numeric(d["quantity_sold"], errors="coerce")
    d["unit_price"] = pd.to_numeric(d["unit_price"], errors="coerce").fillna(0.0)
    d["promotion"] = pd.to_numeric(d["promotion"], errors="coerce").fillna(0)
    d["product_id"] = d["product_id"].astype(str).str.strip()
    d["product_name"] = d["product_name"].fillna("").astype(str)
    d["category"] = d["category"].fillna("").astype(str)

    d = d[d["date"].notna() & d["quantity_sold"].notna()
          & (d["product_id"] != "") & (d["product_id"].str.lower() != "nan")
          & (d["quantity_sold"] >= 0)]
    if d.empty:
        raise ValueError("No valid rows found (check date format and non-negative quantities).")

    d = d.sort_values("date")
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    d["promotion"] = (d["promotion"] > 0).astype(int)
    clean = (d.groupby(["date", "product_id"], as_index=False)
               .agg(product_name=("product_name", "last"), category=("category", "last"),
                    quantity_sold=("quantity_sold", "sum"), unit_price=("unit_price", "mean"),
                    promotion=("promotion", "max")))
    return clean, int(n_in - len(d))


def import_sales(df: pd.DataFrame):
    """Insert/update sales. Re-uploading the same data never creates duplicates."""
    clean, skipped = clean_sales_df(df)
    rows = [(r.date, r.product_id, r.product_name, r.category,
             float(r.quantity_sold), float(r.unit_price), int(r.promotion))
            for r in clean.itertuples(index=False)]
    pids = sorted(clean["product_id"].unique())

    c = get_conn()
    try:
        c.executemany("""
            INSERT INTO sales(date, product_id, product_name, category, quantity_sold, unit_price, promotion)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(date, product_id) DO UPDATE SET
                product_name=excluded.product_name, category=excluded.category,
                quantity_sold=excluded.quantity_sold, unit_price=excluded.unit_price,
                promotion=excluded.promotion
        """, rows)
        c.executemany("INSERT OR IGNORE INTO products(product_id, product_name, category, unit_price) VALUES(?,?,?,?)",
                      [(p, "", "", 0.0) for p in pids])
        c.executemany("INSERT OR IGNORE INTO inventory(product_id, current_stock, incoming_stock, safety_stock) VALUES(?,0,0,0)",
                      [(p,) for p in pids])
        # Product master data always reflects the most recent sales row.
        c.executemany("""
            UPDATE products SET
              product_name=(SELECT product_name FROM sales WHERE product_id=products.product_id ORDER BY date DESC LIMIT 1),
              category=(SELECT category FROM sales WHERE product_id=products.product_id ORDER BY date DESC LIMIT 1),
              unit_price=(SELECT unit_price FROM sales WHERE product_id=products.product_id ORDER BY date DESC LIMIT 1)
            WHERE product_id=?
        """, [(p,) for p in pids])
        c.commit()
    finally:
        c.close()

    return {"rows_in_file": int(len(df)), "rows_imported": int(len(rows)), "rows_skipped": skipped,
            "products_affected": int(len(pids)),
            "date_from": str(clean["date"].min()), "date_to": str(clean["date"].max())}
