"""Generate a SYNTHETIC two-year retail sales dataset (for demos and testing only).

The data are simulated: weekly seasonality, yearly seasonality, a slow trend,
festival peaks, promotions with price discounts, random noise and a few closed
days (missing dates). They are NOT real store data - for the final report,
replace them with a real dataset (see README) and say so explicitly.

Usage:  python scripts/generate_synthetic_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_sales_2yr_synthetic.csv"

# id, name, category, base price, base daily demand, weekend factor
PRODUCTS = [
    ("P001", "Rice 5kg", "Grocery", 42.0, 28, 1.10),
    ("P002", "Cooking Oil 1L", "Grocery", 16.5, 20, 1.05),
    ("P003", "Milk 1L", "Dairy", 6.5, 40, 0.95),
    ("P004", "Biscuits Pack", "Snacks", 3.0, 53, 1.30),
    ("P005", "Shampoo 200ml", "Personal Care", 18.0, 13, 1.15),
    ("P006", "Bread Loaf", "Bakery", 2.5, 60, 1.20),
    ("P007", "Tea 250g", "Beverages", 9.0, 22, 1.00),
    ("P008", "Detergent 1kg", "Household", 12.0, 15, 1.10),
]
FESTIVALS = [("2024-10-25", "2024-11-03"), ("2025-10-14", "2025-10-23")]
CLOSED_DAYS = ["2025-01-26", "2025-03-14", "2025-08-15", "2025-12-25", "2026-01-26", "2026-08-15"]


def main(seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-10-01", "2026-09-30", freq="D")
    dow_shape = np.array([0.92, 0.90, 0.94, 0.98, 1.08, 1.22, 1.10])   # Mon..Sun
    rows = []
    for pid, name, cat, price, base, weekend in PRODUCTS:
        # promotions come in 3-day blocks
        promo = np.zeros(len(dates), dtype=int)
        for i in np.where(rng.random(len(dates)) < 0.035)[0]:
            promo[i:i + 3] = 1
        for t, d in enumerate(dates):
            dow = d.dayofweek
            season = 1 + 0.12 * np.sin(2 * np.pi * (d.dayofyear - 60) / 365.25)
            trend = 1 + 0.0004 * t
            week = dow_shape[dow] * (weekend if dow >= 5 else 1.0)
            fest = 1.35 if any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in FESTIVALS) else 1.0
            unit_price = round(price * (1 + 0.02 * (t // 180)) * (0.9 if promo[t] else 1.0), 2)
            elasticity = (unit_price / price) ** -0.6
            mean = base * season * trend * week * fest * (1.25 if promo[t] else 1.0) * elasticity
            qty = int(rng.poisson(mean))
            rows.append((d.strftime("%Y-%m-%d"), pid, name, cat, qty, unit_price, int(promo[t])))
    df = pd.DataFrame(rows, columns=["date", "product_id", "product_name", "category",
                                     "quantity_sold", "unit_price", "promotion"])
    df = df[~df["date"].isin(CLOSED_DAYS)]
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows ({df.product_id.nunique()} products, "
          f"{df.date.min()} to {df.date.max()}) -> {OUT}")


if __name__ == "__main__":
    main()
