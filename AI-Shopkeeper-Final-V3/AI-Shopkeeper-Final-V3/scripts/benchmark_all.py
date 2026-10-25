"""Train and compare every model on every product and export a results table.

Usage:
    python scripts/benchmark_all.py                 # standard models
    python scripts/benchmark_all.py --tune          # + hyper-parameter search (slower)
    python scripts/benchmark_all.py --lstm          # + LSTM (needs TensorFlow)

Output: results/model_comparison.csv  and a summary printed to the terminal.
Use these numbers (not hand-picked screenshots) in the report's results chapter.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.database import init_db, get_conn  # noqa: E402
from ml.forecast import train_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore cached models")
    args = ap.parse_args()

    init_db()
    c = get_conn()
    pids = [r[0] for r in c.execute("SELECT DISTINCT product_id FROM sales ORDER BY product_id")]
    c.close()
    if not pids:
        sys.exit("No sales data. Import a CSV first (scripts/import_csv.py).")

    rows = []
    for pid in pids:
        print(f"Training {pid} ...", flush=True)
        meta = train_models(pid, tune=args.tune, force=args.force)
        for r in meta["results"]:
            rows.append({"product_id": pid, "model": r["model"], "mae": r["mae"], "rmse": r["rmse"],
                         "wape": r["wape"], "cv_wape": r["cv_wape"], "wape_7d": r["wape_7d"],
                         "selected": r["model"] == meta["best_model"],
                         "test_rows": meta["test_rows"], "tuned": meta["tuned"]})
        if args.lstm:
            try:
                from ml.lstm import get_or_train_lstm
                m = get_or_train_lstm(pid)
                rows.append({"product_id": pid, "model": "LSTM", "mae": m["mae"], "rmse": m["rmse"],
                             "wape": m["wape"], "cv_wape": None, "wape_7d": None, "selected": False,
                             "test_rows": m["test_rows"], "tuned": False})
            except RuntimeError as e:
                print("  LSTM skipped:", e)
                args.lstm = False

    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "model_comparison.csv", index=False)

    pd.set_option("display.width", 140)
    print("\nMean error over all products (lower is better):")
    summary = df.groupby("model")[["mae", "rmse", "wape", "wape_7d"]].mean().sort_values("wape")
    print((summary * [1, 1, 100, 100]).round(2).rename(columns={"wape": "wape_%", "wape_7d": "wape_7d_%"}))
    print("\nTimes each model was selected as best:")
    print(df[df["selected"]].groupby("model").size().to_string())
    print(f"\nSaved {out / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
