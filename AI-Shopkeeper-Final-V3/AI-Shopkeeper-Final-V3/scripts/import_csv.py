"""Import a sales CSV into the database from the command line.

Usage:
    python scripts/import_csv.py data/my_sales.csv            # add / update rows
    python scripts/import_csv.py data/my_sales.csv --reset    # wipe the DB first
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.database import init_db, reset_data  # noqa: E402
from backend.importer import import_sales  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--reset", action="store_true", help="delete all existing data first")
    args = ap.parse_args()

    init_db()
    if args.reset:
        reset_data()
        print("Existing sales, products, inventory and logged results deleted.")
    summary = import_sales(pd.read_csv(args.csv))
    print("Imported:", summary)


if __name__ == "__main__":
    main()
