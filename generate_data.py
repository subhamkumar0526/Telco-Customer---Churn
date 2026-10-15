"""
Generates a realistic (synthetic) retail sales dataset -> data/sales_raw.csv

Intentionally includes messy data (duplicates, missing values, inconsistent text)
so that the cleaning step in analysis.py is a real exercise.
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_ORDERS = 6000
N_CUSTOMERS = 900

CATALOG = {
    "Technology": {"Phones": (250, 900), "Laptops": (500, 1800), "Accessories": (10, 120)},
    "Furniture": {"Chairs": (60, 450), "Tables": (120, 800), "Bookcases": (80, 400)},
    "Office Supplies": {"Paper": (5, 40), "Binders": (4, 30), "Storage": (15, 150)},
}
# cost as a fraction of price (profit margin differs per category)
COST_RATIO = {"Technology": 0.78, "Furniture": 0.82, "Office Supplies": 0.62}

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_P = [0.20, 0.25, 0.15, 0.28, 0.12]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SHIP_MODES = ["Standard", "Second Class", "First Class", "Same Day"]

# --- dates with growth trend + festive-season spike (Oct-Dec) ---
days = pd.date_range("2023-01-01", "2025-12-31")
weights = np.linspace(1.0, 1.8, len(days))
weights *= np.where(days.month.isin([10, 11, 12]), 1.6, 1.0)
weights *= np.where(days.dayofweek >= 5, 1.2, 1.0)
weights /= weights.sum()
order_dates = rng.choice(days, size=N_ORDERS, p=weights)

customers = [f"CUST-{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
cust_segment = dict(zip(customers, rng.choice(SEGMENTS, N_CUSTOMERS, p=[0.52, 0.30, 0.18])))
cust_region = dict(zip(customers, rng.choice(REGIONS, N_CUSTOMERS, p=REGION_P)))
# some customers buy far more often (80/20 behaviour)
cust_p = rng.pareto(2.0, N_CUSTOMERS) + 1
cust_p /= cust_p.sum()

rows = []
for i in range(N_ORDERS):
    cust = rng.choice(customers, p=cust_p)
    category = rng.choice(list(CATALOG), p=[0.30, 0.25, 0.45])
    sub = rng.choice(list(CATALOG[category]))
    lo, hi = CATALOG[category][sub]
    price = round(rng.uniform(lo, hi), 2)
    qty = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]))
    discount = float(rng.choice([0, 0.05, 0.10, 0.20, 0.30, 0.40], p=[0.45, 0.15, 0.15, 0.12, 0.08, 0.05]))
    sales = round(price * qty * (1 - discount), 2)
    cost = round(price * qty * COST_RATIO[category] * rng.uniform(0.95, 1.05), 2)
    rows.append(
        {
            "order_id": f"ORD-{100000 + i}",
            "order_date": pd.Timestamp(order_dates[i]).strftime("%Y-%m-%d"),
            "customer_id": cust,
            "segment": cust_segment[cust],
            "region": cust_region[cust],
            "category": category,
            "sub_category": sub,
            "quantity": qty,
            "unit_price": price,
            "discount": discount,
            "ship_mode": rng.choice(SHIP_MODES, p=[0.55, 0.20, 0.17, 0.08]),
            "sales": sales,
            "cost": cost,
        }
    )

df = pd.DataFrame(rows)

# --- make it messy ---
df.loc[rng.choice(df.index, 120, replace=False), "region"] = np.nan
df.loc[rng.choice(df.index, 80, replace=False), "discount"] = np.nan
df.loc[rng.choice(df.index, 60, replace=False), "ship_mode"] = np.nan
df.loc[rng.choice(df.index, 150, replace=False), "category"] = df["category"].str.upper()
df.loc[rng.choice(df.index, 100, replace=False), "region"] = df["region"].str.lower()
df = pd.concat([df, df.sample(40, random_state=1)], ignore_index=True)  # duplicates
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/sales_raw.csv", index=False)
print(f"Saved data/sales_raw.csv  ->  {df.shape[0]} rows, {df.shape[1]} columns")
