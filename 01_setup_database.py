"""
01_setup_database.py
====================
• Loads DataCoSupplyChainDataset.csv into SQLite  (table: orders)
• Overwrites geography with India-specific hubs
• Creates synthetic Alternate_Vendors table (2-3 vendors per product)
Run once before anything else.
"""
import sqlite3, random, re
import pandas as pd
import numpy as np
from config import (DB_PATH, CSV_PATH, INDIAN_STATES, INDIAN_CITIES,
                    VENDOR_POOL)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load CSV
# ─────────────────────────────────────────────────────────────────────────────
print("📂  Loading DataCo CSV …")
try:
    df = pd.read_csv(CSV_PATH, encoding="latin-1", low_memory=False)
except FileNotFoundError:
    raise SystemExit(
        f"\n❌  CSV not found at:\n    {CSV_PATH}\n"
        "    Place DataCoSupplyChainDataset.csv inside the Data\\ folder."
    )

print(f"    ✔  {len(df):,} rows  |  {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Normalise column names (strip spaces / special chars)
# ─────────────────────────────────────────────────────────────────────────────
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
      .str.replace(r"__+", "_", regex=True)
      .str.strip("_")
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. India geography overwrite
# ─────────────────────────────────────────────────────────────────────────────
print("🇮🇳  Applying India geography overlay …")
rng = np.random.default_rng(42)

def random_state():
    return random.choice(INDIAN_STATES)

def random_city(state):
    return random.choice(INDIAN_CITIES.get(state, ["Mumbai"]))

states = [random_state() for _ in range(len(df))]
cities = [random_city(s) for s in states]

# Try common column name variants
for col in ["Order_Country","order_country","Order Country"]:
    if col in df.columns:
        df[col] = states
for col in ["Order_City","order_city","Order City"]:
    if col in df.columns:
        df[col] = cities

df["Derived_State"] = states
df["Derived_City"]  = cities

# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────
# Identify real/scheduled shipping day columns (flexible naming)
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

real_col  = find_col(df, ["Days_for_shipping_real","Days_for_shipping__real_","Days for shipping (real)"])
sched_col = find_col(df, ["Days_for_shipment_scheduled","Days_for_shipment__scheduled_","Days for shipment (scheduled)"])

if real_col and sched_col:
    df["Shipping_Delay_Days"] = pd.to_numeric(df[real_col], errors="coerce") - \
                                pd.to_numeric(df[sched_col], errors="coerce")
else:
    df["Shipping_Delay_Days"] = 0
    print("    ⚠  Could not find shipping day columns; Shipping_Delay_Days set to 0")

# Order date parsing → Days_Since_Order_Placed (for Ghost PO detection)
date_col = find_col(df, ["order_date_DateOrders","Order_Date","order date (DateOrders)"])
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    reference_date = df[date_col].max()
    df["Days_Since_Order"] = (reference_date - df[date_col]).dt.days.fillna(0).astype(int)
else:
    df["Days_Since_Order"] = rng.integers(1, 365, size=len(df))
    print("    ⚠  Order date column not found; Days_Since_Order randomised")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Write to SQLite
# ─────────────────────────────────────────────────────────────────────────────
print(f"💾  Writing {len(df):,} rows → SQLite …")
conn = sqlite3.connect(DB_PATH)
df.to_sql("orders", conn, if_exists="replace", index=False, chunksize=5000)

# Index for speed
conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON orders(Order_Status)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_product ON orders(Product_Name)")
conn.commit()
print("    ✔  Table 'orders' created")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Alternate_Vendors synthetic table
# ─────────────────────────────────────────────────────────────────────────────
print("🏭  Building Alternate_Vendors table …")

product_col = find_col(df, ["Product_Name","product_name","Product Name"])
category_col= find_col(df, ["Category_Name","category_name","Category Name","Product_Category_Name"])

if product_col:
    products = df[product_col].dropna().unique().tolist()
else:
    products = [f"Product_{i}" for i in range(200)]

rows = []
for prod in products:
    cat = ""
    if category_col:
        mask = df[product_col] == prod
        vals = df.loc[mask, category_col].dropna()
        cat  = vals.iloc[0] if len(vals) else ""

    # 2-3 vendors per product
    n_vendors = random.randint(2, 3)
    selected  = random.sample(VENDOR_POOL, min(n_vendors, len(VENDOR_POOL)))
    for rank, vendor in enumerate(selected, start=1):
        state = random.choice(INDIAN_STATES)
        rows.append({
            "Product_Name":      prod,
            "Category":          cat,
            "Vendor_Name":       vendor,
            "Vendor_State":      state,
            "Vendor_City":       random_city(state),
            "Base_Price_INR":    round(random.uniform(500, 50000), 2),
            "Lead_Time_Days":    random.randint(3, 21),
            "On_Time_Rate_Pct":  round(random.uniform(70, 99), 1),
            "Preference_Rank":   rank,
        })

vendors_df = pd.DataFrame(rows)
vendors_df.to_sql("Alternate_Vendors", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_vend_prod ON Alternate_Vendors(Product_Name)")
conn.commit()
conn.close()

print(f"    ✔  {len(vendors_df):,} vendor records for {len(products):,} products")
print("\n✅  Database setup complete!")
print(f"    DB : {DB_PATH}")
