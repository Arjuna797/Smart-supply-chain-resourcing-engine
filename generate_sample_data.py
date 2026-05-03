"""
generate_sample_data.py
=======================
Generates a realistic synthetic DataCo-like CSV when the real
DataCoSupplyChainDataset.csv is not available (e.g. on Streamlit Cloud).
Produces ~20,000 rows with all columns the project expects.
"""
import os
import random
import numpy as np
import pandas as pd
from config import CSV_PATH, DATA_DIR, INDIAN_STATES, INDIAN_CITIES

os.makedirs(DATA_DIR, exist_ok=True)

SHIPPING_MODES  = ["Standard Class", "Second Class", "First Class", "Same Day"]
ORDER_STATUSES  = ["COMPLETE", "CLOSED", "PENDING", "PROCESSING",
                   "PENDING_PAYMENT", "CANCELED", "SUSPECTED_FRAUD"]
STATUS_WEIGHTS  = [0.45, 0.20, 0.10, 0.08, 0.07, 0.06, 0.04]
CATEGORIES      = ["Electronics","Furniture","Office Supplies","Clothing",
                   "Auto Parts","Pharmaceuticals","Food & Beverage",
                   "Industrial Equipment","Chemicals","Textiles"]
PRODUCTS = {
    "Electronics":          ["Laptop","Mobile Phone","Tablet","LED Monitor","Printer"],
    "Furniture":            ["Office Chair","Standing Desk","Bookshelf","Filing Cabinet"],
    "Office Supplies":      ["A4 Paper Ream","Pen Set","Stapler","Whiteboard"],
    "Clothing":             ["Safety Jacket","Work Boots","Gloves","Helmet"],
    "Auto Parts":           ["Brake Pads","Engine Oil Filter","Spark Plug","Alternator"],
    "Pharmaceuticals":      ["Paracetamol Bulk","IV Saline","Surgical Gloves","Bandages"],
    "Food & Beverage":      ["Packaged Rice","Refined Oil","Sugar 50kg","Tea Leaves"],
    "Industrial Equipment": ["Hydraulic Jack","Air Compressor","Drill Machine","Lathe Part"],
    "Chemicals":            ["Hydrochloric Acid","Sodium Hydroxide","Acetone","Ethanol"],
    "Textiles":             ["Cotton Fabric","Polyester Roll","Denim Cloth","Silk Blend"],
}
SEGMENTS = ["Consumer", "Corporate", "Home Office"]

def random_city(state):
    return random.choice(INDIAN_CITIES.get(state, ["Mumbai"]))

print(f"🔧  Generating synthetic DataCo dataset (~20,000 rows) …")
rng   = np.random.default_rng(42)
N     = 20_000
rows  = []

for i in range(N):
    cat      = random.choice(CATEGORIES)
    product  = random.choice(PRODUCTS[cat])
    state    = random.choice(INDIAN_STATES)
    city     = random_city(state)
    mode     = random.choice(SHIPPING_MODES)
    status   = random.choices(ORDER_STATUSES, weights=STATUS_WEIGHTS)[0]
    sched    = random.randint(2, 10)
    real     = sched + random.randint(-1, 5)
    real     = max(1, real)
    late     = 1 if real > sched else 0
    qty      = random.randint(1, 50)
    price    = round(random.uniform(200, 15000), 2)
    discount = round(random.uniform(0, 0.30), 4)
    profit_r = round(random.uniform(-0.10, 0.35), 4)
    sales    = round(price * qty * (1 - discount), 2)
    profit   = round(sales * profit_r, 2)
    benefit  = round(random.uniform(-500, 2000), 2)
    days_ago = random.randint(1, 730)
    order_dt = pd.Timestamp("2024-01-01") - pd.Timedelta(days=days_ago)

    rows.append({
        "Order_Id":                    i + 100000,
        "Order_Status":                status,
        "order_date_DateOrders":       order_dt.strftime("%m/%d/%Y"),
        "Shipping_Mode":               mode,
        "Days_for_shipping_real":      real,
        "Days_for_shipment_scheduled": sched,
        "Late_delivery_risk":          late,
        "Category_Name":               cat,
        "Product_Name":                product,
        "Order_Item_Quantity":         qty,
        "Product_Price":               price,
        "Order_Item_Discount_Rate":    discount,
        "Order_Item_Profit_Ratio":     profit_r,
        "Order_Item_Total":            sales,
        "Sales":                       sales,
        "Order_Profit_Per_Order":      profit,
        "Benefit_per_order":           benefit,
        "Sales_per_customer":          round(sales * random.uniform(0.8, 1.2), 2),
        "Customer_Segment":            random.choice(SEGMENTS),
        "Order_Country":               state,
        "Order_City":                  city,
        "Derived_State":               state,
        "Derived_City":                city,
    })

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)
print(f"✅  Synthetic dataset saved → {CSV_PATH}  ({len(df):,} rows)")
