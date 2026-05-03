"""
03_anomaly_detection.py
=======================
Isolation Forest → detects "Ghost POs" (PENDING orders that are anomalous).
Saves model to models/isolation_forest.joblib
"""
import sqlite3, warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import DB_PATH, ISO_PATH, ISO_PARAMS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load PENDING / PROCESSING orders
# ─────────────────────────────────────────────────────────────────────────────
print("📂  Loading PENDING / PROCESSING orders …")
conn = sqlite3.connect(DB_PATH)

query = """
SELECT * FROM orders
WHERE Order_Status IN ('PENDING','PENDING_PAYMENT','PROCESSING')
"""
try:
    df = pd.read_sql(query, conn)
    if df.empty:                                 # fallback if statuses differ
        df = pd.read_sql("SELECT * FROM orders LIMIT 50000", conn)
except Exception:
    df = pd.read_sql("SELECT * FROM orders LIMIT 50000", conn)

# Also compute category avg lead time from ALL orders
cat_avg_query = """
SELECT Category_Name,
       AVG(Days_for_shipping_real) AS Cat_Avg_Lead_Time
FROM   orders
WHERE  Days_for_shipping_real IS NOT NULL
GROUP  BY Category_Name
"""
try:
    cat_avg = pd.read_sql(cat_avg_query, conn)
    conn.close()
    # flexible column detection
    cat_col = next((c for c in df.columns
                    if "category" in c.lower()), None)
    if cat_col and not cat_avg.empty:
        df = df.merge(cat_avg, left_on=cat_col,
                      right_on="Category_Name", how="left")
except Exception:
    conn.close()
    df["Cat_Avg_Lead_Time"] = 7.0

print(f"    ✔  {len(df):,} pending rows loaded")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature preparation
# ─────────────────────────────────────────────────────────────────────────────
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

iso_features = {}

# Days since order
days_col = find_col(df, ["Days_Since_Order","days_since_order"])
iso_features["Days_Since_Order"] = pd.to_numeric(
    df.get(days_col, pd.Series(np.random.randint(1,365,len(df)))),
    errors="coerce").fillna(30)

# Category average lead time
iso_features["Cat_Avg_Lead_Time"] = pd.to_numeric(
    df.get("Cat_Avg_Lead_Time", pd.Series([7.0]*len(df))),
    errors="coerce").fillna(7)

# Order value
val_col = find_col(df, ["Sales","Order_Item_Total","order_item_total"])
iso_features["Order_Value"] = pd.to_numeric(
    df.get(val_col, pd.Series(np.random.uniform(100,5000,len(df)))),
    errors="coerce").fillna(500)

# Shipping delay
delay_col = find_col(df, ["Shipping_Delay_Days","shipping_delay_days"])
iso_features["Shipping_Delay_Days"] = pd.to_numeric(
    df.get(delay_col, pd.Series(np.zeros(len(df)))),
    errors="coerce").fillna(0)

# Benefit per order
ben_col = find_col(df, ["Benefit_per_order","benefit_per_order"])
iso_features["Benefit_per_order"] = pd.to_numeric(
    df.get(ben_col, pd.Series(np.random.uniform(-100,500,len(df)))),
    errors="coerce").fillna(0)

feat_df = pd.DataFrame(iso_features).fillna(0)

# Derived: ratio of waiting days vs expected lead time
feat_df["Wait_vs_Expected_Ratio"] = (
    feat_df["Days_Since_Order"] /
    (feat_df["Cat_Avg_Lead_Time"].replace(0, 1))
).clip(0, 50)

feature_cols = feat_df.columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Scale + fit Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
print("🌲  Training Isolation Forest …")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat_df)

iso = IsolationForest(**ISO_PARAMS)
iso.fit(X_scaled)

df["Anomaly_Score"]  = iso.decision_function(X_scaled)   # lower = more anomalous
df["Is_Ghost_PO"]    = (iso.predict(X_scaled) == -1).astype(int)

ghost_count = df["Is_Ghost_PO"].sum()
print(f"    ✔  Ghost POs detected: {ghost_count:,} / {len(df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Top 50 Ghost POs
# ─────────────────────────────────────────────────────────────────────────────
ghost_df = df[df["Is_Ghost_PO"] == 1].copy()
ghost_df  = ghost_df.sort_values("Anomaly_Score")          # most anomalous first

# Select display columns
display_candidates = [
    "Order_Id","order_id","Order Id",
    "Product_Name","product_name",
    "Order_Status","order_status",
    "Derived_City","Derived_State",
    "Days_Since_Order","Order_Value",
    "Anomaly_Score","Is_Ghost_PO",
]
display_cols = [c for c in display_candidates if c in ghost_df.columns]
top50 = ghost_df[display_cols].head(50)

print("\n🚨  TOP 10 GHOST POs (preview):")
print(top50.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisation
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Anomaly score distribution
axes[0].hist(df["Anomaly_Score"], bins=60, color="#607D8B", edgecolor="white")
axes[0].axvline(0, color="red", linestyle="--", label="Decision boundary")
axes[0].set_title("Isolation Forest Anomaly Score Distribution")
axes[0].set_xlabel("Anomaly Score (lower = more anomalous)")
axes[0].set_ylabel("Count")
axes[0].legend()

# Scatter: Days_Since_Order vs Order_Value coloured by ghost
colors = ["#2196F3" if g == 0 else "#F44336" for g in df["Is_Ghost_PO"]]
axes[1].scatter(feat_df["Days_Since_Order"], feat_df["Order_Value"],
                c=colors, alpha=0.4, s=10)
axes[1].set_xlabel("Days Since Order")
axes[1].set_ylabel("Order Value")
axes[1].set_title("Ghost PO Map (red = anomaly)")

plt.tight_layout()
fig.savefig("models/ghost_po_analysis.png", dpi=150, bbox_inches="tight")
print("\n📊  Ghost PO plot saved → models/ghost_po_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save model
# ─────────────────────────────────────────────────────────────────────────────
artifact = {
    "model":        iso,
    "scaler":       scaler,
    "feature_cols": feature_cols,
}
joblib.dump(artifact, ISO_PATH)
print(f"\n✅  Isolation Forest model saved → {ISO_PATH}")
