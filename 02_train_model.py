"""
02_train_model.py
=================
Trains an XGBoost classifier to predict Late_delivery_risk.
Saves the trained model + feature list to models/delay_predictor.joblib
"""
import sqlite3, warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
from config import DB_PATH, MODEL_PATH, XGB_PARAMS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load completed orders
# ─────────────────────────────────────────────────────────────────────────────
print("📊  Loading completed orders from SQLite …")
conn = sqlite3.connect(DB_PATH)

# Pull all orders that have a resolved status
query = """
SELECT * FROM orders
WHERE Order_Status IN ('COMPLETE','CLOSED','SUSPECTED_FRAUD','CANCELED')
"""
try:
    df = pd.read_sql(query, conn)
except Exception:
    df = pd.read_sql("SELECT * FROM orders LIMIT 100000", conn)

conn.close()
print(f"    ✔  {len(df):,} rows loaded")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Identify target column (flexible)
# ─────────────────────────────────────────────────────────────────────────────
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

target_col = find_col(df, [
    "Late_delivery_risk","Late delivery risk","late_delivery_risk"
])
if target_col is None:
    raise SystemExit("❌  'Late_delivery_risk' column not found. Run 01_setup_database.py first.")

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)
print(f"    Target distribution:\n{df[target_col].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
print("🔧  Engineering features …")

numeric_candidates = [
    "Days_for_shipping_real","Days_for_shipping__real_",
    "Days_for_shipment_scheduled","Days_for_shipment__scheduled_",
    "Shipping_Delay_Days","Days_Since_Order",
    "Benefit_per_order","Sales","Order_Item_Quantity",
    "Order_Item_Discount_Rate","Order_Item_Profit_Ratio",
    "Order_Item_Total","Sales_per_customer",
    "Product_Price","Order_Profit_Per_Order",
]
categorical_candidates = [
    "Shipping_Mode","shipping_mode","Shipment_mode",
    "Category_Name","category_name","Product_Category_Name",
    "Customer_Segment","customer_segment",
    "Derived_State","Order_Status",
]

# Keep only columns that exist
num_cols = [c for c in numeric_candidates if c in df.columns]
cat_cols = [c for c in categorical_candidates if c in df.columns]

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

le = LabelEncoder()
for c in cat_cols:
    df[c] = le.fit_transform(df[c].astype(str))

feature_cols = num_cols + cat_cols
X = df[feature_cols]
y = df[target_col]

print(f"    Features used ({len(feature_cols)}): {feature_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train / test split
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("🚀  Training XGBoost …")
model = xgb.XGBClassifier(**XGB_PARAMS)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n📈  EVALUATION METRICS")
print("─" * 50)
print(classification_report(y_test, y_pred,
      target_names=["On Time", "Late"]))
print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

# 5-fold CV
cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5),
                            scoring="roc_auc", n_jobs=-1)
print(f"CV ROC-AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

disp = ConfusionMatrixDisplay(cm, display_labels=["On Time","Late"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix")

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances.nlargest(15).sort_values().plot(kind="barh", ax=axes[1], color="#2196F3")
axes[1].set_title("Top 15 Feature Importances")
axes[1].set_xlabel("XGBoost Importance Score")

plt.tight_layout()
fig.savefig("models/model_evaluation.png", dpi=150, bbox_inches="tight")
print("\n📊  Evaluation plot saved → models/model_evaluation.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save model + metadata
# ─────────────────────────────────────────────────────────────────────────────
artifact = {
    "model":        model,
    "feature_cols": feature_cols,
    "num_cols":     num_cols,
    "cat_cols":     cat_cols,
    "roc_auc":      roc_auc_score(y_test, y_proba),
}
joblib.dump(artifact, MODEL_PATH)
print(f"\n✅  Model saved → {MODEL_PATH}")
