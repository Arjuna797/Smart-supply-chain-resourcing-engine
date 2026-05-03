"""
modules/vendor_scorer.py
========================
Weighted scoring engine to rank Alternate_Vendors for a flagged PO.
"""
import sqlite3
import pandas as pd
import numpy as np
from config import DB_PATH, WEIGHT_LEAD_TIME, WEIGHT_COST


def get_alternate_vendors(product_name: str, top_n: int = 3) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT  Vendor_Name, Vendor_State, Vendor_City,
                    Base_Price_INR, Lead_Time_Days, On_Time_Rate_Pct,
                    Preference_Rank
            FROM    Alternate_Vendors
            WHERE   Product_Name = ?
            ORDER BY Lead_Time_Days ASC
        """
        df = pd.read_sql(query, conn, params=(product_name,))
        conn.close()
    except Exception as e:
        print(f"get_alternate_vendors error: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    def norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9)

    df["Score_LeadTime"]  = 1 - norm(df["Lead_Time_Days"])
    df["Score_Cost"]      = 1 - norm(df["Base_Price_INR"])
    df["Composite_Score"] = (WEIGHT_LEAD_TIME * df["Score_LeadTime"] +
                             WEIGHT_COST      * df["Score_Cost"])
    df = df.sort_values("Composite_Score", ascending=False).head(top_n)
    df["Rank"] = range(1, len(df) + 1)

    return df[["Rank","Vendor_Name","Vendor_City","Vendor_State",
               "Lead_Time_Days","Base_Price_INR","On_Time_Rate_Pct",
               "Composite_Score"]]


def get_high_risk_orders(model_artifact: dict, threshold: float = 0.30,
                          limit: int = 500) -> pd.DataFrame:
    """Always returns a DataFrame — never None."""
    try:
        from sklearn.preprocessing import LabelEncoder

        if model_artifact is None:
            return pd.DataFrame()

        model        = model_artifact.get("model")
        feature_cols = model_artifact.get("feature_cols", [])
        num_cols     = model_artifact.get("num_cols", [])
        cat_cols     = model_artifact.get("cat_cols", [])

        if model is None or not feature_cols:
            return pd.DataFrame()

        conn = sqlite3.connect(DB_PATH)

        # Try PENDING first
        try:
            df = pd.read_sql(
                """SELECT * FROM orders
                   WHERE Order_Status IN
                   ('PENDING','PENDING_PAYMENT','PROCESSING',
                    'pending','processing','Pending','Processing')
                   LIMIT ?""",
                conn, params=(limit,))
        except Exception:
            df = pd.DataFrame()

        # Fallback 1 — all orders
        if df is None or df.empty:
            try:
                df = pd.read_sql(f"SELECT * FROM orders LIMIT {limit}", conn)
            except Exception:
                df = pd.DataFrame()

        # Fallback 2 — random sample
        if df is None or df.empty:
            try:
                df = pd.read_sql(
                    f"SELECT * FROM orders ORDER BY RANDOM() LIMIT {limit}", conn)
            except Exception:
                df = pd.DataFrame()

        conn.close()

        if df is None or df.empty:
            return pd.DataFrame()

        # Prepare features
        available_num = [c for c in num_cols if c in df.columns]
        if available_num:
            df[available_num] = df[available_num].apply(
                pd.to_numeric, errors="coerce").fillna(0)

        le = LabelEncoder()
        for c in cat_cols:
            if c in df.columns:
                df[c] = le.fit_transform(df[c].astype(str))
            else:
                df[c] = 0

        X = df.reindex(columns=feature_cols, fill_value=0)

        # Predict
        proba = model.predict_proba(X)
        if proba is None or len(proba) == 0:
            return pd.DataFrame()

        df["Delay_Probability"] = proba[:, 1]
        df["Risk_Level"] = pd.cut(
            df["Delay_Probability"],
            bins=[0, 0.35, 0.60, 1.0],
            labels=["Low ✅", "Medium ⚠", "High 🚨"]
        )

        high_risk = df[df["Delay_Probability"] >= threshold].copy()

        # Always show something — top 20 if nothing meets threshold
        if high_risk is None or high_risk.empty:
            high_risk = df.nlargest(20, "Delay_Probability").copy()
            high_risk["_fallback_note"] = (
                "⚠ No orders exceeded the threshold — showing top 20 highest-risk orders.")

        return high_risk.sort_values("Delay_Probability", ascending=False)

    except Exception as e:
        print(f"get_high_risk_orders error: {e}")
        return pd.DataFrame()
