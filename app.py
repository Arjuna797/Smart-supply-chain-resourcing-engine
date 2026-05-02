"""
app.py — Dynamic Supplier Re-Sourcing Engine + Ghost PO Anomaly Detector
Streamlit Dashboard
Run: streamlit run app.py
"""
import os, sys, sqlite3, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from config import DB_PATH, MODEL_PATH, ISO_PATH
from modules.vendor_scorer import get_alternate_vendors, get_high_risk_orders

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vendor Re-Sourcing Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main {background-color: #0f1117;}
  h1 {color: #1E88E5;}
  .stMetric {background: #1e2130; border-radius: 10px; padding: 12px;}
  .stMetric label {color: #90caf9 !important;}
  .block-container {padding-top: 1rem;}
  div[data-testid="stTab"] button {font-size: 16px; font-weight: 600;}
  .risk-high   {background:#ff1744;color:white;padding:4px 10px;border-radius:6px;font-weight:700;}
  .risk-medium {background:#ff6d00;color:white;padding:4px 10px;border-radius:6px;font-weight:700;}
  .risk-low    {background:#00c853;color:white;padding:4px 10px;border-radius:6px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_iso():
    if not os.path.exists(ISO_PATH):
        return None
    return joblib.load(ISO_PATH)

@st.cache_data(ttl=300)
def load_summary_stats():
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    stats = {}
    try:
        stats["total_orders"]   = pd.read_sql("SELECT COUNT(*) AS n FROM orders", conn).iloc[0,0]
        stats["pending_orders"] = pd.read_sql(
            "SELECT COUNT(*) AS n FROM orders WHERE Order_Status IN ('PENDING','PENDING_PAYMENT','PROCESSING')",
            conn).iloc[0,0]
        stats["total_vendors"]  = pd.read_sql(
            "SELECT COUNT(DISTINCT Vendor_Name) AS n FROM Alternate_Vendors", conn).iloc[0,0]
        stats["late_pct"] = pd.read_sql(
            "SELECT ROUND(AVG(CAST(Late_delivery_risk AS FLOAT))*100,1) AS p FROM orders", conn).iloc[0,0]
    except Exception:
        pass
    conn.close()
    return stats

@st.cache_data(ttl=60)
def load_orders_sample(n=2000):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM orders LIMIT ?", conn, params=(n,))
    conn.close()
    return df

@st.cache_data(ttl=60)
def load_ghost_orders(n=500):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM orders WHERE Order_Status IN ('PENDING','PENDING_PAYMENT','PROCESSING') LIMIT ?",
            conn, params=(n,))
    except Exception:
        df = pd.read_sql("SELECT * FROM orders LIMIT ?", conn, params=(n,))
    conn.close()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=60)
    st.title("⚙ Control Panel")
    st.divider()

    db_ok    = os.path.exists(DB_PATH)
    model_ok = os.path.exists(MODEL_PATH)
    iso_ok   = os.path.exists(ISO_PATH)

    st.markdown("**System Status**")
    st.markdown(f"{'🟢' if db_ok else '🔴'}  Database:  {'Ready' if db_ok else 'Not Found'}")
    st.markdown(f"{'🟢' if model_ok else '🔴'}  Delay Model: {'Loaded' if model_ok else 'Not Trained'}")
    st.markdown(f"{'🟢' if iso_ok else '🔴'}  Ghost PO Model: {'Loaded' if iso_ok else 'Not Trained'}")

    if not db_ok:
        st.error("Run `01_setup_database.py` first!")
    elif not model_ok:
        st.warning("Run `02_train_model.py` to enable predictions.")
    elif not iso_ok:
        st.warning("Run `03_anomaly_detection.py` for Ghost PO detection.")

    st.divider()
    risk_threshold = st.slider("🎯 High-Risk Threshold", 0.40, 0.95, 0.60, 0.05)
    top_n_vendors  = st.slider("🏭 Vendors to Show",     1,    5,    3)
    max_orders     = st.slider("📦 Max Orders to Scan",  50, 500, 200, 50)
    st.divider()
    st.caption("Dynamic Supplier Re-Sourcing Engine v1.0\n📍 Pune, Maharashtra")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER + KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏭 Dynamic Supplier Re-Sourcing Engine")
st.caption("Predictive Delay Detection  •  Ghost PO Anomaly Detector  •  India Supply Chain Intelligence")

if db_ok:
    stats = load_summary_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Orders",   f"{stats.get('total_orders',0):,}")
    c2.metric("⏳ Pending Orders", f"{stats.get('pending_orders',0):,}")
    c3.metric("🏭 Alt. Vendors",   f"{stats.get('total_vendors',0):,}")
    c4.metric("⚠ Late Delivery %", f"{stats.get('late_pct',0):.1f}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔄 Re-Sourcing Engine",
    "👻 Ghost PO Cleaner",
    "📊 Analytics Dashboard",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RE-SOURCING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔄 Supplier Re-Sourcing Engine")
    st.markdown(
        "Scans **PENDING** orders through the XGBoost delay predictor and "
        "surfaces alternative Indian vendors ranked by lead-time & cost."
    )

    model_artifact = load_model()
    if model_artifact is None:
        st.error("⚠ Delay Prediction model not found. Run `02_train_model.py` first.")
    else:
        col_btn, col_info = st.columns([2, 5])
        scan_btn = col_btn.button("🚀 Scan High-Risk POs", type="primary", use_container_width=True)
        col_info.info(f"Risk threshold: **{risk_threshold:.0%}** · Scanning up to **{max_orders}** orders")

        if scan_btn or "high_risk_df" in st.session_state:
            if scan_btn:
                with st.spinner("Running XGBoost predictions on pending orders …"):
                    hr_df = get_high_risk_orders(model_artifact, risk_threshold, max_orders)
                    st.session_state["high_risk_df"] = hr_df

            hr_df = st.session_state.get("high_risk_df", pd.DataFrame())

            if hr_df.empty:
                st.success("✅ No high-risk orders found at current threshold.")
            else:
                st.error(f"🚨  **{len(hr_df)}** high-risk orders detected!")

                # Display columns
                display_cols = [c for c in [
                    "Order_Id","Product_Name","Derived_City","Derived_State",
                    "Order_Status","Days_Since_Order","Delay_Probability","Risk_Level"
                ] if c in hr_df.columns]

                display_df = hr_df[display_cols].copy() if display_cols else hr_df.head(20)
                if "Delay_Probability" in display_df:
                    display_df["Delay_Probability"] = display_df["Delay_Probability"].apply(lambda x: f"{x:.1%}")

                st.dataframe(display_df, use_container_width=True, height=280)

                # ── Vendor finder ──────────────────────────────────────────
                st.subheader("🏭 Find Alternate Vendor")
                prod_col = next((c for c in hr_df.columns if "Product_Name" in c), None)

                if prod_col:
                    products = hr_df[prod_col].dropna().unique().tolist()
                    sel_prod = st.selectbox("Select flagged product:", products)

                    if st.button("🔍 Get Alternate Vendors", type="secondary"):
                        vendors = get_alternate_vendors(sel_prod, top_n_vendors)
                        if vendors.empty:
                            st.warning("No alternate vendors found for this product.")
                        else:
                            st.success(f"✅  Top {len(vendors)} alternate vendors for **{sel_prod}**")

                            # Colour score column
                            def colour_score(val):
                                clr = "#00c853" if val >= 0.7 else "#ff6d00" if val >= 0.4 else "#f44336"
                                return f"background-color:{clr};color:white;font-weight:700"

                            styled = vendors.style.applymap(
                                colour_score, subset=["Composite_Score"]
                            ).format({"Base_Price_INR": "₹{:,.0f}",
                                      "Composite_Score": "{:.3f}"})
                            st.dataframe(styled, use_container_width=True)

                            # Bar chart
                            fig = px.bar(
                                vendors, x="Vendor_Name", y="Composite_Score",
                                color="Lead_Time_Days",
                                color_continuous_scale="Blues_r",
                                title=f"Vendor Score Comparison — {sel_prod}",
                                labels={"Composite_Score":"Score","Lead_Time_Days":"Lead Days"},
                                template="plotly_dark",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Product name column not available in this dataset slice.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GHOST PO CLEANER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("👻 Ghost PO Anomaly Detector")
    st.markdown(
        "Uses **Isolation Forest** to find POs stuck in *Processing/Pending* "
        "that deviate from normal operational patterns — freeing locked working capital."
    )

    iso_artifact = load_iso()
    if iso_artifact is None:
        st.error("⚠ Isolation Forest model not found. Run `03_anomaly_detection.py` first.")
    else:
        run_ghost = st.button("🔍 Detect Ghost POs", type="primary")

        if run_ghost or "ghost_result" in st.session_state:
            if run_ghost:
                with st.spinner("Running Isolation Forest …"):
                    pend_df = load_ghost_orders(500)

                    if not pend_df.empty:
                        iso   = iso_artifact["model"]
                        scaler= iso_artifact["scaler"]
                        fcols = iso_artifact["feature_cols"]

                        # Rebuild feature matrix
                        feat = pd.DataFrame(index=pend_df.index)
                        for c in fcols:
                            if c in pend_df.columns:
                                feat[c] = pd.to_numeric(pend_df[c], errors="coerce").fillna(0)
                            else:
                                feat[c] = 0.0

                        X_s = scaler.transform(feat)
                        pend_df["Anomaly_Score"] = iso.decision_function(X_s)
                        pend_df["Is_Ghost_PO"]   = (iso.predict(X_s) == -1).astype(int)
                        st.session_state["ghost_result"] = pend_df

            ghost_df = st.session_state.get("ghost_result", pd.DataFrame())

            if ghost_df.empty:
                st.success("No pending orders found.")
            else:
                ghosts = ghost_df[ghost_df["Is_Ghost_PO"] == 1].sort_values("Anomaly_Score")
                normal = ghost_df[ghost_df["Is_Ghost_PO"] == 0]

                g1, g2, g3 = st.columns(3)
                g1.metric("Total Pending",    f"{len(ghost_df):,}")
                g2.metric("🚨 Ghost POs",     f"{len(ghosts):,}", delta=f"{len(ghosts)/len(ghost_df):.1%}")
                g3.metric("✅ Normal Orders", f"{len(normal):,}")

                # Anomaly score distribution
                fig_hist = px.histogram(
                    ghost_df, x="Anomaly_Score", color="Is_Ghost_PO",
                    barmode="overlay", nbins=60,
                    color_discrete_map={0:"#1E88E5", 1:"#F44336"},
                    labels={"Is_Ghost_PO":"Ghost PO","Anomaly_Score":"Anomaly Score"},
                    title="Anomaly Score Distribution (Red = Ghost POs)",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.subheader("🚨 Top Ghost POs — Recommended for Cancellation")
                disp_cols = [c for c in [
                    "Order_Id","Product_Name","Order_Status",
                    "Derived_City","Derived_State",
                    "Days_Since_Order","Anomaly_Score"
                ] if c in ghosts.columns]

                top_ghosts = ghosts[disp_cols].head(50) if disp_cols else ghosts.head(50)

                if "Anomaly_Score" in top_ghosts.columns:
                    top_ghosts["Anomaly_Score"] = top_ghosts["Anomaly_Score"].round(4)

                st.dataframe(top_ghosts, use_container_width=True, height=320)

                # Scatter
                if "Days_Since_Order" in ghost_df.columns:
                    val_col = next((c for c in ["Sales","Order_Item_Total"] if c in ghost_df.columns), None)
                    if val_col:
                        fig_sc = px.scatter(
                            ghost_df.sample(min(1000, len(ghost_df))),
                            x="Days_Since_Order", y=val_col,
                            color=ghost_df["Is_Ghost_PO"].map({0:"Normal",1:"Ghost PO"}).values[:min(1000,len(ghost_df))],
                            color_discrete_map={"Normal":"#1E88E5","Ghost PO":"#F44336"},
                            title="Ghost PO Map — Days Pending vs Order Value",
                            template="plotly_dark", opacity=0.6,
                        )
                        st.plotly_chart(fig_sc, use_container_width=True)

                st.warning(
                    f"💡 Bulk-cancelling these **{len(ghosts)}** Ghost POs "
                    "would unfreeze working capital and clean master data."
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Analytics Dashboard")

    if not db_ok:
        st.error("Database not found.")
    else:
        df_sample = load_orders_sample(3000)

        if df_sample.empty:
            st.info("No data loaded yet.")
        else:
            col1, col2 = st.columns(2)

            # ── Order Status breakdown ─────────────────────────────────────
            status_col = next((c for c in df_sample.columns if "Order_Status" in c), None)
            if status_col:
                status_counts = df_sample[status_col].value_counts().reset_index()
                status_counts.columns = ["Status","Count"]
                fig_pie = px.pie(status_counts, names="Status", values="Count",
                                 title="Order Status Distribution",
                                 template="plotly_dark", hole=0.4)
                col1.plotly_chart(fig_pie, use_container_width=True)

            # ── Late delivery by state ─────────────────────────────────────
            late_col = next((c for c in df_sample.columns if "Late_delivery_risk" in c), None)
            if late_col and "Derived_State" in df_sample.columns:
                df_sample[late_col] = pd.to_numeric(df_sample[late_col], errors="coerce")
                state_late = (df_sample.groupby("Derived_State")[late_col]
                              .mean().mul(100).round(1).reset_index())
                state_late.columns = ["State","Late_%"]
                fig_bar = px.bar(state_late.sort_values("Late_%", ascending=False),
                                 x="State", y="Late_%",
                                 color="Late_%", color_continuous_scale="Reds",
                                 title="Late Delivery Rate by Indian State (%)",
                                 template="plotly_dark")
                col2.plotly_chart(fig_bar, use_container_width=True)

            # ── Shipping mode ──────────────────────────────────────────────
            ship_col = next((c for c in df_sample.columns
                             if "shipping_mode" in c.lower() or "Shipping_Mode" in c), None)
            if ship_col and late_col:
                ship_late = (df_sample.groupby(ship_col)[late_col]
                             .agg(["mean","count"]).reset_index())
                ship_late.columns = ["Shipping_Mode","Late_Rate","Count"]
                ship_late["Late_Rate"] *= 100
                fig_mode = px.bar(ship_late, x="Shipping_Mode", y="Late_Rate",
                                  size="Count", color="Late_Rate",
                                  color_continuous_scale="OrRd",
                                  title="Late Delivery % by Shipping Mode",
                                  template="plotly_dark")
                st.plotly_chart(fig_mode, use_container_width=True)

            # ── Vendor table preview ───────────────────────────────────────
            st.subheader("🏭 Alternate Vendor Database Preview")
            conn = sqlite3.connect(DB_PATH)
            try:
                vdf = pd.read_sql("SELECT * FROM Alternate_Vendors LIMIT 100", conn)
                st.dataframe(vdf, use_container_width=True, height=250)
            except Exception:
                st.info("Vendor table not yet created.")
            finally:
                conn.close()
