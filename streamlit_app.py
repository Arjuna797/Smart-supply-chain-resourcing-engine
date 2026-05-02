"""
streamlit_app.py
================
Entry point for Streamlit Cloud deployment.
Auto-initialises database and trains models on first run.
Does NOT modify any existing project files.
"""
import os, sys, sqlite3, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from config import DB_PATH, CSV_PATH, MODEL_PATH, ISO_PATH, DATA_DIR, MODELS_DIR

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-INITIALISATION  (runs silently on first deploy)
# ─────────────────────────────────────────────────────────────────────────────
def auto_init():
    steps = []

    # 1. Generate synthetic data if real CSV missing
    if not os.path.exists(CSV_PATH):
        steps.append("📂 Generating synthetic supply chain dataset …")

    # 2. Build database
    if not os.path.exists(DB_PATH):
        steps.append("💾 Building SQLite database …")

    # 3. Train models
    if not os.path.exists(MODEL_PATH):
        steps.append("🤖 Training XGBoost delay model …")

    if not os.path.exists(ISO_PATH):
        steps.append("🌲 Training Isolation Forest Ghost PO model …")

    if not steps:
        return   # Everything already initialised

    placeholder = st.empty()
    with placeholder.container():
        st.title("🏭 Smart Supply Chain Re-Sourcing Engine")
        st.info("⏳ **First-time setup running** — this takes ~2 minutes. Please wait …")
        progress = st.progress(0)
        status   = st.empty()

        # Step 1 — Generate data
        if not os.path.exists(CSV_PATH):
            status.write("📂 Generating synthetic DataCo dataset (20,000 orders) …")
            progress.progress(10)
            import generate_sample_data   # noqa — runs on import

        # Step 2 — Build DB
        if not os.path.exists(DB_PATH):
            status.write("💾 Loading data into SQLite database …")
            progress.progress(35)
            import importlib, importlib.util
            spec = importlib.util.spec_from_file_location(
                "setup_db", os.path.join(os.path.dirname(__file__), "01_setup_database.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        # Step 3 — Train XGBoost
        if not os.path.exists(MODEL_PATH):
            status.write("🤖 Training XGBoost delay predictor …")
            progress.progress(60)
            spec = importlib.util.spec_from_file_location(
                "train_model", os.path.join(os.path.dirname(__file__), "02_train_model.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        # Step 4 — Train Isolation Forest
        if not os.path.exists(ISO_PATH):
            status.write("🌲 Training Isolation Forest Ghost PO detector …")
            progress.progress(85)
            spec = importlib.util.spec_from_file_location(
                "anomaly", os.path.join(os.path.dirname(__file__), "03_anomaly_detection.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        progress.progress(100)
        status.write("✅ All models ready!")

    placeholder.empty()
    st.rerun()

auto_init()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MAIN APP after init is complete
# ─────────────────────────────────────────────────────────────────────────────
from modules.vendor_scorer import get_alternate_vendors, get_high_risk_orders

st.set_page_config(
    page_title="Smart Supply Chain Re-Sourcing Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main  {background-color:#0f1117;}
  h1     {color:#1E88E5;}
  .stMetric {background:#1e2130;border-radius:10px;padding:12px;}
  .stMetric label {color:#90caf9!important;}
  .block-container {padding-top:1rem;}
  div[data-testid="stTab"] button {font-size:16px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@st.cache_resource
def load_iso():
    return joblib.load(ISO_PATH) if os.path.exists(ISO_PATH) else None

@st.cache_data(ttl=300)
def load_summary_stats():
    conn  = sqlite3.connect(DB_PATH)
    stats = {}
    try:
        stats["total_orders"]   = pd.read_sql("SELECT COUNT(*) AS n FROM orders", conn).iloc[0,0]
        stats["pending_orders"] = pd.read_sql(
            "SELECT COUNT(*) AS n FROM orders WHERE Order_Status IN ('PENDING','PENDING_PAYMENT','PROCESSING')",
            conn).iloc[0,0]
        stats["total_vendors"]  = pd.read_sql(
            "SELECT COUNT(DISTINCT Vendor_Name) AS n FROM Alternate_Vendors", conn).iloc[0,0]
        stats["late_pct"]       = pd.read_sql(
            "SELECT ROUND(AVG(CAST(Late_delivery_risk AS FLOAT))*100,1) AS p FROM orders",
            conn).iloc[0,0]
    except Exception:
        pass
    conn.close()
    return stats

@st.cache_data(ttl=60)
def load_orders_sample(n=2000):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM orders LIMIT {n}", conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def load_ghost_orders(n=500):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM orders WHERE Order_Status IN "
            "('PENDING','PENDING_PAYMENT','PROCESSING') LIMIT ?",
            conn, params=(n,))
        if df.empty:
            df = pd.read_sql(f"SELECT * FROM orders LIMIT {n}", conn)
    except Exception:
        df = pd.read_sql(f"SELECT * FROM orders LIMIT {n}", conn)
    conn.close()
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=60)
    st.title("⚙ Control Panel")
    st.divider()

    db_ok    = os.path.exists(DB_PATH)
    model_ok = os.path.exists(MODEL_PATH)
    iso_ok   = os.path.exists(ISO_PATH)

    st.markdown("**System Status**")
    st.markdown(f"{'🟢' if db_ok    else '🔴'}  Database")
    st.markdown(f"{'🟢' if model_ok else '🔴'}  XGBoost Delay Model")
    st.markdown(f"{'🟢' if iso_ok   else '🔴'}  Ghost PO Model")

    st.divider()
    risk_threshold = st.slider("🎯 High-Risk Threshold", 0.10, 0.95, 0.30, 0.05)
    top_n_vendors  = st.slider("🏭 Vendors to Show",     1,    5,    3)
    max_orders     = st.slider("📦 Max Orders to Scan",  50, 1000, 500, 50)
    st.divider()
    st.caption("Smart Supply Chain Re-Sourcing Engine\n📍 India Operations Intelligence")

# ── Header KPIs ───────────────────────────────────────────────────────────────
st.title("🏭 Smart Supply Chain Re-Sourcing Engine")
st.caption("Predictive Delay Detection • Ghost PO Detector • India Vendor Intelligence")

if db_ok:
    stats = load_summary_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Orders",   f"{stats.get('total_orders',0):,}")
    c2.metric("⏳ Pending Orders", f"{stats.get('pending_orders',0):,}")
    c3.metric("🏭 Alt. Vendors",   f"{stats.get('total_vendors',0):,}")
    c4.metric("⚠ Late Delivery %", f"{stats.get('late_pct',0):.1f}%")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔄 Re-Sourcing Engine",
    "👻 Ghost PO Cleaner",
    "📊 Analytics Dashboard",
])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔄 Supplier Re-Sourcing Engine")
    st.markdown("Scans orders through XGBoost and surfaces top Indian alternate vendors.")

    model_artifact = load_model()
    if model_artifact is None:
        st.error("Model not ready. Please refresh the page.")
    else:
        col_btn, col_info = st.columns([2, 5])
        scan_btn = col_btn.button("🚀 Scan High-Risk POs", type="primary", use_container_width=True)
        col_info.info(f"Threshold: **{risk_threshold:.0%}** · Scanning **{max_orders}** orders")

        if scan_btn or "high_risk_df" in st.session_state:
            if scan_btn:
                with st.spinner("Running XGBoost predictions …"):
                    hr_df = get_high_risk_orders(model_artifact, risk_threshold, max_orders)
                    st.session_state["high_risk_df"] = hr_df

            hr_df = st.session_state.get("high_risk_df", pd.DataFrame())

            if hr_df.empty:
                st.warning("No orders found.")
            else:
                is_fallback = "_fallback_note" in hr_df.columns
                if is_fallback:
                    st.warning(hr_df["_fallback_note"].iloc[0])
                    st.info(f"Showing top **{len(hr_df)}** highest-risk orders.")
                else:
                    st.error(f"🚨 **{len(hr_df)}** orders exceed **{risk_threshold:.0%}** risk threshold!")

                display_cols = [c for c in [
                    "Order_Id","Product_Name","Derived_City","Derived_State",
                    "Order_Status","Days_Since_Order","Delay_Probability","Risk_Level"
                ] if c in hr_df.columns]

                display_df = hr_df[display_cols].copy() if display_cols else hr_df.head(20)
                if "Delay_Probability" in display_df.columns:
                    display_df["Delay_Probability"] = display_df["Delay_Probability"].apply(
                        lambda x: f"{x:.1%}")
                st.dataframe(display_df, use_container_width=True, height=280)

                st.subheader("🏭 Find Alternate Vendor")
                prod_col = next((c for c in hr_df.columns if "Product_Name" in c), None)
                if prod_col:
                    products = hr_df[prod_col].dropna().unique().tolist()
                    sel_prod = st.selectbox("Select flagged product:", products)
                    if st.button("🔍 Get Alternate Vendors", type="secondary"):
                        vendors = get_alternate_vendors(sel_prod, top_n_vendors)
                        if vendors.empty:
                            st.warning("No alternate vendors found.")
                        else:
                            st.success(f"✅ Top {len(vendors)} vendors for **{sel_prod}**")
                            def colour_score(val):
                                c = "#00c853" if val>=0.7 else "#ff6d00" if val>=0.4 else "#f44336"
                                return f"background-color:{c};color:white;font-weight:700"
                            styled = vendors.style.applymap(
                                colour_score, subset=["Composite_Score"]
                            ).format({"Base_Price_INR":"₹{:,.0f}","Composite_Score":"{:.3f}"})
                            st.dataframe(styled, use_container_width=True)
                            fig = px.bar(vendors, x="Vendor_Name", y="Composite_Score",
                                         color="Lead_Time_Days",
                                         color_continuous_scale="Blues_r",
                                         title=f"Vendor Scores — {sel_prod}",
                                         template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.header("👻 Ghost PO Anomaly Detector")
    st.markdown("Isolation Forest detects POs stuck abnormally long — freeing working capital.")

    iso_artifact = load_iso()
    if iso_artifact is None:
        st.error("Ghost PO model not ready. Please refresh.")
    else:
        if st.button("🔍 Detect Ghost POs", type="primary"):
            with st.spinner("Running Isolation Forest …"):
                pend_df = load_ghost_orders(500)
                if not pend_df.empty:
                    iso    = iso_artifact["model"]
                    scaler = iso_artifact["scaler"]
                    fcols  = iso_artifact["feature_cols"]
                    feat   = pd.DataFrame(index=pend_df.index)
                    for c in fcols:
                        feat[c] = pd.to_numeric(pend_df.get(c, 0), errors="coerce").fillna(0)
                    X_s = scaler.transform(feat)
                    pend_df["Anomaly_Score"] = iso.decision_function(X_s)
                    pend_df["Is_Ghost_PO"]   = (iso.predict(X_s) == -1).astype(int)
                    st.session_state["ghost_result"] = pend_df

        if "ghost_result" in st.session_state:
            ghost_df = st.session_state["ghost_result"]
            ghosts = ghost_df[ghost_df["Is_Ghost_PO"]==1].sort_values("Anomaly_Score")
            normal = ghost_df[ghost_df["Is_Ghost_PO"]==0]

            g1, g2, g3 = st.columns(3)
            g1.metric("Total Scanned",   f"{len(ghost_df):,}")
            g2.metric("🚨 Ghost POs",    f"{len(ghosts):,}")
            g3.metric("✅ Normal Orders", f"{len(normal):,}")

            fig_hist = px.histogram(ghost_df, x="Anomaly_Score", color="Is_Ghost_PO",
                                    barmode="overlay", nbins=60,
                                    color_discrete_map={0:"#1E88E5",1:"#F44336"},
                                    title="Anomaly Score Distribution",
                                    template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)

            disp_cols = [c for c in ["Order_Id","Product_Name","Order_Status",
                                      "Derived_City","Days_Since_Order","Anomaly_Score"]
                         if c in ghosts.columns]
            top_g = ghosts[disp_cols].head(50).copy() if disp_cols else ghosts.head(50)
            if "Anomaly_Score" in top_g.columns:
                top_g["Anomaly_Score"] = top_g["Anomaly_Score"].round(4)
            st.dataframe(top_g, use_container_width=True, height=320)
            st.warning(f"💡 Cancelling **{len(ghosts)}** Ghost POs frees blocked working capital.")

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Analytics Dashboard")
    df_sample = load_orders_sample(3000)

    if df_sample.empty:
        st.info("No data available.")
    else:
        col1, col2 = st.columns(2)

        status_col = next((c for c in df_sample.columns if "Order_Status" in c), None)
        if status_col:
            sc = df_sample[status_col].value_counts().reset_index()
            sc.columns = ["Status","Count"]
            col1.plotly_chart(px.pie(sc, names="Status", values="Count",
                                     title="Order Status Distribution",
                                     template="plotly_dark", hole=0.4),
                              use_container_width=True)

        late_col = next((c for c in df_sample.columns if "Late_delivery_risk" in c), None)
        if late_col and "Derived_State" in df_sample.columns:
            df_sample[late_col] = pd.to_numeric(df_sample[late_col], errors="coerce")
            sl = df_sample.groupby("Derived_State")[late_col].mean().mul(100).round(1).reset_index()
            sl.columns = ["State","Late_%"]
            col2.plotly_chart(px.bar(sl.sort_values("Late_%", ascending=False),
                                     x="State", y="Late_%", color="Late_%",
                                     color_continuous_scale="Reds",
                                     title="Late Delivery % by Indian State",
                                     template="plotly_dark"),
                              use_container_width=True)

        ship_col = next((c for c in df_sample.columns
                         if "shipping_mode" in c.lower() or "Shipping_Mode" in c), None)
        if ship_col and late_col:
            sm = df_sample.groupby(ship_col)[late_col].agg(["mean","count"]).reset_index()
            sm.columns = ["Shipping_Mode","Late_Rate","Count"]
            sm["Late_Rate"] *= 100
            fig_m = px.bar(sm, x="Shipping_Mode", y="Late_Rate",
                           color="Late_Rate", text="Count",
                           color_continuous_scale="OrRd",
                           title="Late Delivery % by Shipping Mode",
                           template="plotly_dark")
            fig_m.update_traces(texttemplate="%{text} orders", textposition="outside")
            st.plotly_chart(fig_m, use_container_width=True)

        st.subheader("🏭 Alternate Vendor Database")
        conn = sqlite3.connect(DB_PATH)
        try:
            st.dataframe(pd.read_sql("SELECT * FROM Alternate_Vendors LIMIT 100", conn),
                         use_container_width=True, height=250)
        except Exception:
            st.info("Vendor table loading …")
        finally:
            conn.close()
