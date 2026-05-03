"""
streamlit_app.py — Streamlit Cloud Entry Point
Fully self-contained — no external module imports needed.
"""
import os, sys, sqlite3, warnings, random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (DB_PATH, CSV_PATH, MODEL_PATH, ISO_PATH,
                    DATA_DIR, MODELS_DIR, INDIAN_STATES, INDIAN_CITIES,
                    VENDOR_POOL, XGB_PARAMS, ISO_PARAMS,
                    WEIGHT_LEAD_TIME, WEIGHT_COST)

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# VENDOR SCORING (inline — no modules/ import needed)
# ─────────────────────────────────────────────────────────────────────────────
def get_alternate_vendors(product_name: str, top_n: int = 3) -> pd.DataFrame:
    try:
        conn  = sqlite3.connect(DB_PATH)
        df    = pd.read_sql(
            "SELECT Vendor_Name, Vendor_State, Vendor_City, Base_Price_INR, "
            "Lead_Time_Days, On_Time_Rate_Pct, Preference_Rank "
            "FROM Alternate_Vendors WHERE Product_Name = ? ORDER BY Lead_Time_Days ASC",
            conn, params=(product_name,))
        conn.close()
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df["Score_LeadTime"]  = 1 - norm(df["Lead_Time_Days"])
    df["Score_Cost"]      = 1 - norm(df["Base_Price_INR"])
    df["Composite_Score"] = WEIGHT_LEAD_TIME * df["Score_LeadTime"] + WEIGHT_COST * df["Score_Cost"]
    df = df.sort_values("Composite_Score", ascending=False).head(top_n)
    df["Rank"] = range(1, len(df)+1)
    return df[["Rank","Vendor_Name","Vendor_City","Vendor_State",
               "Lead_Time_Days","Base_Price_INR","On_Time_Rate_Pct","Composite_Score"]]


def get_high_risk_orders(model_artifact: dict, threshold: float = 0.30,
                          limit: int = 500) -> pd.DataFrame:
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
        try:
            df = pd.read_sql(
                "SELECT * FROM orders WHERE Order_Status IN "
                "('PENDING','PENDING_PAYMENT','PROCESSING') LIMIT ?",
                conn, params=(limit,))
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            try:
                df = pd.read_sql(f"SELECT * FROM orders LIMIT {limit}", conn)
            except Exception:
                df = pd.DataFrame()
        conn.close()

        if df is None or df.empty:
            return pd.DataFrame()

        avail_num = [c for c in num_cols if c in df.columns]
        if avail_num:
            df[avail_num] = df[avail_num].apply(pd.to_numeric, errors="coerce").fillna(0)
        le = LabelEncoder()
        for c in cat_cols:
            df[c] = le.fit_transform(df[c].astype(str)) if c in df.columns else 0

        X = df.reindex(columns=feature_cols, fill_value=0)
        df["Delay_Probability"] = model.predict_proba(X)[:, 1]
        df["Risk_Level"] = pd.cut(df["Delay_Probability"],
                                  bins=[0,0.35,0.60,1.0],
                                  labels=["Low ✅","Medium ⚠","High 🚨"])
        high_risk = df[df["Delay_Probability"] >= threshold].copy()
        if high_risk is None or high_risk.empty:
            high_risk = df.nlargest(20, "Delay_Probability").copy()
            high_risk["_fallback_note"] = "⚠ No orders exceeded threshold — showing top 20 highest-risk."
        return high_risk.sort_values("Delay_Probability", ascending=False)
    except Exception as e:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-INIT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _generate_data():
    CATEGORIES = ["Electronics","Furniture","Office Supplies","Clothing",
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
    SHIPPING_MODES = ["Standard Class","Second Class","First Class","Same Day"]
    ORDER_STATUSES = ["COMPLETE","CLOSED","PENDING","PROCESSING","PENDING_PAYMENT","CANCELED","SUSPECTED_FRAUD"]
    STATUS_WEIGHTS = [0.45,0.20,0.10,0.08,0.07,0.06,0.04]
    SEGMENTS       = ["Consumer","Corporate","Home Office"]

    rows = []
    for i in range(20000):
        cat     = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS[cat])
        state   = random.choice(INDIAN_STATES)
        city    = random.choice(INDIAN_CITIES.get(state, ["Mumbai"]))
        mode    = random.choice(SHIPPING_MODES)
        status  = random.choices(ORDER_STATUSES, weights=STATUS_WEIGHTS)[0]
        sched   = random.randint(2, 10)
        real    = max(1, sched + random.randint(-1, 5))
        qty     = random.randint(1, 50)
        price   = round(random.uniform(200, 15000), 2)
        disc    = round(random.uniform(0, 0.30), 4)
        prof_r  = round(random.uniform(-0.10, 0.35), 4)
        sales   = round(price * qty * (1 - disc), 2)
        days_ago= random.randint(1, 730)
        order_dt= (pd.Timestamp("2024-01-01") - pd.Timedelta(days=days_ago)).strftime("%m/%d/%Y")
        rows.append({
            "Order_Id":i+100000, "Order_Status":status,
            "order_date_DateOrders":order_dt, "Shipping_Mode":mode,
            "Days_for_shipping_real":real, "Days_for_shipment_scheduled":sched,
            "Late_delivery_risk":1 if real>sched else 0,
            "Category_Name":cat, "Product_Name":product,
            "Order_Item_Quantity":qty, "Product_Price":price,
            "Order_Item_Discount_Rate":disc, "Order_Item_Profit_Ratio":prof_r,
            "Order_Item_Total":sales, "Sales":sales,
            "Order_Profit_Per_Order":round(sales*prof_r,2),
            "Benefit_per_order":round(random.uniform(-500,2000),2),
            "Sales_per_customer":round(sales*random.uniform(0.8,1.2),2),
            "Customer_Segment":random.choice(SEGMENTS),
            "Order_Country":state, "Order_City":city,
            "Derived_State":state, "Derived_City":city,
        })
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)


def _setup_database():
    df = pd.read_csv(CSV_PATH, encoding="latin-1", low_memory=False)
    df.columns = (df.columns.str.strip()
                  .str.replace(r"[^a-zA-Z0-9_]","_",regex=True)
                  .str.replace(r"__+","_",regex=True).str.strip("_"))

    def find_col(df, cands):
        for c in cands:
            if c in df.columns: return c
        return None

    real_col  = find_col(df,["Days_for_shipping_real","Days_for_shipping__real_"])
    sched_col = find_col(df,["Days_for_shipment_scheduled","Days_for_shipment__scheduled_"])
    df["Shipping_Delay_Days"] = (
        pd.to_numeric(df[real_col],errors="coerce") -
        pd.to_numeric(df[sched_col],errors="coerce")
    ) if real_col and sched_col else 0

    date_col = find_col(df,["order_date_DateOrders","Order_Date"])
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["Days_Since_Order"] = (df[date_col].max() - df[date_col]).dt.days.fillna(0).astype(int)
    else:
        df["Days_Since_Order"] = np.random.randint(1, 365, len(df))

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("orders", conn, if_exists="replace", index=False, chunksize=5000)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON orders(Order_Status)")

    prod_col = find_col(df,["Product_Name","product_name"])
    cat_col  = find_col(df,["Category_Name","category_name"])
    products = df[prod_col].dropna().unique().tolist() if prod_col else []
    vrows = []
    for prod in products:
        cat = ""
        if cat_col:
            vals = df.loc[df[prod_col]==prod, cat_col].dropna()
            cat  = vals.iloc[0] if len(vals) else ""
        n = random.randint(2, 3)
        for rank, vendor in enumerate(random.sample(VENDOR_POOL, min(n, len(VENDOR_POOL))), 1):
            state = random.choice(INDIAN_STATES)
            vrows.append({
                "Product_Name":prod, "Category":cat, "Vendor_Name":vendor,
                "Vendor_State":state,
                "Vendor_City":random.choice(INDIAN_CITIES.get(state,["Mumbai"])),
                "Base_Price_INR":round(random.uniform(500,50000),2),
                "Lead_Time_Days":random.randint(3,21),
                "On_Time_Rate_Pct":round(random.uniform(70,99),1),
                "Preference_Rank":rank,
            })
    pd.DataFrame(vrows).to_sql("Alternate_Vendors", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def _train_model():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM orders WHERE Order_Status IN ('COMPLETE','CLOSED','CANCELED','SUSPECTED_FRAUD')", conn)
        if df.empty: df = pd.read_sql("SELECT * FROM orders LIMIT 50000", conn)
    except: df = pd.read_sql("SELECT * FROM orders LIMIT 50000", conn)
    conn.close()

    def find_col(df, cands):
        for c in cands:
            if c in df.columns: return c
        return None

    target = find_col(df,["Late_delivery_risk","late_delivery_risk"])
    if not target: return
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    num_cols = [c for c in ["Days_for_shipping_real","Days_for_shipment_scheduled",
                             "Shipping_Delay_Days","Days_Since_Order","Sales",
                             "Order_Item_Quantity","Order_Item_Discount_Rate",
                             "Order_Item_Profit_Ratio","Order_Item_Total","Benefit_per_order"]
                if c in df.columns]
    cat_cols = [c for c in ["Shipping_Mode","Category_Name","Customer_Segment",
                             "Derived_State","Order_Status"] if c in df.columns]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    le = LabelEncoder()
    for c in cat_cols:
        df[c] = le.fit_transform(df[c].astype(str))

    feat_cols = num_cols + cat_cols
    X, y = df[feat_cols], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_te,y_te)], verbose=False)
    joblib.dump({"model":model,"feature_cols":feat_cols,"num_cols":num_cols,"cat_cols":cat_cols}, MODEL_PATH)


def _train_iso():
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM orders WHERE Order_Status IN ('PENDING','PENDING_PAYMENT','PROCESSING') LIMIT 5000", conn)
        if df.empty: df = pd.read_sql("SELECT * FROM orders LIMIT 5000", conn)
    except: df = pd.read_sql("SELECT * FROM orders LIMIT 5000", conn)
    conn.close()

    def find_col(df, cands):
        for c in cands:
            if c in df.columns: return c
        return None

    feat = pd.DataFrame()
    feat["Days_Since_Order"]    = pd.to_numeric(df.get("Days_Since_Order", pd.Series([30]*len(df))), errors="coerce").fillna(30)
    feat["Cat_Avg_Lead_Time"]   = 7.0
    vc = find_col(df,["Sales","Order_Item_Total"])
    feat["Order_Value"]         = pd.to_numeric(df[vc] if vc else pd.Series([500]*len(df)), errors="coerce").fillna(500)
    feat["Shipping_Delay_Days"] = pd.to_numeric(df.get("Shipping_Delay_Days", pd.Series([0]*len(df))), errors="coerce").fillna(0)
    feat["Benefit_per_order"]   = pd.to_numeric(df.get("Benefit_per_order", pd.Series([0]*len(df))), errors="coerce").fillna(0)
    feat["Wait_vs_Expected"]    = (feat["Days_Since_Order"] / feat["Cat_Avg_Lead_Time"].replace(0,1)).clip(0,50)
    feat = feat.fillna(0)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(feat)
    iso = IsolationForest(**ISO_PARAMS)
    iso.fit(X_s)
    joblib.dump({"model":iso,"scaler":scaler,"feature_cols":feat.columns.tolist()}, ISO_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-INIT RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_auto_init():
    needs = [
        not os.path.exists(CSV_PATH),
        not os.path.exists(DB_PATH),
        not os.path.exists(MODEL_PATH),
        not os.path.exists(ISO_PATH),
    ]
    if not any(needs):
        return

    st.set_page_config(page_title="Setting up…", page_icon="⏳", layout="centered")
    st.title("🏭 Smart Supply Chain Re-Sourcing Engine")
    st.info("⏳ **First-time setup — please wait ~3 minutes…**")
    bar = st.progress(0)
    msg = st.empty()

    if needs[0]:
        msg.write("📂 Generating 20,000 synthetic orders…")
        bar.progress(10)
        _generate_data()

    if needs[1]:
        msg.write("💾 Building SQLite database…")
        bar.progress(35)
        _setup_database()

    if needs[2]:
        msg.write("🤖 Training XGBoost delay model…")
        bar.progress(60)
        _train_model()

    if needs[3]:
        msg.write("🌲 Training Isolation Forest…")
        bar.progress(85)
        _train_iso()

    bar.progress(100)
    msg.write("✅ Ready! Loading dashboard…")
    st.rerun()


run_auto_init()

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Supply Chain Re-Sourcing Engine",
    page_icon="🏭", layout="wide", initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .main{background-color:#0f1117;} h1{color:#1E88E5;}
  .stMetric{background:#1e2130;border-radius:10px;padding:12px;}
  .stMetric label{color:#90caf9!important;}
  .block-container{padding-top:1rem;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@st.cache_resource
def load_iso():
    return joblib.load(ISO_PATH) if os.path.exists(ISO_PATH) else None

@st.cache_data(ttl=300)
def load_summary_stats():
    conn = sqlite3.connect(DB_PATH)
    stats = {}
    try:
        stats["total_orders"]   = pd.read_sql("SELECT COUNT(*) AS n FROM orders",conn).iloc[0,0]
        stats["pending_orders"] = pd.read_sql("SELECT COUNT(*) AS n FROM orders WHERE Order_Status IN ('PENDING','PENDING_PAYMENT','PROCESSING')",conn).iloc[0,0]
        stats["total_vendors"]  = pd.read_sql("SELECT COUNT(DISTINCT Vendor_Name) AS n FROM Alternate_Vendors",conn).iloc[0,0]
        stats["late_pct"]       = pd.read_sql("SELECT ROUND(AVG(CAST(Late_delivery_risk AS FLOAT))*100,1) AS p FROM orders",conn).iloc[0,0]
    except: pass
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
        df = pd.read_sql("SELECT * FROM orders WHERE Order_Status IN ('PENDING','PENDING_PAYMENT','PROCESSING') LIMIT ?",conn,params=(n,))
        if df is None or df.empty:
            df = pd.read_sql(f"SELECT * FROM orders LIMIT {n}",conn)
    except:
        df = pd.read_sql(f"SELECT * FROM orders LIMIT {n}",conn)
    conn.close()
    return df if df is not None else pd.DataFrame()

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
    st.markdown(f"{'🟢' if model_ok else '🔴'}  XGBoost Model")
    st.markdown(f"{'🟢' if iso_ok   else '🔴'}  Ghost PO Model")
    st.divider()
    risk_threshold = st.slider("🎯 High-Risk Threshold", 0.10, 0.95, 0.30, 0.05)
    top_n_vendors  = st.slider("🏭 Vendors to Show",     1, 5, 3)
    max_orders     = st.slider("📦 Max Orders to Scan",  50, 1000, 500, 50)
    st.divider()
    st.caption("Smart Supply Chain Re-Sourcing Engine\n📍 India Operations Intelligence")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏭 Smart Supply Chain Re-Sourcing Engine")
st.caption("Predictive Delay Detection  •  Ghost PO Detector  •  India Vendor Intelligence")
if db_ok:
    s = load_summary_stats()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📦 Total Orders",   f"{s.get('total_orders',0):,}")
    c2.metric("⏳ Pending Orders", f"{s.get('pending_orders',0):,}")
    c3.metric("🏭 Alt. Vendors",   f"{s.get('total_vendors',0):,}")
    c4.metric("⚠ Late Delivery %", f"{s.get('late_pct',0):.1f}%")
st.divider()

tab1,tab2,tab3 = st.tabs(["🔄 Re-Sourcing Engine","👻 Ghost PO Cleaner","📊 Analytics Dashboard"])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔄 Supplier Re-Sourcing Engine")
    ma = load_model()
    if ma is None:
        st.error("Model not ready — please refresh.")
    else:
        cb,ci = st.columns([2,5])
        scan_btn = cb.button("🚀 Scan High-Risk POs", type="primary", use_container_width=True)
        ci.info(f"Threshold: **{risk_threshold:.0%}** · Scanning **{max_orders}** orders")

        if scan_btn or "high_risk_df" in st.session_state:
            if scan_btn:
                with st.spinner("Running XGBoost…"):
                    hr = get_high_risk_orders(ma, risk_threshold, max_orders)
                    st.session_state["high_risk_df"] = hr if hr is not None else pd.DataFrame()

            hr = st.session_state.get("high_risk_df", pd.DataFrame())
            if hr is None: hr = pd.DataFrame()

            if hr.empty:
                st.warning("No orders found.")
            else:
                if "_fallback_note" in hr.columns:
                    st.warning(hr["_fallback_note"].iloc[0])
                    st.info(f"Showing top **{len(hr)}** highest-risk orders.")
                else:
                    st.error(f"🚨 **{len(hr)}** orders exceed **{risk_threshold:.0%}** threshold!")

                dc = [c for c in ["Order_Id","Product_Name","Derived_City","Derived_State","Order_Status","Days_Since_Order","Delay_Probability","Risk_Level"] if c in hr.columns]
                ddf = hr[dc].copy() if dc else hr.head(20)
                if "Delay_Probability" in ddf.columns:
                    ddf["Delay_Probability"] = ddf["Delay_Probability"].apply(lambda x: f"{x:.1%}")
                st.dataframe(ddf, use_container_width=True, height=280)

                st.subheader("🏭 Find Alternate Vendor")
                pc = next((c for c in hr.columns if "Product_Name" in c), None)
                if pc:
                    sp = st.selectbox("Select flagged product:", hr[pc].dropna().unique().tolist())
                    if st.button("🔍 Get Alternate Vendors", type="secondary"):
                        vd = get_alternate_vendors(sp, top_n_vendors)
                        if vd is None or vd.empty:
                            st.warning("No alternate vendors found.")
                        else:
                            st.success(f"✅ Top {len(vd)} vendors for **{sp}**")
                            def colour_score(val):
                                c = "#00c853" if val>=0.7 else "#ff6d00" if val>=0.4 else "#f44336"
                                return f"background-color:{c};color:white;font-weight:700"
                            try:
                                st.dataframe(vd.style.map(colour_score,subset=["Composite_Score"])
                                             .format({"Base_Price_INR":"₹{:,.0f}","Composite_Score":"{:.3f}"}),
                                             use_container_width=True)
                            except:
                                st.dataframe(vd, use_container_width=True)
                            st.plotly_chart(px.bar(vd,x="Vendor_Name",y="Composite_Score",
                                                   color="Lead_Time_Days",color_continuous_scale="Blues_r",
                                                   title=f"Vendor Scores — {sp}",template="plotly_dark"),
                                            use_container_width=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.header("👻 Ghost PO Anomaly Detector")
    st.markdown("**Isolation Forest** finds POs stuck abnormally long — freeing working capital.")
    ia = load_iso()
    if ia is None:
        st.error("Ghost PO model not ready — refresh.")
    else:
        if st.button("🔍 Detect Ghost POs", type="primary"):
            with st.spinner("Running Isolation Forest…"):
                pdf = load_ghost_orders(500)
                if pdf is None: pdf = pd.DataFrame()
                if not pdf.empty:
                    iso    = ia["model"]
                    scaler = ia["scaler"]
                    fcols  = ia["feature_cols"]
                    feat = pd.DataFrame(index=pdf.index)
                    for c in fcols:
                        feat[c] = pd.to_numeric(pdf.get(c,0), errors="coerce").fillna(0)
                    Xs = scaler.transform(feat)
                    pdf["Anomaly_Score"] = iso.decision_function(Xs)
                    pdf["Is_Ghost_PO"]   = (iso.predict(Xs)==-1).astype(int)
                    st.session_state["ghost_result"] = pdf

        if "ghost_result" in st.session_state:
            gdf = st.session_state["ghost_result"]
            if gdf is None: gdf = pd.DataFrame()
            if not gdf.empty:
                gh = gdf[gdf["Is_Ghost_PO"]==1].sort_values("Anomaly_Score")
                nm = gdf[gdf["Is_Ghost_PO"]==0]
                g1,g2,g3 = st.columns(3)
                g1.metric("Total Scanned",  f"{len(gdf):,}")
                g2.metric("🚨 Ghost POs",   f"{len(gh):,}", delta=f"{len(gh)/max(len(gdf),1):.1%}")
                g3.metric("✅ Normal",      f"{len(nm):,}")
                st.plotly_chart(px.histogram(gdf,x="Anomaly_Score",color="Is_Ghost_PO",
                                             barmode="overlay",nbins=60,
                                             color_discrete_map={0:"#1E88E5",1:"#F44336"},
                                             title="Anomaly Score Distribution",template="plotly_dark"),
                                use_container_width=True)
                dc = [c for c in ["Order_Id","Product_Name","Order_Status","Derived_City","Days_Since_Order","Anomaly_Score"] if c in gh.columns]
                tg = gh[dc].head(50).copy() if dc else gh.head(50).copy()
                if "Anomaly_Score" in tg.columns:
                    tg["Anomaly_Score"] = tg["Anomaly_Score"].round(4)
                st.dataframe(tg, use_container_width=True, height=320)
                st.warning(f"💡 Cancelling **{len(gh)}** Ghost POs frees blocked working capital.")

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Analytics Dashboard")
    dfs = load_orders_sample(3000)
    if dfs is None: dfs = pd.DataFrame()
    if dfs.empty:
        st.info("No data available.")
    else:
        c1,c2 = st.columns(2)
        sc = next((c for c in dfs.columns if "Order_Status" in c), None)
        if sc:
            sv = dfs[sc].value_counts().reset_index()
            sv.columns = ["Status","Count"]
            c1.plotly_chart(px.pie(sv,names="Status",values="Count",
                                   title="Order Status Distribution",
                                   template="plotly_dark",hole=0.4), use_container_width=True)
        lc = next((c for c in dfs.columns if "Late_delivery_risk" in c), None)
        if lc and "Derived_State" in dfs.columns:
            dfs[lc] = pd.to_numeric(dfs[lc], errors="coerce")
            sl = dfs.groupby("Derived_State")[lc].mean().mul(100).round(1).reset_index()
            sl.columns = ["State","Late_%"]
            c2.plotly_chart(px.bar(sl.sort_values("Late_%",ascending=False),
                                   x="State",y="Late_%",color="Late_%",
                                   color_continuous_scale="Reds",
                                   title="Late Delivery % by Indian State",
                                   template="plotly_dark"), use_container_width=True)
        shc = next((c for c in dfs.columns if "shipping_mode" in c.lower() or "Shipping_Mode" in c), None)
        if shc and lc:
            sm = dfs.groupby(shc)[lc].agg(["mean","count"]).reset_index()
            sm.columns = ["Shipping_Mode","Late_Rate","Count"]
            sm["Late_Rate"] *= 100
            fm = px.bar(sm,x="Shipping_Mode",y="Late_Rate",color="Late_Rate",text="Count",
                        color_continuous_scale="OrRd",title="Late Delivery % by Shipping Mode",
                        template="plotly_dark")
            fm.update_traces(texttemplate="%{text} orders",textposition="outside")
            st.plotly_chart(fm, use_container_width=True)
        st.subheader("🏭 Alternate Vendor Database")
        conn = sqlite3.connect(DB_PATH)
        try:
            st.dataframe(pd.read_sql("SELECT * FROM Alternate_Vendors LIMIT 100",conn),
                         use_container_width=True, height=250)
        except: st.info("Loading vendors…")
        finally: conn.close()
