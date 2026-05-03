"""
config.py — Central configuration for Dynamic Supplier Re-Sourcing Engine
Automatically detects Streamlit Cloud and uses /tmp for writable storage.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Detect Streamlit Cloud (read-only source mount) ──────────────────────────
# On Streamlit Cloud, source is at /mount/src/... which is read-only.
# We use /tmp which is always writable.
IS_CLOUD = BASE_DIR.startswith("/mount/src") or os.environ.get("STREAMLIT_CLOUD", "") == "1"

if IS_CLOUD:
    WRITABLE_DIR = "/tmp/supply_chain"
else:
    WRITABLE_DIR = BASE_DIR

DATA_DIR   = os.path.join(WRITABLE_DIR, "Data")
MODELS_DIR = os.path.join(WRITABLE_DIR, "models")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Key Paths ─────────────────────────────────────────────────────────────────
DB_PATH    = os.path.join(DATA_DIR,   "supply_chain.db")
CSV_PATH   = os.path.join(DATA_DIR,   "DataCoSupplyChainDataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "delay_predictor.joblib")
ISO_PATH   = os.path.join(MODELS_DIR, "isolation_forest.joblib")

# ── India Geography ───────────────────────────────────────────────────────────
INDIAN_STATES = [
    "Maharashtra","Gujarat","Tamil Nadu","Karnataka","Rajasthan",
    "Telangana","Uttar Pradesh","West Bengal","Punjab","Haryana",
    "Delhi","Madhya Pradesh",
]
INDIAN_CITIES = {
    "Maharashtra":    ["Pune","Mumbai","Nashik","Aurangabad","Nagpur"],
    "Gujarat":        ["Ahmedabad","Surat","Vadodara","Rajkot"],
    "Tamil Nadu":     ["Chennai","Coimbatore","Madurai","Tiruchirappalli"],
    "Karnataka":      ["Bengaluru","Mysuru","Hubli","Mangaluru"],
    "Rajasthan":      ["Jaipur","Jodhpur","Udaipur","Kota"],
    "Telangana":      ["Hyderabad","Warangal","Karimnagar"],
    "Uttar Pradesh":  ["Lucknow","Kanpur","Agra","Varanasi"],
    "West Bengal":    ["Kolkata","Asansol","Siliguri"],
    "Punjab":         ["Ludhiana","Amritsar","Jalandhar"],
    "Haryana":        ["Gurugram","Faridabad","Panipat"],
    "Delhi":          ["New Delhi","Dwarka","Rohini"],
    "Madhya Pradesh": ["Indore","Bhopal","Jabalpur"],
}

# ── Synthetic Indian Vendor Pool ──────────────────────────────────────────────
VENDOR_POOL = [
    "Tata Steel Ltd","Reliance Industries","Mahindra Logistics",
    "Infosys BPO","Wipro Manufacturing","Pune Auto Parts Pvt Ltd",
    "Bharat Forge","Kirloskar Industries","Motherson Sumi",
    "Bajaj Auto Ancillaries","Raymond Fabrics","ITC Agri-Business",
    "Godrej Consumer","Dabur Supply Chain","Marico Commodities",
    "Sun Pharma Inputs","Cipla Raw Materials","Aarti Industries",
    "Deepak Nitrite","Gujarat Fluorochemicals","Dixon Technologies",
    "Optiemus Infracom","Amber Enterprises","PG Electroplast",
    "Kaynes Technology","Blue Dart Express","DTDC Courier",
    "Gati-KWE","Delhivery Pvt Ltd","Ecom Express",
    "Nashik Precision Tools","Coimbatore Castings",
    "Ludhiana Hardware Works","Jaipur Gems & Jewels",
    "Chennai Auto Components","Pune Polymers Pvt Ltd",
    "Nagpur Steel Works","Surat Textiles Ltd",
]

# ── XGBoost hyper-params ──────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
)

# ── Isolation Forest ──────────────────────────────────────────────────────────
ISO_PARAMS = dict(n_estimators=100, contamination=0.08, random_state=42)

# ── Vendor Scoring weights ────────────────────────────────────────────────────
WEIGHT_LEAD_TIME = 0.70
WEIGHT_COST      = 0.30
