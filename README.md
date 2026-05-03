# 🏭 Smart Supply Chain Re-Sourcing Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://smart-supply-chain-resourcing-engine-fe4j5q65lgwjbmzkyekmbw.streamlit.app/)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![IsolationForest](https://img.shields.io/badge/ML-Isolation%20Forest-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightblue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**AI-powered procurement intelligence dashboard built for Indian manufacturing operations.**  
Predicts supplier delays before they happen. Detects Ghost POs. Finds alternate vendors instantly.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-supply-chain-resourcing-engine.streamlit.app)

</div>

---

## 👨‍💻 About This Project
I built this project to bridge the gap between raw procurement data and intelligent, automated decision-making for materials management teams. Having seen how manufacturing companies lose crores annually due to reactive supply chain management, I wanted to build something that shifts the approach from **"find out after it's too late"** to **"predict and act before it happens."**

This project combines my interest in **Machine Learning**, **ERP/procurement operations**, and **real-world Indian business context** — using a free, open-source tech stack deployable by anyone.

- 📍 Based in Maharashtra India
- 🎓 Interest in Supply Chain Analytics, ML for Tabular Data, Business Intelligence
- 🔧 Tech: Python · Pandas · XGBoost · Scikit-Learn · Streamlit · SQLite · Plotly

---

## 🧠 Problem Statement

In any Indian manufacturing or trading company, the Purchase Department raises hundreds of Purchase Orders every month. Two critical problems silently drain working capital:

| Problem | Real-World Impact |
|---|---|
| **Late Deliveries** discovered only after deadline | Production line halts, penalty clauses trigger, customer commitments missed |
| **Ghost POs** stuck in "Processing" for months | Budget locked, material never arrives, master data polluted |

> **This system solves both problems — before they happen — using Machine Learning.**

---

## ⚙ What This Project Does

### 🔄 Module 1 — Dynamic Supplier Re-Sourcing Engine
- Trains an **XGBoost classifier** on historical order data to predict `Late_delivery_risk`
- Scans all pending Purchase Orders in real time through the trained model
- If a PO is flagged as high-risk, the system immediately surfaces **Top 3 alternate Indian vendors**
- Vendors are ranked using a **weighted scoring algorithm**: `70% Lead Time + 30% Cost`

### 👻 Module 2 — Ghost PO Anomaly Detector
- Uses **Isolation Forest** (unsupervised ML) — no labelled data required
- Learns what a "normal" pending order looks like in terms of wait time, order value, and category
- Flags orders that deviate abnormally — these are your **Ghost POs**
- Output: a prioritised list of POs recommended for bulk cancellation to free working capital

---

## 📊 Dashboard Features

| Tab | Feature |
|---|---|
| 🔄 Re-Sourcing Engine | Scan pending POs → XGBoost delay score → alternate vendor finder with bar chart |
| 👻 Ghost PO Cleaner | Isolation Forest scan → anomaly score distribution → top 50 ghost POs table |
| 📊 Analytics | Order status pie chart · Late delivery by Indian state · Shipping mode performance · Vendor DB |

**KPI Cards (live from database):**
- 📦 Total Orders · ⏳ Pending Orders · 🏭 Alternate Vendors · ⚠ Late Delivery %

---

## 🗂 Project Structure

```
smart-supply-chain-resourcing-engine/
├── streamlit_app.py          ← Streamlit Cloud entry point (fully self-contained)
├── app.py                    ← Local run entry point
├── config.py                 ← Paths, Indian geography, model params
├── generate_sample_data.py   ← Auto-generates 20k synthetic orders on cloud
├── 01_setup_database.py      ← CSV → SQLite + India overlay + vendor table
├── 02_train_model.py         ← XGBoost training + evaluation
├── 03_anomaly_detection.py   ← Isolation Forest training
├── modules/
│   └── vendor_scorer.py      ← Weighted vendor ranking engine
├── requirements.txt
└── .streamlit/
    └── config.toml           ← Dark theme
```

---

## 🛠 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Database | SQLite | Zero-config, portable, handles 180k+ rows |
| Data Processing | Pandas + NumPy | Industry standard for tabular data |
| ML — Delay Prediction | **XGBoost** | Best-in-class for tabular classification |
| ML — Anomaly Detection | **Isolation Forest** (Scikit-Learn) | Unsupervised — no labels needed |
| Dashboard | **Streamlit** | Production-quality UI in pure Python |
| Charts | Plotly Express | Interactive, dark-theme visualisations |

---

## 📍 India-Specific Features

- Geography mapped to **12 Indian states** and major logistics hubs (Pune, Mumbai, Chennai, Bengaluru, Hyderabad, Ahmedabad etc.)
- Synthetic vendor table includes **real Indian company names**: Tata Steel, Bharat Forge, Delhivery, Blue Dart, Mahindra Logistics, Reliance Industries, Cipla, Dixon Technologies and more
- Vendor scoring weighted for **Indian procurement priorities**: speed of delivery over cost
- Dataset context adapted for **Indian manufacturing, pharma, FMCG, auto ancillary** sectors

---


>

---

## 💻 Run Locally

```bash
git clone https://github.com/Arjuna797/Smart-supply-chain-resourcing-engine.git
cd Smart-supply-chain-resourcing-engine

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

python 01_setup_database.py
python 02_train_model.py
python 03_anomaly_detection.py
streamlit run app.py
```

---


## 📁 Dataset

- Based on **DataCo Smart Supply Chain Dataset** (180,000+ real orders)
- On Streamlit Cloud: a **20,000-row synthetic dataset** is auto-generated matching the same schema
- Geography overwritten with Indian states and cities for local relevance

---

## 📄 License

MIT License — free to use, modify, and deploy.

---

<div align="center">
Built with ❤️ in Pune, Maharashtra 🇮🇳<br>
<b>Arjun</b> · <a href="https://github.com/Arjuna797">github.com/Arjuna797</a>
</div>
