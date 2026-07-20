# 📊 ClientPulse — B2B Fintech Product Analytics System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Churn%20Model-9cf)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama--3.1-F55036)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811)

ClientPulse is an **end-to-end product-analytics system** for a B2B fintech (trading/treasury) SaaS platform. It fuses product usage, support tickets, and revenue data into a single pipeline that answers the two questions a product analyst lives by: **which customers are about to churn, and what should we build or fix next.**

Built with Python, LightGBM, an LLM (Groq/Llama-3.1), and Power BI, ClientPulse demonstrates the full **data → metrics → prediction → opportunity → decision** workflow — turning raw customer and product data into a ranked, revenue-weighted roadmap decision.

---

## 🌐 Links

- **GitHub Repository:** https://github.com/shreyas-krishnan7/ClientPulse
-  **Deployed URL:** https://clientpulse-paf5xvbtmsnwgy5qyifokr.streamlit.app/
---

## 📌 Overview

Enterprise SaaS platforms serve thousands of accounts that each generate three disconnected streams of signal — how they use the product, what they complain about, and how much they pay. No single view exists, so churn, product pain points, and opportunities stay hidden until an account cancels.

ClientPulse solves this by:

- **Defining and tracking key product & business metrics** (adoption, engagement, account-health score)
- **Predicting churn** with a validated machine-learning model and explaining *why* each account is at risk
- **Turning customer feedback into product opportunities** using an LLM-powered classification engine
- **Ranking opportunities by business impact** (frequency × severity × revenue-at-risk)
- **Delivering the insight** through interactive dashboards and an auto-generated decision memo

---

## 📸 Output Screenshots

**Power BI — Customer Health & Churn Risk Overview**
<img width="911" height="510" alt="image" src="https://github.com/user-attachments/assets/c7a47330-3d05-4423-8927-8e3831d6d4e9" />

**Power BI — Product Opportunities (Voice of Customer)**
<img width="910" height="507" alt="image" src="https://github.com/user-attachments/assets/1bcfc99f-97d9-4dc2-9525-efb3174ea5a3" />

**Streamlit — Interactive Dashboard**
<img width="1600" height="760" alt="image" src="https://github.com/user-attachments/assets/3427ea8a-85e7-4e26-ae3a-0e8e49f3772e" />
<img width="1600" height="760" alt="image" src="https://github.com/user-attachments/assets/108957ad-0ffb-4bb1-9b5a-bff47dd9ba6e" />
<img width="1600" height="760" alt="image" src="https://github.com/user-attachments/assets/d8c8564a-3036-4ded-a95b-5a10a950471e" />
<img width="1600" height="760" alt="image" src="https://github.com/user-attachments/assets/d9aeffe1-b8a1-4d02-a4a3-bbadfd816b58" />

**Churn Drivers (SHAP)**
<img width="940" height="792" alt="image" src="https://github.com/user-attachments/assets/baa8f6c1-2e47-4061-8d29-86595e32921b" />


---

## 📈 Results & Key Metrics

| Metric | Result |
|---|---|
| Accounts analyzed | **5,000** |
| Churn model performance | **ROC-AUC 0.79** |
| At-risk accounts flagged | **1,021** |
| Support tickets classified | **5,000** |
| Ticket classification accuracy (by product area) | **97.8%** |
| Emergent themes surfaced | **12** |
| Top product opportunity | **Reconciliation — ~$1.5M revenue at risk** |

---

## ✨ Features

### 📊 Product Metrics Layer
- Feature adoption, seat utilization, and login stickiness
- Composite **account-health score** (0–100)
- Revenue-at-risk = churn probability × MRR

### 🤖 Churn Prediction
- Validated **LightGBM** model (ROC-AUC 0.79)
- **SHAP** explainability — surfaces *why* each account is at risk
- Per-account drill-down with the top churn drivers

### 🗣️ Voice-of-Customer → Opportunity Engine
- **LLM-based** ticket classification (Groq/Llama-3.1) by product area & severity
- VADER sentiment analysis
- Theme clustering to surface recurring pain points
- **Opportunity scoring**: frequency × severity × revenue-at-risk

### 📈 Dashboards
- **Streamlit** app: KPIs, at-risk leaderboard, churn drivers, opportunity backlog, account drill-down
- **Power BI** executive dashboard: customer health, churn risk, and product opportunities

### 📝 Automated Decision Memo
- Auto-generated one-page recommendation for Product, Engineering & business stakeholders

---

## 🏗️ System Architecture

```
        ┌──────────────────────────────────────────────┐
        │              DATA FOUNDATION                  │
        │  accounts · usage · support tickets · revenue │
        └───────────────────────┬──────────────────────┘
                                │
             ┌──────────────────┼───────────────────┐
             ▼                  ▼                   ▼
   ┌──────────────────┐ ┌───────────────┐ ┌────────────────────┐
   │  METRICS LAYER   │ │  CHURN MODEL  │ │  VoC ENGINE (LLM)  │
   │ adoption, health │ │  LightGBM +   │ │ classify → cluster │
   │ stickiness, KPIs │ │  SHAP (0.79)  │ │ → rank opportunities│
   └────────┬─────────┘ └───────┬───────┘ └─────────┬──────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
             ┌───────────────────────────────────────┐
             │      DASHBOARDS  +  DECISION MEMO      │
             │  Power BI · Streamlit · recommendation │
             └───────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Data & Analysis | Pandas, NumPy |
| Machine Learning | scikit-learn, LightGBM |
| Explainability | SHAP |
| LLM / NLP | Groq API (Llama-3.1), VADER |
| Dashboards | Streamlit, Plotly, Power BI |
| Config | python-dotenv |

---

## 📂 Project Structure

```
clientpulse
│
├── generate_data.py        # synthetic account dataset (usage + support + revenue)
├── pipeline.py             # product KPIs + LightGBM churn model
├── train_model.py          # trains churn model, prints AUC, saves SHAP plot
├── generate_tickets.py     # synthetic support-ticket text (linked to accounts)
├── voc_engine.py           # LLM classify → cluster → rank opportunities
├── export_powerbi.py       # export Power BI-ready CSVs
├── generate_memo.py        # auto-generated decision memo
├── app.py                  # Streamlit dashboard
│
├── requirements.txt
├── .env
├── .gitignore
│
├── assets/                 # SHAP summary plot
├── screenshots/            # dashboard screenshots
├── powerbi/                # ClientPulse.pbix
└── reports/                # decision_memo.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A free **Groq API key** (https://console.groq.com) — for the LLM classification step
- Power BI Desktop (optional, for the Power BI dashboard)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/clientpulse.git
cd clientpulse
```

### 2. Set up the environment
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 3. Add your API key
Copy the template and paste your key:
```bash
cp .env
```
```
GROQ_API_KEY=your_groq_key_here
```

### 4. Run the pipeline (in order)
```bash
python generate_data.py       # -> data/accounts.csv
python train_model.py         # churn model + SHAP plot (assets/shap_summary.png)
python generate_tickets.py    # -> data/tickets.csv
python voc_engine.py          # classify + rank -> data/opportunities.csv
python export_powerbi.py      # -> powerbi/*.csv for Power BI
python generate_memo.py       # -> reports/decision_memo.md
```

### 5. Launch the dashboard
```bash
streamlit run app.py          # opens http://localhost:8501
```

> **Tip:** In `voc_engine.py`, keep `SAMPLE = 5000` for a fast, reliable LLM run that stays within the Groq free-tier daily limit.

---

## 🎮 How It Works

1. **Data foundation** — generates a realistic dataset of 5,000 accounts with usage, support, and revenue signals, plus linked support-ticket text.
2. **Metrics layer** — computes product KPIs and a composite account-health score.
3. **Churn model** — a LightGBM classifier predicts churn (AUC 0.79); SHAP explains the drivers.
4. **VoC engine** — an LLM classifies each support ticket, clusters them into themes, and ranks product opportunities by frequency × severity × revenue-at-risk.
5. **Dashboards** — Power BI and Streamlit surface health, churn risk, and the opportunity backlog.
6. **Decision** — an auto-generated memo delivers a prioritized recommendation.

---

## 💡 Technical Highlights

### Predictive Modelling with Explainability
A validated LightGBM churn model (ROC-AUC 0.79) paired with SHAP, so every at-risk flag comes with a *reason* — not just a score.

### LLM-Powered Voice-of-Customer
Support tickets are classified by an LLM (Groq/Llama-3.1) at 97.8% accuracy, then clustered and scored — converting unstructured feedback into a ranked, revenue-weighted product backlog.

### Business-Weighted Prioritization
Opportunities are ranked by **frequency × severity × revenue-at-risk**, tying every product decision back to revenue impact.

### End-to-End Data-to-Decision Pipeline
From raw data generation through metrics, prediction, NLP, dashboards, and a final recommendation memo — the complete workflow a product analyst runs.

---

## ⚠️ A Note on the Data

All data in ClientPulse is **synthetically generated** with documented processes (`generate_data.py`, `generate_tickets.py`). Ticket labels are ground-truth, so classification accuracy is measurable — but metrics reflect the model and pipeline on a synthetic benchmark, not a live production system. On real-world data, expect noisier results. Swap in real product telemetry and support tickets to productionize.

---

## 📈 Future Enhancements

- Real product telemetry + support-ticket integration
- Time-series retention cohorts and churn early-warning
- CI pipeline for model retraining and monitoring

