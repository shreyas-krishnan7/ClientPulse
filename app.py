"""
ClientPulse — Streamlit dashboard.

Surfaces the full chain: KPIs -> churn risk -> revenue-at-risk -> per-account
"why", plus (Module 3) the Voice-of-Customer opportunity backlog.

    python generate_data.py
    python generate_tickets.py      # Module 3
    python voc_engine.py            # Module 3
    streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from pipeline import load_data, compute_kpis, train_churn_model, FEATURE_COLS

st.set_page_config(page_title="ClientPulse", layout="wide")


@st.cache_data
def get_scored():
    df = compute_kpis(load_data())
    model, auc, scored, _ = train_churn_model(df)
    importances = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    scored["revenue_at_risk"] = (scored["churn_risk"] * scored["mrr"]).round(0)
    return scored, auc, importances


scored, auc, importances = get_scored()

st.title("📈 ClientPulse — B2B Fintech Product Analytics")
st.caption("Data → Metrics → Prediction → Decision · simulated trading/treasury SaaS")

# ---------- KPI header ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accounts", f"{len(scored):,}")
c2.metric("Avg Health", f"{scored['health_score'].mean():.0f}/100")
c3.metric("Avg Adoption", f"{scored['adoption_rate'].mean():.0%}")
c4.metric("At-Risk (>50%)", f"{int((scored['churn_risk'] > 0.5).sum()):,}")
c5.metric("Churn Model AUC", f"{auc:.3f}")

st.divider()
left, right = st.columns([1.3, 1])

with left:
    st.subheader("🚨 At-Risk Accounts — ranked by revenue-at-risk")
    cols = ["account_id", "industry", "plan_tier", "mrr", "health_score",
            "adoption_rate", "unresolved_tickets", "churn_risk", "revenue_at_risk"]
    top = scored.sort_values("revenue_at_risk", ascending=False)[cols].head(20).copy()
    top["churn_risk"] = (top["churn_risk"] * 100).round(0).astype(int).astype(str) + "%"
    top["adoption_rate"] = (top["adoption_rate"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(top, use_container_width=True, hide_index=True)

with right:
    st.subheader("🔍 Churn Drivers (global)")
    fig = px.bar(importances.head(8).sort_values("importance"),
                 x="importance", y="feature", orientation="h")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
a, b = st.columns(2)
with a:
    st.subheader("Health Score Distribution")
    fig2 = px.histogram(scored, x="health_score", nbins=30)
    fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
with b:
    st.subheader("Adoption vs Churn Risk")
    sample = scored.sample(min(1500, len(scored)), random_state=1)
    fig3 = px.scatter(sample, x="adoption_rate", y="churn_risk",
                      color="plan_tier", opacity=0.5)
    fig3.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("🧭 Account Drill-down — why is this account at risk?")
choices = scored.sort_values("revenue_at_risk", ascending=False)["account_id"].head(50)
acct = st.selectbox("Select an account", choices)
row = scored[scored["account_id"] == acct].iloc[0]

d1, d2, d3, d4 = st.columns(4)
d1.metric("Churn Risk", f"{row['churn_risk'] * 100:.0f}%")
d2.metric("Health", f"{row['health_score']:.0f}/100")
d3.metric("Adoption", f"{row['adoption_rate'] * 100:.0f}%")
d4.metric("Revenue at Risk", f"${row['revenue_at_risk']:,.0f}")
st.info(
    f"**Why flagged** — adoption {row['adoption_rate'] * 100:.0f}% "
    f"(benchmark {scored['adoption_rate'].mean() * 100:.0f}%), "
    f"{int(row['unresolved_tickets'])} unresolved tickets, "
    f"{int(row['logins_last_30d'])} logins in last 30d, "
    f"NPS {int(row['nps_last_survey'])}/10."
)

# ---------- Module 3: Product Opportunities (Voice-of-Customer) ----------
st.divider()
st.header("🎯 Product Opportunities — Voice-of-Customer")
if os.path.exists("data/opportunities.csv"):
    opp = pd.read_csv("data/opportunities.csv")
    o1, o2 = st.columns([1, 1.25])
    with o1:
        figo = px.bar(opp.sort_values("opportunity_score").tail(10),
                      x="opportunity_score", y="theme", orientation="h",
                      color="opportunity_score", color_continuous_scale="Reds")
        figo.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0),
                           coloraxis_showscale=False)
        st.plotly_chart(figo, use_container_width=True)
    with o2:
        show = opp[["theme", "top_category", "tickets", "accounts_affected",
                    "revenue_at_risk", "opportunity_score"]].copy()
        show["revenue_at_risk"] = show["revenue_at_risk"].map(lambda v: f"${v:,.0f}")
        st.dataframe(show, use_container_width=True, hide_index=True)
    t = opp.iloc[0]
    st.success(
        f"**Top opportunity:** {t['theme']} — score {t['opportunity_score']}, "
        f"{int(t['tickets'])} tickets across {int(t['accounts_affected'])} accounts, "
        f"${t['revenue_at_risk']:,.0f} revenue at risk."
    )
else:
    st.info("Run `python generate_tickets.py` then `python voc_engine.py` "
            "to populate the opportunity backlog.")
