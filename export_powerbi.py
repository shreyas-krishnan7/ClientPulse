"""
<<<<<<< HEAD
export Power BI-ready data.
=======
ClientPulse — export Power BI-ready data.
>>>>>>> 037b3c99143a17b23ea28aff7c4f4971d79d91db

Produces a /powerbi folder of clean CSVs to import into Power BI Desktop:
  accounts_scored.csv  — accounts + KPIs + churn_risk + revenue_at_risk
  tickets.csv          — classified tickets (linked to accounts by account_id)
  opportunities.csv    — ranked opportunity backlog
  metrics.csv          — headline KPIs for cards (incl. model AUC)

<<<<<<< HEAD

=======
Run (after voc_engine.py):  python export_powerbi.py
>>>>>>> 037b3c99143a17b23ea28aff7c4f4971d79d91db
"""
import os
import pandas as pd
from pipeline import load_data, compute_kpis, train_churn_model

OUT = "powerbi"


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. scored accounts (KPIs + churn risk + revenue-at-risk)
    df = compute_kpis(load_data())
    model, auc, scored, _ = train_churn_model(df)
    scored["revenue_at_risk"] = (scored["churn_risk"] * scored["mrr"]).round(0)
    scored.to_csv(f"{OUT}/accounts_scored.csv", index=False)

    # 2. classified tickets (fact table, linked to accounts by account_id)
    if os.path.exists("data/tickets_classified.csv"):
        t = pd.read_csv("data/tickets_classified.csv")
        cols = [c for c in ["ticket_id", "account_id", "created_date", "pred_category",
                            "pred_severity", "sentiment", "theme"] if c in t.columns]
        t[cols].to_csv(f"{OUT}/tickets.csv", index=False)

    # 3. opportunity backlog
    if os.path.exists("data/opportunities.csv"):
        pd.read_csv("data/opportunities.csv").to_csv(f"{OUT}/opportunities.csv", index=False)

    # 4. headline KPIs for cards
    pd.DataFrame([
        {"metric": "Total Accounts", "value": len(scored)},
        {"metric": "Avg Health Score", "value": round(scored["health_score"].mean(), 1)},
        {"metric": "Avg Adoption %", "value": round(scored["adoption_rate"].mean() * 100, 1)},
        {"metric": "At-Risk Accounts", "value": int((scored["churn_risk"] > 0.5).sum())},
        {"metric": "Total Revenue at Risk", "value": round(scored["revenue_at_risk"].sum(), 0)},
        {"metric": "Churn Model AUC", "value": round(auc, 3)},
    ]).to_csv(f"{OUT}/metrics.csv", index=False)

    print(f"Exported Power BI files to ./{OUT}/:")
    for f in ["accounts_scored.csv", "tickets.csv", "opportunities.csv", "metrics.csv"]:
        p = f"{OUT}/{f}"
        if os.path.exists(p):
            print(f"  {p}  ({len(pd.read_csv(p)):,} rows)")


if __name__ == "__main__":
    main()
