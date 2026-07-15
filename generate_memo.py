"""

Reads the scored accounts + opportunity backlog and writes a one-page
recommendation memo (Markdown) for Product / Engineering / business stakeholders.


"""
import os
import pandas as pd
from datetime import datetime
from pipeline import load_data, compute_kpis, train_churn_model

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USE_LLM = False
GROQ_MODEL = "llama-3.1-8b-instant"
OUT = "reports"


def money(x):
    return f"${x:,.0f}"


def build_facts():
    df = compute_kpis(load_data())
    _, auc, scored, _ = train_churn_model(df)
    scored["revenue_at_risk"] = scored["churn_risk"] * scored["mrr"]

    facts = {
        "total_accounts": len(scored),
        "avg_health": scored["health_score"].mean(),
        "avg_adoption": scored["adoption_rate"].mean() * 100,
        "at_risk": int((scored["churn_risk"] > 0.5).sum()),
        "total_rev_at_risk": scored["revenue_at_risk"].sum(),
        "auc": auc,
    }
    top_accounts = (scored.sort_values("revenue_at_risk", ascending=False)
                    .head(5)[["account_id", "industry", "plan_tier", "mrr",
                              "health_score", "churn_risk", "revenue_at_risk"]])

    opp = None
    if os.path.exists("data/opportunities.csv"):
        opp = pd.read_csv("data/opportunities.csv")
        opp["label"] = opp["top_category"] if "top_category" in opp.columns else opp["theme"]
    return facts, top_accounts, opp


def default_summary(facts, opp):
    s = (f"Across {facts['total_accounts']:,} accounts, {facts['at_risk']:,} are at high churn risk, "
         f"putting {money(facts['total_rev_at_risk'])} of revenue in play. ")
    if opp is not None:
        top = opp.iloc[0]
        s += (f"The highest-impact product opportunity is **{top['label']}** "
              f"({int(top['tickets'])} tickets, {money(top['revenue_at_risk'])} at risk), "
              f"which should be prioritised on the roadmap.")
    return s


def llm_summary(facts, opp):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    top = opp.iloc[0] if opp is not None else None
    prompt = (
        "Write a concise 3-sentence executive summary for a product decision memo "
        "for a B2B fintech SaaS. Facts: "
        f"{facts['at_risk']} accounts at high churn risk, {money(facts['total_rev_at_risk'])} total "
        f"revenue at risk, average health {facts['avg_health']:.0f}/100."
        + (f" Top product opportunity: '{top['label']}' ({int(top['tickets'])} tickets, "
           f"{money(top['revenue_at_risk'])} at risk)." if top is not None else "")
        + " Be direct and business-focused. No preamble, no bullet points."
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL, temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    facts, top_accounts, opp = build_facts()
    today = datetime.today().strftime("%d %B %Y")
    summary = llm_summary(facts, opp) if USE_LLM else default_summary(facts, opp)

    L = []
    L.append("# ClientPulse — Product Decision Memo")
    L.append(f"*Prepared: {today}  ·  Audience: Product, Engineering & Business stakeholders*\n")

    L.append("## Executive Summary")
    L.append(summary + "\n")

    L.append("## Portfolio Health")
    L.append(f"- **Accounts:** {facts['total_accounts']:,}")
    L.append(f"- **Average health score:** {facts['avg_health']:.0f}/100")
    L.append(f"- **Average feature adoption:** {facts['avg_adoption']:.0f}%")
    L.append(f"- **Accounts at high churn risk (>50%):** {facts['at_risk']:,}")
    L.append(f"- **Total revenue at risk:** {money(facts['total_rev_at_risk'])}")
    L.append(f"- *(Churn model validation AUC: {facts['auc']:.3f})*\n")

    L.append("## Accounts to Save — top 5 by revenue at risk")
    L.append("| Account | Industry | Tier | MRR | Health | Churn risk | Revenue at risk |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in top_accounts.iterrows():
        L.append(f"| {r['account_id']} | {r['industry']} | {r['plan_tier']} | {money(r['mrr'])} | "
                 f"{r['health_score']:.0f} | {r['churn_risk']*100:.0f}% | {money(r['revenue_at_risk'])} |")
    L.append("")

    if opp is not None:
        L.append("## Product Opportunities to Build/Fix — ranked")
        L.append("| Rank | Area | Tickets | Accounts | Revenue at risk | Score |")
        L.append("|---|---|---|---|---|---|")
        for i, r in opp.head(5).iterrows():
            L.append(f"| {i+1} | {r['label']} | {int(r['tickets'])} | {int(r['accounts_affected'])} | "
                     f"{money(r['revenue_at_risk'])} | {r['opportunity_score']} |")
        L.append("")

        top = opp.iloc[0]
        L.append("## Recommendation")
        L.append(f"**Prioritise {top['label']} as the next roadmap item.** It is the highest-scoring "
                 f"opportunity ({top['opportunity_score']}/10), affecting {int(top['accounts_affected'])} "
                 f"accounts and {money(top['revenue_at_risk'])} of revenue. In parallel, trigger proactive "
                 f"outreach to the top at-risk accounts above to protect near-term revenue.\n")

    L.append("---")

    path = f"{OUT}/decision_memo.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote {path}")
    print("Preview it in VS Code (Ctrl+Shift+V), or convert to PDF for your portfolio.")


if __name__ == "__main__":
    main()
