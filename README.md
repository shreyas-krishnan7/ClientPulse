# ClientPulse — B2B Fintech Product Analytics

> Turning raw customer + product data into a ranked, actionable product decision —
> the workflow a Technical Product Analyst runs for a B2B trading/treasury SaaS.

**Status:** v1 shipped (data → metrics → prediction → dashboard). **Module 3
shipped** (Voice-of-Customer → opportunity engine). Automation + decision memo
on the roadmap.

---

## The idea
A fintech platform used by ~5,000 enterprise accounts generates three disconnected
signal streams — **product usage**, **support tickets**, and **account/revenue** data.
No single view exists, so churn, product complaints, and opportunities get missed.
ClientPulse fuses them into one system that answers: *who's at risk, why, and what
should we build/fix next.*

The spine: **data → metrics → prediction → opportunity → decision.**

## Architecture
| Layer | What it does | File |
|---|---|---|
| Data foundation | Generates a realistic 3-signal account dataset | `generate_data.py` |
| Metrics layer | Product KPIs (adoption, seat utilisation, stickiness, health score) | `pipeline.py` |
| Predictive engine | LightGBM churn model, validated, scores every account | `pipeline.py` |
| Model trainer | Prints ROC-AUC + saves a SHAP driver plot | `train_model.py` |
| **VoC generator** | Generates linked support-ticket text (Module 3) | `generate_tickets.py` |
| **Opportunity engine** | Classify → cluster → rank opportunities (Module 3) | `voc_engine.py` |
| Dashboard | KPIs, revenue-at-risk leaderboard, drivers, **opportunity backlog** | `app.py` |

## The opportunity score (Module 3)
For each emergent theme surfaced from support tickets:

```
opportunity_score = (frequency × avg_severity × revenue_at_risk) ^ (1/3), scaled 0–10
```
where `revenue_at_risk` reuses v1's churn model (`mrr × churn_risk` of affected accounts).

## Run it — in order
```bash
pip install -r requirements.txt

python generate_data.py       # -> data/accounts.csv   (v1)
python train_model.py         # prints churn AUC (+ SHAP plot)   (v1)

python generate_tickets.py    # -> data/tickets.csv     (Module 3)
python voc_engine.py          # -> data/opportunities.csv (Module 3)

streamlit run app.py          # dashboard incl. opportunity backlog
```

## Roadmap (v2 — in progress)
- [x] Voice-of-Customer → opportunity engine (classify + cluster + rank)
- [ ] **Automation**: weekly PM digest + at-risk-account alerts
- [ ] **Power BI version** of the dashboard
- [ ] **Decision memo**: one-page auto-generated product recommendation
- [ ] Time dimension → retention cohorts + churn early-warning

## Notes on the data
All data is **synthetically generated** with documented processes. Tickets carry
ground-truth labels so classifier accuracy is measurable. Swap in real product
telemetry + support tickets to productionise.
