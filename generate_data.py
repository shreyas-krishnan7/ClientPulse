"""


Models a B2B fintech (trading/treasury) SaaS with ~5,000 enterprise accounts.
Each account has engagement, support, and revenue signals. A churn label is
generated from a realistic latent-risk process so the predictive model has
genuine signal to learn (low adoption / low logins / unresolved tickets -> churn).

"""
import os
import numpy as np
import pandas as pd

TOTAL_FEATURES = 12  # number of product features the platform exposes


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    industries = ["Banking", "Asset Mgmt", "Insurance", "Corporate Treasury", "Brokerage"]
    tiers = ["Starter", "Growth", "Enterprise"]

    plan_tier = rng.choice(tiers, size=n, p=[0.40, 0.40, 0.20])
    industry = rng.choice(industries, size=n)
    tenure = rng.integers(1, 60, size=n)

    seats = np.where(
        plan_tier == "Starter", rng.integers(3, 15, n),
        np.where(plan_tier == "Growth", rng.integers(10, 60, n), rng.integers(50, 300, n)),
    )
    base_mrr = np.where(plan_tier == "Starter", 500, np.where(plan_tier == "Growth", 2500, 12000))
    mrr = (base_mrr * (0.7 + 0.6 * rng.random(n))).round(0)

    # engagement is driven by a latent "adoption" factor
    adoption = rng.beta(2, 2, n)                                   # 0..1
    features_adopted = (adoption * TOTAL_FEATURES).round().astype(int).clip(0, TOTAL_FEATURES)
    active_seats = (seats * (0.30 + 0.60 * adoption) * (0.6 + 0.8 * rng.random(n))).round().astype(int)
    active_seats = np.minimum(active_seats, seats)
    logins = (adoption * 40 * (0.5 + rng.random(n))).round().astype(int).clip(0, 120)
    session_min = (5 + adoption * 40 + rng.normal(0, 5, n)).clip(1, 90).round(1)
    tickets = rng.poisson(3, n)
    unresolved = np.minimum(tickets, rng.poisson(1.2 * (1 - adoption) + 0.3, n))
    resolution = (2 + (1 - adoption) * 10 + rng.normal(0, 2, n)).clip(0.5, 40).round(1)
    nps = (adoption * 10 + rng.normal(0, 1.5, n)).clip(0, 10).round(0)

    # latent churn risk -> probability -> label
    z = (
        1.5
        - 3.0 * adoption
        - 0.02 * logins
        - 1.5 * (active_seats / np.maximum(seats, 1))
        + 0.35 * unresolved
        + 0.03 * resolution
        - 0.12 * nps
        - 0.01 * tenure
        + rng.normal(0, 0.6, n)
    )
    p = 1 / (1 + np.exp(-z))
    churned = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "account_id": [f"ACC{100000 + i}" for i in range(n)],
        "industry": industry,
        "plan_tier": plan_tier,
        "mrr": mrr,
        "tenure_months": tenure,
        "seats_purchased": seats,
        "active_seats_last_30d": active_seats,
        "logins_last_30d": logins,
        "features_adopted": features_adopted,
        "avg_session_min": session_min,
        "support_tickets_last_90d": tickets,
        "unresolved_tickets": unresolved,
        "avg_resolution_days": resolution,
        "nps_last_survey": nps,
        "churned": churned,
    })


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/accounts.csv", index=False)
    print(f"Wrote data/accounts.csv | {len(df):,} accounts | churn rate {df['churned'].mean():.1%}")
