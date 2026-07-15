"""
ClientPulse Module 3 — support-ticket text generator.

Creates data/tickets.csv, linked to the accounts in data/accounts.csv, so the
Voice-of-Customer engine has real text to classify. Each ticket keeps a
true_category / true_severity label so classifier accuracy can be measured.
Templates carry light paraphrase noise so classification isn't trivially perfect.

Run (after generate_data.py):  python generate_tickets.py
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CATEGORIES = ["Reconciliation", "Settlement", "Reporting", "API/Integration",
              "Performance", "Onboarding", "Access/Permissions", "Billing",
              "Data Accuracy", "UI/Usability"]
# uneven mix so a few themes clearly dominate the opportunity ranking
CAT_P = np.array([0.16, 0.14, 0.13, 0.11, 0.10, 0.08, 0.08, 0.07, 0.08, 0.05])
CAT_P = CAT_P / CAT_P.sum()

SEVERITIES = ["Critical", "High", "Medium", "Low"]  # order matters for the prob lists below

# severity depends on the category (settlement/data issues skew critical,
# UI issues skew low) so the ticket text actually carries a severity signal.
SEV_BY_CAT = {
    "Settlement":         [0.35, 0.40, 0.20, 0.05],
    "Data Accuracy":      [0.35, 0.35, 0.25, 0.05],
    "Reconciliation":     [0.25, 0.40, 0.30, 0.05],
    "Access/Permissions": [0.30, 0.35, 0.25, 0.10],
    "API/Integration":    [0.15, 0.35, 0.35, 0.15],
    "Performance":        [0.15, 0.35, 0.35, 0.15],
    "Reporting":          [0.08, 0.30, 0.42, 0.20],
    "Onboarding":         [0.05, 0.20, 0.45, 0.30],
    "Billing":            [0.05, 0.20, 0.45, 0.30],
    "UI/Usability":       [0.02, 0.10, 0.38, 0.50],
}

# explicit severity cue phrases appended to each ticket so the label is learnable
SEV_CUES = {
    "Critical": ["This is blocking all trading and needs an urgent fix.",
                 "Production is down for us — critical impact.",
                 "We cannot operate until this is resolved."],
    "High":     ["This is seriously impacting our desk today.",
                 "High priority, please escalate.",
                 "It's affecting multiple users right now."],
    "Medium":   ["Disrupting our workflow but we have a workaround.",
                 "Moderate impact, we'd like this fixed soon.",
                 "Not urgent but it keeps recurring."],
    "Low":      ["Minor issue, low priority.",
                 "Just a nice-to-have improvement.",
                 "No rush on this one."],
}

TEMPLATES = {
    "Reconciliation": [
        "Reconciliation break between our ledger and your settlement report for {date}.",
        "End-of-day balances don't match after recon — we're off by {amt}.",
        "Recon job flagged {n} unmatched trades again this morning."],
    "Settlement": [
        "Trade {id} failed to settle and is stuck in pending status.",
        "Settlement instruction was rejected with no clear reason.",
        "Several settlements missed the cutoff window yesterday."],
    "Reporting": [
        "The daily P&L export is missing columns since the last release.",
        "Can't schedule the regulatory report — it times out on generation.",
        "Exported report totals don't tie out to the dashboard figures."],
    "API/Integration": [
        "Our API integration returns 500 errors on the positions endpoint.",
        "Webhook events stopped firing after we rotated our API keys.",
        "The REST API rate limit is blocking our overnight batch sync."],
    "Performance": [
        "The dashboard takes over 30 seconds to load during market open.",
        "Query performance has degraded badly this week.",
        "Latency spikes make the blotter unusable at peak hours."],
    "Onboarding": [
        "New user setup is confusing — we can't complete account configuration.",
        "Onboarding docs don't match the current UI, we're stuck at SSO setup.",
        "Data migration during onboarding dropped several historical trades."],
    "Access/Permissions": [
        "A user can see accounts they shouldn't have permission to view.",
        "Admin role can't grant entitlements after the latest update.",
        "MFA lockout is blocking half our desk this morning."],
    "Billing": [
        "This month's invoice double-charged us for seats we removed.",
        "Billing shows Enterprise tier but we're on Growth.",
        "We were charged for overages we don't understand."],
    "Data Accuracy": [
        "Pricing feed shows stale marks for several instruments.",
        "FX rates in the app don't match our custodian's rates.",
        "Position quantities are wrong after the corporate action."],
    "UI/Usability": [
        "The new layout hides the export button — very hard to find.",
        "Filters reset every time we change tabs, it's frustrating.",
        "Dark mode makes several fields unreadable."],
}

NOISE = ["Please advise urgently.", "This is impacting our team.", "Second time reporting this.",
         "Any update would help.", "Happy to jump on a call.", "", ""]


def _fill(t, rng):
    return t.format(
        date=f"{rng.integers(1, 28):02d}/{rng.integers(1, 12):02d}",
        amt=f"${rng.integers(1, 900)}k",
        n=rng.integers(2, 40),
        id=f"TRD{rng.integers(10000, 99999)}",
    )


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    accounts = pd.read_csv("data/accounts.csv")
    today = datetime.today()

    rows, tid = [], 100000
    for _, a in accounts.iterrows():
        for _ in range(int(a["support_tickets_last_90d"])):
            cat = rng.choice(CATEGORIES, p=CAT_P)
            sev = rng.choice(SEVERITIES, p=SEV_BY_CAT[cat])
            text = " ".join([
                _fill(rng.choice(TEMPLATES[cat]), rng),
                rng.choice(SEV_CUES[sev]),
                rng.choice(NOISE),
            ]).strip()
            created = (today - timedelta(days=int(rng.integers(0, 90)))).date().isoformat()
            rows.append((f"TK{tid}", a["account_id"], created, text, cat, sev))
            tid += 1

    return pd.DataFrame(rows, columns=[
        "ticket_id", "account_id", "created_date", "raw_text",
        "true_category", "true_severity"])


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/tickets.csv", index=False)
    print(f"Wrote data/tickets.csv | {len(df):,} tickets across {df['account_id'].nunique():,} accounts")
