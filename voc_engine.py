"""
Pipeline:
  1. classify each ticket   (category + severity + sentiment)
  2. cluster tickets into emergent themes
  3. score & rank product opportunities  (frequency x severity x revenue-at-risk)

Classifier paths:
  * USE_LLM = True   -> Groq LLM (few-shot, batched, temperature=0)  [set GROQ_API_KEY]
  * USE_LLM = False  -> offline TF-IDF + LogisticRegression (no API key)
Outputs:  data/tickets_classified.csv , data/opportunities.csv
"""
import os
import json
import time
import numpy as np
import pandas as pd

# Load variables from a local .env file (e.g. GROQ_API_KEY) if present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---------------- config ----------------
USE_LLM = True                 # you generated a Groq key -> use the LLM path
SAMPLE = 5000                # START small to smoke-test the wiring; set to None for the full run
GROQ_MODEL = "llama-3.1-8b-instant"   # fast + free-tier friendly (or "llama-3.3-70b-versatile" for accuracy)
BATCH_SIZE = 20                # tickets classified per API call
SEV_WEIGHT = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
N_THEMES = 12

CATEGORIES = ["Reconciliation", "Settlement", "Reporting", "API/Integration",
              "Performance", "Onboarding", "Access/Permissions", "Billing",
              "Data Accuracy", "UI/Usability"]
SEVERITIES = ["Critical", "High", "Medium", "Low"]

SYSTEM_PROMPT = f"""You are a support-ticket classifier for a B2B fintech (trading/treasury) SaaS.
For EACH ticket, assign exactly one category and one severity.

Categories: {CATEGORIES}
Severities: {SEVERITIES}

Examples:
- "Trade TRD12345 failed to settle and is stuck in pending." -> category: Settlement, severity: High
- "The new layout hides the export button." -> category: UI/Usability, severity: Low
- "Pricing feed shows stale marks for several instruments." -> category: Data Accuracy, severity: Critical

Return ONLY JSON of this exact shape, one object per ticket, SAME order as input:
{{"results": [{{"i": 1, "category": "...", "severity": "..."}}]}}
No prose, no markdown."""


# ---------- 1a. LLM classification (Groq) ----------
def _coerce_category(c):
    if not c:
        return CATEGORIES[0]
    for cat in CATEGORIES:                       # case-insensitive match
        if c.strip().lower() == cat.lower():
            return cat
    return CATEGORIES[0]


def _classify_batch(client, batch, attempts=3):
    user = "Classify these tickets:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
    for a in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
            )
            results = json.loads(resp.choices[0].message.content).get("results", [])
            out = []
            for i in range(len(batch)):
                item = results[i] if i < len(results) else {}
                sev = item.get("severity")
                out.append({
                    "category": _coerce_category(item.get("category")),
                    "severity": sev if sev in SEVERITIES else "Medium",
                })
            return out
        except Exception as e:
            if a == attempts - 1:                # give up on this batch -> safe defaults
                print(f"  [warn] batch failed ({e}); using defaults")
                return [{"category": CATEGORIES[0], "severity": "Medium"} for _ in batch]
            time.sleep(2 * (a + 1))              # backoff on rate limit / transient errors


def llm_classify(texts):
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to a .env file "
                           "(GROQ_API_KEY=your_key) or export it in your shell.")
    client = Groq(api_key=key)

    cats, sevs = [], []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for b, start in enumerate(range(0, len(texts), BATCH_SIZE), 1):
        parsed = _classify_batch(client, texts[start:start + BATCH_SIZE])
        cats.extend(p["category"] for p in parsed)
        sevs.extend(p["severity"] for p in parsed)
        print(f"  classified batch {b}/{n_batches}", end="\r")
        time.sleep(0.5)                          # gentle on the free-tier rate limit
    print()
    return cats, sevs


# ---------- 1b. offline classification (fallback) ----------
def offline_classify(df):
    vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(df["raw_text"])
    results = {}
    for target in ["category", "severity"]:
        y = df[f"true_{target}"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000, C=4.0).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        results[target] = (accuracy_score(yte, pred), f1_score(yte, pred, average="macro"))
        df[f"pred_{target}"] = LogisticRegression(max_iter=1000, C=4.0).fit(X, y).predict(X)
    return df, vec, X, results


def add_sentiment(df):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        df["sentiment"] = df["raw_text"].apply(lambda t: sia.polarity_scores(t)["compound"])
    except Exception:
        df["sentiment"] = 0.0
    return df


# ---------- 2. theme clustering ----------
def cluster_themes(df, X, vec, n_themes=N_THEMES):
    km = KMeans(n_clusters=n_themes, random_state=42, n_init=10)
    df["theme_id"] = km.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    names = {c: ", ".join(terms[km.cluster_centers_[c].argsort()[::-1][:3]])
             for c in range(n_themes)}
    df["theme"] = df["theme_id"].map(names)
    return df


# ---------- 3. opportunity scoring ----------
def score_opportunities(df):
    accounts = pd.read_csv("data/accounts.csv")
    if "churn_risk" not in accounts.columns:          # reuse v1's churn model
        from pipeline import load_data, compute_kpis, train_churn_model
        _, _, accounts, _ = train_churn_model(compute_kpis(load_data()))

    g = df.merge(accounts[["account_id", "mrr", "churn_risk"]], on="account_id", how="left")
    g["churn_risk"] = g["churn_risk"].fillna(g["churn_risk"].median())
    g["sev_w"] = g["pred_severity"].map(SEV_WEIGHT)
    g["rev_at_risk"] = g["mrr"] * g["churn_risk"]

    opp = g.groupby("theme_id").agg(
        theme=("theme", "first"),
        top_category=("pred_category", lambda s: s.mode().iat[0]),
        tickets=("ticket_id", "count"),
        avg_severity=("sev_w", "mean"),
        accounts_affected=("account_id", "nunique"),
        revenue_at_risk=("rev_at_risk", "sum"),
        avg_sentiment=("sentiment", "mean"),
    ).reset_index(drop=True)

    for col in ["tickets", "avg_severity", "revenue_at_risk"]:
        lo, hi = opp[col].min(), opp[col].max()
        opp[f"{col}_n"] = (opp[col] - lo) / (hi - lo + 1e-9)
    opp["opportunity_score"] = (
        (opp["tickets_n"] * opp["avg_severity_n"] * opp["revenue_at_risk_n"]) ** (1 / 3) * 10
    ).round(1)
    opp["revenue_at_risk"] = opp["revenue_at_risk"].round(0)
    return opp.sort_values("opportunity_score", ascending=False).reset_index(drop=True)


def main():
    df = pd.read_csv("data/tickets.csv")

    if SAMPLE:
        df = df.sample(min(SAMPLE, len(df)), random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df):,} tickets (set SAMPLE=None in voc_engine.py for the full run)")

    if USE_LLM:
        print(f"Classifier: Groq LLM ({GROQ_MODEL})")
        df["pred_category"], df["pred_severity"] = llm_classify(df["raw_text"].tolist())
        vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2)).fit(df["raw_text"])
        X = vec.transform(df["raw_text"])
        if "true_category" in df.columns:            # honest out-of-sample accuracy
            print(f"  vs ground truth -> category {accuracy_score(df['true_category'], df['pred_category']):.1%} "
                  f"| severity {accuracy_score(df['true_severity'], df['pred_severity']):.1%}")
    else:
        print("Classifier: offline TF-IDF + LogisticRegression")
        df, vec, X, results = offline_classify(df)
        for t, (acc, f1) in results.items():
            print(f"  {t:9s} accuracy {acc:.1%} | macro-F1 {f1:.2f}")

    df = add_sentiment(df)
    df = cluster_themes(df, X, vec)
    df.to_csv("data/tickets_classified.csv", index=False)

    opp = score_opportunities(df)
    opp.to_csv("data/opportunities.csv", index=False)

    print(f"\nProcessed {len(df):,} tickets -> {opp.shape[0]} themes")
    print("Top 3 product opportunities:")
    for _, r in opp.head(3).iterrows():
        print(f"  [{r['opportunity_score']:>4}] {r['theme']:<32} "
              f"{int(r['tickets']):>4} tickets | {int(r['accounts_affected']):>4} accounts "
              f"| ${r['revenue_at_risk']:,.0f} at risk")


if __name__ == "__main__":
    main()
