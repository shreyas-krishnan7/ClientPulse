"""
ClientPulse — standalone model trainer.

Trains + validates the churn model, prints the ROC-AUC, and (if SHAP is
installed) saves a SHAP summary plot to assets/shap_summary.png for the README.

Run:  python train_model.py
"""
import os
from pipeline import load_data, compute_kpis, train_churn_model, prepare_features


def main():
    df = compute_kpis(load_data())
    model, auc, scored, _ = train_churn_model(df)

    at_risk = int((scored["churn_risk"] > 0.5).sum())
    print(f"Validation ROC-AUC : {auc:.3f}")
    print(f"Accounts scored    : {len(scored):,}")
    print(f"Flagged at-risk    : {at_risk:,} (>50% churn probability)")

    # optional explainability artifact for the README
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X = prepare_features(df)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):        # binary classifier -> take positive class
            sv = sv[1]
        os.makedirs("assets", exist_ok=True)
        shap.summary_plot(sv, X, show=False)
        plt.tight_layout()
        plt.savefig("assets/shap_summary.png", dpi=120, bbox_inches="tight")
        print("Saved assets/shap_summary.png")
    except Exception as e:
        print(f"(SHAP plot skipped: {e})")


if __name__ == "__main__":
    main()
