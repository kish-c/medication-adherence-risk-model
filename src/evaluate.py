"""
Evaluates both trained models on the held-out test set and saves all
figures + a metrics summary to reports/.

Run:
    python -m src.evaluate
"""

import os
import pickle
import json

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from src.config import (
    TEST_FILE, DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR,
    FEATURE_COLUMNS, TARGET_COLUMN,
    FIGURE_DPI, FIGURE_STYLE,
)
from src.utils import get_logger, ensure_dirs

log = get_logger(__name__)
MODELS_DIR = os.path.join(DATA_PROCESSED_DIR, "models")

plt.style.use(FIGURE_STYLE)
COLORS = {"lr": "#5B8DB8", "rf": "#E05C5C"}


def _load(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _plot_roc(models: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (model, color) in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Medication Non-Adherence Prediction", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curve.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_confusion_matrix(model, model_name: str, X_test, y_test, color: str) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Adherent", "Non-Adherent"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    log.info("Saved %s", path)
    # Also save primary as confusion_matrix.png
    if "logistic" in model_name.lower():
        import shutil
        shutil.copy(path, os.path.join(FIGURES_DIR, "confusion_matrix.png"))


def _plot_feature_importance(lr_model, rf_model, feature_names: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Logistic Regression coefficients
    coef = lr_model.named_steps["clf"].coef_[0]
    sorted_idx = np.argsort(np.abs(coef))
    ax = axes[0]
    colors = [COLORS["lr"] if c > 0 else "#E05C5C" for c in coef[sorted_idx]]
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        coef[sorted_idx],
        color=colors, edgecolor="white", height=0.7,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Logistic Regression\nCoefficients (scaled)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coefficient value")

    # Random Forest feature importances
    importances = rf_model.named_steps["clf"].feature_importances_
    sorted_idx  = np.argsort(importances)
    ax = axes[1]
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=COLORS["rf"], edgecolor="white", height=0.7,
    )
    ax.set_title("Random Forest\nFeature Importances", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance score")

    fig.suptitle("Model Interpretability", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_score_distribution(models: dict, X_test, y_test) -> None:
    """Risk score distributions for adherent vs. non-adherent patients."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, (label, (model, color)) in zip(axes, models.items()):
        proba = model.predict_proba(X_test)[:, 1]
        df_tmp = pd.DataFrame({"score": proba, "label": y_test.values})
        for val, grp_label, c in [(0, "Adherent", COLORS["lr"]), (1, "Non-Adherent", COLORS["rf"])]:
            subset = df_tmp[df_tmp["label"] == val]["score"]
            ax.hist(subset, bins=30, alpha=0.65, color=c, label=grp_label, density=True)
        ax.axvline(0.5, color="black", linestyle="--", lw=1.2, label="Threshold 0.5")
        ax.set_title(f"Risk Score Distribution\n{label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Probability (non-adherent)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "score_distribution.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    log.info("Saved %s", path)


def main() -> None:
    ensure_dirs(FIGURES_DIR, REPORTS_DIR)

    log.info("Loading test data …")
    test = pd.read_csv(TEST_FILE)
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    log.info("Loading trained models …")
    lr = _load("logistic_regression")
    rf = _load("random_forest")

    models = {
        "Logistic Regression": (lr, COLORS["lr"]),
        "Random Forest":       (rf, COLORS["rf"]),
    }

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics_summary = {}
    for model_name, (model, _) in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        report  = classification_report(y_test, y_pred, output_dict=True)
        auc     = roc_auc_score(y_test, y_proba)
        avg_p   = average_precision_score(y_test, y_proba)

        metrics_summary[model_name] = {
            "roc_auc":       round(auc, 4),
            "avg_precision": round(avg_p, 4),
            "precision_1":   round(report["1"]["precision"], 4),
            "recall_1":      round(report["1"]["recall"], 4),
            "f1_1":          round(report["1"]["f1-score"], 4),
            "accuracy":      round(report["accuracy"], 4),
        }
        log.info("\n%s\n%s", model_name,
                 classification_report(y_test, y_pred,
                                       target_names=["Adherent", "Non-Adherent"]))

    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("Generating plots …")
    _plot_roc(models, X_test, y_test)
    _plot_confusion_matrix(lr, "Logistic Regression", X_test, y_test, COLORS["lr"])
    _plot_confusion_matrix(rf, "Random Forest",       X_test, y_test, COLORS["rf"])
    _plot_feature_importance(lr, rf, FEATURE_COLUMNS)
    _plot_score_distribution(models, X_test, y_test)

    log.info("Evaluation complete. All figures in %s", FIGURES_DIR)

    # Print summary table
    print("\n" + "="*60)
    print("  MODEL PERFORMANCE SUMMARY")
    print("="*60)
    header = f"{'Model':<25} {'ROC-AUC':>8} {'F1':>6} {'Recall':>8} {'Precision':>10}"
    print(header)
    print("-"*60)
    for name, m in metrics_summary.items():
        print(f"{name:<25} {m['roc_auc']:>8.3f} {m['f1_1']:>6.3f} {m['recall_1']:>8.3f} {m['precision_1']:>10.3f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
