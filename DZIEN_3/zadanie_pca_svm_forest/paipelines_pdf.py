# ===============================================
# Pima Indians Diabetes — AUTO RUN (PyCharm)
# ===============================================

from __future__ import annotations

import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

warnings.filterwarnings("ignore")

# ===============================================
# KONFIGURACJA — EDYTUJ TYLKO TO
# ===============================================

CSV_PATH = "diabetes_with_headers.csv"     # plik musi być w katalogu projektu
TARGET_COL = "Outcome"
OUTPUT_DIR = "out"

# ===============================================

RANDOM_STATE = 42


def build_preprocessor(feature_names: List[str]) -> Tuple[ColumnTransformer, List[str]]:
    zero_as_missing_candidates = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zero_as_missing = [c for c in zero_as_missing_candidates if c in feature_names]
    other_cols = [c for c in feature_names if c not in zero_as_missing]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("zero_as_missing_num", numeric_pipe, zero_as_missing),
            ("plain_num", numeric_pipe, other_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, zero_as_missing


def apply_zero_to_nan(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X2 = X.copy()
    for c in cols:
        X2[c] = X2[c].replace(0, np.nan)
    return X2


def predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def main():

    print("=== START REPORT BUILD ===")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    # ---- LOAD ----
    df = pd.read_csv(CSV_PATH)
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL]).apply(pd.to_numeric, errors="coerce")

    preprocessor, zero_cols = build_preprocessor(list(X.columns))
    X_clean = apply_zero_to_nan(X, zero_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---- MODELE ----
    rf_model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ))
    ])

    pca_svm = Pipeline([
        ("prep", preprocessor),
        ("pca", PCA()),
        ("svm", SVC(probability=True))
    ])

    stack = StackingClassifier(
        estimators=[
            ("pca_svm", pca_svm),
            ("rf", rf_model),
        ],
        final_estimator=LogisticRegression(max_iter=500),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )

    models = {
        "RF": rf_model,
        "PCA+SVM": pca_svm,
        "STACK": stack
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        proba = predict_proba_safe(model, X_test)
        y_pred = (proba >= 0.5).astype(int)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_pos": precision_score(y_test, y_pred),
            "recall_pos": recall_score(y_test, y_pred),
            "f1_pos": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        results.append(metrics)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure()
        ConfusionMatrixDisplay(cm).plot()
        plt.title(name)
        plt.savefig(out_dir / f"cm_{name}.png")
        plt.close(fig)

    summary_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    # ---- ROC ----
    fig = plt.figure()
    for name, model in models.items():
        proba = predict_proba_safe(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curves")
    plt.savefig(out_dir / "roc.png")
    plt.close(fig)

    # ---- HTML ----
    html = f"""
    <h1>Pima Diabetes Report</h1>
    {summary_df.to_html(index=False)}
    <h2>ROC</h2>
    <img src="roc.png" width="500">
    """

    (out_dir / "report.html").write_text(html, encoding="utf-8")

    # ---- PDF ----
    doc = SimpleDocTemplate(str(out_dir / "report.pdf"), pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph("Pima Diabetes Report", styles["Title"]))
    story.append(Spacer(1, 12))

    table_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
    t = Table(table_data)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(PageBreak())

    story.append(Image(str(out_dir / "roc.png"), width=400, height=300))

    doc.build(story)

    print("=== REPORT READY ===")
    print("Folder:", out_dir.resolve())


if __name__ == "__main__":
    main()
