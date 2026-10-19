"""
Week 5 — Comprehensive Data Science Project
Combines the model-development workflow from Week 4 into a reproducible final project.
Dataset: Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.
Positive class: malignant (1); negative class: benign (0).

Run:
    pip install -r requirements.txt
    python final_project.py

Educational demonstration only. Not for medical diagnosis.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

OUT = Path("visualizations")
OUT.mkdir(exist_ok=True)

dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = (dataset.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42))
    ])
}

rows, predictions, probabilities, cv = [], {}, {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    predictions[name], probabilities[name] = pred, prob
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    cv[name] = (cv_scores.mean(), cv_scores.std())
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "CV ROC-AUC mean": cv_scores.mean(),
        "CV ROC-AUC std": cv_scores.std()
    })

metrics = pd.DataFrame(rows)
print(metrics.to_string(index=False))
metrics.to_csv("model_metrics.csv", index=False)

# Performance comparison
ax = metrics.set_index("Model")[["Accuracy","Precision","Recall","F1-score","ROC-AUC"]].plot(
    kind="bar", figsize=(10,6)
)
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0,1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT/"01_model_performance_comparison.png", dpi=180)
plt.close()

# Confusion matrix
plt.figure(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions["Logistic Regression"], labels=[0,1],
    display_labels=["Benign","Malignant"], colorbar=False
)
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT/"02_logistic_confusion_matrix.png", dpi=180)
plt.close()

# ROC curve
plt.figure(figsize=(7,6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc = roc_auc_score(y_test, probabilities[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"03_roc_curve_comparison.png", dpi=180)
plt.close()

# Cross-validation chart
plt.figure(figsize=(8,5))
names = list(cv)
plt.bar(names, [cv[n][0] for n in names], yerr=[cv[n][1] for n in names], capsize=7)
plt.ylim(0,1.05)
plt.ylabel("ROC-AUC")
plt.title("Five-Fold Cross-Validation on Training Data")
plt.tight_layout()
plt.savefig(OUT/"04_cross_validation.png", dpi=180)
plt.close()

print("\\nImportant: educational dataset only; not for clinical use.")
