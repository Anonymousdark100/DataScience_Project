# Week 5 — Comprehensive Data Science Project

## Executive overview
This final project consolidates the machine-learning workflow into a reproducible report: dataset selection, preparation, model training, validation, performance comparison, and stakeholder recommendations.

## Dataset
Wisconsin Diagnostic Breast Cancer dataset, loaded through scikit-learn. It contains 569 observations and 30 numeric features. The positive class is malignant (1), negative class benign (0). This is strictly an educational dataset/model demonstration, not a clinical diagnostic tool.

## Models
- Logistic Regression (median imputation + scaling)
- Decision Tree (median imputation + constrained depth and leaf size)

## Validation
- Stratified 75/25 train/test split (random_state=42)
- Five-fold cross-validation on training data using ROC-AUC
- Held-out test metrics: accuracy, precision, recall, F1-score, ROC-AUC

## Visual evidence
1. Model performance comparison
2. Confusion matrix
3. ROC curve comparison
4. Five-fold cross-validation comparison

## Run
```bash
pip install -r requirements.txt
python final_project.py
```

## Key recommendation
Logistic Regression is the stronger baseline on this test split. However, if this were a real high-stakes application, do not deploy based on these results alone. Use independent validation, calibration, careful threshold analysis, bias assessment, and expert governance.

## References
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- https://scikit-learn.org/stable/modules/model_evaluation.html
- https://scikit-learn.org/stable/modules/compose.html#pipeline
