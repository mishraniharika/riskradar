import pandas as pd
import joblib
from sklearn.metrics import (
    precision_score, recall_score, precision_recall_curve,
    auc, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# Load the saved model and test set from Day 3
model = joblib.load("../data/xgb_model.pkl")
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()  # squeeze turns single-column df into a plain Series

# Get predictions
# predict() gives hard 0/1 decisions using a default 0.5 threshold
y_pred = model.predict(X_test)

# predict_proba() gives the raw risk score (0 to 1) before any threshold is applied
# we need this for precision-recall curve and PR-AUC
y_proba = model.predict_proba(X_test)[:, 1]

# 1. Precision and Recall at default threshold (0.5)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# 2. Confusion matrix - shows exact counts of each outcome type
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix breakdown:")
print(f"True Negatives (correctly caught as normal): {tn}")
print(f"False Positives (wrongly flagged as fraud): {fp}")
print(f"False Negatives (fraud MISSED): {fn}")
print(f"True Positives (fraud correctly caught): {tp}")

# 3. Full classification report (precision/recall/f1 for both classes)
print("\n", classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

# 4. PR-AUC - the single summary score across all thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recalls, precisions)
print(f"PR-AUC: {pr_auc:.3f}")

# 5. Plot the Precision-Recall curve
plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall (fraud caught)")
plt.ylabel("Precision (accuracy of fraud flags)")
plt.title("Precision-Recall Curve - RiskRadar Model")
plt.legend()
plt.savefig("../results/pr_curve.png")
plt.close()

print("\nSaved PR curve to results/pr_curve.png")

# ---- Threshold comparison: what if we lower the cutoff to catch more fraud? ----
from sklearn.metrics import precision_score, recall_score

thresholds_to_try = [0.5, 0.3, 0.2]

print("\nThreshold comparison:")
print(f"{'Threshold':<10}{'Precision':<12}{'Recall':<10}{'Fraud Caught':<15}{'False Alarms'}")

for t in thresholds_to_try:
    y_pred_t = (y_proba >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    print(f"{t:<10}{p:<12.3f}{r:<10.3f}{tp_t:<15}{fp_t}")