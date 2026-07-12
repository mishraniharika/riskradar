import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model and test data
model = joblib.load("../data/xgb_model.pkl")
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

# Create a SHAP explainer for our XGBoost model
# TreeExplainer is optimized specifically for tree-based models (like XGBoost)
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the whole test set
# This tells us, for every transaction, how much each feature pushed the score up or down
shap_values = explainer.shap_values(X_test)

# ---- Global view: which features matter most OVERALL across all transactions ----
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("../results/shap_summary.png")
plt.close()
print("Saved overall feature importance to results/shap_summary.png")

# ---- Local view: explain ONE specific flagged transaction ----
# Find a transaction the model flagged as fraud (high probability)
y_proba = model.predict_proba(X_test)[:, 1]
flagged_indices = X_test[y_proba > 0.5].index

# Pick the first flagged transaction as an example
example_idx = flagged_indices[0]
example_position = X_test.index.get_loc(example_idx)

print(f"\nExplaining transaction at index {example_idx}")
print(f"Model's risk score for this transaction: {y_proba[example_position]:.3f}")
print(f"Actual label: {'FRAUD' if y_test.iloc[example_position] == 1 else 'NORMAL'}")

# Show the top features that drove THIS specific prediction
shap_values_single = shap_values[example_position]
feature_names = X_test.columns
contributions = pd.Series(shap_values_single, index=feature_names).sort_values(key=abs, ascending=False)

print("\nTop 5 features driving this prediction:")
print(contributions.head(5))

# Save a visual explanation for this single transaction
plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_single,
        base_values=explainer.expected_value,
        data=X_test.iloc[example_position],
        feature_names=list(feature_names)
    ),
    show=False
)
plt.tight_layout()
plt.savefig("../results/shap_single_transaction.png")
plt.close()
print("\nSaved single-transaction explanation to results/shap_single_transaction.png")

# For contrast: explain a transaction that was CORRECTLY caught as fraud
correct_fraud_mask = (y_proba > 0.5) & (y_test.values == 1)
correct_fraud_indices = X_test[correct_fraud_mask].index

if len(correct_fraud_indices) > 0:
    idx2 = correct_fraud_indices[0]
    pos2 = X_test.index.get_loc(idx2)
    print(f"\n--- Contrast example: correctly caught fraud ---")
    print(f"Transaction at index {idx2}, risk score: {y_proba[pos2]:.3f}")
    contributions2 = pd.Series(shap_values[pos2], index=feature_names).sort_values(key=abs, ascending=False)
    print("Top 5 features:")
    print(contributions2.head(5))

# Save the explainer itself for use in the backend next week
joblib.dump(explainer, "../data/shap_explainer.pkl")
print("\nSaved SHAP explainer to data/shap_explainer.pkl")