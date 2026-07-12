\# Changelog



\## Week 1

\- Day 1: Documented understanding of fraud detection problem (false positives/negatives, key clues, why accuracy is misleading)

\- Day 2: Loaded and inspected Kaggle Credit Card Fraud dataset (284,807 transactions, 492 fraud)

\- Day 3: EDA on class imbalance (0.17% fraud) and transaction amount patterns

\- Day 4: Set up project structure, virtual environment, requirements.txt, .gitignore, GitHub Actions CI, CHANGELOG



\## Week 2

\- Day 1: Engineered time-window, velocity, and amount features

\- Day 2: Handled class imbalance via class weighting (scale\_pos\_weight)

\- Day 3: Trained XGBoost model with stratified train/test split

\- Day 4: Evaluated model (Precision 0.854, Recall 0.837, PR-AUC 0.868); tested threshold tuning, kept default

\- Day 5: Added SHAP explainability; documented false-positive vs correct-catch comparison

