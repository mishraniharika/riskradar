import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

# Load the engineered dataset from Week 2 Day 1
df = pd.read_csv("../data/creditcard_features.csv")

# Drop the helper column we created (not a real feature)
df = df.drop(columns=["Time_td"])

# Separate features (X) from the label (y)
X = df.drop(columns=["Class"])
y = df["Class"]

# Stratified train/test split - keeps the same fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% held out for testing
    random_state=42,     # makes the split reproducible - same split every time you run this
    stratify=y            # ensures fraud ratio stays consistent in both train and test
)

print(f"Train set: {X_train.shape[0]} rows, fraud cases: {y_train.sum()}")
print(f"Test set: {X_test.shape[0]} rows, fraud cases: {y_test.sum()}")

# Calculate scale_pos_weight the same way as Day 2, but on the TRAINING set only
# (important: always calculate this from train data, never from test data,
# to avoid "leaking" information about the test set into training decisions)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight (train-based): {scale_pos_weight:.2f}")

# Build and train the XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,          # number of decision trees the model builds
    max_depth=5,               # how deep each tree can grow (controls complexity)
    learning_rate=0.1,         # how much each tree corrects the previous ones' mistakes
    scale_pos_weight=scale_pos_weight,  # tells the model fraud mistakes cost ~578x more
    eval_metric="aucpr",       # optimizes for precision-recall, not plain accuracy
    random_state=42
)

model.fit(X_train, y_train)

# Save the trained model and the train/test split for tomorrow's evaluation
joblib.dump(model, "../data/xgb_model.pkl")
X_test.to_csv("../data/X_test.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("\nModel trained and saved to data/xgb_model.pkl")