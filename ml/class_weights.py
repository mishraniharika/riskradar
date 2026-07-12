import pandas as pd

df = pd.read_csv("../data/creditcard_features.csv")

# Count how many of each class we have
class_counts = df["Class"].value_counts()
normal_count = class_counts[0]
fraud_count = class_counts[1]

# scale_pos_weight tells XGBoost how much more to penalize
# mistakes on the minority (fraud) class.
# Standard formula: (number of majority class) / (number of minority class)
scale_pos_weight = normal_count / fraud_count

print(f"Normal transactions: {normal_count}")
print(f"Fraud transactions: {fraud_count}")
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")