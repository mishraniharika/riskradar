import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("../data/creditcard.csv")

# 1. Class imbalance - exact numbers and percentage
fraud_count = df["Class"].value_counts()
fraud_percent = df["Class"].value_counts(normalize=True) * 100
print("Counts:\n", fraud_count)
print("\nPercentage:\n", fraud_percent)

# 2. Visualize the imbalance as a bar chart
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Normal (0) vs Fraud (1) Transaction Counts")
plt.savefig("../data/class_imbalance.png")
plt.close()

# 3. Compare transaction amounts: fraud vs normal
print("\nAmount stats for NORMAL transactions:")
print(df[df["Class"] == 0]["Amount"].describe())

print("\nAmount stats for FRAUD transactions:")
print(df[df["Class"] == 1]["Amount"].describe())

# 4. Visualize amount distribution for both classes
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=df)
plt.ylim(0, 500)  # zoom in, since a few huge outliers can squash the chart
plt.title("Transaction Amount: Normal vs Fraud")
plt.savefig("../data/amount_comparison.png")
plt.close()

print("\nSaved charts: class_imbalance.png and amount_comparison.png in the data folder")