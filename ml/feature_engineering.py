import pandas as pd
import numpy as np

# Load the raw dataset
df = pd.read_csv("../data/creditcard.csv")

# 1. Hour of day
# Time is seconds elapsed since the first transaction.
# We convert it into hour-of-day (0-23) using modulo 86400 (seconds in a day).
df["hour_of_day"] = (df["Time"] % 86400) // 3600

# 2. Time since previous transaction
# .diff() calculates the difference between each row and the one before it.
df["time_since_prev"] = df["Time"].diff().fillna(0)

# 3. Transaction count in rolling 60-second window
# We sort by Time first to make sure rolling windows are in correct order.
df = df.sort_values("Time").reset_index(drop=True)
# Convert Time (plain seconds) into a proper timedelta type,
# since pandas needs a real time-aware column for time-based rolling windows.
df["Time_td"] = pd.to_timedelta(df["Time"], unit="s")
df["txn_count_last_60s"] = df.rolling("60s", on="Time_td")["Time_td"].count()

# 4. Log-transformed Amount
# log1p = log(1 + x), safely handles Amount = 0 without breaking (log(0) is undefined).
df["log_amount"] = np.log1p(df["Amount"])

# 5. Amount relative to rolling average (last 50 transactions)
df["rolling_avg_amount"] = df["Amount"].rolling(window=50, min_periods=1).mean()
df["amount_vs_rolling_avg"] = df["Amount"] / (df["rolling_avg_amount"] + 1e-5)
# +1e-5 is a tiny number added to avoid dividing by zero

# Sanity check: show the new columns
print(df[["Time", "hour_of_day", "time_since_prev", "txn_count_last_60s",
          "Amount", "log_amount", "rolling_avg_amount",
          "amount_vs_rolling_avg", "Class"]].head(10))

print("\nShape of dataset after feature engineering:", df.shape)
print("\nAny missing values introduced?\n", df.isnull().sum().sum())

print(df[["Time", "time_since_prev", "txn_count_last_60s"]].head(15))
print(df[["Time", "txn_count_last_60s"]].iloc[500:515])
# Save the engineered dataset for use in Week 2 model training
df.to_csv("../data/creditcard_features.csv", index=False)
print("\nSaved engineered dataset to data/creditcard_features.csv")