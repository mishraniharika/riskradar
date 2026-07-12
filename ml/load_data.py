import pandas as pd

# Load the dataset
df = pd.read_csv("../data/creditcard.csv")

# Show the first 5 rows
print(df.head())

# Show basic info: column names, data types, missing values
print(df.info())

# Show how many fraud vs non-fraud transactions exist
print(df["Class"].value_counts())