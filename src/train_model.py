import pandas as pd

df = pd.read_csv(
    "data/processed/training_dataset.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nFirst 5 Samples:")
print(df.head())