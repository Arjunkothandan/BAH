import pandas as pd

df = pd.read_csv(
    "data/processed/training_dataset.csv"
)

print(df.shape)

print(df.head())