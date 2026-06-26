import pandas as pd

df = pd.read_csv(
    "data/raw/transit_targets.csv"
)

print(df.head())

print("\nShape:")
print(df.shape)