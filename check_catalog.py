import pandas as pd

df = pd.read_csv(
    "data/raw/exo_CTL_08.01.csv",
    header=None
)

print("Shape:")
print(df.shape)

print("\nFirst 20 Rows:")
print(df.head(20))

print("\nColumn Statistics:")
print(df.describe(include="all"))