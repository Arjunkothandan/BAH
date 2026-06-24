import pandas as pd

df = pd.read_csv(
    "data/raw/exo_CTL_08.01.csv",
    header=None
)

print(df[2].value_counts().head(20))