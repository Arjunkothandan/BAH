import pandas as pd

df = pd.read_csv("data/training_dataset.csv")

print(df["label"].value_counts())