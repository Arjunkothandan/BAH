import pandas as pd
import random

df = pd.read_csv(
    "data/raw/transit_targets.csv"
)

tic_id = random.choice(
    df["tic_id"].tolist()
)

print(
    f"Selected TIC ID: {tic_id}"
)