# generate_transit_targets.py

import pandas as pd

df = pd.read_csv(
    "data/raw/exo_CTL_08.01.csv",
    header=None
)

transit_df = df[
    df[2] == "planetcandidate"
]

transit_df = transit_df.head(100)

result = pd.DataFrame({
    "tic_id": transit_df[0],
    "label": "Transit"
})

result.to_csv(
    "data/raw/transit_targets.csv",
    index=False
)

print(result.head())

print(
    f"\nTotal Transit Stars: {len(result)}"
)