import pandas as pd
import lightkurve as lk
from feature_extraction import extract_features

targets = pd.read_csv(
    "data/raw/transit_targets.csv"
)

dataset = []

for idx, row in targets.iterrows():

    tic_id = row["tic_id"]
    label = row["label"]

    print(
        f"Processing TIC {tic_id}"
    )

    try:

        search = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS"
        )

        lc = search.download()

        features = extract_features(
            lc
        )

        features["label"] = label

        dataset.append(
            features
        )

    except Exception as e:

        print(
            f"Failed {tic_id}: {e}"
        )

df = pd.DataFrame(
    dataset
)

df.to_csv(
    "data/processed/training_dataset.csv",
    index=False
)

print(
    "\nDataset Saved!"
)

print(df.head())