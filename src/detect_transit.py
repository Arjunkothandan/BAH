import lightkurve as lk

from .feature_extraction import extract_features

tic_id = input(
    "Enter TIC ID: "
)

search = lk.search_lightcurve(
    f"TIC {tic_id}",
    mission="TESS"
)

lc = search.download()

features = extract_features(
    lc
)

print("\nRESULTS")

for key, value in features.items():
    print(f"{key}: {value}")

exo_score = (
    features["snr"]
    * features["significance"]
)

print(
    f"\nExoScore: {exo_score:.2f}"
)

if exo_score > 1:
    print(
        "Candidate Strength: HIGH"
    )
else:
    print(
        "Candidate Strength: LOW"
    )