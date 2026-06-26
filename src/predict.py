import lightkurve as lk

from .feature_extraction import extract_features

print("\n=== AI Exoplanet Detector ===\n")

tic_id = input(
    "Enter TIC ID: "
)

print(
    "\nDownloading TESS Light Curve..."
)

search = lk.search_lightcurve(
    f"TIC {tic_id}",
    mission="TESS"
)

lc = search.download()

print(
    "Light Curve Downloaded Successfully!"
)

features = extract_features(
    lc
)

snr = features["snr"]

significance = features["significance"]

exo_score = snr * significance

if exo_score > 1:
    confidence = "HIGH"

elif exo_score > 0.3:
    confidence = "MEDIUM"

else:
    confidence = "LOW"

print("\n==========================")
print("EXOPLANET ANALYSIS REPORT")
print("==========================")

print(
    f"Period: {features['period']:.2f} days"
)

print(
    f"Duration: {features['duration']:.2f} days"
)

print(
    f"Depth: {features['depth']:.5f}"
)

print(
    f"SNR: {features['snr']:.2f}"
)

print(
    f"Significance: {features['significance']:.3f}"
)

print(
    f"ExoScore: {exo_score:.2f}"
)

print(
    f"Confidence: {confidence}"
)

print("==========================")