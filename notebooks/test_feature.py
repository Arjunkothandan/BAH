import sys
import os

project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(project_root)

import lightkurve as lk

from src.feature_extraction import extract_features

search = lk.search_lightcurve(
    "TIC 25155310",
    mission="TESS"
)

lc = search[0].download()

features = extract_features(lc)

print(features)