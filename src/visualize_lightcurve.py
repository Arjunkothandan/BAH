import lightkurve as lk
import pandas as pd
import matplotlib.pyplot as plt

tic_id = input(
    "Enter TIC ID: "
)

print(
    "\nDownloading Light Curve..."
)

search = lk.search_lightcurve(
    f"TIC {tic_id}",
    mission="TESS"
)

lc = search.download()

df = lc.to_pandas()

df = df[
    df["quality"] == 0
]

df = df.dropna(
    subset=["pdcsap_flux"]
)

# Find lowest flux point
min_idx = df["pdcsap_flux"].idxmin()

min_flux = df.loc[
    min_idx,
    "pdcsap_flux"
]

plt.figure(
    figsize=(12,6)
)

plt.plot(
    df.index,
    df["pdcsap_flux"]
)

plt.scatter(
    min_idx,
    min_flux,
    s=100
)

plt.annotate(
    "Possible Transit",
    (
        min_idx,
        min_flux
    )
)

plt.title(
    f"TIC {tic_id} Light Curve"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Flux"
)

plt.grid()

plt.savefig(
    f"plots/tic_{tic_id}.png"
)

plt.show()

print(
    f"\nSaved: plots/tic_{tic_id}.png"
)