import sys
import os

# Add src folder to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src"
        )
    )
)

import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt

try:
    from feature_extraction import extract_features  # type: ignore
    from preprocessing import clean_lightcurve  # type: ignore
except ImportError as e:
    st.error(f"Failed to import modules: {str(e)}")
    st.stop()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Exoplanet Detector",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 AI Exoplanet Detector")

st.write(
    """
    Analyze TESS Light Curves and identify
    potential exoplanet transit signals.
    """
)

# --------------------------------------------------
# INPUT
# --------------------------------------------------

tic_id = st.text_input(
    "Enter TIC ID",
    value="80423805"
)

# --------------------------------------------------
# BUTTON
# --------------------------------------------------

if st.button("Analyze Star"):

    try:

        with st.spinner(
            "Downloading TESS Light Curve..."
        ):

            search = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS"
            )

            lc = search.download()
            st.write("Type of lc:", type(lc))

        # ----------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------

        # use previously imported clean_lightcurve

        features = extract_features(lc)

        # ----------------------------------------
        # CONFIDENCE SCORE
        # ----------------------------------------

        exo_score = features["snr"] * features["significance"]

        if exo_score >= 15:
            confidence = "🟢 Strong Exoplanet Candidate"

        elif exo_score >= 5:
            confidence = "🟡 Possible Transit Candidate"

        else:
            confidence = "🔴 No Significant Transit Detected"

        # ----------------------------------------
        # METRICS
        # ----------------------------------------

        st.success(
            "Analysis Complete"
        )

        st.subheader(
            "Analysis Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Period (days)",
                f"{features['period']:.2f}"
            )

            st.metric(
                "Duration (days)",
                f"{features['duration']:.2f}"
            )

        with col2:
            st.metric(
                "SNR",
                f"{features['snr']:.2f}"
            )

            st.metric(
                "Depth",
                f"{features['depth']:.5f}"
            )

        with col3:
            st.metric(
                "Significance",
                f"{features['significance']:.3f}"
            )

            st.metric(
                "ExoScore",
                f"{exo_score:.2f}"
            )

        st.subheader("Classification")

        st.success(confidence)

        # ----------------------------------------
        # LIGHT CURVE
        # ----------------------------------------

        fig, ax = plt.subplots(figsize=(12,5))

        ax.plot(
            lc.time.value,
            lc.flux.value,
            ".",
            markersize=2
        )

        ax.set_title(f"TIC {tic_id}")

        ax.set_xlabel("Time")

        ax.set_ylabel("Normalized Flux")

        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )