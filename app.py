import streamlit as st
import pandas as pd, numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from pathlib import Path
from notebooks.court import draw_half_court

# Streamlit page configuration
st.set_page_config(
    page_title="Curry shot predictor",
    page_icon="⛹🏻‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_data():
    df = pd.read_csv('data/processed/cleaned_curry_2009to25.csv')

    feature_names = ['shot_distance', 'shot_angle', 'time_remaining', 'is_3pt', 'is_clutch']
    X = df[feature_names]
    y = df['shot_made_flag']

    model_path = Path('models/rf_curry.pkl')
    model_source = "pickle"

    try:
        with open(model_path, 'rb') as f:
            header = f.read(64)
            f.seek(0)

            # Git LFS pointer files start with plain text, not pickle bytes.
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Model file is a Git LFS pointer, not a pickle binary.")

            model = pickle.load(f)
    except Exception:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)
        model_source = "retrained_fallback"

    return df, model, model_source

@st.cache_resource
def load_marker_images():
    base = Path(__file__).resolve().parent
    front = Image.open(base / "sc-front.png").convert("RGBA")
    side = Image.open(base / "sc-side.png").convert("RGBA")
    return {
        "front": np.array(front),
        "side": np.array(side),
    }

def get_marker_view(angle_deg):
    # Shots near center lane use front view; wider side angles use side view.
    return "side" if abs(angle_deg) >= 30 else "front"

df, model, model_source = load_model_data()
marker_images = load_marker_images()

if model_source == "retrained_fallback":
    st.warning(
        "`models/rf_curry.pkl` appears unavailable locally (likely Git LFS pointer). "
        "Using an auto-trained fallback model from `data/processed/cleaned_curry_2009to25.csv`."
    )

st.title("Steph Curry Shot Predictor")

st.write("""
This project predicts the prob% if Curry will make a shot based on:
- Shot distance (ft)
- Shot angle (°)
- Time remaining (s)
- 3-point shot?
- Clutch situation (<10s left)

Data was fetched from `nba_api` from 2009 to 2025 season

Model: Random Forest Classifier
""")

# Side-by-side layout: controls on the left, chart on the right.
controls_col, chart_col = st.columns([1, 1.35], gap="large")

with controls_col:
    st.subheader("Shot Setup")
    dist = st.slider("Shot distance", 0, 30, 15)
    angle = st.slider("Shot angle (°)", -90, 90, 0)
    time_rem = st.slider("Time remaining (s)", 0, 2880, 720)
    is_3pt = st.checkbox("3-point shot? `3pt = 23ft 9in | corner 3pt = 22ft`")
    is_clutch = st.checkbox("Clutch (<10s)?")

# Predict
feature_names = ['shot_distance', 'shot_angle', 'time_remaining', 'is_3pt', 'is_clutch']
features_df = pd.DataFrame([[dist, angle, time_rem, int(is_3pt), int(is_clutch)]], columns=feature_names)
prob = model.predict_proba(features_df)[0,1]

# Display probability with color
color = "green" if prob >= 0.4 else "red"
with controls_col:
    st.markdown(
        f"<span style='font-size:2em;'>Probability of Make: <span style='color:{color};'><b>{prob:.2%}</b></span></span>",
        unsafe_allow_html=True
    )

# Plot on court
fig, ax = plt.subplots(figsize=(8, 7.2))
draw_half_court(ax)

# Tighten the visible court so there is less unused space above the arc.
ax.set_ylim(-47.5, 320)
ax.margins(x=0, y=0)

# Convert distance to inches for plotting
dist_in = dist * 12
x = dist_in * np.sin(np.radians(angle))
y = dist_in * np.cos(np.radians(angle))

# Orientation-aware Curry marker (front for central/top angle, side for wide angle).
marker_view = get_marker_view(angle)
marker_img = marker_images[marker_view]

# Flip side view when shot is on the right side so the body faces inward.
if marker_view == "side" and x > 0:
    marker_img = np.fliplr(marker_img)

img_h_px, img_w_px = marker_img.shape[:2]
img_aspect = img_w_px / img_h_px

# Keep a consistent visual marker size while respecting each image's aspect ratio.
# Bumped up size range so the half-body marker reads more clearly.
marker_height = float(np.clip(68 - (dist_in * 0.06), 44, 68))
marker_width = marker_height * img_aspect
ax.imshow(
    marker_img,
    extent=(x - marker_width / 2, x + marker_width / 2, y - marker_height / 2, y + marker_height / 2),
    zorder=5,
)
fig.tight_layout(pad=0.2)

with chart_col:
    st.subheader("Shot Location")
    st.pyplot(fig)

with controls_col:
    st.write(f"Shot coordinates: ({x:.2f}, {y:.2f}) inches from hoop")
    st.write(f"Shot distance: {dist_in:.2f} inches")
    st.caption(f"Marker view: {marker_view}")

# Other stuff you can add in streamlit
st.sidebar.header("About")
st.sidebar.write("""
This app predicts the probability of Steph Curry making a shot based on various factors, the model is trained on historical shot data from the NBA.
""")

# Author and GitHub repo link
st.sidebar.markdown("---")
st.sidebar.markdown("**Created by:** Kriston")
st.sidebar.markdown("**Repository:** [beyond-the-arc](https://github.com/Kr1s7on/beyond-the-arc)")