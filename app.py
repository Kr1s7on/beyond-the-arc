import streamlit as st
import pandas as pd, numpy as np
import pickle
import matplotlib.pyplot as plt
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

# Features the model was trained on. The UI only collects 5 of them; the rest
# are derived from those 5 or held at typical-game values.
MODEL_FEATURES = [
    'log_dist', 'abs_angle', 'time_frac', 'is_clutch', 'is_3pt', 'is_overtime',
    'is_home', 'attempt_idx', 'attempt_frac', 'game_fg_sofar', 'game_3fg_sofar',
    'days_rest', 'roll5_3fg', 'action_type', 'shot_zone_basic', 'period', 'opp',
]

DEFAULTS = {
    'is_overtime': 0, 'is_home': 0, 'attempt_idx': 9, 'attempt_frac': 0.47,
    'game_fg_sofar': 0.474, 'game_3fg_sofar': 0.422, 'days_rest': 2,
    'roll5_3fg': 0.409, 'period': 3, 'opp': 'PHX',
}

@st.cache_resource
def load_model_data():
    df = pd.read_csv('data/processed/cleaned_curry_2009to25.csv')
    model_path = Path('models/curry_shot_model.pkl')

    with open(model_path, 'rb') as f:
        header = f.read(64)
        f.seek(0)

        # Git LFS pointer files start with plain text, not pickle bytes.
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Model file is a Git LFS pointer, not a pickle binary.")

        bundle = pickle.load(f)

    return df, bundle['model'], bundle['threshold']

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

def build_features(dist, angle, time_rem, is_3pt):
    # derive shot zone and action type from distance + angle
    abs_angle = abs(angle)

    if is_3pt:
        if abs_angle >= 55 and dist <= 23.75:
            zone = 'Left Corner 3' if angle < 0 else 'Right Corner 3'
        else:
            zone = 'Above the Break 3'
    elif dist <= 4:
        zone = 'Restricted Area'
    elif dist <= 8:
        zone = 'In The Paint (Non-RA)'
    else:
        zone = 'Mid-Range'

    if is_3pt or dist >= 16:
        action = 'Pullup Jump shot' if abs_angle >= 40 else 'Jump Shot'
    elif dist <= 4:
        action = 'Driving Layup Shot'
    else:
        action = 'Floating Jump shot'

    row = {
        'log_dist': np.log1p(dist),
        'abs_angle': abs_angle,
        'time_frac': time_rem / 720.0,
        'is_clutch': int(time_rem < 10),  # derived from the time slider
        'is_3pt': int(is_3pt),
        'action_type': action,
        'shot_zone_basic': zone,
    }
    row.update(DEFAULTS)
    return pd.DataFrame([row])[MODEL_FEATURES]

df, model, threshold = load_model_data()
marker_images = load_marker_images()

st.title("Steph Curry Shot Predictor")

st.write("""
This project predicts the prob% if Curry will make a shot based on:
- Shot distance (ft)
- Shot angle (°)
- Time remaining (s)
- 3-point shot?

Data was fetched from `nba_api` from 2009 to 2025 season

Model: Gradient Boosting (shot type + location + form features)

Clutch is derived automatically from the time slider (<10s left).
""")

# Side-by-side layout: controls on the left, chart on the right.
controls_col, chart_col = st.columns([1, 1.35], gap="large")

with controls_col:
    st.subheader("Shot Setup")
    dist = st.slider("Shot distance", 0, 30, 15)
    angle = st.slider("Shot angle (°)", -90, 90, 0)
    time_rem = st.slider("Time remaining (s)", 0, 2880, 720)
    is_3pt = st.checkbox("3-point shot? `3pt = 23ft 9in | corner 3pt = 22ft`")

# Predict
features_df = build_features(dist, angle, time_rem, is_3pt)
prob = model.predict_proba(features_df)[0,1]

# Display probability with color (threshold tuned in the notebook)
color = "green" if prob >= threshold else "red"
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
# mock commit
