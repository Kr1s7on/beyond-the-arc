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

# Features the location-first app model was trained on. The UI collects the
# first two; the rest are held at typical-game values. Shot type/zone are
# deliberately not features: the UI can't observe them, and deriving them from
# distance with hard-coded cutoffs made predictions jump at 4/16/24 ft and stay
# flat in between.
MODEL_FEATURES = [
    'log_dist', 'abs_angle', 'time_frac', 'is_clutch', 'is_overtime',
    'is_home', 'attempt_idx', 'attempt_frac', 'game_fg_sofar', 'game_3fg_sofar',
    'days_rest', 'roll5_3fg', 'period', 'opp',
]

DEFAULTS = {
    'is_overtime': 0, 'is_home': 0, 'attempt_idx': 9, 'attempt_frac': 0.47,
    'game_fg_sofar': 0.474, 'game_3fg_sofar': 0.422, 'days_rest': 2,
    'roll5_3fg': 0.409, 'period': 3, 'opp': 'PHX',
}

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / 'models' / 'curry_location_model.pkl'

    with open(model_path, 'rb') as f:
        header = f.read(64)
        f.seek(0)

        # Git LFS pointer files start with plain text, not pickle bytes.
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Model file is a Git LFS pointer, not a pickle binary.")

        bundle = pickle.load(f)

    return bundle['model'], bundle['threshold']

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

def build_features(dist, angle, is_clutch):
    # The clutch toggle maps to the final 5 seconds; otherwise assume a full
    # quarter remains so the continuous time_frac feature stays sensible.
    time_rem = 5.0 if is_clutch else 720.0
    row = {
        'log_dist': np.log1p(dist),
        'abs_angle': abs(angle),
        'time_frac': time_rem / 720.0,
        'is_clutch': int(is_clutch),
    }
    row.update(DEFAULTS)
    return pd.DataFrame([row])[MODEL_FEATURES]

model, threshold = load_model()
marker_images = load_marker_images()

st.title("Steph Curry Shot Predictor")

st.write("""
This project predicts the prob% if Curry will make a shot based on:
- Shot distance (ft)
- Shot angle (°)
- Clutch shot (5s left)

Data was fetched from `nba_api` from 2009 to 2025 season

Model: Logistic Regression (location + form/context features)
""")

# Side-by-side layout: controls on the left, chart on the right.
controls_col, chart_col = st.columns([1, 1.35], gap="large")

with controls_col:
    st.subheader("Shot Setup")
    angle = st.slider("Shot angle (°)", -90, 90, 0)
    st.caption(f"Selected angle: **{angle}°**")

    # Court is 50 ft wide, so a shot stays in bounds only while
    # dist * sin|angle| <= 25 ft. Drive the angle freely and cap the distance
    # to match: straight on -> 42 ft (half court), 36 deg -> 42 ft,
    # corner (90 deg) -> 25 ft.
    s = abs(np.sin(np.radians(angle)))
    max_dist = 42 if s < 1e-6 else min(42, int(25 / s))
    if "shot_distance" not in st.session_state:
        st.session_state.shot_distance = 15
    st.session_state.shot_distance = int(np.clip(st.session_state.shot_distance, 0, max_dist))
    dist = st.slider("Shot distance", 0, max_dist, value=None, key="shot_distance")
    st.caption(f"Selected distance: **{dist} ft** (0–{max_dist} ft)")

    is_clutch = st.checkbox("Clutch shot? (5s left)")

# 3-pt status comes from the court geometry, not a toggle: behind the 23.75 ft
# arc, or past the 22 ft corner line (x = dist*sin|angle| >= 22 ft).
is_3pt = (dist >= 23.75) or (dist * abs(np.sin(np.radians(angle))) >= 22)

with controls_col:
    st.checkbox("3-point shot? (auto)", value=is_3pt, disabled=True,
                help="Auto-enabled when the shot is behind the 3-pt line: past the 23.75 ft arc, or past the 22 ft corner line.")

# Predict
features_df = build_features(dist, angle, is_clutch)
prob = model.predict_proba(features_df)[0,1]

# Display probability with color (threshold tuned on out-of-fold predictions)
color = "green" if prob >= threshold else "red"
with controls_col:
    st.markdown(
        f"<span style='font-size:2em;'>Probability of Make: <span style='color:{color};'><b>{prob:.2%}</b></span></span>",
        unsafe_allow_html=True
    )

# Plot on court. The figure ratio matches the view (640 units wide, 505 tall)
# so the equal-aspect court fills the canvas without letterboxing.
fig, ax = plt.subplots(figsize=(8, 6.31))
draw_half_court(ax)

# Show the full half court, including the half-court line and center circle.
# The extra headroom above the line keeps the marker from being clipped when
# the shot is at the maximum distance.
ax.set_ylim(-55, 450)
ax.margins(x=0, y=0)

# Court units are tenths of a foot (10 units == 1 ft).
dist_u = dist * 10
x = dist_u * np.sin(np.radians(angle))
y = dist_u * np.cos(np.radians(angle))

# Orientation-aware Curry marker (front for central/top angle, side for wide angle).
marker_view = get_marker_view(angle)
marker_img = marker_images[marker_view]

# Flip side view when shot is on the right side so the body faces inward.
if marker_view == "side" and x > 0:
    marker_img = np.fliplr(marker_img)

img_h_px, img_w_px = marker_img.shape[:2]
img_aspect = img_w_px / img_h_px

# Keep a consistent visual marker size while respecting each image's aspect
# ratio. Slightly larger overall, with the side profile bumped a touch more so
# it reads as clearly as the front at the same distance.
scale = 1.12 * (1.10 if marker_view == "side" else 1.0)
marker_height = float(np.clip(68 - (dist_u * 0.072), 44, 68)) * scale
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
    st.write(f"Shot coordinates: ({x/10:.2f}, {y/10:.2f}) ft from hoop")
    st.write(f"Shot distance: {dist} ft")
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
