import streamlit as st
import pandas as pd, numpy as np
import pickle
import matplotlib.pyplot as plt
from notebooks.court import draw_half_court

@st.cache_resource
def load_model_data():
    df = pd.read_csv('data/processed/cleaned_curry_2009to25.csv')
    model = pickle.load(open('models/rf_curry.pkl','rb'))
    return df, model

df, model = load_model_data()

st.title("Steph Curry Shot Predictor")

st.write("""
This project predicts the prob% if Chef Curry will make a shot based on:
- Shot distance (ft)
- Shot angle (°)
- Time remaining (s)
- 3-point shot?
- Clutch situation (<10s left)

Data was fetched from `nba_api` from 2009 to 2025 season
         
(thank you mr swar patel)

Model: Random Forest Classifier
""")

# User inputs
dist = st.slider("Shot distance", 0, 30, 15)
angle = st.slider("Shot angle (°)", -90, 90, 0)
time_rem = st.slider("Time remaining (s)", 0, 2880, 720)
is_3pt = st.checkbox("3-point shot?")
is_clutch = st.checkbox("Clutch (<10s)?")

# Predict
feature_names = ['shot_distance', 'shot_angle', 'time_remaining', 'is_3pt', 'is_clutch']
features_df = pd.DataFrame([[dist, angle, time_rem, int(is_3pt), int(is_clutch)]], columns=feature_names)
prob = model.predict_proba(features_df)[0,1]

# Display probability with color
color = "green" if prob >= 0.4 else "red"
st.markdown(
    f"<span style='font-size:2em;'>Probability of Make: <span style='color:{color};'><b>{prob:.2%}</b></span></span>",
    unsafe_allow_html=True
)

# Plot on court
fig, ax = plt.subplots(figsize=(6,6))
draw_half_court(ax)

# Convert distance to inches for plotting
dist_in = dist * 12
x = dist_in * np.sin(np.radians(angle))
y = dist_in * np.cos(np.radians(angle))
ax.scatter([x], [y], c='blue', s=150)
st.pyplot(fig)

# Just for info
st.write(f"Shot coordinates: ({x:.2f}, {y:.2f}) inches from hoop")
st.write(f"Shot distance: {dist_in:.2f} inches")