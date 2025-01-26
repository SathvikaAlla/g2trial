import cv2
import numpy as np
import ultralytics
import matplotlib.pyplot as plt
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import time

from sprint_criteria_checks import evaluate_sprint_running, evaluate_sprint_start, get_player_coords
from longjump_criteria_checks import evaluate_long_jump
from highjump_criteria_checks import evaluate_high_jump

import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os
import shutil
import pandas as pd

# Streamlit App
st.set_page_config("Athelete Assist", layout="centered")

st.title("Athlete Assist")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@100..900&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Geist', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Geist', sans-serif;
        }
        html, div, span, p, a, li, ul, button {
        font-family: 'Geist', sans-serif !important;
        }

    </style>
""", unsafe_allow_html=True)

# # Sport Selection
sport = st.sidebar.selectbox(
    "**Which sport would you like to analyze?**",
    ("Sprint Starting Technique", "Sprint Running Technique", "Long Jump", "High Jump"),
)

st.write(f"**You selected:** {sport}")
uploaded_file = st.sidebar.file_uploader("**Choose a video...**", type=["mp4", "avi", "mov"], on_change=st.session_state.clear)

if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

# Play video function with play/pause functionality and frame tracking
def play_video(results, image_placeholder):
    while st.session_state.current_frame < len(results) and st.session_state.is_playing:
        result = results[st.session_state.current_frame]
        annotated_frame = result.plot()
        image_placeholder.image(annotated_frame, channels="BGR", caption=f"Frame {st.session_state.current_frame + 1}")
        time.sleep(0.02)  # Adjust for smoother playback if necessary
        st.session_state.current_frame += 1

# Load and display the uploaded video
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.video(temp_video_path)  # Display the uploaded video

    if "results" not in st.session_state:
        st.write("Processing video...")
        model = YOLO('src/yolo11m-pose.pt')
        st.session_state.results = model.track(source=temp_video_path)
    else:
        st.write("Using cached results...")

    results = st.session_state.results

    image_placeholder = st.empcaty()

    # Play and Pause buttons
    option_map = {
        0: ":material/play_circle:",
        1: ":material/pause_circle:"
    }
    selection = st.pills(
        "Controls",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        label_visibility='hidden'
    )
    # Handle play/pause button clicks
    if selection == 0:
        st.session_state.is_playing = True
        play_video(results, image_placeholder=image_placeholder)
    if selection == 1:
        st.session_state.is_playing = False

    # Reset frame position if video ends
    if st.session_state.current_frame >= len(results):
        st.session_state.current_frame = 0

    if sport == "Sprint Starting Technique":
        # st.write("Evaluating sprint starting technique...")
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)
        player_coords = get_player_coords(player, results)  
        scoring, eval_frames = evaluate_sprint_start(player_coords=player_coords)
        scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
        st.data_editor(scoring_df, column_config={
                                "Criteria": st.column_config.TextColumn(width='large'),
                                "Score": st.column_config.NumberColumn(
                                    format="%d ⭐")})
    
    elif sport == "Sprint Running Technique":
        # st.write("Evaluating sprint running technique...")
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)
        player_coords = get_player_coords(player, results)  
        scoring, eval_frames = evaluate_sprint_running(player_coords=player_coords)
        scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
        st.data_editor(scoring_df, column_config={
                                "Criteria": st.column_config.TextColumn(width='large'),
                                "Score": st.column_config.NumberColumn(
                                    format="%d ⭐")})
        
        
    elif sport == "Long Jump":
        # st.write("Evaluating long jump technique...")
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)
        player_coords = get_player_coords(player, results, True, True) # include boxes
        scoring, eval_frames = evaluate_long_jump(player_coords=player_coords)
        scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
        st.data_editor(scoring_df, column_config={
                                "Criteria": st.column_config.TextColumn(width='large'),
                                "Score": st.column_config.NumberColumn(
                                    format="%d ⭐")})
    
    elif sport == "High Jump":
        # st.write("Evaluating high jump technique...")
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)
        player_coords = get_player_coords(player, results, True, True) # include boxes
        scoring, eval_frames = evaluate_high_jump(player_coords=player_coords)
        scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
        st.data_editor(scoring_df, column_config={
                                "Criteria": st.column_config.TextColumn(width='large'),
                                "Score": st.column_config.NumberColumn(
                                    format="%d ⭐")})
        
    
    
    os.remove(temp_video_path)