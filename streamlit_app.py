!pip install --upgrade tensorflow

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib


RNN_model = joblib.load('trueskill_sigma.sav')

num_points = st.slider("GPM", 1, 10000, 1100)
num_turns = st.slider("XPM", 1, 300, 31)

if(st.session_state.bt and st.session_state.pred is False):
    with col1:
        col1_data = st.number_input('GPM', min_value=0, value=0, step=1)
    with col2:
        col2_data = st.number_input('XPM', min_value=0, value=0, step=1)
    with col3:
        col3_data = st.number_input('Kills', min_value=0, value=0, step=1)
    with col4:
        col4_data = st.number_input('Assists', min_value=0, value=0, step=1)
    with col5:
        col5_data = st.number_input('Last hits', min_value=0, value=0, step=1)
    with col6:
        col6_data = st.number_input('Hero damage', min_value=0, value=0, step=1)
    with col7:
        col7_data = st.number_input('Tower damage', min_value=0, value=0, step=1)
    with col8:
        col8_data = st.number_input('XP from hero kills', min_value=0, value=0, step=1)
    with col9:
        col9_data = st.number_input('XP from creeps', min_value=0, value=0, step=1)
    with col10:
        col10_data = st.number_input('Other XP', min_value=0, value=0, step=1)
    with col11:
        col11_data = st.number_input('Gold for killing heros', min_value=0, value=0, step=1)
    with col12:
        col12_data = st.number_input('Gold for killing creeps', min_value=0, value=0, step=1)
    with col13:
        col13_data = st.number_input('Total matches', min_value=0, value=0, step=1)
    with col14:
        col14_data = st.number_input('Total wins', min_value=0, value=0, step=1)
        
  
    PredictionButton = st.button('Predict trueskill_sigma!')
    if(PredictionButton):
        prediction = RNN_model.predict([[col1_data,col2_data,col3_data,col4_data,col5_data,col6_data,col7_data,col8_data,col9_data,col10_data,col11_data,col12_data,col13_data,col14_data]])
        saveResult(prediction)
        pred()
