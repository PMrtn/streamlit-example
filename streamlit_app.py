#pip install --upgrade tensorflow
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
#import joblib


#RNN_model = joblib.load('trueskill_sigma.sav')

gpm = st.slider('GPM', min_value=0, max_value=999, value=0, step=1)
xpm = st.slider('XPM', min_value=0, max_value=999, value=0, step=1)
kills = st.slider('Kills', min_value=0, max_value=999, value=0, step=1)
assists = st.slider('Assists', min_value=0, max_value=999, value=0, step=1)
last_hits = st.slider('Last hits', min_value=0, max_value=999, value=0, step=1)
hero_damage = st.slider('Hero damage', min_value=0, max_value=999, value=0, step=1)
tower_damage = st.slider('Tower damage', min_value=0, max_value=999, value=0, step=1)
xp_hero_kills = st.slider('XP from hero kills', min_value=0, max_value=999, value=0, step=1)
xp_creeps = st.slider('XP from creeps', min_value=0, max_value=999, value=0, step=1)
other_xp = st.slider('Other XP', min_value=0, max_value=999, value=0, step=1)
gold_killing_heros = st.slider('Gold for killing heros', min_value=0, max_value=999, value=0, step=1)
gold_killing_creeps = st.slider('Gold for killing creeps', min_value=0, max_value=999, value=0, step=1)
total_matches = st.slider('Total matches', min_value=0, max_value=999, value=0, step=1)
total_wins = st.slider('Total wins', min_value=0, max_value=999, value=0, step=1)

        
PredictionButton = st.button('Predict trueskill_sigma!')
if(PredictionButton):
    prediction = RNN_model.predict([[col1_data,col2_data,col3_data,col4_data,col5_data,col6_data,col7_data,col8_data,col9_data,col10_data,col11_data,col12_data,col13_data,col14_data]])
    saveResult(prediction)
    pred()
