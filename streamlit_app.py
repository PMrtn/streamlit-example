#pip install --upgrade tensorflow
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
#import joblib

st.write("""
         Trueskill_sigma regression
         Pintér Martin
         """)

#RNN_model = joblib.load('trueskill_sigma.sav')

gpm = st.slider('GPM', min_value=0, max_value=1200, value=650, step=1)
xpm = st.slider('XPM', min_value=0, max_value=1200, value=700, step=1)
kills = st.slider('Kills', min_value=0, max_value=100, value=10, step=1)
deaths = st.slider('Deaths', min_value=0, max_value=100, value=4, step=1)
assists = st.slider('Assists', min_value=0, max_value=100, value=16, step=1)
last_hits = st.slider('Last hits', min_value=0, max_value=1500, value=642, step=1)
hero_damage = st.slider('Hero damage', min_value=0, max_value=150000, value=68469, step=1)
tower_damage = st.slider('Tower damage', min_value=0, max_value=30000, value=12467, step=1)
xp_hero_kills = st.slider('XP from hero kills', min_value=0, max_value=20000, value=0, step=1)
xp_creeps = st.slider('XP from creeps', min_value=0, max_value=999, value=0, step=1)
other_xp = st.slider('Other XP', min_value=0, max_value=999, value=0, step=1)
gold_killing_heroes = st.slider('Gold for killing heroes', min_value=0, max_value=999, value=0, step=1)
gold_killing_creeps = st.slider('Gold for killing creeps', min_value=0, max_value=999, value=0, step=1)
total_matches = st.slider('Total matches', min_value=0, max_value=999, value=0, step=1)
total_wins = st.slider('Total wins', min_value=0, max_value=999, value=0, step=1)

prediction = 0

PredictionButton = st.button('Predict trueskill_sigma!')
if(PredictionButton):
    prediction = RNN_model.predict([[gpm,xpm,kills,deaths,assists,last_hits,hero_damage,tower_damage,xp_hero_kills,xp_creeps,other_xp,gold_killing_heroes,gold_killing_creeps,total_matches,total_wins]])
    st.subheader(f'The predicted trueskill_sigma is: {[prediction[0]]}')
         
         
