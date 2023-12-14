#pip install --upgrade tensorflow
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


st.subheader("""Trueskill_sigma regression""")
st.write("Pintér Martin")
st.write()
st.write()


player_stats_filtered = pd.read_csv("player_stats_filtered.csv")
st.write(player_stats_filtered.head())


X = player_stats_filtered[['gold_per_min', 'xp_per_min', 'kills', 'deaths', 'assists', 'last_hits',
           'hero_damage', 'tower_damage', 'xp_hero', 'xp_creep', 'xp_roshan', 'xp_other',
           'gold_killing_heros', 'gold_killing_creeps', 'total_wins', 'total_matches']]
y = player_stats_filtered['trueskill_sigma']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

scaler = StandardScaler()
X = scaler.fit_transform(X)

FNN_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

FNN_model.compile(optimizer='adam', loss='mean_squared_error')
FNN_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)



gpm = st.slider('GPM', min_value=0, max_value=1200, value=650, step=1)
xpm = st.slider('XPM', min_value=0, max_value=1200, value=700, step=1)
kills = st.slider('Kills', min_value=0, max_value=100, value=10, step=1)
deaths = st.slider('Deaths', min_value=0, max_value=100, value=4, step=1)
assists = st.slider('Assists', min_value=0, max_value=100, value=16, step=1)
last_hits = st.slider('Last hits', min_value=0, max_value=1500, value=642, step=1)
hero_damage = st.slider('Hero damage', min_value=0, max_value=150000, value=68469, step=1)
tower_damage = st.slider('Tower damage', min_value=0, max_value=30000, value=12467, step=1)
xp_hero_kills = st.slider('XP from hero kills', min_value=0, max_value=20000, value=8261, step=1)
xp_creeps = st.slider('XP from creeps', min_value=0, max_value=20000, value=13749, step=1)
xp_rosh = st.slider('XP from Roshan', min_value=0, max_value=5000, value=1490, step=1)
other_xp = st.slider('Other XP', min_value=0, max_value=10000, value=2718, step=1)
gold_killing_heroes = st.slider('Gold for killing heroes', min_value=0, max_value=100000, value=8911, step=1)
gold_killing_creeps = st.slider('Gold for killing creeps', min_value=0, max_value=100000, value=23770, step=1)
total_matches = st.slider('Total matches', min_value=0, max_value=5000, value=4891, step=1)
total_wins = st.slider('Total wins', min_value=0, max_value=5000, value=2727, step=1)

predict_data = np.reshape([gpm,xpm,kills,deaths,assists,last_hits,hero_damage,tower_damage,xp_hero_kills,xp_creeps,xp_rosh,other_xp,gold_killing_heroes,gold_killing_creeps,total_matches,total_wins], (1, 16))

prediction = 0

PredictionButton = st.button('Predict trueskill_sigma!')
if(PredictionButton):
    prediction = FNN_model.predict(predict_data)
    st.subheader(f'The predicted trueskill_sigma is: {[prediction[0][0]]}')
