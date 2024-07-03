import streamlit as st
import torch
import numpy as np
import pandas as pd
import requests
import time
import asyncio
import websockets
import json
from distraservice import load_model, predict_rainfall
import folium
from streamlit_folium import st_folium

# Function to get IP location
def get_ip_location():
    try:
        response = requests.get("http://ipinfo.io/json")
        data = response.json()
        loc = data['loc'].split(',')
        return float(loc[0]), float(loc[1])
    except Exception as e:
        st.error(f"Could not fetch location: {e}")
        return None, None

# Default coordinates for Cameroon (approximately central)
DEFAULT_LAT = 4.0587
DEFAULT_LON = 9.7085

model = load_model()

# Streamlit app
st.title("Rainfall Prediction Dashboard")

# Layout: Two columns
col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed

with col1:
    # Input: Amount of rainfall
    rainfall_amount = st.number_input("Enter the amount of rainfall (mm):", min_value=0.0, step=0.1)

    # Predict and display
    if st.button("Predict"):
        prediction = predict_rainfall(model, rainfall_amount)
        st.write(f"Predicted rainfall outcome: {prediction}")

    # Enable live updates
    if st.checkbox("Enable live updates"):
        st.write("Live updates enabled. Press the button below to simulate new data.")
        interval = st.slider("Update interval (seconds):", min_value=1, max_value=10, value=5)
        while True:
            time.sleep(interval)
            simulated_rainfall = np.random.uniform(0, 100)
            prediction = predict_rainfall(model, simulated_rainfall)
            st.write(f"Simulated rainfall amount: {simulated_rainfall} mm")
            st.write(f"Predicted outcome: {prediction}")
            st.experimental_rerun()

with col2:
    # Display map with user's location or default to Cameroon
    st.subheader("Location Map")
    lat, lon = get_ip_location()
    if lat and lon:
        location = pd.DataFrame({
            'lat': [lat],
            'lon': [lon]
        })
    else:
        # Default location for Cameroon
        location = pd.DataFrame({
            'lat': [DEFAULT_LAT],
            'lon': [DEFAULT_LON]
        })
    
    m = folium.Map(location=[location['lat'][0], location['lon'][0]], zoom_start=6)
    folium.Marker([location['lat'][0], location['lon'][0]], tooltip="User Location").add_to(m)
    
    # Function to add alerts to the map
    def add_alert_to_map(alert):
        folium.Marker(
            location=[alert['lat'], alert['lon']],
            popup=folium.Popup(f"Alert: {alert['message']}<br>Details: {alert['details']}", max_width=300),
            icon=folium.Icon(color='red')
        ).add_to(m)
    
    async def receive_alerts():
        uri = "ws://your-websocket-server-url"
        async with websockets.connect(uri) as websocket:
            while True:
                alert_data = await websocket.recv()
                alert = json.loads(alert_data)
                add_alert_to_map(alert)
                st_folium(m, width=700, height=500)

    asyncio.run(receive_alerts())
