import streamlit as st
import torch
import numpy as np
import pandas as pd
import requests
import time
from distraservicemodel import load_model, predict_rainfall
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

st.set_page_config(page_title="Resilix", page_icon="🚨",layout="wide")
# Layout: Two columns
col1, col2 = st.columns([1, 3])  

with col1:
        # Display map with user's location or default to Cameroon
    st.subheader("Resilix" icon="🚨")
    # Input: Upload file
    uploaded_file = st.file_uploader("Upload a JSON or Excel file with rainfall data", type=["json", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.type == "application/json":
            data = pd.read_json(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(uploaded_file)
        
        st.write("Uploaded Data:")
        st.write(data)
        
        # Extract rainfall data
        if 'rainfall' in data.columns and 'date' in data.columns:
            rainfall_data = data['rainfall'].tolist()
            
            if len(rainfall_data) == 13:
                prediction, flood_state = predict_rainfall(model, rainfall_data)
                st.write(f"Prediction: {prediction}, Flood State: {flood_state}")
            else:
                st.error("The file must contain exactly 13 intervals of rainfall data.")
        else:
            st.error("The file must contain 'date' and 'rainfall' columns.")

    # Enable live updates
    if st.checkbox("Enable live updates"):
        st.write("Live updates enabled. Press the button below to simulate new data.")
        interval = st.slider("Update interval (seconds):", min_value=1, max_value=10, value=5)
        while st.checkbox("Enable live updates"):  # Check if checkbox is still ticked
            time.sleep(interval)
            simulated_rainfall = np.random.uniform(0, 100, 13).tolist()  # Simulate 13 intervals of data
            prediction, flood_state = predict_rainfall(model, simulated_rainfall)
            st.write(f"Simulated rainfall amount: {simulated_rainfall}")
            st.write(f"Predicted outcome: {prediction}, Flood State: {flood_state}")
            st.experimental_rerun()

with col2:
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

    m = folium.Map(location=[location['lat'][0], location['lon'][0]], zoom_start=13)
    folium.Marker([location['lat'][0], location['lon'][0]], tooltip="User Location").add_to(m)

    # Function to add alerts to the map
    def add_alert_to_map(alert):
        folium.Marker(
            location=[alert['lat'], alert['lon']],
            popup=folium.Popup(f"Alert: {alert['message']}<br>Details: {alert['details']}", max_width=300),
            icon=folium.Icon(color='red')
        ).add_to(m)

    # Function to generate dummy alerts within Buea (replace with real-time data source)
    def generate_dummy_alerts():
        dummy_alerts = [
            {"lat": 4.1540, "lon": 9.2414, "message": "High Rainfall Warning", "details": "Expected rainfall: 100mm"},
            {"lat": 4.1550, "lon": 9.2315, "message": "Flood Alert", "details": "Heavy rain causing potential floods"},
            {"lat": 4.1580, "lon": 9.2516, "message": "Landslide Risk", "details": "Rain-induced landslides possible"},
            {"lat": 4.1470, "lon": 9.2420, "message": "Severe Thunderstorm Warning", "details": "Thunderstorm expected"},
            {"lat": 4.1490, "lon": 9.2600, "message": "Tornado Warning", "details": "Possible tornado formation"}
        ]
        return dummy_alerts

    # Add dummy alerts to the map
    dummy_alerts = generate_dummy_alerts()
    for alert in dummy_alerts:
        add_alert_to_map(alert)
    
    st_folium(m, width=700, height=500)
