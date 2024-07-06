import streamlit as st
import numpy as np
import pandas as pd
import time
from distraservicemodel import load_model, predict_rainfall
import folium
from streamlit_folium import st_folium

# Static coordinates for Fako Division
FAKO_LAT = 4.0739
FAKO_LON = 9.7038

model = load_model()

st.set_page_config(page_title="Resilix", page_icon="🚨", layout="wide")

# Sidebar for prediction and file upload
with st.sidebar:
    st.header("Resilix 🚨")
    
    # Input: Upload file
    uploaded_file = st.file_uploader("Upload a JSON or Excel file with rainfall data", type=["json", "xlsx"])
    
    if uploaded_file:
        try:
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
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Enable live updates
    if st.checkbox("Enable live updates"):
        st.write("Live updates enabled. Press the button below to simulate new data.")
        interval = st.slider("Update interval (seconds):", min_value=1, max_value=10, value=5)
        if 'live_updates' not in st.session_state:
            st.session_state.live_updates = True
        while st.session_state.live_updates:  # Check if live updates checkbox is still ticked
            time.sleep(interval)
            simulated_rainfall = np.random.uniform(0, 100, 13).tolist()  # Simulate 13 intervals of data
            prediction, flood_state = predict_rainfall(model, simulated_rainfall)
            st.write(f"Simulated rainfall amount: {simulated_rainfall}")
            st.write(f"Predicted outcome: {prediction}, Flood State: {flood_state}")
            st.experimental_rerun()

# Layout: Two columns
col1, col2 = st.columns([4, 1])  


with col1:
    st.subheader("Location Map")
    
    # Create map with static location
    location = pd.DataFrame({
        'lat': [FAKO_LAT],
        'lon': [FAKO_LON]
    })
    
    m = folium.Map(location=[location['lat'][0], location['lon'][0]], zoom_start=13)
    folium.Marker([location['lat'][0], location['lon'][0]], tooltip="Fako Division").add_to(m)
    
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
    
    st_folium(m, width=800, height=500)
    
with col2:
    st.subheader("Alerts")

    # Placeholder for alerts
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []

    for i, alert in enumerate(st.session_state.alerts):
        st.write(f"Alert {i+1}: {alert['message']}")
        if st.button(f"Raise Emergency for Alert {i+1}", key=f"emergency_{i}"):
            st.write(f"Emergency raised for Alert {i+1}")
