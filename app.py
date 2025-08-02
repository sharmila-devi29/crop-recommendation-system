import streamlit as st
import numpy as np
import pickle
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# Load the model and label mapping
with open("data.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)

# Set static background image
st.markdown("""
<style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
</style>
""", unsafe_allow_html=True)

# Load animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login page
def login():
    st.title("Crop Recommendation System - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful. Welcome.")
        else:
            st.error("Invalid credentials. Please try again.")

if not st.session_state.logged_in:
    login()
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Try the Model", "Evaluation", "Roadmap", "About"])
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()

# Pages
if page == "Home":
    st.title("Crop Recommendation System")
    st.write("Welcome to the Smart Crop Advisor powered by Machine Learning.")
   # st_lottie(load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_tll0j4bb.json"), height=250)

elif page == "Try the Model":
    st.header("Enter Soil and Weather Information")
    use_weather = st.checkbox("Auto-fill using city weather data")

    temp = humidity = rainfall = None
    if use_weather:
        city = st.text_input("Enter your city (e.g., Chennai)")
        if city:
            def get_weather(city_name):
                api_key = "4c8a01a9309e06dc8df3f5a34defcf4a"  # Replace with your valid API key
                try:
                    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        return data["main"]["temp"], data["main"]["humidity"], data.get("rain", {}).get("1h", 0)
                    else:
                        st.warning("Unable to fetch weather data. Please enter manually.")
                        return None, None, None
                except:
                    st.error("Error fetching weather data.")
                    return None, None, None
            temp, humidity, rainfall = get_weather(city)

    col1 = st.number_input("Nitrogen (N):")
    col2 = st.number_input("Phosphorus (P):")
    col3 = st.number_input("Potassium (K):")
    col4 = st.number_input("Temperature (°C):", value=temp if temp is not None else 0.0)
    col5 = st.number_input("Humidity (%):", value=humidity if humidity is not None else 0.0)
    col6 = st.number_input("pH:")
    col7 = st.number_input("Rainfall (mm):", value=rainfall if rainfall is not None else 0.0)

    if st.button("Recommend"):
        features = np.array([[col1, col2, col3, col4, col5, col6, col7]])
        pred = model.predict(features)[0]
        crop = label_mapping.get(pred, "Unknown")
        st.success(f"Recommended Crop: {crop}")

elif page == "Evaluation":
    st.header("Model Evaluation Metrics")
    with open("metrics.txt", "r") as f:
        st.code(f.read(), language="text")

    st.subheader("Feature Importance")
    fi_df = pd.read_csv("feature_importance.csv")
    st.bar_chart(fi_df.set_index("Feature"))

elif page == "Roadmap":
    st.header("Project Roadmap")
    st.markdown("""
    - Dataset Preprocessing  
    - Model Training using Random Forest  
    - Model Evaluation  
    - Streamlit Web App Development  
    - UI Enhancements and Navigation  
    - Deployment to Streamlit Cloud (Upcoming)
    """)

elif page == "About":
    st.header("About the Developer")
    st.write("Sharmila Devi, 3rd Year B.E. Engineering Student at Thiagarajar College.")
    st.write("This project showcases skills in Machine Learning, Data Preprocessing, and Interactive Web App Development using Streamlit.")
