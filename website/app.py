import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SkyFare | Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "../model/flight_model.pkl")
)

model_columns = joblib.load(
    os.path.join(BASE_DIR, "../model/model_columns.pkl")
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #071B33, #123D5A, #176B78);
    color: white;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

.hero {
    text-align: center;
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(120deg, #102A43, #176B87);
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.2);
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    color: #D5EAF2;
    font-size: 17px;
}

.section {
    background: rgba(255,255,255,0.10);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.18);
    margin-bottom: 20px;
}

.stButton > button {
    background: linear-gradient(90deg, #FF8A5B, #FFB36B);
    color: #102A43;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
    font-size: 17px;
}

.stButton > button:hover {
    background: #FFD08A;
    color: #102A43;
}

label, .stMarkdown, p {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>✈️ SKYFARE</h1>
    <h3>Your Journey, Your Budget</h3>
    <p>Discover smarter flight prices with Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## 🌍 Plan Your Journey")
st.write("Enter your travel details to estimate your flight ticket price.")

# ---------------- INPUT FORM ----------------

with st.container():

    st.markdown('<div class="section">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            "✈️ Select Airline",
            ["AirAsia", "Air India", "GO_FIRST",
             "Indigo", "SpiceJet", "Vistara"]
        )

        source_city = st.selectbox(
            "📍 Source City",
            ["Delhi", "Mumbai", "Bangalore",
             "Kolkata", "Hyderabad", "Chennai"]
        )

        destination_city = st.selectbox(
            "🎯 Destination City",
            ["Mumbai", "Delhi", "Bangalore",
             "Kolkata", "Hyderabad", "Chennai"]
        )

        travel_class = st.selectbox(
            "💺 Travel Class",
            ["Economy", "Business"]
        )

        stops = st.selectbox(
            "🔄 Number of Stops",
            ["zero", "one", "two_or_more"]
        )

    with col2:
        departure_time = st.selectbox(
            "🕒 Departure Time",
            ["Early_Morning", "Morning", "Afternoon",
             "Evening", "Night", "Late_Night"]
        )

        arrival_time = st.selectbox(
            "🕒 Arrival Time",
            ["Early_Morning", "Morning", "Afternoon",
             "Evening", "Night", "Late_Night"]
        )

        duration = st.number_input(
            "⏱️ Flight Duration (hours)",
            min_value=0.5,
            max_value=50.0,
            value=2.5,
            step=0.5
        )

        days_left = st.number_input(
            "📅 Days Left Before Journey",
            min_value=1,
            max_value=50,
            value=15
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------

st.markdown("### 💰 Ready to discover your fare?")

if st.button("✈️ Predict Flight Price"):

    if source_city == destination_city:
        st.error("Source city and destination city cannot be the same.")

    else:
        input_data = pd.DataFrame([{
            "airline": airline,
            "source_city": source_city,
            "destination_city": destination_city,
            "class": travel_class,
            "stops": stops,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "duration": duration,
            "days_left": days_left
        }])

        input_encoded = pd.get_dummies(input_data)

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        prediction = model.predict(input_encoded)[0]

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(120deg,#D7FFF1,#A8E6CF);
                padding: 25px;
                border-radius: 18px;
                text-align: center;
                margin-top: 25px;
            ">
                <h3 style="color:#123D3A;">
                    🎉 Your Estimated Flight Price
                </h3>
                <h1 style="color:#087F5B;">
                    ₹{prediction:,.2f}
                </h1>
                <p style="color:#245B50;">
                    Have a wonderful journey!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("""
<hr>
<p style="text-align:center; color:#D5EAF2;">
    SKYFARE | Flight Price Prediction Project
    <br>
    Powered by Machine Learning
</p>
""", unsafe_allow_html=True)