"""Streamlit App for Flight Fare Prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.model_loader import load_model, load_scaler, load_feature_names, load_model_metadata
from utils.preprocessor import (
    preprocess_input, get_season,
    SOURCE_AIRPORTS, DESTINATION_AIRPORTS, AIRLINES,
    STOPOVERS, AIRCRAFT_TYPES, CLASSES, BOOKING_SOURCES, SEASONALITIES,
)

# Page configuration
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        color: #1565C0;
        font-weight: bold;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Build display labels for airports: "City (CODE)"
SOURCE_LABELS = {code: f"{name.split(',')[-1].strip() if ',' in name else name.replace(' Airport', '')} ({code})"
                 for code, name in SOURCE_AIRPORTS.items()}
DEST_LABELS = {code: f"{name.split(',')[-1].strip() if ',' in name else name.replace(' Airport', '')} ({code})"
               for code, name in DESTINATION_AIRPORTS.items()}


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">✈️ Flight Fare Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict flight ticket prices for Bangladesh routes using Machine Learning</p>', unsafe_allow_html=True)

    # Load model and artifacts
    try:
        model = load_model()
        scaler = load_scaler()
        feature_names = load_feature_names()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please ensure the model has been trained and artifacts are in the 'models/' folder.")
        return

    # Sidebar - Model Information (optional metadata)
    with st.sidebar:
        st.header("📊 Model Information")
        try:
            metadata = load_model_metadata()
            st.write(f"**Model Type:** {metadata['model_name']}")
            st.write("**Performance Metrics:**")
            st.metric("R² Score", f"{metadata['metrics']['R2']:.4f}")
            st.metric("RMSE", f"{metadata['metrics']['RMSE']:.2f} BDT")
            st.metric("MAE", f"{metadata['metrics']['MAE']:.2f} BDT")
            st.metric("MAPE", f"{metadata['metrics']['MAPE']:.2f}%")
        except FileNotFoundError:
            st.write("*Model metadata not available.*")

        st.markdown("---")
        st.info("💡 This model was trained on historical Bangladesh flight data to predict total fare.")

    # --- Input Form ---
    st.header("🎫 Enter Flight Details")

    # Row 1: Route & Airline
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox("Airline", options=AIRLINES)

    with col2:
        source_label = st.selectbox("Source Airport", options=list(SOURCE_LABELS.values()))
        # Reverse-lookup code from label
        source_code = [k for k, v in SOURCE_LABELS.items() if v == source_label][0]

    with col3:
        dest_label = st.selectbox("Destination Airport", options=list(DEST_LABELS.values()))
        dest_code = [k for k, v in DEST_LABELS.items() if v == dest_label][0]

    # Row 2: Date, Time, Duration
    col4, col5, col6 = st.columns(3)

    with col4:
        travel_date = st.date_input(
            "Departure Date",
            value=datetime.now(),
            min_value=datetime.now()
        )

    with col5:
        travel_time = st.time_input(
            "Departure Time",
            value=datetime.now().time()
        )

    with col6:
        duration = st.number_input(
            "Duration (hrs)",
            min_value=0.5,
            max_value=24.0,
            value=3.0,
            step=0.5,
        )

    # Row 3: Booking details
    col7, col8, col9 = st.columns(3)

    with col7:
        days_before = st.number_input(
            "Days Before Departure",
            min_value=1,
            max_value=90,
            value=30,
            step=1,
        )

    with col8:
        stopovers = st.selectbox("Stopovers", options=STOPOVERS)

    with col9:
        aircraft_type = st.selectbox("Aircraft Type", options=AIRCRAFT_TYPES)

    # Row 4: Class, Booking Source, Seasonality
    col10, col11, col12 = st.columns(3)

    with col10:
        ticket_class = st.selectbox("Class", options=CLASSES)

    with col11:
        booking_source = st.selectbox("Booking Source", options=BOOKING_SOURCES)

    with col12:
        seasonality = st.selectbox("Seasonality", options=SEASONALITIES)

    # Validate source != destination
    if source_code == dest_code:
        st.warning("⚠️ Source and Destination cannot be the same!")
        return

    # Derive date features
    month = travel_date.month
    day = travel_date.day
    weekday = travel_date.weekday()
    hour = travel_time.hour

    # Prepare input data
    input_data = {
        'Airline': airline,
        'Source': source_code,
        'Destination': dest_code,
        'Month': month,
        'Day': day,
        'Weekday': weekday,
        'Hour': hour,
        'Duration (hrs)': duration,
        'Days Before Departure': days_before,
        'Stopovers': stopovers,
        'Aircraft Type': aircraft_type,
        'Class': ticket_class,
        'Booking Source': booking_source,
        'Seasonality': seasonality,
    }

    # Show derived info
    season = get_season(month)
    route = f"{source_code} -> {dest_code}"
    st.caption(f"Season: **{season}** | Route: **{route}**")

    # Prediction button
    if st.button("🔮 Predict Fare", use_container_width=True):
        try:
            # Preprocess input (includes scaling)
            input_df = preprocess_input(input_data, feature_names, scaler)

            # Make prediction
            prediction = model.predict(input_df)[0]

            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Predicted Fare")
            st.markdown(f'<p class="prediction-value">{prediction:,.2f} BDT</p>', unsafe_allow_html=True)
            st.markdown(f"*Approximately ${prediction/110:,.2f} USD*")
            st.markdown('</div>', unsafe_allow_html=True)

            # Display input summary
            st.subheader("📋 Flight Summary")
            col_a, col_b = st.columns(2)

            with col_a:
                st.write(f"**Route:** {SOURCE_AIRPORTS[source_code].split(',')[0]} → {DESTINATION_AIRPORTS[dest_code].split(',')[0]}")
                st.write(f"**Airline:** {airline}")
                st.write(f"**Date:** {travel_date.strftime('%B %d, %Y')}")
                st.write(f"**Time:** {travel_time.strftime('%H:%M')}")
                st.write(f"**Duration:** {duration} hrs")

            with col_b:
                st.write(f"**Class:** {ticket_class}")
                st.write(f"**Stopovers:** {stopovers}")
                st.write(f"**Aircraft:** {aircraft_type}")
                st.write(f"**Booking Source:** {booking_source}")
                st.write(f"**Booked:** {days_before} days before departure")

            # Confidence interval estimation (rough estimate)
            st.info(f"💡 **Estimated Price Range:** {prediction * 0.9:,.2f} - {prediction * 1.1:,.2f} BDT")

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.error("Please check your input and try again.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit & Scikit-learn |
        <a href='https://github.com/Airlectric/Amalitech-DEM09-Machine-Learning-Deployment-of-Flight-Price-Analysis-.git' target='_blank'>View on GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
