"""Streamlit App for Flight Fare Prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.model_loader import load_model, load_scaler, load_feature_names, load_model_metadata
from utils.preprocessor import preprocess_input

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

    # Main content - Input Form
    st.header("🎫 Enter Flight Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox(
            "Airline",
            options=[
                'Biman Bangladesh Airlines', 'US-Bangla Airlines', 'Novoair',
                'Regent Airways', 'Air Astra'
            ]
        )

        source = st.selectbox(
            "Source City",
            options=['Dhaka', 'Chittagong', 'Sylhet', 'Cox\'s Bazar', 'Jessore']
        )

    with col2:
        destination = st.selectbox(
            "Destination City",
            options=['Dhaka', 'Chittagong', 'Sylhet', 'Cox\'s Bazar', 'Jessore']
        )

        travel_date = st.date_input(
            "Departure Date",
            value=datetime.now(),
            min_value=datetime.now()
        )

    with col3:
        travel_time = st.time_input(
            "Departure Time",
            value=datetime.now().time()
        )

        season = st.selectbox(
            "Season",
            options=['Spring', 'Summer', 'Autumn', 'Winter']
        )

    # Validate input
    if source == destination:
        st.warning("⚠️ Source and Destination cannot be the same!")
        return

    # Prepare input data
    input_data = {
        'Airline': airline,
        'Source': source,
        'Destination': destination,
        'Month': travel_date.month,
        'Day': travel_date.day,
        'Weekday': travel_date.weekday(),
        'Hour': travel_time.hour,
        'Season': season
    }

    # Prediction button
    if st.button("🔮 Predict Fare", use_container_width=True):
        try:
            # Preprocess input
            input_df = preprocess_input(input_data, feature_names)

            # Make prediction
            prediction = model.predict(input_df)[0]

            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Predicted Fare")
            st.markdown(f'<p class="prediction-value">{prediction:.2f} BDT</p>', unsafe_allow_html=True)
            st.markdown(f"*Approximately ${prediction/110:.2f} USD*")
            st.markdown('</div>', unsafe_allow_html=True)

            # Display input summary
            st.subheader("📋 Flight Summary")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Route:** {source} → {destination}")
                st.write(f"**Airline:** {airline}")
                st.write(f"**Date:** {travel_date.strftime('%B %d, %Y')}")

            with col2:
                st.write(f"**Time:** {travel_time.strftime('%H:%M')}")
                st.write(f"**Season:** {season}")
                st.write(f"**Day of Week:** {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][travel_date.weekday()]}")

            # Confidence interval estimation (rough estimate)
            st.info(f"💡 **Estimated Price Range:** {prediction * 0.9:.2f} - {prediction * 1.1:.2f} BDT")

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
