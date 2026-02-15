"""Input preprocessing utilities for inference."""

import pandas as pd
import numpy as np

def preprocess_input(input_data: dict, feature_names: list) -> pd.DataFrame:
    """Preprocess user input to match training data format.

    Args:
        input_data: Dictionary with user input (Airline, Source, Destination, etc.)
        feature_names: List of feature names from training

    Returns:
        pd.DataFrame: Preprocessed input ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])

    # Encode categorical features (one-hot encoding)
    # We need to match the exact feature names from training
    categorical_cols = ['Airline', 'Source', 'Destination', 'Season']

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Create a DataFrame with all training features initialized to 0
    final_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # Fill in the values from our encoded input
    for col in df_encoded.columns:
        if col in feature_names:
            final_df[col] = df_encoded[col].values[0]

    return final_df

def validate_input(input_data: dict) -> tuple:
    """Validate user input.

    Args:
        input_data: Dictionary with user input

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    # Check required fields
    required_fields = ['Airline', 'Source', 'Destination', 'Month', 'Day', 'Weekday', 'Hour', 'Season']

    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            return False, f"Missing required field: {field}"

    # Validate ranges
    if not (1 <= input_data['Month'] <= 12):
        return False, "Month must be between 1 and 12"

    if not (1 <= input_data['Day'] <= 31):
        return False, "Day must be between 1 and 31"

    if not (0 <= input_data['Weekday'] <= 6):
        return False, "Weekday must be between 0 and 6"

    if not (0 <= input_data['Hour'] <= 23):
        return False, "Hour must be between 0 and 23"

    # Check source != destination
    if input_data['Source'] == input_data['Destination']:
        return False, "Source and Destination cannot be the same"

    return True, ""
