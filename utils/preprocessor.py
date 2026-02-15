"""Input preprocessing utilities for inference."""

import pandas as pd
import numpy as np

# Airport code → airport name mappings
SOURCE_AIRPORTS = {
    'BZL': 'Barisal Airport',
    'CGP': 'Shah Amanat International Airport, Chittagong',
    'CXB': "Cox's Bazar Airport",
    'DAC': 'Hazrat Shahjalal International Airport, Dhaka',
    'JSR': 'Jessore Airport',
    'RJH': 'Shah Makhdum Airport, Rajshahi',
    'SPD': 'Saidpur Airport',
    'ZYL': 'Osmani International Airport, Sylhet',
}

DESTINATION_AIRPORTS = {
    'BKK': 'Suvarnabhumi Airport, Bangkok',
    'BZL': 'Barisal Airport',
    'CCU': 'Netaji Subhas Chandra Bose International Airport, Kolkata',
    'CGP': 'Shah Amanat International Airport, Chittagong',
    'CXB': "Cox's Bazar Airport",
    'DAC': 'Hazrat Shahjalal International Airport, Dhaka',
    'DEL': 'Indira Gandhi International Airport, Delhi',
    'DOH': 'Hamad International Airport, Doha',
    'DXB': 'Dubai International Airport',
    'IST': 'Istanbul Airport',
    'JED': 'King Abdulaziz International Airport, Jeddah',
    'JFK': 'John F. Kennedy International Airport, New York',
    'JSR': 'Jessore Airport',
    'KUL': 'Kuala Lumpur International Airport',
    'LHR': 'London Heathrow Airport',
    'RJH': 'Shah Makhdum Airport, Rajshahi',
    'SIN': 'Singapore Changi Airport',
    'SPD': 'Saidpur Airport',
    'YYZ': 'Toronto Pearson International Airport',
    'ZYL': 'Osmani International Airport, Sylhet',
}

AIRLINES = [
    'Air Arabia', 'Air Astra', 'Air India', 'AirAsia',
    'Biman Bangladesh Airlines', 'British Airways', 'Cathay Pacific',
    'Emirates', 'Etihad Airways', 'FlyDubai', 'Gulf Air', 'IndiGo',
    'Kuwait Airways', 'Lufthansa', 'Malaysian Airlines', 'NovoAir',
    'Qatar Airways', 'Saudia', 'Singapore Airlines', 'SriLankan Airlines',
    'Thai Airways', 'Turkish Airlines', 'US-Bangla Airlines', 'Vistara',
]

STOPOVERS = ['1 Stop', '2 Stops', 'Direct']
AIRCRAFT_TYPES = ['Airbus A320', 'Airbus A350', 'Boeing 737', 'Boeing 777', 'Boeing 787']
CLASSES = ['Business', 'Economy', 'First Class']
BOOKING_SOURCES = ['Direct Booking', 'Online Website', 'Travel Agency']
SEASONALITIES = ['Eid', 'Hajj', 'Regular', 'Winter Holidays']


def get_season(month: int) -> str:
    """Derive season from month (matches training pipeline)."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def preprocess_input(input_data: dict, feature_names: list, scaler=None) -> pd.DataFrame:
    """Preprocess user input to match the 248 training features.

    Args:
        input_data: Dictionary with raw user input fields.
        feature_names: List of 248 feature names from training.
        scaler: Fitted scaler from training (RobustScaler). If provided,
                numerical features are scaled to match training.

    Returns:
        pd.DataFrame: Single-row DataFrame ready for model.predict().
    """
    # Initialize all features to 0
    final_df = pd.DataFrame(0, index=[0], columns=feature_names, dtype=float)

    # --- Numerical features ---
    final_df['Duration (hrs)'] = input_data['Duration (hrs)']
    final_df['Days Before Departure'] = input_data['Days Before Departure']
    final_df['Month'] = input_data['Month']
    final_df['Day'] = input_data['Day']
    final_df['Weekday'] = input_data['Weekday']
    final_df['Hour'] = input_data['Hour']

    # --- One-hot encoded categorical features ---
    # Helper: set a one-hot column to 1 if it exists in feature_names
    def set_onehot(prefix, value):
        col = f"{prefix}_{value}"
        if col in final_df.columns:
            final_df[col] = 1

    # Airline
    set_onehot('Airline', input_data['Airline'])

    # Source airport code and name
    source_code = input_data['Source']
    set_onehot('Source', source_code)
    if source_code in SOURCE_AIRPORTS:
        set_onehot('Source Name', SOURCE_AIRPORTS[source_code])

    # Destination airport code and name
    dest_code = input_data['Destination']
    set_onehot('Destination', dest_code)
    if dest_code in DESTINATION_AIRPORTS:
        set_onehot('Destination Name', DESTINATION_AIRPORTS[dest_code])

    # Route Combined
    route = f"{source_code} -> {dest_code}"
    set_onehot('Route_Combined', route)

    # Stopovers
    set_onehot('Stopovers', input_data['Stopovers'])

    # Aircraft Type
    set_onehot('Aircraft Type', input_data['Aircraft Type'])

    # Class
    set_onehot('Class', input_data['Class'])

    # Booking Source
    set_onehot('Booking Source', input_data['Booking Source'])

    # Seasonality
    set_onehot('Seasonality', input_data['Seasonality'])

    # Season (derived from month)
    season = get_season(input_data['Month'])
    set_onehot('Season', season)

    # --- Apply scaler (only to the numerical features it was fit on) ---
    if scaler is not None:
        NUMERICAL_COLS = ['Duration (hrs)', 'Days Before Departure', 'Month', 'Day', 'Weekday', 'Hour']
        final_df[NUMERICAL_COLS] = scaler.transform(final_df[NUMERICAL_COLS])

    return final_df


def validate_input(input_data: dict) -> tuple:
    """Validate user input.

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    required_fields = [
        'Airline', 'Source', 'Destination', 'Month', 'Day', 'Weekday',
        'Hour', 'Duration (hrs)', 'Days Before Departure', 'Stopovers',
        'Aircraft Type', 'Class', 'Booking Source', 'Seasonality',
    ]

    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            return False, f"Missing required field: {field}"

    if not (1 <= input_data['Month'] <= 12):
        return False, "Month must be between 1 and 12"

    if not (1 <= input_data['Day'] <= 31):
        return False, "Day must be between 1 and 31"

    if not (0 <= input_data['Weekday'] <= 6):
        return False, "Weekday must be between 0 and 6"

    if not (0 <= input_data['Hour'] <= 23):
        return False, "Hour must be between 0 and 23"

    if input_data['Duration (hrs)'] <= 0:
        return False, "Duration must be greater than 0"

    if input_data['Days Before Departure'] < 1:
        return False, "Days Before Departure must be at least 1"

    if input_data['Source'] == input_data['Destination']:
        return False, "Source and Destination cannot be the same"

    return True, ""
