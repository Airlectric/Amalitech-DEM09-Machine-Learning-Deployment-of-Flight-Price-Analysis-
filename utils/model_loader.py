"""Model and artifact loading utilities."""

import pickle
import os

def get_artifact_path(filename: str) -> str:
    """Get the full path to an artifact file."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    artifact_path = os.path.join(base_path, 'training_pipeline', 'artifacts', filename)
    return artifact_path

def load_model(use_tuned: bool = True):
    """Load the trained model.

    Args:
        use_tuned: If True, load the tuned model if available, else load best model

    Returns:
        Trained model object
    """
    if use_tuned:
        tuned_path = get_artifact_path('best_model_tuned.pkl')
        if os.path.exists(tuned_path):
            with open(tuned_path, 'rb') as f:
                return pickle.load(f)

    # Fallback to best model
    model_path = get_artifact_path('best_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model

def load_scaler():
    """Load the fitted scaler."""
    scaler_path = get_artifact_path('scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return scaler

def load_feature_names():
    """Load the feature names used during training."""
    feature_path = get_artifact_path('feature_names.pkl')
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature names not found at {feature_path}. Please train the model first.")

    with open(feature_path, 'rb') as f:
        feature_names = pickle.load(f)

    return feature_names

def load_model_metadata():
    """Load model metadata (name, metrics, etc.)."""
    metadata_path = get_artifact_path('best_model_metadata.pkl')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Please train the model first.")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return metadata
