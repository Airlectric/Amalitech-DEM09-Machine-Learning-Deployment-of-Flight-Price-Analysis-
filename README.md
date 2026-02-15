# Flight Fare Prediction - Deployment App

A Streamlit web application for predicting Bangladesh flight fares using a trained ML model. The project is fully self-contained and ready for deployment.

## Project Structure

```
official-deployment/
├── app/
│   ├── __init__.py
│   └── main.py              # Streamlit application
├── utils/
│   ├── __init__.py
│   ├── model_loader.py      # Model and artifact loading
│   └── preprocessor.py      # Input preprocessing and validation
├── models/                   # Trained model artifacts
│   ├── best_model.pkl        # Trained ML model
│   ├── scaler.pkl            # Fitted scaler
│   └── feature_names.txt     # Feature names from training
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

## Model Artifacts

All model artifacts live in the `models/` directory. These are required:

| File | Description |
|------|-------------|
| `best_model.pkl` | Trained scikit-learn model |
| `scaler.pkl` | Fitted scaler for feature scaling |
| `feature_names.txt` | Feature names used during training (one per line) |

Optional:

| File | Description |
|------|-------------|
| `best_model_tuned.pkl` | Hyperparameter-tuned model (loaded preferentially if present) |
| `best_model_metadata.pkl` | Model name and performance metrics (displayed in sidebar) |

The model was trained on Kaggle using the following library versions:

- numpy==2.0.2
- scikit-learn==1.6.1
- pandas==2.2.2
- joblib==1.5.3

## Features

### Prediction Interface
- Select airline, source/destination cities, date, time, and season
- One-click fare prediction with results in BDT and approximate USD
- Estimated price range displayed alongside the prediction
- Input validation (e.g., source and destination cannot match)

### Supported Input Options
- **Airlines:** Biman Bangladesh Airlines, US-Bangla Airlines, Novoair, Regent Airways, Air Astra
- **Cities:** Dhaka, Chittagong, Sylhet, Cox's Bazar, Jessore
- **Seasons:** Spring, Summer, Autumn, Winter

### Model Info Sidebar
If `best_model_metadata.pkl` is provided, the sidebar displays the model type and metrics (R2, RMSE, MAE, MAPE). Otherwise it gracefully shows a fallback message.

## Usage Example

```python
from utils.model_loader import load_model, load_scaler, load_feature_names
from utils.preprocessor import preprocess_input

model = load_model()
scaler = load_scaler()
feature_names = load_feature_names()

input_data = {
    'Airline': 'Biman Bangladesh Airlines',
    'Source': 'Dhaka',
    'Destination': 'Chittagong',
    'Month': 6,
    'Day': 15,
    'Weekday': 2,
    'Hour': 14,
    'Season': 'Summer'
}

input_df = preprocess_input(input_data, feature_names)
prediction = model.predict(input_df)[0]
print(f"Predicted fare: {prediction:.2f} BDT")
```

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repository and set `app/main.py` as the entry point

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t flight-fare-predictor .
docker run -p 8501:8501 flight-fare-predictor
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'numpy._core'` | numpy version mismatch | Install exact versions from `requirements.txt` |
| `FileNotFoundError: Model not found` | Missing `.pkl` files in `models/` | Export artifacts from training and place in `models/` |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` |

---

**Built with Streamlit & Scikit-learn** | [View on GitHub](https://github.com/Airlectric/Amalitech-DEM09-Machine-Learning-Deployment-of-Flight-Price-Analysis-.git)
