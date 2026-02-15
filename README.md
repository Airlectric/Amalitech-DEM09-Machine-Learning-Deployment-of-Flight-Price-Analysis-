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
The app collects 12 user inputs and automatically derives the remaining features (Season, Route, airport names) to build all 248 model features internally.

### Input Fields

| # | Field | Type | Options |
|---|-------|------|---------|
| 1 | Airline | Dropdown | 24 airlines (Air Arabia, Biman Bangladesh, Emirates, Qatar Airways, etc.) |
| 2 | Source Airport | Dropdown | 8 Bangladesh airports (DAC, CGP, CXB, JSR, ZYL, RJH, SPD, BZL) |
| 3 | Destination Airport | Dropdown | 20 airports (domestic + international: DXB, LHR, JFK, SIN, etc.) |
| 4 | Departure Date | Date picker | Derives Month, Day, Weekday, and Season automatically |
| 5 | Departure Time | Time picker | Derives Hour |
| 6 | Duration (hrs) | Number | Flight duration (0.5 - 24 hrs) |
| 7 | Days Before Departure | Number | Booking lead time (1 - 90 days) |
| 8 | Stopovers | Dropdown | Direct, 1 Stop, 2 Stops |
| 9 | Aircraft Type | Dropdown | Airbus A320, Airbus A350, Boeing 737, Boeing 777, Boeing 787 |
| 10 | Class | Dropdown | Economy, Business, First Class |
| 11 | Booking Source | Dropdown | Direct Booking, Online Website, Travel Agency |
| 12 | Seasonality | Dropdown | Regular, Hajj, Eid, Winter Holidays |

### From 17 Dataset Columns to 12 Frontend Fields

The raw dataset has 17 columns, but not all are user inputs. The app reduces them to 12 fields:

| Raw Dataset Column | Frontend | Reason |
|---|---|---|
| `Airline` | User selects | Direct input |
| `Source` | User selects airport | Direct input |
| `Source Name` | Auto-derived | Mapped from Source airport code |
| `Destination` | User selects airport | Direct input |
| `Destination Name` | Auto-derived | Mapped from Destination airport code |
| `Departure Date & Time` | Date + Time pickers | Split into 2 fields |
| `Arrival Date & Time` | Not needed | Duration replaces it |
| `Duration (hrs)` | User enters | Direct input |
| `Stopovers` | User selects | Direct input |
| `Aircraft Type` | User selects | Direct input |
| `Class` | User selects | Direct input |
| `Booking Source` | User selects | Direct input |
| `Base Fare (BDT)` | Excluded | Dropped during training (data leakage — sums to target) |
| `Tax & Surcharge (BDT)` | Excluded | Dropped during training (data leakage — sums to target) |
| `Total Fare (BDT)` | Excluded | This is the target variable we predict |
| `Seasonality` | User selects | Direct input |
| `Days Before Departure` | User enters | Direct input |

The preprocessor then auto-generates additional derived features (`Route_Combined`, `Season`, `Month`, `Day`, `Weekday`, `Hour`) and one-hot encodes everything to produce all 248 model features.

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
    'Source': 'DAC',                   # Airport code
    'Destination': 'CGP',              # Airport code
    'Month': 6,
    'Day': 15,
    'Weekday': 2,
    'Hour': 14,
    'Duration (hrs)': 1.5,
    'Days Before Departure': 30,
    'Stopovers': 'Direct',
    'Aircraft Type': 'Boeing 737',
    'Class': 'Economy',
    'Booking Source': 'Online Website',
    'Seasonality': 'Regular',
}

input_df = preprocess_input(input_data, feature_names, scaler)
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
