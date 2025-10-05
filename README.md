# Amazon-Time-Prediction
📦 Delivery Time Prediction Capstone Project
Overview
This repository contains a production-ready Streamlit web application for predicting the total delivery time (in minutes) for e-commerce and food orders. The project is built using a robust machine learning pipeline that transforms complex logistical, geographical, and temporal data into actionable business predictions.

The final model, a Tuned Gradient Boosting Regressor (GBR), provides highly accurate estimates, crucial for improving customer satisfaction and optimizing dispatch logistics.

Key Features of the Application
Real-time Prediction: Uses agent, location (store and drop-off coordinates), weather, and traffic data to generate delivery time predictions instantly.

Feature Engineering: Incorporates critical feature engineering logic directly into the app, including Haversine Distance calculation and Time-to-Pickup determination.

Cyclical Encoding: Correctly handles time-based features (like the hour of the day) using Sine and Cosine transformations to capture rush-hour effects.

Intuitive UI: Deployed via Streamlit with a clean, Amazon-inspired (Yellow/Black) theme for ease of use by logistics managers or agents.

Model Performance
The final model was optimized using Grid Search Cross-Validation, yielding the following performance metrics on the test set:

Metric

Value

Interpretation

Model

Tuned Gradient Boosting Regressor

Robust ensemble method for regression.

Mean Absolute Error (MAE)

17.30 minutes

On average, the prediction is off by less than 18 minutes. (High Business Value)

R 
2
  Score

0.814

The model explains over 81% of the variance in delivery time.

Top 3 Feature Importances
The most significant drivers of the predicted delivery time were identified as:

Travel_Distance_km: The primary physical constraint on delivery speed.

Time_to_Pickup_minutes: The efficiency (or delay) of the agent at the store.

Cyclical Order Hour: Captures peak traffic and demand windows (e.g., lunch and dinner rushes).

Repository Structure
This repository follows best practices for ML deployment, separating the application code from the model assets.

/Delivery-Time-Predictor/
├── app.py                      # Main Streamlit application script
├── requirements.txt            # List of all necessary Python dependencies
├── .streamlit/                 # Directory for Streamlit configuration
│   └── config.toml             # Custom theme file (Amazon Yellow/Black)
└── model/                      # Directory for serialized ML assets
    ├── best_delivery_time_predictor.joblib # The final GBR model
    └── feature_scaler.joblib               # The fitted StandardScaler for feature transformation

Setup and Running the Application
To run this application locally, follow these steps:

1. Clone the Repository
git clone [https://github.com/YourUsername/Delivery-Time-Prediction-Capstone.git](https://github.com/YourUsername/Delivery-Time-Prediction-Capstone.git)
cd Delivery-Time-Prediction-Capstone

2. Create and Activate Virtual Environment
It is highly recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
Install all required Python libraries using the provided requirements.txt file.

pip install -r requirements.txt

4. Run the Streamlit App
Launch the application directly from your terminal.

streamlit run app.py

The application will automatically open in your web browser, ready for real-time predictions.
