import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from math import radians, sin, cos, sqrt, atan2

# --- CONFIGURATION AND MODEL LOADING ---

# Define paths relative to the app.py file (assuming model files are in the root directory)
current_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.join(current_dir, 'best_delivery_time_predictor.joblib')
SCALER_PATH = os.path.join(current_dir, 'feature_scaler.joblib')

try:
    # Load the trained model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.sidebar.success("Model and Scaler loaded successfully.")

    # --- DEFINITIVE FIX: EXACT FEATURE LIST FROM feature_scaler.joblib (Attempt 3 - Removing Trailing Spaces) ---
    # The previous attempt with trailing spaces failed, suggesting they were artifacts of 
    # the joblib serialization log, not the true feature names. We are cleaning them up.
    EXPECTED_FEATURES = [
        # Numerical and Engineered Features (Order based on joblib metadata)
        'Agent_Age', 
        'Agent_Rating', 
        'Travel_Distance_km', 
        'Time_to_Pickup_minutes',
        'Is_Weekend',              
        'Order_Hour_sin', 
        'Order_Hour_cos', 
        
        # Ordinal Encoding
        'Traffic_Encoded', 
        
        # Weather OHE (Baseline: Cloudy/Rainy)
        'Weather_Fog', 
        'Weather_Sandstorms', 
        'Weather_Stormy', 
        'Weather_Sunny', 
        'Weather_Windy', 
        
        # Vehicle OHE (Cleaned names - assuming no trailing spaces)
        'Vehicle_motorcycle', 
        'Vehicle_scooter', 
        'Vehicle_van',
        
        # Area OHE (Cleaned names - assuming no trailing spaces)
        'Area_Other', 
        'Area_Semi-Urban', 
        'Area_Urban', 
        
        # Category OHE (Baseline: Food/Home)
        'Category_Books', 
        'Category_Clothing', 
        'Category_Cosmetics', 
        'Category_Electronics', 
        'Category_Grocery', 
        'Category_Toys',
        
        # Day of Week OHE (Keeping '.0' suffix as it appears in metadata)
        'Order_DayOfWeek_1.0', 
        'Order_DayOfWeek_2.0', 
        'Order_DayOfWeek_3.0',
        'Order_DayOfWeek_4.0', 
        'Order_DayOfWeek_5.0', 
        'Order_DayOfWeek_6.0'
    ]
    # --- END DEFINITIVE FIX ---

except FileNotFoundError:
    st.error(f"File not found. Please ensure your model files are in the root directory next to this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during model loading: {type(e).__name__}: {e}")
    st.stop()


# --- HELPER FUNCTIONS FOR FEATURE ENGINEERING ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points (in km)."""
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def create_temporal_features(order_dt, pickup_dt):
    """Calculates time-to-pickup, cyclical hour, and day-of-week features."""
    
    # Time to Pickup (in minutes)
    time_diff = pickup_dt - order_dt
    time_to_pickup_minutes = max(0, time_diff.total_seconds() / 60)

    # Temporal Features
    order_hour = order_dt.hour
    order_dayofweek = order_dt.dayofweek # Monday=0, Sunday=6
    is_weekend = 1 if order_dayofweek >= 5 else 0

    # Cyclical Encoding
    order_hour_sin = np.sin(2 * np.pi * order_hour / 24)
    order_hour_cos = np.cos(2 * np.pi * order_hour / 24)

    return time_to_pickup_minutes, order_hour_sin, order_hour_cos, is_weekend, order_dayofweek

def preprocess_inputs(user_inputs, expected_features):
    """
    Transforms raw user inputs into the final, standardized feature vector 
    expected by the trained ML model.
    """
    
    # 1. Initialize DataFrame with ALL expected features set to 0
    df = pd.DataFrame(0, index=[0], columns=expected_features)

    # 2. Add Numerical and Engineered Features (Note: Is_Weekend is included here, matching its position in EXPECTED_FEATURES)
    for key in ['Agent_Age', 'Agent_Rating', 'Travel_Distance_km', 'Time_to_Pickup_minutes',
                'Is_Weekend', 'Order_Hour_sin', 'Order_Hour_cos']:
        # Ensure we only try to assign keys that are in the raw_inputs dictionary
        if key in user_inputs: 
            df[key] = user_inputs[key]
    
    # 3. Ordinal Encoding for Traffic (0=Low, 1=Medium, 2=High, 3=Jammed)
    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jammed': 3}
    df['Traffic_Encoded'] = traffic_map.get(user_inputs['Traffic'], 1)

    # 4. One-Hot Encoding (Set the relevant feature column to 1)
    
    # Weather (Note: 'Cloudy' and 'Rainy' inputs are now mapped to the implicit baseline if they were chosen)
    weather_map = {
        'Sunny': 'Weather_Sunny', 'Fog': 'Weather_Fog', 'Sandstorms': 'Weather_Sandstorms',
        'Stormy': 'Weather_Stormy', 'Windy': 'Weather_Windy', 
        # Inputs like 'Cloudy' and 'Rainy' will now fall to the baseline (all OHE columns remain 0)
    }
    weather_col = weather_map.get(user_inputs['Weather'], None)
    if weather_col and weather_col in df.columns: 
        df[weather_col] = 1

    # Vehicle (Handling lowercase names, NOW ASSUMING NO TRAILING SPACES)
    # The maps are updated to reflect the new, cleaner feature names.
    vehicle_map = {
        'Motorcycle': 'Vehicle_motorcycle', 
        'Scooter': 'Vehicle_scooter', 
        'Van': 'Vehicle_van',
        # 'Electric_Vehicle' inputs will now fall to the baseline (all OHE columns remain 0)
    }
    vehicle_col = vehicle_map.get(user_inputs['Vehicle'], None)
    if vehicle_col and vehicle_col in df.columns: 
        df[vehicle_col] = 1

    # Area (Handling, NOW ASSUMING NO TRAILING SPACES)
    # The maps are updated to reflect the new, cleaner feature names.
    area_map = {
        'Urban': 'Area_Urban', 
        'Semi-Urban': 'Area_Semi-Urban', 
        'Other': 'Area_Other'
        # 'Metro' and 'Rural' inputs will now fall to the baseline
    }
    area_col = area_map.get(user_inputs['Area'], None)
    if area_col and area_col in df.columns: 
        df[area_col] = 1
    
    # Category (Handling missing categories)
    category_map = {
        'Books': 'Category_Books', 
        'Clothing': 'Category_Clothing', 
        'Cosmetics': 'Category_Cosmetics', 
        'Electronics': 'Category_Electronics', 
        'Grocery': 'Category_Grocery', 
        'Toys': 'Category_Toys',
        # 'Food' and 'Home' inputs will now fall to the baseline
    }
    category_col = category_map.get(user_inputs['Category'], None)
    if category_col and category_col in df.columns: 
        df[category_col] = 1
    
    # Day of Week (Handling the '.0' suffix)
    order_dayofweek = user_inputs['Order_DayOfWeek']
    if order_dayofweek > 0:
        day_col = f'Order_DayOfWeek_{order_dayofweek}.0'
        if day_col in df.columns: df[day_col] = 1

    # 5. Explicitly return columns in the EXPECTED_FEATURES order
    return df[expected_features]


# --- STREAMLIT APP LAYOUT & LOGIC ---

st.set_page_config(
    page_title="Delivery Time Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the theme elements not covered by the config.toml
st.markdown("""
<style>
/* Header/Title */
h1 {
    color: #FF9900; /* Amazon Yellow */
    font-weight: 700;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}

/* Sidebar Styling - Secondary Background color from config.toml handles main bar */
.stSidebar > div:first-child {
    color: white; /* Text in dark sidebar */
}

/* Prediction Button (Styled like Amazon 'Proceed' button) */
.stButton>button {
    background-color: #FF9900; /* Amazon Yellow */
    color: #111111;
    font-weight: bold;
    border-radius: 6px;
    padding: 10px 20px;
    margin-top: 20px;
    border: none;
    transition: all 0.2s;
}

.stButton>button:hover {
    background-color: #ffb84d;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

/* Metric Boxes */
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #FF9900;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("📦 Optimized Delivery Time Predictor")
st.markdown("Predict the total delivery time (in minutes) using the Gradient Boosting Model.")

# --- USER INPUTS ---

with st.sidebar:
    st.header("Agent & Order Details")

    # Agent Features
    agent_age = st.number_input("Agent Age", min_value=18, max_value=60, value=25, step=1)
    agent_rating = st.slider("Agent Rating", min_value=1.0, max_value=5.0, value=4.7, step=0.1)

    # Time Features
    order_date = st.date_input("Order Date", value=pd.to_datetime('today'))
    order_time_str = st.text_input("Order Time (HH:MM:SS)", value="18:30:00")
    pickup_time_str = st.text_input("Pickup Time (HH:MM:SS)", value="18:45:00")
    
    try:
        order_dt = pd.to_datetime(f"{order_date} {order_time_str}")
        pickup_dt = pd.to_datetime(f"{order_date} {pickup_time_str}")
    except ValueError:
        st.error("Please ensure time is in HH:MM:SS format.")
        st.stop()


col1, col2 = st.columns(2)

with col1:
    st.subheader("Store & Drop-off Location")
    store_lat = st.number_input("Store Latitude", value=12.93, format="%.4f")
    store_lon = st.number_input("Store Longitude", value=77.62, format="%.4f")
    
    drop_lat = st.number_input("Drop-off Latitude", value=12.98, format="%.4f")
    drop_lon = st.number_input("Drop-off Longitude", value=77.58, format="%.4f")
    
    # UPDATED: Only areas present in the model training data
    area = st.selectbox("Delivery Area Type", options=['Urban', 'Other', 'Semi-Urban'], index=0)

with col2:
    st.subheader("Environmental & Order Factors")
    
    traffic = st.selectbox("Traffic Condition", options=['Low', 'Medium', 'High', 'Jammed'], index=1)
    # Note: 'Cloudy' and 'Rainy' removed as they weren't OHE in your model
    weather = st.selectbox("Weather Condition", options=['Sunny', 'Fog', 'Sandstorms', 'Stormy', 'Windy'], index=0) 
    
    # UPDATED: Only vehicles present in the model training data (Van instead of Electric_Vehicle)
    vehicle = st.selectbox("Vehicle Type", options=['Scooter', 'Motorcycle', 'Van'], index=0)
    
    # UPDATED: Only categories present in the model training data (Toys added; Food/Home removed)
    category = st.selectbox("Order Category", options=['Electronics', 'Grocery', 'Books', 'Clothing', 'Cosmetics', 'Toys'], index=0)


# --- PREDICTION TRIGGER ---

if st.button("Predict Delivery Time"):
    
    # 1. Feature Engineering
    travel_distance_km = haversine(store_lat, store_lon, drop_lat, drop_lon)
    time_to_pickup_minutes, order_hour_sin, order_hour_cos, is_weekend, order_dayofweek = \
        create_temporal_features(order_dt, pickup_dt)
    
    # 2. Collect all raw inputs into a dictionary
    raw_inputs = {
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Travel_Distance_km': travel_distance_km,
        'Time_to_Pickup_minutes': time_to_pickup_minutes,
        'Is_Weekend': is_weekend, # Order adjusted
        'Order_Hour_sin': order_hour_sin, # Order adjusted
        'Order_Hour_cos': order_hour_cos, # Order adjusted
        'Traffic': traffic,
        'Weather': weather,
        'Vehicle': vehicle,
        'Area': area,
        'Category': category,
        'Order_DayOfWeek': order_dayofweek
    }

    # 3. Preprocess and get the exact feature vector
    X_predict = preprocess_inputs(raw_inputs, EXPECTED_FEATURES)

    # 4. Scaling (This line should now pass!)
    X_scaled = scaler.transform(X_predict)
    
    # 5. Prediction
    prediction = model.predict(X_scaled)[0]
    
    # 6. Display Results
    
    predicted_time = max(1, round(prediction))
    hours = int(predicted_time // 60)
    minutes = int(predicted_time % 60)
    
    st.markdown("## Prediction Result")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    col_res1.metric(label="Predicted Delivery Time", value=f"~{predicted_time} min")
    col_res2.metric(label="Equivalent Time", value=f"{hours}h {minutes}m")
    col_res3.metric(label="Time to Pickup", value=f"{time_to_pickup_minutes:.1f} min")

    st.success(f"The estimated time for this delivery is **{predicted_time} minutes** based on the current {traffic} traffic and {weather} conditions.")
