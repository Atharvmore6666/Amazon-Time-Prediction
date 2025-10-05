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

    # --- EXPECTED FEATURES (REQUIRED FOR SCALER COMPATIBILITY) ---
    # These feature names must match the names the scaler was trained on exactly.
    EXPECTED_FEATURES = [
        'Agent_Age', 
        'Agent_Rating', 
        'Travel_Distance_km', 
        'Time_to_Pickup_minutes',
        'Is_Weekend',              
        'Order_Hour_sin', 
        'Order_Hour_cos', 
        'Traffic_Encoded', 
        'Weather_Fog', 
        'Weather_Sandstorms', 
        'Weather_Stormy', 
        'Weather_Sunny', 
        'Weather_Windy', 
        'Vehicle_motorcycle ', 
        'Vehicle_scooter ', 
        'Vehicle_van',
        'Area_Other',     
        'Area_Semi-Urban ', 
        'Area_Urban ', 
        'Category_Books', 
        'Category_Clothing', 
        'Category_Cosmetics', 
        'Category_Electronics', 
        'Category_Grocery', 
        'Category_Home',       
        'Category_Jewelry',    
        'Category_Kitchen',    
        'Category_Outdoors', 
        'Category_Pet Supplies',
        'Category_Shoes',
        'Category_Skincare',
        'Category_Snacks',
        'Category_Sports', 
        'Category_Toys', 
        'Order_DayOfWeek_1.0', 
        'Order_DayOfWeek_2.0', 
        'Order_DayOfWeek_3.0',
        'Order_DayOfWeek_4.0', 
        'Order_DayOfWeek_5.0', 
        'Order_DayOfWeek_6.0'
    ]
    # --- END EXPECTED FEATURES ---

except FileNotFoundError:
    st.error(f"File not found. Please ensure your model files are in the root directory next to this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during model loading: {type(e).__name__}: {e}")
    st.info("Check the file paths and ensure the joblib files are valid.")
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
    
    time_diff = pickup_dt - order_dt
    time_to_pickup_minutes = max(0, time_diff.total_seconds() / 60)

    order_hour = order_dt.hour
    order_dayofweek = order_dt.dayofweek # Monday=0, Sunday=6
    is_weekend = 1 if order_dayofweek >= 5 else 0

    order_hour_sin = np.sin(2 * np.pi * order_hour / 24)
    order_hour_cos = np.cos(2 * np.pi * order_hour / 24)

    return time_to_pickup_minutes, order_hour_sin, order_hour_cos, is_weekend, order_dayofweek

def preprocess_inputs(user_inputs, expected_features):
    """
    Transforms raw user inputs into the final, standardized feature vector, 
    using the exact feature names (including control characters/spaces).
    """
    
    # 1. Initialize DataFrame with ALL expected features set to 0
    df = pd.DataFrame(0, index=[0], columns=expected_features)

    # 2. Add Numerical and Engineered Features
    for key in ['Agent_Age', 'Agent_Rating', 'Travel_Distance_km', 'Time_to_Pickup_minutes',
                'Is_Weekend', 'Order_Hour_sin', 'Order_Hour_cos']:
        if key in user_inputs: 
            df[key] = user_inputs[key]
    
    # 3. Ordinal Encoding for Traffic (0=Low, 1=Medium, 2=High, 3=Jammed)
    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jammed': 3}
    df['Traffic_Encoded'] = traffic_map.get(user_inputs['Traffic'], 1)

    # 4. One-Hot Encoding (Map friendly names to the required feature names)
    
    # Weather 
    weather_map = {
        'Sunny': 'Weather_Sunny', 
        'Fog': 'Weather_Fog', 
        'Sandstorms': 'Weather_Sandstorms',
        'Stormy': 'Weather_Stormy', 
        'Windy': 'Weather_Windy', 
    }
    weather_col = weather_map.get(user_inputs['Weather'], None)
    if weather_col and weather_col in df.columns: 
        df[weather_col] = 1

    # Vehicle 
    vehicle_map = {
        'Motorcycle': 'Vehicle_motorcycle ', 
        'Scooter': 'Vehicle_scooter ', 
        'Van': 'Vehicle_van',
    }
    vehicle_col = vehicle_map.get(user_inputs['Vehicle'], None)
    if vehicle_col and vehicle_col in df.columns: 
        df[vehicle_col] = 1

    # Area 
    area_map = {
        'Urban': 'Area_Urban ', 
        'Semi-Urban': 'Area_Semi-Urban ', 
        'Other': 'Area_Other' 
    }
    area_col = area_map.get(user_inputs['Area'], None)
    if area_col and area_col in df.columns: 
        df[area_col] = 1
    
    # Category 
    category_map = {
        'Books': 'Category_Books', 
        'Clothing': 'Category_Clothing', 
        'Category_Cosmetics': 'Category_Cosmetics', 
        'Electronics': 'Category_Electronics', 
        'Grocery': 'Category_Grocery', 
        'Toys': 'Category_Toys', 
        'Home': 'Category_Home',
        'Jewelry': 'Category_Jewelry',
        'Kitchen': 'Category_Kitchen',
        'Outdoors': 'Category_Outdoors',
        'Pet Supplies': 'Category_Pet Supplies',
        'Shoes': 'Category_Shoes',
        'Skincare': 'Category_Skincare',
        'Snacks': 'Category_Snacks',
        'Sports': 'Category_Sports', 
    }
    category_col = category_map.get(user_inputs['Category'], None)
    if category_col and category_col in df.columns: 
        df[category_col] = 1
    
    # Day of Week 
    order_dayofweek = user_inputs['Order_DayOfWeek'] + 1 
    if order_dayofweek > 0 and order_dayofweek <= 6: 
        day_col = f'Order_DayOfWeek_{order_dayofweek}.0'
        if day_col in df.columns: df[day_col] = 1

    # 5. Explicitly return columns in the EXPECTED_FEATURES order
    return df[expected_features]


# --- STREAMLIT APP LAYOUT & LOGIC ---

st.set_page_config(
    page_title="Amazon Logistics Delivery Time Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the Amazon Theme (Primary: #FF9900, Secondary: #232F3E)
st.markdown("""
<style>
/* Header/Title - Using the secondary Amazon Navy Blue */
h1 {
    color: #232F3E; 
    font-weight: 700;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    margin-top: 10px;
}

/* Prediction Button - Primary Amazon Yellow/Orange */
.stButton>button {
    background-color: #FF9900; 
    color: #111111; /* Black text on yellow button */
    font-weight: bold;
    border-radius: 6px;
    padding: 10px 20px;
    margin-top: 20px;
    border: none;
    transition: all 0.2s;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.stButton>button:hover {
    background-color: #F3A847; /* Slightly darker orange on hover */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}

/* Metric Boxes */
.stMetric {
    background-color: #FFFFFF;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #232F3E; /* Navy border for contrast */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Metric Label (title) - Dark Gray */
.stMetric label {
    color: #111111 !important; /* Black */
    font-weight: 500;
}

/* FIX: Metric Value Color - Primary Amazon Yellow/Orange */
.stMetric .stMetricValue {
    color: #FF9900 !important; /* Amazon Primary Yellow/Orange */
    font-weight: 900 !important; 
}

/* Logo Placeholder - Styled to reflect the brand's primary colors */
.logo-container {
    padding-bottom: 20px;
    padding-top: 10px;
}

.logo-placeholder {
    display: inline-block;
    font-size: 28px;
    font-weight: 800;
    color: #232F3E; /* Navy Text */
}

/* Highlight for the 'smile' effect in the logo */
.logo-highlight {
    color: #FF9900; /* Yellow/Orange */
    margin-left: 2px;
}
</style>
""", unsafe_allow_html=True)

# Logo Placeholder (Mimicking the look of the Amazon logo/branding)
st.markdown("""
<div class="logo-container">
    <span class="logo-placeholder">
        amazon
        <span class="logo-highlight">Logistics</span>
    </span>
</div>
""", unsafe_allow_html=True)

st.title("Delivery Time Predictor")
st.markdown("Use this tool to predict the total delivery time (in minutes) for **Amazon Logistics** deliveries.")

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
    
    area = st.selectbox("Delivery Area Type", options=['Urban', 'Other', 'Semi-Urban'], index=0)

with col2:
    st.subheader("Environmental & Order Factors")
    
    traffic = st.selectbox("Traffic Condition", options=['Low', 'Medium', 'High', 'Jammed'], index=1)
    weather = st.selectbox("Weather Condition", options=['Sunny', 'Fog', 'Sandstorms', 'Stormy', 'Windy'], index=0) 
    
    vehicle = st.selectbox("Vehicle Type", options=['Scooter', 'Motorcycle', 'Van'], index=0)
    
    category = st.selectbox("Order Category", options=['Electronics', 'Grocery', 'Books', 'Clothing', 'Cosmetics', 'Toys', 'Sports'], index=0)


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
        'Is_Weekend': is_weekend,
        'Order_Hour_sin': order_hour_sin,
        'Order_Hour_cos': order_hour_cos,
        'Traffic': traffic,
        'Weather': weather,
        'Vehicle': vehicle,
        'Area': area,
        'Category': category,
        'Order_DayOfWeek': order_dayofweek 
    }

    # 3. Preprocess and get the exact feature vector
    X_predict = preprocess_inputs(raw_inputs, EXPECTED_FEATURES)

    # 4. Scaling
    try:
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


    except ValueError as e:
        st.error(f"Prediction Feature Mismatch Error: {e}")
        st.warning("The feature names still do not match the fitted scaler. Please ensure the model files are correct.")
        st.write("Generated Features (X_predict.columns):")
        st.code(list(X_predict.columns))
        st.stop()
