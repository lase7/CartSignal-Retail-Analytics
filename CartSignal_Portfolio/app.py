import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Model (Simplified: No Caching to prevent errors) ---
def load_model():
    # This expects 'cartsignal_model.pkl' to be in the same folder
    return joblib.load('cartsignal_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'cartsignal_model.pkl' not found. Make sure it is in the same folder as this app.py file!")
    st.stop()

# --- 2. App Interface ---
st.set_page_config(page_title="CartSignal AI", page_icon="🛒")

st.title("🛒 CartSignal: Parent Propensity Predictor")
st.markdown("""
**Architect's Dashboard:** Adjust the customer behavior sliders on the left.
The AI will predict if this user is transitioning into the "New Parent" segment.
""")

# --- 3. Sidebar Inputs ---
st.sidebar.header("Customer Behavior Signals")

def user_input_features():
    # Behavioral Features
    total_orders = st.sidebar.slider('Total Lifetime Orders', 0, 100, 20)
    avg_days = st.sidebar.slider('Avg Days Between Orders', 0, 30, 7)
    avg_hour = st.sidebar.slider('Avg Order Hour (24h)', 0, 24, 14)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Basket DNA")
    healthy_ratio = st.sidebar.slider('Healthy Ratio (Produce/Dairy)', 0.0, 1.0, 0.5)
    junk_ratio = st.sidebar.slider('Junk Ratio (Snacks/Vice)', 0.0, 1.0, 0.2)
    reorder_rate = st.sidebar.slider('Loyalty (Reorder Rate)', 0.0, 1.0, 0.6)
    
    # Derived Feature Engineering (Must match training data)
    health_junk_diff = healthy_ratio - junk_ratio
    
    data = {
        'total_orders': total_orders,
        'avg_days_between_orders': avg_days,
        'avg_order_hour': avg_hour,
        'healthy_ratio': healthy_ratio,
        'junk_ratio': junk_ratio,
        'reorder_rate': reorder_rate,
        'health_junk_diff': health_junk_diff
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 4. Prediction Logic ---
st.subheader("Real-Time Prediction")

# Display the input data for transparency
st.text("Input Data Vector:")
st.dataframe(input_df)

if st.button("Analyze Customer"):
    # Get the raw probability
    prediction_proba = model.predict_proba(input_df)[0][1]
    
    # ---------------------------------------------------------
    # ARCHITECT'S CALIBRATION
    # Since the model is strict, we lower the threshold.
    # In this specific model, anything above 10% is a STRONG signal.
    # ---------------------------------------------------------
    decision_threshold = 0.10  # Lowered from 0.50
    
    if prediction_proba > decision_threshold:
        prediction = 1
        status = "TARGET IDENTIFIED: Likely New Parent"
    else:
        prediction = 0
        status = "Segment: Standard Shopper"
    # ---------------------------------------------------------

    # Visual Output
    st.markdown("---")
    
    if prediction == 1:
        st.success(f"👶 **{status}**")
        st.info(f" Model Confidence: {prediction_proba:.1%} (Threshold Met)")
        st.markdown(f"""
        **Analysis:**
        * The model detected a significant deviation from the baseline.
        * **Action:** Trigger 'New Family' Bundle Offer.
        """)
        st.balloons()
    else:
        st.info(f"🛒 **{status}**")
        st.write(f"Model Confidence: {prediction_proba:.1%} (Below Threshold)")
        st.write("Behavior is consistent with standard consumption patterns.")