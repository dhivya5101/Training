import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load the deployed model from the specific run ID
def load_model():
    logged_model_uri = "runs:/5b111bce24544cb4a3ecb5eaec998cb9/model"
    model = mlflow.sklearn.load_model(logged_model_uri)
    return model

# Streamlit App Title
st.title("💸 Online Retail Price Prediction App")

st.markdown("Enter the product details below to get a prediction for the total price:")

# Input fields for features
quantity = st.number_input("Quantity", min_value=1, step=1, value=10)
unit_price = st.number_input("Unit Price (£)", min_value=0.01, step=0.01, value=1.00)

# Automatically calculate TotalPrice (used as a feature)
total_price_feature = quantity * unit_price

# Predict button
if st.button("Predict Total Price"):
    model = load_model()
    input_data = np.array([[quantity, unit_price, total_price_feature]])
    prediction = model.predict(input_data)

    st.success(f"✅ Predicted Total Price: **£{prediction[0]:.2f}**")
