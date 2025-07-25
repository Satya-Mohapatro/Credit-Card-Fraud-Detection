import streamlit as st
import pandas as pd
import joblib

st.title("💳 Credit Card Fraud Detection")
st.markdown("""
Upload credit card transactions to predict fraud probabilities using a Gradient Boosting (XGBoost) model within a clean sklearn pipeline.
""")

@st.cache_resource
def load_pipeline():
    pipeline = joblib.load('xgb_fraud_pipeline.pkl')  # Ensure you have saved your trained pipeline as this
    return pipeline

pipeline = load_pipeline()

# Upload CSV
uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    # Predict
    predictions = pipeline.predict(data)
    probabilities = pipeline.predict_proba(data)[:, 1]

    # Set custom fraud detection threshold
threshold = st.slider("Set Fraud Probability Threshold", 0.0, 1.0, 0.1, 0.01)

# Predict probabilities
probabilities = pipeline.predict_proba(data)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# Display predictions
# Display predictions with message

result_df = data.copy()
result_df['Fraud_Prediction'] = predictions
result_df['Fraud_Probability'] = probabilities

# Display table
st.subheader("📊 Predictions with Fraud Status:")
st.dataframe(result_df.head())

st.subheader("🚨 Detection Results:")

for idx, prob in enumerate(probabilities):
    if prob >= threshold:
        st.error(f"Transaction {idx}: 🚩 Fraud Detected (Probability: {prob:.4f})")
    else:
        st.success(f"Transaction {idx}: ✅ No Fraud Detected (Probability: {prob:.4f})")


# Option to download
st.download_button(
    "Download Predictions as CSV",
    data=result_df.to_csv(index=False).encode('utf-8'),
    file_name="fraud_predictions.csv",
    mime="text/csv"
)

