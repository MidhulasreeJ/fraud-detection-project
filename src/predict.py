import joblib
from src.preprocessing import preprocess

model = joblib.load("models/fraud_model.pkl")

def predict_fraud(input_df):
    processed = preprocess(input_df)
    probability = model.predict_proba(processed)[0][1]
    return probability