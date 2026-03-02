import joblib

scaler = joblib.load("models/scaler.pkl")

def preprocess(data):
    data[['Time', 'Amount']] = scaler.transform(data[['Time', 'Amount']])
    return data