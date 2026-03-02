# src/train_model.py

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE


def load_data(data_path):
    """Load dataset"""
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df):
    """Split features and target"""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def split_data(X, y):
    """Train test split"""
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_data(X_train, X_test, models_path):
    """Scale Time and Amount features"""

    scaler = StandardScaler()

    X_train[['Time', 'Amount']] = scaler.fit_transform(
        X_train[['Time', 'Amount']]
    )

    X_test[['Time', 'Amount']] = scaler.transform(
        X_test[['Time', 'Amount']]
    )

    # Save scaler
    os.makedirs(models_path, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))

    print("Scaler saved")

    return X_train, X_test


def balance_data(X_train, y_train):
    """Handle imbalanced data using SMOTE"""

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_train, y_train
    )

    return X_resampled, y_resampled


def train_model(X_train, y_train):
    """Train Logistic Regression model"""

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))


def save_model(model, models_path):
    """Save trained model"""

    joblib.dump(model, os.path.join(models_path, "fraud_model.pkl"))

    print("Model saved successfully")


def main():

    DATA_PATH = "data/creditcard.csv"
    MODELS_PATH = "models"

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Scaling data...")
    X_train, X_test = scale_data(X_train, X_test, MODELS_PATH)

    print("Handling imbalance...")
    X_train_res, y_train_res = balance_data(X_train, y_train)

    print("Training model...")
    model = train_model(X_train_res, y_train_res)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving model...")
    save_model(model, MODELS_PATH)

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()