"""
Train a pest prediction model using the generated pest dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_and_preprocess_data():
    """Load and preprocess the pest dataset."""
    # Load the dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'datasets', 'pest_prediction', 'pest_data.csv')
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    
    # Select features and target
    feature_columns = [
        'crop', 'region', 'season', 'temperature', 'humidity', 'rainfall', 'wind_speed',
        'soil_moisture', 'soil_ph', 'soil_type', 'nitrogen', 'phosphorus', 'potassium',
        'weather_condition', 'irrigation_method', 'previous_crop', 'days_since_planting',
        'plant_density'
    ]
    
    target_column = 'pest'
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Encode categorical variables
    categorical_columns = [
        'crop', 'region', 'season', 'soil_type', 'weather_condition', 
        'irrigation_method', 'previous_crop'
    ]
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    print("Data preprocessing completed")
    
    return X, y, label_encoders

def train_model(X, y):
    """Train the pest prediction model."""
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Create and train the model
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred

def save_model(model, label_encoders, X):
    """Save the trained model and preprocessor."""
    # Create preprocessor pipeline
    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Fit preprocessor on training data
    preprocessor.fit(X)
    
    # Save model and preprocessor
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pest_model.pkl')
    preprocessor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessor.pkl')
    encoders_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'label_encoders.pkl')
    
    print(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)
    
    print(f"Saving preprocessor to: {preprocessor_path}")
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"Saving label encoders to: {encoders_path}")
    joblib.dump(label_encoders, encoders_path)
    
    print("Model, preprocessor, and label encoders saved successfully")

def main():
    """Main function to train the pest prediction model."""
    print("=== PEST PREDICTION MODEL TRAINING ===")
    
    # Load and preprocess data
    X, y, label_encoders = load_and_preprocess_data()
    
    # Train model
    model, X_test, y_test, y_pred = train_model(X, y)
    
    # Save model and preprocessor
    save_model(model, label_encoders, X)
    
    print("=== TRAINING COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()