"""
Data preprocessing for crop recommendation dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class RecommendationPreprocessor:
    """Preprocessor for crop recommendation data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, file_path):
        """
        Load crop recommendation data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, data, target_columns=['Crop', 'Fertilizer']):
        """
        Preprocess the crop recommendation data
        
        Args:
            data (pandas.DataFrame): Input data
            target_columns (list): List of target column names
            
        Returns:
            tuple: (X, y_crop, y_fertilizer) - Processed features and targets
        """
        # Remove the 'Link' column if it exists
        if 'Link' in data.columns:
            data = data.drop('Link', axis=1)
        
        # Separate features and targets
        feature_columns = [col for col in data.columns if col not in target_columns]
        self.feature_columns = feature_columns
        
        X = data[feature_columns].copy()
        y_crop = data[target_columns[0]] if len(target_columns) > 0 else None
        y_fertilizer = data[target_columns[1]] if len(target_columns) > 1 else None
        
        # Handle categorical features
        categorical_columns = ['District_Name', 'Soil_color']
        numerical_columns = [col for col in feature_columns if col not in categorical_columns]
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        # Encode targets
        if y_crop is not None:
            if 'crop_encoder' not in self.label_encoders:
                self.label_encoders['crop_encoder'] = LabelEncoder()
                y_crop = self.label_encoders['crop_encoder'].fit_transform(y_crop)
            else:
                y_crop = self.label_encoders['crop_encoder'].transform(y_crop)
        
        if y_fertilizer is not None:
            if 'fertilizer_encoder' not in self.label_encoders:
                self.label_encoders['fertilizer_encoder'] = LabelEncoder()
                y_fertilizer = self.label_encoders['fertilizer_encoder'].fit_transform(y_fertilizer)
            else:
                y_fertilizer = self.label_encoders['fertilizer_encoder'].transform(y_fertilizer)
        
        return X, y_crop, y_fertilizer
    
    def save_preprocessor(self, file_path):
        """
        Save the preprocessor to disk
        
        Args:
            file_path (str): Path to save the preprocessor
        """
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, file_path)
        print(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path):
        """
        Load the preprocessor from disk
        
        Args:
            file_path (str): Path to load the preprocessor from
        """
        try:
            data = joblib.load(file_path)
            self.label_encoders = data['label_encoders']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            print(f"Preprocessor loaded from {file_path}")
        except Exception as e:
            print(f"Error loading preprocessor: {str(e)}")
    
    def preprocess_single_sample(self, sample_data):
        """
        Preprocess a single sample for prediction
        
        Args:
            sample_data (dict): Dictionary with feature names as keys and values as values
            
        Returns:
            numpy.ndarray: Preprocessed feature vector
        """
        # Convert to DataFrame
        df = pd.DataFrame([sample_data])
        
        # Apply the same preprocessing steps
        categorical_columns = ['District_Name', 'Soil_color']
        numerical_columns = [col for col in self.feature_columns if col not in categorical_columns]
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in df.columns and col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError as e:
                    print(f"Warning: Unknown category in {col}, using default value")
                    df[col] = 0  # Default to first category
        
        # Scale numerical features
        df[numerical_columns] = self.scaler.transform(df[numerical_columns])
        
        return df[self.feature_columns].values

# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = RecommendationPreprocessor()
    
    # Load data
    data_path = os.path.join("datasets", "crop_recommendation", "Crop and fertilizer dataset.csv")
    data = preprocessor.load_data(data_path)
    
    if data is not None:
        # Preprocess data
        X, y_crop, y_fertilizer = preprocessor.preprocess_data(data)
        
        print(f"Features shape: {X.shape}")
        print(f"Crop targets shape: {y_crop.shape if y_crop is not None else 'None'}")
        print(f"Fertilizer targets shape: {y_fertilizer.shape if y_fertilizer is not None else 'None'}")
        
        # Save preprocessor
        preprocessor.save_preprocessor("crop_recommendation/preprocessor.pkl")