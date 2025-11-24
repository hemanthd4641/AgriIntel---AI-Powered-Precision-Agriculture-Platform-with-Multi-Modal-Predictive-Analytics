"""
Enhanced training script for crop recommendation models with detailed insights and explanations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys
import os
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crop_recommendation.preprocessing.recommendation_preprocessor import RecommendationPreprocessor
from crop_recommendation.models.crop_recommendation_models import (
    CropRecommendationRandomForest,
    CropRecommendationXGBoost,
    CropRecommendationNeuralNet
)


def create_sample_crop_data():
    """Create sample crop recommendation dataset for demonstration."""
    # In a real implementation, you would load data from a file
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    data = {
        'N': np.random.uniform(0, 140, n_samples),
        'P': np.random.uniform(5, 145, n_samples),
        'K': np.random.uniform(5, 205, n_samples),
        'temperature': np.random.uniform(9, 44, n_samples),
        'humidity': np.random.uniform(14, 100, n_samples),
        'ph': np.random.uniform(3.5, 10, n_samples),
        'rainfall': np.random.uniform(20, 300, n_samples),
    }
    
    # Create crop labels based on some simple rules
    crops = []
    for i in range(n_samples):
        n, p, k, temp, humidity, ph, rainfall = (
            data['N'][i], data['P'][i], data['K'][i], 
            data['temperature'][i], data['humidity'][i], 
            data['ph'][i], data['rainfall'][i]
        )
        
        # Simple rule-based crop assignment for demonstration
        if n > 100 and p > 50 and k > 50 and 20 < temp < 30:
            crop = 'rice'
        elif n > 80 and p > 40 and k > 40 and 15 < temp < 25:
            crop = 'wheat'
        elif n > 120 and p > 60 and k > 60 and 25 < temp < 35:
            crop = 'maize'
        elif n > 60 and p > 30 and k > 30 and 20 < temp < 30:
            crop = 'cotton'
        elif n > 150 and p > 50 and k > 150 and 25 < temp < 35:
            crop = 'sugarcane'
        else:
            # Randomly assign one of the common crops
            crop = np.random.choice(['rice', 'wheat', 'maize', 'cotton', 'sugarcane'])
        
        crops.append(crop)
    
    data['label'] = crops
    return pd.DataFrame(data)


def generate_crop_insights(model, X_train, y_train, feature_names, class_names):
    """Generate insights about the crop recommendation model."""
    insights = {
        'model_type': model.model_type,
        'training_samples': len(X_train),
        'features': feature_names,
        'classes': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
        'feature_importance': {},
        'model_performance': {}
    }
    
    # Get feature importance if available
    if hasattr(model, 'get_feature_importance'):
        try:
            importance_scores = model.get_feature_importance()
            feature_importance = dict(zip(feature_names, importance_scores))
            # Sort by importance
            insights['feature_importance'] = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        except:
            insights['feature_importance'] = {}
    
    return insights


def generate_crop_recommendation_explanation(crop, soil_conditions, weather_conditions, insights):
    """
    Generate a natural language explanation for crop recommendation using rule-based logic.
    In a real implementation, this would use rule-based logic.

    """
    explanation = f"Based on the analysis of your soil and weather conditions, {crop.title()} is the recommended crop for your farm.\n\n"
    
    explanation += "Key factors supporting this recommendation:\n"
    
    # Add soil condition analysis
    n, p, k, ph = soil_conditions['N'], soil_conditions['P'], soil_conditions['K'], soil_conditions['ph']
    explanation += f"- Soil nutrients (N:{n:.1f}, P:{p:.1f}, K:{k:.1f}) are "
    if n > 100 and p > 50 and k > 50:
        explanation += "well-balanced for this crop.\n"
    else:
        explanation += "adequate for this crop.\n"
    
    # Add pH analysis
    explanation += f"- Soil pH ({ph:.1f}) is "
    if 6.0 <= ph <= 7.5:
        explanation += "optimal for most crops including this one.\n"
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        explanation += "slightly outside optimal range but still suitable.\n"
    else:
        explanation += "outside optimal range; consider soil amendments.\n"
    
    # Add weather analysis
    temp, humidity, rainfall = weather_conditions['temperature'], weather_conditions['humidity'], weather_conditions['rainfall']
    explanation += f"- Weather conditions (Temp:{temp:.1f}°C, Humidity:{humidity:.1f}%, Rainfall:{rainfall:.1f}mm) are "
    if 20 <= temp <= 30 and 60 <= humidity <= 80 and 100 <= rainfall <= 200:
        explanation += "ideal for this crop.\n"
    else:
        explanation += "generally suitable for this crop.\n"
    
    # Add feature importance insights if available
    if insights.get('feature_importance'):
        explanation += "\nModel Insights:\n"
        top_features = list(insights['feature_importance'].items())[:3]
        for feature, importance in top_features:
            explanation += f"- {feature.title()} is a key factor ({importance:.2%} importance) in this recommendation.\n"
    
    explanation += f"\n{crop.title()} is known for:\n"
    crop_info = {
        'rice': "- High water requirement, good for areas with adequate rainfall\n- Thrives in warm, humid conditions\n- Good source of carbohydrates",
        'wheat': "- Moderate climate crop, prefers cooler temperatures\n- Requires well-drained soil\n- Staple food crop worldwide",
        'maize': "- Warm season crop, needs plenty of sunlight\n- High nitrogen requirement\n- Versatile crop used for food and feed",
        'cotton': "- Warm climate crop, needs long frost-free period\n- Requires well-drained, fertile soil\n- Important fiber crop",
        'sugarcane': "- Tropical crop, needs high temperatures and rainfall\n- Heavy feeder, requires rich soil\n- Major source of sugar"
    }
    explanation += crop_info.get(crop.lower(), "- A suitable crop for your conditions")
    
    return explanation


def save_model_metadata(metadata, file_path):
    """Save model metadata to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to {file_path}")
    except Exception as e:
        print(f"Error saving model metadata: {e}")


def train_crop_recommendation_model_enhanced():
    """Train the enhanced crop recommendation model with detailed insights."""
    print("Creating sample crop dataset...")
    df = create_sample_crop_data()
    print(f"Dataset created with {len(df)} samples")
    print(f"Crop distribution:\n{df['label'].value_counts()}")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = RecommendationPreprocessor()
    # Create a copy of the dataframe with the correct column structure for preprocessing
    df_for_preprocessing = df.copy()
    df_for_preprocessing.rename(columns={'label': 'Crop'}, inplace=True)
    X, y, _ = preprocessor.preprocess_data(df_for_preprocessing, target_columns=['Crop'])
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    class_names = df['label'].unique()
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = CropRecommendationRandomForest(n_estimators=100, random_state=42)
    rf_model.train(X_train, y_train)
    
    # Evaluate Random Forest model
    rf_pred, rf_prob = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = CropRecommendationXGBoost(n_estimators=100, max_depth=6, learning_rate=0.1)
    xgb_model.train(X_train, y_train)
    
    # Evaluate XGBoost model
    xgb_pred, xgb_prob = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Select best model based on accuracy
    if rf_accuracy >= xgb_accuracy:
        best_model = rf_model
        best_model_name = "RandomForest"
        print(f"Selected Random Forest as best model with accuracy: {rf_accuracy:.4f}")
    else:
        best_model = xgb_model
        best_model_name = "XGBoost"
        print(f"Selected XGBoost as best model with accuracy: {xgb_accuracy:.4f}")
    
    # Generate insights
    print("Generating model insights...")
    insights = generate_crop_insights(best_model, X_train, y_train, feature_names, class_names)
    insights['model_accuracy'] = float(max(rf_accuracy, xgb_accuracy))
    
    # Create sample explanation
    sample_conditions = {
        'soil_conditions': {'N': 120, 'P': 60, 'K': 80, 'ph': 6.5},
        'weather_conditions': {'temperature': 25, 'humidity': 70, 'rainfall': 150}
    }
    
    # Generate sample explanation for one of the crops
    sample_crop = class_names[0] if len(class_names) > 0 else 'wheat'
    explanation = generate_crop_recommendation_explanation(
        sample_crop, 
        sample_conditions['soil_conditions'], 
        sample_conditions['weather_conditions'], 
        insights
    )
    
    # Save metadata
    metadata = {
        'model_type': best_model_name,
        'accuracy': float(max(rf_accuracy, xgb_accuracy)),
        'features': feature_names,
        'classes': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
        'insights': insights,
        'sample_explanation': explanation,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    # Save the best model, preprocessor, and metadata
    print("Saving model, preprocessor, and metadata...")
    os.makedirs('crop_recommendation/saved_models', exist_ok=True)
    best_model.save_model('crop_recommendation/saved_models/crop_model_enhanced.pkl')
    joblib.dump(preprocessor, 'crop_recommendation/saved_models/crop_preprocessor_enhanced.pkl')
    save_model_metadata(metadata, 'crop_recommendation/saved_models/crop_model_metadata.json')
    
    print("Enhanced crop recommendation model and preprocessor saved successfully!")
    
    return best_model, preprocessor, metadata


if __name__ == "__main__":
    train_crop_recommendation_model_enhanced()