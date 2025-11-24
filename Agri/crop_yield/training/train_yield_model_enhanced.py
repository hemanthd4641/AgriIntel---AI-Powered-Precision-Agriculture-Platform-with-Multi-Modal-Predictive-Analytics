"""
Enhanced Training script for crop yield prediction model with rule-based integration

This script trains a crop yield prediction model and integrates with rule-based systems
to provide detailed explanations and recommendations for predictions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys
import json
from datetime import datetime
from django.utils import timezone

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from crop_yield_prediction.preprocessing.yield_preprocessor import YieldPreprocessor

def generate_yield_insights(predicted_yield, input_data, model_performance):
    """
    Generate detailed insights about the yield prediction using rule-based logic.
    In a production environment, this would be replaced with more sophisticated rule-based systems.
    
    Args:
        predicted_yield (float): The predicted yield in tons per hectare
        input_data (dict): Dictionary containing input features
        model_performance (dict): Dictionary with model performance metrics
        
    Returns:
        dict: Dictionary containing detailed insights and recommendations
    """
    insights = {
        'prediction_summary': '',
        'key_factors': [],
        'recommendations': [],
        'risk_assessment': '',
        'comparison_to_benchmarks': ''
    }
    
    # Extract input data
    crop = input_data.get('Crop', 'Unknown')
    rainfall = input_data.get('Rainfall_mm', 0)
    temperature = input_data.get('Temperature_Celsius', 0)
    soil_type = input_data.get('Soil_Type', 'Unknown')
    fertilizer_used = input_data.get('Fertilizer_Used', False)
    irrigation_used = input_data.get('Irrigation_Used', False)
    weather_condition = input_data.get('Weather_Condition', 'Unknown')
    days_to_harvest = input_data.get('Days_to_Harvest', 0)
    
    # Generate prediction summary
    if predicted_yield > 5.0:
        insights['prediction_summary'] = f"Excellent yield prediction of {predicted_yield:.2f} tons/hectare for {crop}."
    elif predicted_yield > 3.0:
        insights['prediction_summary'] = f"Good yield prediction of {predicted_yield:.2f} tons/hectare for {crop}."
    elif predicted_yield > 2.0:
        insights['prediction_summary'] = f"Moderate yield prediction of {predicted_yield:.2f} tons/hectare for {crop}."
    else:
        insights['prediction_summary'] = f"Below average yield prediction of {predicted_yield:.2f} tons/hectare for {crop}."
    
    # Analyze key factors
    if rainfall < 300:
        insights['key_factors'].append("Low rainfall may limit yield potential")
    elif rainfall > 1200:
        insights['key_factors'].append("High rainfall may cause waterlogging issues")
    else:
        insights['key_factors'].append("Rainfall is within optimal range")
    
    if temperature < 15:
        insights['key_factors'].append("Cool temperatures may slow growth")
    elif temperature > 35:
        insights['key_factors'].append("High temperatures may stress plants")
    else:
        insights['key_factors'].append("Temperature is within optimal range")
    
    if fertilizer_used:
        insights['key_factors'].append("Fertilizer application will support growth")
    else:
        insights['key_factors'].append("Lack of fertilizer may limit yield")
    
    if irrigation_used:
        insights['key_factors'].append("Irrigation provides consistent moisture")
    else:
        insights['key_factors'].append("Dependence on rainfall for moisture")
    
    # Generate recommendations
    if not fertilizer_used:
        insights['recommendations'].append("Consider applying balanced fertilizer to improve yield potential")
    
    if not irrigation_used and rainfall < 500:
        insights['recommendations'].append("Implement irrigation system to supplement low rainfall")
    
    if days_to_harvest < 90:
        insights['recommendations'].append("Monitor crop closely as harvest approaches")
    elif days_to_harvest > 150:
        insights['recommendations'].append("Plan for extended growing season and potential pest pressure")
    
    # Risk assessment
    risk_factors = 0
    if rainfall < 300 or rainfall > 1200:
        risk_factors += 1
    if temperature < 10 or temperature > 40:
        risk_factors += 1
    if not fertilizer_used:
        risk_factors += 1
    if not irrigation_used and rainfall < 400:
        risk_factors += 1
    
    if risk_factors == 0:
        insights['risk_assessment'] = "Low risk - conditions are favorable for good yield"
    elif risk_factors == 1:
        insights['risk_assessment'] = "Moderate risk - some conditions may impact yield"
    else:
        insights['risk_assessment'] = "High risk - multiple factors may significantly impact yield"
    
    # Comparison to benchmarks
    crop_benchmarks = {
        'Wheat': 3.5,
        'Rice': 4.2,
        'Maize': 5.1,
        'Cotton': 2.8,
        'Soybean': 2.9,
        'Potato': 20.0,  # tons/hectare
        'Tomato': 25.0   # tons/hectare
    }
    
    if crop in crop_benchmarks:
        benchmark = crop_benchmarks[crop]
        if predicted_yield > benchmark * 1.2:
            insights['comparison_to_benchmarks'] = f"Prediction is {((predicted_yield/benchmark)-1)*100:.1f}% above typical {crop} yield benchmark of {benchmark} tons/hectare"
        elif predicted_yield < benchmark * 0.8:
            insights['comparison_to_benchmarks'] = f"Prediction is {(1-(predicted_yield/benchmark))*100:.1f}% below typical {crop} yield benchmark of {benchmark} tons/hectare"
        else:
            insights['comparison_to_benchmarks'] = f"Prediction is within 20% of typical {crop} yield benchmark of {benchmark} tons/hectare"
    else:
        insights['comparison_to_benchmarks'] = f"No benchmark available for {crop}"
    
    return insights

def generate_rule_based_explanation(predicted_yield, input_data, insights):
    """
    Generate a natural language explanation using rule-based logic.
    
    Args:
        predicted_yield (float): The predicted yield
        input_data (dict): Input data for the prediction
        insights (dict): Previously generated insights
        
    Returns:
        str: Natural language explanation of the prediction
    """
    # In a production environment, this would call an actual LLM
    # For now, we'll generate a comprehensive explanation based on the insights
    
    explanation = f"Based on the analysis of growing conditions, the predicted yield for {input_data.get('Crop', 'the crop')} is {predicted_yield:.2f} tons per hectare. "
    
    # Add prediction summary
    explanation += insights['prediction_summary'] + " "
    
    # Add key factors
    if insights['key_factors']:
        explanation += "Key factors influencing this prediction include: "
        explanation += ", ".join(insights['key_factors']) + ". "
    
    # Add risk assessment
    explanation += "Risk assessment: " + insights['risk_assessment'] + ". "
    
    # Add comparison to benchmarks
    explanation += insights['comparison_to_benchmarks'] + ". "
    
    # Add recommendations
    if insights['recommendations']:
        explanation += "To optimize yield, consider the following recommendations: "
        explanation += " ".join([f"{i+1}. {rec}" for i, rec in enumerate(insights['recommendations'])]) + ". "
    
    # Add model confidence information
    explanation += "This prediction is based on historical data patterns and should be used as a guide for planning purposes."
    
    return explanation

def save_prediction_metadata(model_performance, feature_importance, insights):
    """
    Save metadata about the model and predictions for future reference.
    
    Args:
        model_performance (dict): Model performance metrics
        feature_importance (dict): Feature importance scores
        insights (dict): Generated insights
    """
    metadata = {
        'training_date': timezone.now().isoformat(),
        'model_performance': model_performance,
        'feature_importance': feature_importance,
        'insights_template': insights
    }
    
    models_dir = "models/crop_yield_prediction"
    metadata_path = os.path.join(models_dir, "prediction_metadata.json")
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Prediction metadata saved to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")

def train_yield_prediction_model():
    """Train an enhanced model for crop yield prediction with rule-based integration"""
    
    print("Training enhanced crop yield prediction model...")
    
    # Create preprocessor
    preprocessor = YieldPreprocessor()
    
    # Load data
    data_path = os.path.join("datasets", "crop_yield_prediction", "crop_yield.csv")
    data = preprocessor.load_data(data_path)
    
    if data is None:
        print("Failed to load data")
        return
    
    # For demonstration, let's use a subset of the data to speed up training
    # In a real scenario, you would use the full dataset
    data_subset = data.sample(n=10000, random_state=42)
    print(f"Using subset of {len(data_subset)} samples for training")
    
    # Preprocess data
    X, y = preprocessor.preprocess_data(data_subset)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape if y is not None else 'None'}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train yield prediction model
    print("Training yield prediction model...")
    yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
    yield_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = yield_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_performance = {
        'mean_squared_error': float(mse),
        'mean_absolute_error': float(mae),
        'r2_score': float(r2)
    }
    
    print(f"Yield prediction metrics:")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Get feature importance
    feature_names = preprocessor.feature_columns if preprocessor.feature_columns else [f"Feature_{i}" for i in range(X.shape[1])]
    feature_importance = dict(zip(feature_names, yield_model.feature_importances_))
    
    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Save model and preprocessor
    models_dir = "models/crop_yield_prediction"
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(yield_model, os.path.join(models_dir, "yield_model_enhanced.pkl"))
    preprocessor.save_preprocessor(os.path.join(models_dir, "preprocessor_enhanced.pkl"))
    
    print("Enhanced model saved successfully!")
    
    # Generate sample insights and explanation
    sample_data = {
        'Region': 'West',
        'Soil_Type': 'Sandy',
        'Crop': 'Cotton',
        'Rainfall_mm': 900.0,
        'Temperature_Celsius': 28.0,
        'Fertilizer_Used': True,
        'Irrigation_Used': True,
        'Weather_Condition': 'Sunny',
        'Days_to_Harvest': 120
    }
    
    # Preprocess the sample
    X_sample = preprocessor.preprocess_single_sample(sample_data)
    
    # Make prediction
    yield_pred = yield_model.predict(X_sample)
    predicted_yield = yield_pred[0]
    
    print(f"\nSample prediction:")
    print(f"Input: {sample_data}")
    print(f"Predicted yield: {predicted_yield:.2f} tons per hectare")
    
    # Generate insights
    insights = generate_yield_insights(predicted_yield, sample_data, model_performance)
    
    # Generate rule-based explanation
    explanation = generate_rule_based_explanation(predicted_yield, sample_data, insights)
    
    print(f"\nGenerated Insights:")
    print(f"Prediction Summary: {insights['prediction_summary']}")
    print(f"Key Factors: {', '.join(insights['key_factors'])}")
    print(f"Risk Assessment: {insights['risk_assessment']}")
    print(f"Comparison to Benchmarks: {insights['comparison_to_benchmarks']}")
    print(f"Recommendations: {', '.join(insights['recommendations'])}")
    
    print(f"\nRule-Based Explanation:")
    print(explanation)
    
    # Save metadata
    save_prediction_metadata(model_performance, feature_importance, insights)
    
    # Save sample explanation
    explanation_path = os.path.join(models_dir, "sample_explanation.txt")
    try:
        with open(explanation_path, 'w', encoding='utf-8') as f:
            f.write(f"Sample Input: {sample_data}\n")
            f.write(f"Predicted Yield: {predicted_yield:.2f} tons/hectare\n\n")
            f.write("Generated Explanation:\n")
            f.write(explanation)
        print(f"\nSample explanation saved to {explanation_path}")
    except Exception as e:
        print(f"Error saving explanation: {str(e)}")
    
    return yield_model, preprocessor, insights

def integrate_with_rule_based_system():
    """Provide information about integrating with rule-based explanation systems"""
    print("\n=== Rule-Based Explanation System ===")
    print("This system uses rule-based logic for generating explanations:")
    print("1. Pre-defined rules based on agricultural best practices")
    print("2. Conditional logic for different crop types and conditions")
    print("3. Template-based natural language generation")
    print("4. Configurable recommendation engines")
    print("\nBenefits:")
    print("- No dependency on external APIs")
    print("- Consistent and predictable responses")
    print("- Lower computational requirements")
    print("- Easier to customize and maintain")

if __name__ == "__main__":
    # Train the enhanced model
    model, preprocessor, insights = train_yield_prediction_model()
    
    # Show rule-based integration information
    integrate_with_rule_based_system()