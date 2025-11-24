"""
Enhanced Crop Yield Prediction Script

This script loads the trained crop yield prediction model and provides
detailed explanations and recommendations based on rule-based logic."""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
from datetime import datetime
from django.utils import timezone

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from crop_yield_prediction.preprocessing.yield_preprocessor import YieldPreprocessor

def load_enhanced_model():
    """
    Load the enhanced crop yield prediction model and preprocessor
    
    Returns:
        tuple: (model, preprocessor) or (None, None) if loading fails
    """
    try:
        # Define paths
        models_dir = "models/crop_yield_prediction"
        model_path = os.path.join(models_dir, "yield_model_enhanced.pkl")
        preprocessor_path = os.path.join(models_dir, "preprocessor_enhanced.pkl")
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Enhanced model not found at {model_path}")
            # Try to load the regular model as fallback
            model_path = os.path.join(models_dir, "yield_model.pkl")
            if not os.path.exists(model_path):
                print(f"Regular model also not found at {model_path}")
                return None, None
        
        if not os.path.exists(preprocessor_path):
            print(f"Enhanced preprocessor not found at {preprocessor_path}")
            # Try to load the regular preprocessor as fallback
            preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
            if not os.path.exists(preprocessor_path):
                print(f"Regular preprocessor also not found at {preprocessor_path}")
                return None, None
        
        # Load model and preprocessor
        model = joblib.load(model_path)
        preprocessor_data = joblib.load(preprocessor_path)
        
        # Create a new preprocessor instance and load the saved data
        preprocessor = YieldPreprocessor()
        preprocessor.label_encoders = preprocessor_data['label_encoders']
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.feature_columns = preprocessor_data['feature_columns']
        
        print("Enhanced model and preprocessor loaded successfully!")
        return model, preprocessor
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def get_prediction_metadata():
    """
    Load prediction metadata if available
    
    Returns:
        dict: Metadata dictionary or empty dict if not found
    """
    try:
        models_dir = "models/crop_yield_prediction"
        metadata_path = os.path.join(models_dir, "prediction_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return {}

def generate_yield_insights(predicted_yield, input_data, metadata):
    """
    Generate detailed insights about the yield prediction with enhanced specificity
    
    Args:
        predicted_yield (float): The predicted yield in tons per hectare
        input_data (dict): Dictionary containing input features
        metadata (dict): Model metadata
        
    Returns:
        dict: Dictionary containing detailed insights and recommendations
    """
    insights = {
        'prediction_summary': '',
        'key_factors': [],
        'recommendations': [],
        'risk_assessment': '',
        'comparison_to_benchmarks': '',
        'confidence_level': '',
        'detailed_factors': {},
        'nutrient_analysis': {},
        'water_management': {},
        'timing_advice': {}
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
    region = input_data.get('Region', 'Unknown')
    
    # Generate prediction summary with more detail
    if predicted_yield > 6.0:
        insights['prediction_summary'] = f"Excellent yield prediction of {predicted_yield:.2f} tons/hectare for {crop}. This is significantly above average and indicates optimal growing conditions."
    elif predicted_yield > 4.5:
        insights['prediction_summary'] = f"Very good yield prediction of {predicted_yield:.2f} tons/hectare for {crop}. Conditions are favorable for high productivity."
    elif predicted_yield > 3.0:
        insights['prediction_summary'] = f"Good yield prediction of {predicted_yield:.2f} tons/hectare for {crop}. Expect solid returns with proper management."
    elif predicted_yield > 2.0:
        insights['prediction_summary'] = f"Moderate yield prediction of {predicted_yield:.2f} tons/hectare for {crop}. Some factors may be limiting productivity."
    else:
        insights['prediction_summary'] = f"Below average yield prediction of {predicted_yield:.2f} tons/hectare for {crop}. Significant improvements in management practices are recommended."
    
    # Detailed factor analysis
    insights['detailed_factors'] = {
        'rainfall_analysis': {
            'value': rainfall,
            'optimal_range': '400-1000 mm',
            'assessment': 'Low' if rainfall < 400 else 'High' if rainfall > 1000 else 'Optimal',
            'impact': f'{((rainfall-700)/700)*100:.1f}% from optimal' if rainfall != 0 else 'Unknown'
        },
        'temperature_analysis': {
            'value': temperature,
            'optimal_range': '18-30°C',
            'assessment': 'Low' if temperature < 18 else 'High' if temperature > 30 else 'Optimal',
            'impact': f'{((temperature-24)/24)*100:.1f}% from optimal' if temperature != 0 else 'Unknown'
        },
        'soil_analysis': {
            'type': soil_type,
            'characteristics': {
                'Clay': 'High water retention, slow drainage, rich in nutrients',
                'Sandy': 'Fast drainage, low nutrient retention, easy to work',
                'Loam': 'Balanced drainage and nutrient retention, ideal for most crops',
                'Unknown': 'Soil characteristics unknown'
            }.get(soil_type, 'Specialized soil type'),
            'recommendation': {
                'Clay': 'Improve drainage with organic matter, avoid working when wet',
                'Sandy': 'Add organic matter and fertilizers frequently, use mulch to retain moisture',
                'Loam': 'Maintain with regular organic matter additions, monitor nutrient levels',
                'Unknown': 'Conduct soil test to determine characteristics and needs'
            }.get(soil_type, 'Consult local agricultural extension for specific recommendations')
        }
    }
    
    # Nutrient analysis
    insights['nutrient_analysis'] = {
        'fertilizer_status': 'Adequate' if fertilizer_used else 'Deficient',
        'nitrogen_need': 'High' if not fertilizer_used else 'Managed',
        'phosphorus_need': 'Medium',
        'potassium_need': 'Medium to High',
        'micronutrients': 'Check soil test for zinc, iron, manganese',
        'ph_level': 'Test recommended' if soil_type != 'Unknown' else 'Unknown'
    }
    
    # Water management analysis
    insights['water_management'] = {
        'irrigation_status': 'Adequate' if irrigation_used else 'Dependent on rainfall',
        'water_availability': 'Good' if (irrigation_used or rainfall > 600) else 'Marginal' if rainfall > 400 else 'Poor',
        'drainage_need': 'Improve' if soil_type == 'Clay' else 'Maintain' if soil_type == 'Loam' else 'Monitor' if soil_type == 'Sandy' else 'Unknown',
        'moisture_conservation': 'Mulching recommended' if rainfall < 600 else 'Standard practices adequate'
    }
    
    # Timing advice
    insights['timing_advice'] = {
        'days_to_harvest': days_to_harvest,
        'harvest_timing': 'Early harvest window' if days_to_harvest < 60 else 'Standard harvest window' if days_to_harvest < 120 else 'Extended growing season',
        'critical_periods': 'Monitor flowering and grain filling stages',
        'pest_pressure': 'High' if days_to_harvest > 120 else 'Moderate' if days_to_harvest > 80 else 'Low'
    }
    
    # Analyze key factors with more detail
    if rainfall < 300:
        insights['key_factors'].append("Critical: Low rainfall may severely limit yield potential")
    elif rainfall < 500:
        insights['key_factors'].append("Concern: Below optimal rainfall may reduce yield")
    elif rainfall > 1200:
        insights['key_factors'].append("Caution: Excessive rainfall may cause waterlogging and disease issues")
    elif rainfall > 1000:
        insights['key_factors'].append("Note: High rainfall may require improved drainage")
    else:
        insights['key_factors'].append("Positive: Rainfall is within optimal range for crop growth")
    
    if temperature < 10:
        insights['key_factors'].append("Critical: Cool temperatures may significantly slow growth and development")
    elif temperature < 15:
        insights['key_factors'].append("Concern: Cool temperatures may delay maturity")
    elif temperature > 38:
        insights['key_factors'].append("Critical: High temperatures may cause heat stress and reduce yields")
    elif temperature > 35:
        insights['key_factors'].append("Concern: Elevated temperatures may reduce quality and yield")
    else:
        insights['key_factors'].append("Positive: Temperature is within optimal range for crop development")
    
    if fertilizer_used:
        insights['key_factors'].append("Positive: Fertilizer application will support robust growth and development")
    else:
        insights['key_factors'].append("Critical: Lack of fertilizer may severely limit yield potential")
    
    if irrigation_used:
        insights['key_factors'].append("Positive: Irrigation provides consistent moisture for optimal growth")
    else:
        insights['key_factors'].append("Risk: Dependence on rainfall makes crop vulnerable to drought conditions")
    
    # Generate detailed recommendations
    if not fertilizer_used:
        insights['recommendations'].append("Critical: Apply balanced NPK fertilizer based on soil test results to optimize yield potential")
    
    if not irrigation_used and rainfall < 500:
        insights['recommendations'].append("Urgent: Implement irrigation system to supplement inadequate rainfall and ensure consistent moisture")
    elif not irrigation_used:
        insights['recommendations'].append("Recommendation: Consider supplemental irrigation for critical growth stages to maximize yield")
    
    if days_to_harvest < 60:
        insights['recommendations'].append("Priority: Monitor crop closely as harvest approaches to optimize timing")
    elif days_to_harvest > 150:
        insights['recommendations'].append("Planning: Prepare for extended growing season with additional pest and disease management")
    
    # Additional specific recommendations based on conditions
    if soil_type == 'Clay':
        insights['recommendations'].append("Soil Management: Add organic matter to improve drainage and workability of clay soil")
    elif soil_type == 'Sandy':
        insights['recommendations'].append("Soil Management: Add organic matter and consider frequent, light fertilization for sandy soil")
    
    if weather_condition == 'Sunny' and temperature > 32:
        insights['recommendations'].append("Heat Management: Consider shade cloth or increased irrigation frequency to reduce heat stress")
    elif weather_condition == 'Rainy' and not irrigation_used:
        insights['recommendations'].append("Water Management: Ensure adequate drainage to prevent waterlogging during rainy periods")
    
    # Risk assessment with detailed breakdown
    risk_factors = []
    if rainfall < 300 or rainfall > 1200:
        risk_factors.append("Water stress risk")
    if temperature < 10 or temperature > 40:
        risk_factors.append("Temperature stress risk")
    if not fertilizer_used:
        risk_factors.append("Nutrient deficiency risk")
    if not irrigation_used and rainfall < 400:
        risk_factors.append("Drought risk")
    
    if len(risk_factors) == 0:
        insights['risk_assessment'] = "Low risk - conditions are favorable for excellent yield with minimal concerns"
    elif len(risk_factors) == 1:
        insights['risk_assessment'] = f"Moderate risk - {risk_factors[0]} may impact yield. Proactive management recommended"
    elif len(risk_factors) == 2:
        insights['risk_assessment'] = f"High risk - {', '.join(risk_factors[:-1])} and {risk_factors[-1]} may significantly impact yield. Immediate action needed"
    else:
        insights['risk_assessment'] = f"Critical risk - Multiple factors ({', '.join(risk_factors)}) may severely impact yield. Comprehensive intervention required"
    
    # Comparison to benchmarks with more detail
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
        percentage_diff = ((predicted_yield/benchmark)-1)*100
        if predicted_yield > benchmark * 1.2:
            insights['comparison_to_benchmarks'] = f"Exceptional: Prediction is {percentage_diff:.1f}% above typical {crop} yield benchmark of {benchmark} tons/hectare"
        elif predicted_yield > benchmark * 1.1:
            insights['comparison_to_benchmarks'] = f"Excellent: Prediction is {percentage_diff:.1f}% above typical {crop} yield benchmark of {benchmark} tons/hectare"
        elif predicted_yield > benchmark * 1.05:
            insights['comparison_to_benchmarks'] = f"Very Good: Prediction is {percentage_diff:.1f}% above typical {crop} yield benchmark of {benchmark} tons/hectare"
        elif predicted_yield > benchmark * 0.95:
            insights['comparison_to_benchmarks'] = f"Good: Prediction is within 5% of typical {crop} yield benchmark of {benchmark} tons/hectare"
        elif predicted_yield > benchmark * 0.9:
            insights['comparison_to_benchmarks'] = f"Fair: Prediction is {abs(percentage_diff):.1f}% below typical {crop} yield benchmark of {benchmark} tons/hectare"
        elif predicted_yield > benchmark * 0.8:
            insights['comparison_to_benchmarks'] = f"Below Average: Prediction is {abs(percentage_diff):.1f}% below typical {crop} yield benchmark of {benchmark} tons/hectare"
        else:
            insights['comparison_to_benchmarks'] = f"Poor: Prediction is {abs(percentage_diff):.1f}% below typical {crop} yield benchmark of {benchmark} tons/hectare"
    else:
        insights['comparison_to_benchmarks'] = f"No benchmark available for {crop}. Regional or variety-specific benchmarks recommended"
    
    # Confidence level based on model performance with enhanced detail
    if metadata and 'model_performance' in metadata:
        r2_score = metadata['model_performance'].get('r2_score', 0)
        rmse = metadata['model_performance'].get('rmse', 0)
        if r2_score > 0.85:
            insights['confidence_level'] = f"High confidence (R²={r2_score:.3f}, RMSE={rmse:.2f}) - model explains variance very well"
        elif r2_score > 0.75:
            insights['confidence_level'] = f"Good confidence (R²={r2_score:.3f}, RMSE={rmse:.2f}) - model has strong predictive power"
        elif r2_score > 0.6:
            insights['confidence_level'] = f"Moderate confidence (R²={r2_score:.3f}, RMSE={rmse:.2f}) - model has reasonable predictive power but use with caution"
        else:
            insights['confidence_level'] = f"Low confidence (R²={r2_score:.3f}, RMSE={rmse:.2f}) - model predictions should be used cautiously and validated locally"
    else:
        insights['confidence_level'] = "Confidence level unavailable - no model performance data. Use predictions as general guidance only"
    
    return insights

def generate_rule_based_explanation(predicted_yield, input_data, insights):
    """
    Generate a natural language explanation based on rule-based logic.
    
    Args:
        predicted_yield (float): The predicted yield
        input_data (dict): Input data for the prediction
        insights (dict): Previously generated insights
        
    Returns:
        str: Natural language explanation of the prediction with detailed recommendations
    """
    # Rule-based explanation
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
    
    # Add confidence level
    explanation += "Model confidence: " + insights['confidence_level'] + ". "
    
    # Add enhanced recommendations
    if insights['recommendations']:
        explanation += "To optimize yield, consider the following recommendations: "
        for i, rec in enumerate(insights['recommendations']):
            explanation += f"{i+1}. {rec} "
    
    # Add additional general recommendations
    explanation += "Additional recommendations for yield optimization: "
    explanation += "1. Monitor soil moisture levels regularly. 2. Apply balanced fertilization based on soil tests. 3. Implement integrated pest management. 4. Consider crop rotation to improve soil health. 5. Keep detailed records for future planning. "
    
    # Add model information
    explanation += "This prediction is based on historical data patterns and should be used as a guide for planning purposes."
    
    return explanation

def predict_yield_enhanced(input_data):
    """
    Make an enhanced yield prediction with detailed explanations
    
    Args:
        input_data (dict): Dictionary with input features
        
    Returns:
        dict: Dictionary with prediction, insights, and explanation
    """
    # Load model and preprocessor
    model, preprocessor = load_enhanced_model()
    
    if model is None or preprocessor is None:
        return {
            'error': 'Failed to load model or preprocessor',
            'prediction': None,
            'insights': None,
            'explanation': None
        }
    
    # Load metadata
    metadata = get_prediction_metadata()
    
    try:
        # Preprocess the input data
        X_sample = preprocessor.preprocess_single_sample(input_data)
        
        # Make prediction
        yield_pred = model.predict(X_sample)
        predicted_yield = float(yield_pred[0])
        
        # Generate insights
        insights = generate_yield_insights(predicted_yield, input_data, metadata)
        
        # Generate explanation
        explanation = generate_rule_based_explanation(predicted_yield, input_data, insights)
        
        return {
            'prediction': predicted_yield,
            'insights': insights,
            'explanation': explanation,
            'input_data': input_data,
            'timestamp': timezone.now().isoformat()
        }
        
    except Exception as e:
        return {
            'error': f'Error making prediction: {str(e)}',
            'prediction': None,
            'insights': None,
            'explanation': None
        }

def demonstrate_predictions():
    """Demonstrate the enhanced prediction system with sample data"""
    
    print("=== Enhanced Crop Yield Prediction System ===")
    print("This system provides detailed explanations and recommendations for yield predictions.\n")
    
    # Sample data for demonstration
    sample_data_list = [
        {
            'Region': 'North',
            'Soil_Type': 'Clay',
            'Crop': 'Wheat',
            'Rainfall_mm': 450.0,
            'Temperature_Celsius': 22.0,
            'Fertilizer_Used': True,
            'Irrigation_Used': False,
            'Weather_Condition': 'Partly Cloudy',
            'Days_to_Harvest': 100
        },
        {
            'Region': 'South',
            'Soil_Type': 'Sandy',
            'Crop': 'Rice',
            'Rainfall_mm': 1200.0,
            'Temperature_Celsius': 30.0,
            'Fertilizer_Used': True,
            'Irrigation_Used': True,
            'Weather_Condition': 'Sunny',
            'Days_to_Harvest': 120
        },
        {
            'Region': 'West',
            'Soil_Type': 'Loam',
            'Crop': 'Maize',
            'Rainfall_mm': 800.0,
            'Temperature_Celsius': 26.0,
            'Fertilizer_Used': False,
            'Irrigation_Used': True,
            'Weather_Condition': 'Rainy',
            'Days_to_Harvest': 90
        }
    ]
    
    for i, sample_data in enumerate(sample_data_list, 1):
        print(f"\n--- Sample Prediction {i} ---")
        
        # Make prediction
        result = predict_yield_enhanced(sample_data)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            continue
        
        # Display results
        print(f"Input Data: {sample_data}")
        print(f"Predicted Yield: {result['prediction']:.2f} tons/hectare")
        
        print("\nKey Insights:")
        insights = result['insights']
        if insights:
            print(f"  Summary: {insights['prediction_summary']}")
            print(f"  Risk Assessment: {insights['risk_assessment']}")
            print(f"  Confidence: {insights['confidence_level']}")
            print(f"  Comparison: {insights['comparison_to_benchmarks']}")
            if insights['key_factors']:
                print("  Key Factors:")
                for factor in insights['key_factors']:
                    print(f"    - {factor}")
            if insights['recommendations']:
                print("  Recommendations:")
                for j, rec in enumerate(insights['recommendations'], 1):
                    print(f"    {j}. {rec}")
        
        print("\nDetailed Explanation:")
        print(result['explanation'])
        print("-" * 50)

def interactive_prediction():
    """Allow users to input their own data for prediction"""
    
    print("\n=== Interactive Yield Prediction ===")
    print("Enter your crop and environmental data for a detailed yield prediction.")
    
    # Get user input
    try:
        input_data = {
            'Region': input("Region (e.g., North, South, East, West): ") or "Unknown",
            'Soil_Type': input("Soil Type (e.g., Clay, Sandy, Loam): ") or "Unknown",
            'Crop': input("Crop (e.g., Wheat, Rice, Maize, Cotton): ") or "Unknown",
            'Rainfall_mm': float(input("Rainfall (mm): ") or "0"),
            'Temperature_Celsius': float(input("Temperature (°C): ") or "0"),
            'Fertilizer_Used': input("Fertilizer Used (True/False): ").lower() == 'true',
            'Irrigation_Used': input("Irrigation Used (True/False): ").lower() == 'true',
            'Weather_Condition': input("Weather Condition (e.g., Sunny, Rainy, Cloudy): ") or "Unknown",
            'Days_to_Harvest': int(input("Days to Harvest: ") or "0")
        }
        
        # Make prediction
        result = predict_yield_enhanced(input_data)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Display results
        print(f"\n--- Prediction Results ---")
        print(f"Predicted Yield: {result['prediction']:.2f} tons/hectare")
        
        print("\nKey Insights:")
        insights = result['insights']
        if insights:
            print(f"  Summary: {insights['prediction_summary']}")
            print(f"  Risk Assessment: {insights['risk_assessment']}")
            print(f"  Confidence: {insights['confidence_level']}")
            print(f"  Comparison: {insights['comparison_to_benchmarks']}")
            if insights['key_factors']:
                print("  Key Factors:")
                for factor in insights['key_factors']:
                    print(f"    - {factor}")
            if insights['recommendations']:
                print("  Recommendations:")
                for j, rec in enumerate(insights['recommendations'], 1):
                    print(f"    {j}. {rec}")
        
        print("\nDetailed Explanation:")
        print(result['explanation'])
        
    except ValueError as e:
        print(f"Invalid input: Please enter numeric values for rainfall, temperature, and days to harvest.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Demonstrate with sample data
    demonstrate_predictions()
    
    # Optionally allow interactive prediction
    user_choice = input("\nWould you like to make your own prediction? (y/n): ")
    if user_choice.lower() == 'y':
        interactive_prediction()