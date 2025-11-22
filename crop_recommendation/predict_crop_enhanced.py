"""
Enhanced prediction script for crop recommendation with detailed insights and explanations.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crop_recommendation.preprocessing.recommendation_preprocessor import RecommendationPreprocessor
from crop_recommendation.models.crop_recommendation_models import (
    CropRecommendationRandomForest,
    CropRecommendationXGBoost
)


def load_model_metadata():
    """Load the model metadata."""
    try:
        metadata_path = os.path.join('crop_recommendation', 'saved_models', 'crop_model_metadata.json')
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load model metadata: {e}")
        return {}


def get_prediction_metadata():
    """Get metadata about the prediction model."""
    return load_model_metadata()


def generate_crop_insights(crop, confidence, sample_data, metadata):
    """Generate detailed insights about the crop recommendation."""
    insights = {
        'prediction_summary': f'Recommended crop: {crop} (confidence: {confidence:.2%})',
        'key_factors': [],
        'recommendations': [],
        'risk_assessment': '',
        'comparison_to_benchmarks': '',
        'confidence_level': f'High ({confidence:.2%})' if confidence > 0.8 else f'Moderate ({confidence:.2%})' if confidence > 0.6 else f'Low ({confidence:.2%})'
    }
    
    # Analyze key factors
    n, p, k = sample_data['N'], sample_data['P'], sample_data['K']
    temp, humidity, rainfall = sample_data['temperature'], sample_data['humidity'], sample_data['rainfall']
    ph = sample_data['ph']
    
    # Nutrient analysis
    if n > 100:
        insights['key_factors'].append('High nitrogen levels suitable for nitrogen-demanding crops')
    elif n < 50:
        insights['key_factors'].append('Low nitrogen levels may limit crop growth')
    else:
        insights['key_factors'].append('Moderate nitrogen levels')
        
    if p > 50:
        insights['key_factors'].append('Adequate phosphorus for root development')
    elif p < 30:
        insights['key_factors'].append('Low phosphorus may affect flowering and fruiting')
    else:
        insights['key_factors'].append('Moderate phosphorus levels')
        
    if k > 50:
        insights['key_factors'].append('Good potassium levels for disease resistance')
    elif k < 30:
        insights['key_factors'].append('Low potassium may reduce stress tolerance')
    else:
        insights['key_factors'].append('Moderate potassium levels')
    
    # pH analysis
    if 6.0 <= ph <= 7.5:
        insights['key_factors'].append('Optimal soil pH for most crops')
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        insights['key_factors'].append('Slightly suboptimal pH, may affect nutrient availability')
    else:
        insights['key_factors'].append('Extreme pH levels, may require soil amendments')
    
    # Weather analysis
    if 20 <= temp <= 30:
        insights['key_factors'].append('Ideal temperature range for crop growth')
    else:
        insights['key_factors'].append('Temperature outside optimal range for some crops')
        
    if 60 <= humidity <= 80:
        insights['key_factors'].append('Good humidity levels for plant transpiration')
    else:
        insights['key_factors'].append('Humidity levels may affect plant water relations')
        
    if 100 <= rainfall <= 200:
        insights['key_factors'].append('Adequate rainfall for crop needs')
    elif rainfall < 50:
        insights['key_factors'].append('Low rainfall, irrigation may be needed')
    else:
        insights['key_factors'].append('High rainfall, ensure proper drainage')
    
    # Recommendations
    insights['recommendations'].append(f'Plant {crop} during the appropriate season for your region')
    insights['recommendations'].append('Monitor soil moisture and adjust irrigation as needed')
    insights['recommendations'].append('Regularly test soil to maintain optimal nutrient levels')
    
    if ph < 5.5 or ph > 8.0:
        insights['recommendations'].append('Consider liming or sulfur application to adjust soil pH')
    
    # Risk assessment
    risk_factors = []
    if temp < 10 or temp > 35:
        risk_factors.append('temperature stress')
    if rainfall < 30 or rainfall > 300:
        risk_factors.append('water stress')
    if n < 30 or p < 20 or k < 20:
        risk_factors.append('nutrient deficiency')
        
    if risk_factors:
        insights['risk_assessment'] = f'Potential risks: {", ".join(risk_factors)}. Monitor these factors closely.'
    else:
        insights['risk_assessment'] = 'Low risk factors identified. Growing conditions appear favorable.'
    
    # Comparison to benchmarks
    crop_benchmarks = {
        'rice': {'n': (100, 150), 'p': (50, 80), 'k': (50, 80), 'ph': (5.5, 7.0)},
        'wheat': {'n': (80, 120), 'p': (40, 60), 'k': (40, 60), 'ph': (6.0, 7.5)},
        'maize': {'n': (120, 180), 'p': (60, 90), 'k': (60, 90), 'ph': (5.8, 7.0)},
        'cotton': {'n': (100, 150), 'p': (40, 60), 'k': (80, 120), 'ph': (6.0, 8.5)},
        'sugarcane': {'n': (150, 250), 'p': (50, 80), 'k': (150, 250), 'ph': (6.5, 7.5)}
    }
    
    if crop.lower() in crop_benchmarks:
        benchmarks = crop_benchmarks[crop.lower()]
        comparison = f'{crop} benchmarks: N({benchmarks["n"][0]}-{benchmarks["n"][1]}), '
        comparison += f'P({benchmarks["p"][0]}-{benchmarks["p"][1]}), '
        comparison += f'K({benchmarks["k"][0]}-{benchmarks["k"][1]}), '
        comparison += f'pH({benchmarks["ph"][0]}-{benchmarks["ph"][1]})'
        insights['comparison_to_benchmarks'] = comparison
    
    return insights


def generate_llm_explanation(crop, confidence, sample_data, insights):
    """
    Generate a natural language explanation for the crop recommendation.
    In a real implementation, this would use an LLM.
    """
    explanation = f"Based on the comprehensive analysis of your soil and environmental conditions, {crop.title()} is the optimal crop recommendation for your farm with a confidence level of {confidence:.1%}.\n\n"
    
    explanation += "Detailed Analysis:\n"
    for factor in insights['key_factors']:
        explanation += f"- {factor}\n"
    
    explanation += f"\nWhy {crop.title()}?\n"
    crop_descriptions = {
        'rice': 'Rice is well-suited to your conditions with adequate water availability and warm temperatures. It\'s an excellent staple crop that can provide high yields in suitable environments.',
        'wheat': 'Wheat is ideal for your conditions with moderate temperatures and balanced nutrients. It\'s a versatile cereal crop that performs well in temperate climates.',
        'maize': 'Maize thrives in your warm, sunny conditions with good nutrient levels. It\'s a high-yielding crop that\'s valuable for both human consumption and animal feed.',
        'cotton': 'Cotton is well-matched to your warm climate and good soil conditions. It\'s an important fiber crop that can be very profitable with proper management.',
        'sugarcane': 'Sugarcane is perfectly suited to your high temperatures and adequate rainfall. It\'s a high-value crop that requires significant inputs but can provide excellent returns.'
    }
    explanation += crop_descriptions.get(crop.lower(), f'{crop.title()} is recommended based on the analysis of your soil and weather conditions.')
    
    explanation += f"\n\nRecommendations:\n"
    for rec in insights['recommendations']:
        explanation += f"- {rec}\n"
    
    explanation += f"\nRisk Assessment:\n{insights['risk_assessment']}"
    
    explanation += f"\n\nConfidence Level: {insights['confidence_level']}\n"
    explanation += "This recommendation is based on machine learning analysis of similar conditions and historical data."
    
    return explanation


def predict_crop_enhanced(sample_data):
    """Make an enhanced crop prediction with detailed insights."""
    try:
        # Load model and preprocessor
        model_path = os.path.join('crop_recommendation', 'saved_models', 'crop_model_enhanced.pkl')
        preprocessor_path = os.path.join('crop_recommendation', 'saved_models', 'crop_preprocessor_enhanced.pkl')
        
        # Try enhanced model first, fallback to regular model
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            model_path = os.path.join('crop_recommendation', 'saved_models', 'crop_model.pkl')
            preprocessor_path = os.path.join('crop_recommendation', 'saved_models', 'crop_preprocessor.pkl')
            print("Using regular model")
        else:
            print("Using enhanced model")
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor not found")
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Prepare data for prediction
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Create DataFrame with proper column names for preprocessing
        sample_df = pd.DataFrame([sample_data])
        
        # Preprocess the sample
        X_sample = preprocessor.preprocess_single_sample(sample_data)
        
        # Make prediction
        # Handle different model types
        if hasattr(model, 'predict_proba'):
            # For models with predict_proba method
            predictions = model.predict(X_sample)
            probabilities = model.predict_proba(X_sample)
            predicted_class = predictions[0]
            confidence = np.max(probabilities[0]) if probabilities.ndim > 1 else np.max(probabilities)
        else:
            # For models without predict_proba
            predictions = model.predict(X_sample)
            predicted_class = predictions[0]
            confidence = 0.85  # Default confidence for models without probability estimates
        
        # Decode the predicted class
        if 'crop_encoder' in preprocessor.label_encoders:
            crop = preprocessor.label_encoders['crop_encoder'].inverse_transform([predicted_class])[0]
        else:
            # Fallback to class names from metadata or default names
            metadata = get_prediction_metadata()
            class_names = metadata.get('classes', ['rice', 'wheat', 'maize', 'cotton', 'sugarcane'])
            crop = class_names[predicted_class] if predicted_class < len(class_names) else 'unknown'
        
        # Generate insights
        insights = generate_crop_insights(crop, confidence, sample_data, get_prediction_metadata())
        
        # Generate explanation
        explanation = generate_llm_explanation(crop, confidence, sample_data, insights)
        
        return {
            'recommended_crop': crop,
            'confidence': float(confidence),
            'insights': insights,
            'explanation': explanation
        }
        
    except Exception as e:
        print(f"Error in enhanced crop prediction: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to basic prediction
        return {
            'recommended_crop': 'wheat',
            'confidence': 0.75,
            'insights': {
                'prediction_summary': 'Fallback prediction: wheat',
                'key_factors': ['Error in enhanced prediction system'],
                'recommendations': ['Check system configuration'],
                'risk_assessment': 'System error occurred',
                'comparison_to_benchmarks': 'N/A',
                'confidence_level': 'Low (system error)'
            },
            'explanation': f'An error occurred in the enhanced prediction system: {e}. As a fallback, wheat is recommended as a versatile crop.'
        }


def interactive_crop_prediction():
    """Interactive mode for crop prediction."""
    print("=== Enhanced Crop Recommendation System ===")
    print("Enter the following soil and weather conditions:")
    
    try:
        sample_data = {
            'N': float(input("Nitrogen (N) level (0-140): ")),
            'P': float(input("Phosphorus (P) level (5-145): ")),
            'K': float(input("Potassium (K) level (5-205): ")),
            'temperature': float(input("Temperature (°C) (9-44): ")),
            'humidity': float(input("Humidity (%) (14-100): ")),
            'ph': float(input("Soil pH (3.5-10): ")),
            'rainfall': float(input("Rainfall (mm) (20-300): "))
        }
        
        result = predict_crop_enhanced(sample_data)
        
        print("\n=== Prediction Results ===")
        print(f"Recommended Crop: {result['recommended_crop']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nExplanation:\n{result['explanation']}")
        
        print("\n=== Detailed Insights ===")
        insights = result['insights']
        print(f"Prediction Summary: {insights['prediction_summary']}")
        print(f"Key Factors: {', '.join(insights['key_factors'])}")
        print(f"Recommendations: {', '.join(insights['recommendations'])}")
        print(f"Risk Assessment: {insights['risk_assessment']}")
        print(f"Confidence Level: {insights['confidence_level']}")
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage
    sample_data = {
        'N': 120,
        'P': 60,
        'K': 80,
        'temperature': 25,
        'humidity': 70,
        'ph': 6.5,
        'rainfall': 150
    }
    
    result = predict_crop_enhanced(sample_data)
    print("Enhanced Crop Prediction Result:")
    print(f"Recommended Crop: {result['recommended_crop']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Explanation: {result['explanation'][:200]}...")
    
    # Run interactive mode
    # interactive_crop_prediction()