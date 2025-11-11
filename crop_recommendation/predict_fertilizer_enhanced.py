"""
Enhanced prediction script for fertilizer recommendation with detailed insights and explanations.
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
from crop_recommendation.models.crop_recommendation_models import FertilizerRecommendationModel


def load_model_metadata():
    """Load the model metadata."""
    try:
        metadata_path = os.path.join('crop_recommendation', 'saved_models', 'fertilizer_model_metadata.json')
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load model metadata: {e}")
        return {}


def get_prediction_metadata():
    """Get metadata about the prediction model."""
    return load_model_metadata()


def generate_fertilizer_insights(fertilizer, confidence, sample_data, metadata):
    """Generate detailed insights about the fertilizer recommendation."""
    insights = {
        'prediction_summary': f'Recommended fertilizer: {fertilizer} (confidence: {confidence:.2%})',
        'key_factors': [],
        'recommendations': [],
        'risk_assessment': '',
        'comparison_to_benchmarks': '',
        'confidence_level': f'High ({confidence:.2%})' if confidence > 0.8 else f'Moderate ({confidence:.2%})' if confidence > 0.6 else f'Low ({confidence:.2%})'
    }
    
    # Analyze key factors
    n, p, k = sample_data['Nitrogen'], sample_data['Phosphorus'], sample_data['Potassium']
    ph, moisture = sample_data['pH'], sample_data['Moisture']
    crop_type_encoded = sample_data['Crop_Type_Encoded']
    
    # Nutrient deficiency analysis
    if n < 50:
        insights['key_factors'].append('Severe nitrogen deficiency detected')
    elif n < 80:
        insights['key_factors'].append('Moderate nitrogen deficiency')
    else:
        insights['key_factors'].append('Adequate nitrogen levels')
        
    if p < 30:
        insights['key_factors'].append('Severe phosphorus deficiency detected')
    elif p < 50:
        insights['key_factors'].append('Moderate phosphorus deficiency')
    else:
        insights['key_factors'].append('Adequate phosphorus levels')
        
    if k < 40:
        insights['key_factors'].append('Severe potassium deficiency detected')
    elif k < 60:
        insights['key_factors'].append('Moderate potassium deficiency')
    else:
        insights['key_factors'].append('Adequate potassium levels')
    
    # pH analysis
    if 6.0 <= ph <= 7.5:
        insights['key_factors'].append('Optimal soil pH for nutrient uptake')
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        insights['key_factors'].append('Slightly suboptimal pH, may affect nutrient availability')
    else:
        insights['key_factors'].append('Extreme pH levels, may significantly reduce fertilizer effectiveness')
    
    # Moisture analysis
    if moisture < 30:
        insights['key_factors'].append('Low soil moisture, may affect nutrient uptake')
    elif moisture > 70:
        insights['key_factors'].append('High soil moisture, may cause nutrient leaching')
    else:
        insights['key_factors'].append('Optimal soil moisture for nutrient availability')
    
    # Recommendations
    insights['recommendations'].append(f'Apply {fertilizer} according to crop requirements and soil test results')
    insights['recommendations'].append('Consider split applications for better nutrient uptake efficiency')
    insights['recommendations'].append('Monitor plant response and adjust application rates accordingly')
    
    if ph < 5.0 or ph > 8.5:
        insights['recommendations'].append('Consider soil amendments to optimize pH for better fertilizer effectiveness')
    
    # Risk assessment
    risk_factors = []
    if n < 30 or p < 20 or k < 20:
        risk_factors.append('severe nutrient deficiency')
    if moisture < 20 or moisture > 80:
        risk_factors.append('extreme soil moisture')
    if ph < 4.5 or ph > 9.0:
        risk_factors.append('extreme pH levels')
        
    if risk_factors:
        insights['risk_assessment'] = f'Potential risks: {", ".join(risk_factors)}. These factors may reduce fertilizer effectiveness.'
    else:
        insights['risk_assessment'] = 'Low risk factors identified. Conditions appear favorable for fertilizer application.'
    
    # Comparison to benchmarks
    fertilizer_benchmarks = {
        'Urea': {'n_content': 46, 'best_for': 'Leafy growth, chlorophyll production'},
        'DAP': {'p_content': 46, 'best_for': 'Root development, flowering'},
        'MOP': {'k_content': 60, 'best_for': 'Disease resistance, fruit quality'},
        'SSP': {'p_content': 16, 'ca_content': 12, 'best_for': 'Soil structure, root development'},
        'NPK 15-15-15': {'n_content': 15, 'p_content': 15, 'k_content': 15, 'best_for': 'Balanced nutrition'}
    }
    
    if fertilizer in fertilizer_benchmarks:
        benchmarks = fertilizer_benchmarks[fertilizer]
        comparison = f'{fertilizer} characteristics: '
        for key, value in benchmarks.items():
            comparison += f'{key}={value}, '
        insights['comparison_to_benchmarks'] = comparison.rstrip(', ')
    
    return insights


def generate_llm_explanation(fertilizer, confidence, sample_data, insights):
    """
    Generate a natural language explanation for the fertilizer recommendation.
    In a real implementation, this would use an LLM.
    """
    explanation = f"Based on the comprehensive analysis of your soil conditions, {fertilizer} is the optimal fertilizer recommendation with a confidence level of {confidence:.1%}.\n\n"
    
    explanation += "Detailed Analysis:\n"
    for factor in insights['key_factors']:
        explanation += f"- {factor}\n"
    
    explanation += f"\nWhy {fertilizer}?\n"
    fertilizer_descriptions = {
        'Urea': 'Urea is recommended due to the nitrogen deficiency in your soil. With 46% nitrogen content, it will effectively address the nitrogen shortage and promote healthy leafy growth and chlorophyll production.',
        'DAP': 'DAP (Diammonium Phosphate) is recommended because of the phosphorus deficiency. Its high phosphorus content (46% P2O5) will support root development and flowering, which are crucial for crop establishment.',
        'MOP': 'Muriate of Potash (MOP) is recommended due to potassium deficiency. With 60% K2O content, it will improve disease resistance and enhance fruit quality.',
        'SSP': 'Single Super Phosphate is recommended for your soil conditions. It provides phosphorus (16% P2O5) and calcium (12% CaO) to improve soil structure and root development.',
        'NPK 15-15-15': 'Balanced NPK fertilizer is recommended because your soil shows relatively balanced nutrient levels. This complete fertilizer will provide all essential nutrients in equal proportions for overall plant health.'
    }
    explanation += fertilizer_descriptions.get(fertilizer, f'{fertilizer} is recommended based on the analysis of your soil conditions.')
    
    explanation += f"\n\nApplication Recommendations:\n"
    for rec in insights['recommendations']:
        explanation += f"- {rec}\n"
    
    explanation += f"\nRisk Assessment:\n{insights['risk_assessment']}"
    
    explanation += f"\n\nConfidence Level: {insights['confidence_level']}\n"
    explanation += "This recommendation is based on machine learning analysis of similar conditions and historical data."
    
    return explanation


def predict_fertilizer_enhanced(sample_data):
    """Make an enhanced fertilizer prediction with detailed insights."""
    try:
        # Load model and preprocessor
        model_path = os.path.join('crop_recommendation', 'saved_models', 'fertilizer_model_enhanced.pkl')
        preprocessor_path = os.path.join('crop_recommendation', 'saved_models', 'fertilizer_preprocessor_enhanced.pkl')
        
        # Try enhanced model first, fallback to regular model
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            model_path = os.path.join('crop_recommendation', 'saved_models', 'fertilizer_model.pkl')
            preprocessor_path = os.path.join('crop_recommendation', 'saved_models', 'fertilizer_preprocessor.pkl')
            print("Using regular model")
        else:
            print("Using enhanced model")
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor not found")
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Prepare data for prediction
        feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Crop_Type_Encoded', 'Moisture']
        sample_array = np.array([[sample_data[feature] for feature in feature_names]])
        
        # Create DataFrame with proper column names for preprocessing
        sample_df = pd.DataFrame(sample_array, columns=feature_names)
        
        # Preprocess the sample
        X_sample = preprocessor.preprocess_single_sample(sample_data)
        
        # Make prediction
        predictions, probabilities = model.predict(X_sample)
        predicted_class = predictions[0]
        confidence = np.max(probabilities)
        
        # Decode the predicted class
        if hasattr(preprocessor.label_encoders, 'fertilizer_encoder'):
            fertilizer = preprocessor.label_encoders['fertilizer_encoder'].inverse_transform([predicted_class])[0]
        else:
            # Fallback to class names from metadata or default names
            metadata = get_prediction_metadata()
            class_names = metadata.get('classes', ['Urea', 'DAP', 'MOP', 'SSP', 'NPK 15-15-15'])
            fertilizer = class_names[predicted_class] if predicted_class < len(class_names) else 'NPK 15-15-15'
        
        # Generate insights
        insights = generate_fertilizer_insights(fertilizer, confidence, sample_data, get_prediction_metadata())
        
        # Generate explanation
        explanation = generate_llm_explanation(fertilizer, confidence, sample_data, insights)
        
        return {
            'recommended_fertilizer': fertilizer,
            'confidence': float(confidence),
            'insights': insights,
            'explanation': explanation
        }
        
    except Exception as e:
        print(f"Error in enhanced fertilizer prediction: {e}")
        # Fallback to basic prediction
        return {
            'recommended_fertilizer': 'NPK 15-15-15',
            'confidence': 0.75,
            'insights': {
                'prediction_summary': 'Fallback prediction: NPK 15-15-15',
                'key_factors': ['Error in enhanced prediction system'],
                'recommendations': ['Check system configuration'],
                'risk_assessment': 'System error occurred',
                'comparison_to_benchmarks': 'N/A',
                'confidence_level': 'Low (system error)'
            },
            'explanation': f'An error occurred in the enhanced prediction system: {e}. As a fallback, NPK 15-15-15 is recommended as a balanced fertilizer.'
        }


def interactive_fertilizer_prediction():
    """Interactive mode for fertilizer prediction."""
    print("=== Enhanced Fertilizer Recommendation System ===")
    print("Enter the following soil conditions:")
    
    try:
        sample_data = {
            'Nitrogen': float(input("Nitrogen (N) level (0-140): ")),
            'Phosphorus': float(input("Phosphorus (P) level (5-145): ")),
            'Potassium': float(input("Potassium (K) level (5-205): ")),
            'pH': float(input("Soil pH (3.5-10): ")),
            'Crop_Type_Encoded': int(input("Crop Type Encoded (0-4): ")),
            'Moisture': float(input("Soil Moisture (%) (20-80): "))
        }
        
        result = predict_fertilizer_enhanced(sample_data)
        
        print("\n=== Prediction Results ===")
        print(f"Recommended Fertilizer: {result['recommended_fertilizer']}")
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
        'Nitrogen': 40,
        'Phosphorus': 35,
        'Potassium': 60,
        'pH': 6.5,
        'Crop_Type_Encoded': 1,
        'Moisture': 65
    }
    
    result = predict_fertilizer_enhanced(sample_data)
    print("Enhanced Fertilizer Prediction Result:")
    print(f"Recommended Fertilizer: {result['recommended_fertilizer']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Explanation: {result['explanation'][:200]}...")
    
    # Run interactive mode
    # interactive_fertilizer_prediction()