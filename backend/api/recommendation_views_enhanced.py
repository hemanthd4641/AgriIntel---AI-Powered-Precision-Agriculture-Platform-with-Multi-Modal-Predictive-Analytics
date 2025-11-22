"""
Enhanced views for crop and fertilizer recommendation API endpoints with detailed insights and explanations.
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
import joblib
import numpy as np
import os
import sys
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .models import CropRecommendation, FertilizerRecommendation
from .recommendation_serializers import (
    CropRecommendationSerializer, 
    FertilizerRecommendationSerializer,
    RecommendationRequestSerializer
)

# Global variables for enhanced models (loaded once when module is imported)
crop_model_enhanced = None
crop_preprocessor_enhanced = None
fertilizer_model_enhanced = None
fertilizer_preprocessor_enhanced = None


def load_enhanced_models():
    """Load enhanced recommendation models and preprocessors if not already loaded."""
    global crop_model_enhanced, crop_preprocessor_enhanced, fertilizer_model_enhanced, fertilizer_preprocessor_enhanced
    
    # Load enhanced crop recommendation model
    if crop_model_enhanced is None:
        try:
            crop_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                          'crop_recommendation', 'saved_models', 'crop_model_enhanced.pkl')
            if os.path.exists(crop_model_path):
                crop_model_enhanced = joblib.load(crop_model_path)
                print("Enhanced crop recommendation model loaded successfully")
            else:
                print(f"Enhanced crop model file not found at {crop_model_path}")
        except Exception as e:
            print(f"Error loading enhanced crop model: {e}")
            crop_model_enhanced = None
    
    # Load enhanced crop preprocessor
    if crop_preprocessor_enhanced is None:
        try:
            preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           'crop_recommendation', 'saved_models', 'crop_preprocessor_enhanced.pkl')
            if os.path.exists(preprocessor_path):
                crop_preprocessor_enhanced = joblib.load(preprocessor_path)
                print("Enhanced crop preprocessor loaded successfully")
            else:
                print(f"Enhanced crop preprocessor file not found at {preprocessor_path}")
        except Exception as e:
            print(f"Error loading enhanced crop preprocessor: {e}")
            crop_preprocessor_enhanced = None
    
    # Load enhanced fertilizer recommendation model
    if fertilizer_model_enhanced is None:
        try:
            fertilizer_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                               'crop_recommendation', 'saved_models', 'fertilizer_model_enhanced.pkl')
            if os.path.exists(fertilizer_model_path):
                fertilizer_model_enhanced = joblib.load(fertilizer_model_path)
                print("Enhanced fertilizer recommendation model loaded successfully")
            else:
                print(f"Enhanced fertilizer model file not found at {fertilizer_model_path}")
        except Exception as e:
            print(f"Error loading enhanced fertilizer model: {e}")
            fertilizer_model_enhanced = None
    
    # Load enhanced fertilizer preprocessor
    if fertilizer_preprocessor_enhanced is None:
        try:
            fertilizer_preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                      'crop_recommendation', 'saved_models', 'fertilizer_preprocessor_enhanced.pkl')
            if os.path.exists(fertilizer_preprocessor_path):
                fertilizer_preprocessor_enhanced = joblib.load(fertilizer_preprocessor_path)
                print("Enhanced fertilizer preprocessor loaded successfully")
            else:
                print(f"Enhanced fertilizer preprocessor file not found at {fertilizer_preprocessor_path}")
        except Exception as e:
            print(f"Error loading enhanced fertilizer preprocessor: {e}")
            fertilizer_preprocessor_enhanced = None

# Preload enhanced models when module is imported
print("Preloading enhanced recommendation models and components...")
load_enhanced_models()
print("Enhanced recommendation models and components preloaded successfully")


def load_model_metadata(model_type):
    """Load the model metadata."""
    try:
        if model_type == 'crop':
            metadata_path = os.path.join('crop_recommendation', 'saved_models', 'crop_model_metadata.json')
        elif model_type == 'fertilizer':
            metadata_path = os.path.join('crop_recommendation', 'saved_models', 'fertilizer_model_metadata.json')
        else:
            return {}
            
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {model_type} model metadata: {e}")
        return {}


def generate_crop_insights_enhanced(crop, confidence, sample_data, metadata):
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


def generate_fertilizer_insights_enhanced(fertilizer, confidence, sample_data, metadata):
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


def generate_llm_explanation_enhanced(crop, fertilizer, confidence, sample_data, crop_insights, fertilizer_insights, location=None, season=None):
    """
    Generate a natural language explanation for the recommendations using LLM.
    """
    # Try to use the explainable AI system with LLM, including location/season in the prompt
    try:
        from explainable_ai.llm_interface import AgricultureLLM
        from explainable_ai.rag_system import AgricultureRAG

        # Initialize LLM and RAG system
        llm = AgricultureLLM(model_name="microsoft/Phi-3-mini-4k-instruct")
        rag_system = AgricultureRAG()

        if llm.text_generator:
            # Create a detailed prompt for the LLM that includes location and season
            soil_conditions = f"N:{sample_data.get('N', sample_data.get('Nitrogen', 'N/A'))}, P:{sample_data.get('P', sample_data.get('Phosphorus', 'N/A'))}, K:{sample_data.get('K', sample_data.get('Potassium', 'N/A'))}, pH:{sample_data.get('ph', sample_data.get('pH', 'N/A'))}"
            weather_conditions = f"Temperature:{sample_data.get('temperature', 'N/A')}°C, Humidity:{sample_data.get('humidity', sample_data.get('Moisture', 'N/A'))}%, Rainfall:{sample_data.get('rainfall', 'N/A')}mm"
            location_info = f"Location: {location or 'Unknown'}, Season: {season or 'Unknown'}"

            prompt = f"""
You are an agricultural extension expert. Provide concise, actionable, and location-specific recommendations for a farmer.

{location_info}
Crop Recommendation: {crop} (confidence: {confidence:.1%})
Fertilizer Recommendation: {fertilizer}

Soil Conditions: {soil_conditions}
Weather Conditions: {weather_conditions}

Using local best practices for the specified location and season, provide:
1) A short explanation (2-3 sentences) why the crop and fertilizer were recommended for this location
2) Three specific, practical actions the farmer should take this season in this location
3) Any local considerations (pests, water management, common local constraints)

Be concise and avoid generic statements. Where possible, cite best-practice steps.
"""

            # Use RAG to ground recommendations with local docs (if available)
            try:
                # Use the correct method name from the RAG system
                rag_context = rag_system.query_knowledge_base(prompt, k=3)
            except Exception:
                rag_context = None

            if rag_context:
                # Append retrieved context to the prompt to ground the LLM
                context_text = "\n".join([doc.get('content', '')[:200] for doc in rag_context[:2]])
                prompt = prompt + "\n\nLocal reference:\n" + context_text

            response = llm.text_generator(
                prompt,
                max_new_tokens=300,
                temperature=0.6,
                do_sample=False
            )

            # Strip the prompt from generated text and return the new portion
            try:
                generated = response[0]['generated_text']
                # Remove prompt prefix if present
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                return generated
            except Exception:
                return str(response)
        else:
            # Fallback to rule-based explanation if LLM is not available
            return generate_rule_based_explanation(crop, fertilizer, confidence, sample_data, crop_insights, fertilizer_insights)
    except Exception as e:
        print(f"Error generating LLM-based explanation: {e}")
        # Fallback to rule-based explanation if LLM fails
        return generate_rule_based_explanation(crop, fertilizer, confidence, sample_data, crop_insights, fertilizer_insights)


def generate_rule_based_explanation(crop, fertilizer, confidence, sample_data, crop_insights, fertilizer_insights):
    """
    Generate rule-based explanation (fallback when LLM is not available).
    """
    explanation = f"Based on the comprehensive analysis of your soil and environmental conditions, {crop.title()} is the optimal crop recommendation with a confidence level of {confidence:.1%}.\n\n"
    
    explanation += "🌾 Crop Recommendation Analysis:\n"
    for factor in crop_insights['key_factors']:
        explanation += f"- {factor}\n"
    
    explanation += f"\n✅ Why {crop.title()}?\n"
    crop_descriptions = {
        'rice': 'Rice is well-suited to your conditions with adequate water availability and warm temperatures. It\'s an excellent staple crop that can provide high yields in suitable environments.',
        'wheat': 'Wheat is ideal for your conditions with moderate temperatures and balanced nutrients. It\'s a versatile cereal crop that performs well in temperate climates.',
        'maize': 'Maize thrives in your warm, sunny conditions with good nutrient levels. It\'s a high-yielding crop that\'s valuable for both human consumption and animal feed.',
        'cotton': 'Cotton is well-matched to your warm climate and good soil conditions. It\'s an important fiber crop that can be very profitable with proper management.',
        'sugarcane': 'Sugarcane is perfectly suited to your high temperatures and adequate rainfall. It\'s a high-value crop that requires significant inputs but can provide excellent returns.'
    }
    explanation += crop_descriptions.get(crop.lower(), f'{crop.title()} is recommended based on the analysis of your soil and weather conditions.')
    
    explanation += f"\n\n🧪 Fertilizer Recommendation: {fertilizer}\n"
    explanation += "Fertilizer Analysis:\n"
    for factor in fertilizer_insights['key_factors']:
        explanation += f"- {factor}\n"
    
    explanation += f"\n✅ Why {fertilizer}?\n"
    fertilizer_descriptions = {
        'Urea': 'Urea is recommended due to the nitrogen deficiency in your soil. With 46% nitrogen content, it will effectively address the nitrogen shortage and promote healthy leafy growth and chlorophyll production.',
        'DAP': 'Diammonium Phosphate is recommended because of the phosphorus deficiency. Its high phosphorus content (46% P2O5) will support root development and flowering, which are crucial for crop establishment.',
        'MOP': 'Muriate of Potash (MOP) is recommended due to potassium deficiency. With 60% K2O content, it will improve disease resistance and enhance fruit quality.',
        'SSP': 'Single Super Phosphate is recommended for your soil conditions. It provides phosphorus (16% P2O5) and calcium (12% CaO) to improve soil structure and root development.',
        'NPK 15-15-15': 'Balanced NPK fertilizer is recommended because your soil shows relatively balanced nutrient levels. This complete fertilizer will provide all essential nutrients in equal proportions for overall plant health.'
    }
    explanation += fertilizer_descriptions.get(fertilizer, f'{fertilizer} is recommended based on the analysis of your soil conditions.')
    
    explanation += f"\n\n📋 Crop Recommendations:\n"
    for rec in crop_insights['recommendations']:
        explanation += f"- {rec}\n"
    
    explanation += f"\n📋 Fertilizer Recommendations:\n"
    for rec in fertilizer_insights['recommendations']:
        explanation += f"- {rec}\n"
    
    explanation += f"\n⚠️ Risk Assessment:\n{crop_insights['risk_assessment']}"
    
    explanation += f"\n\n📊 Confidence Level: {crop_insights['confidence_level']}\n"
    explanation += "This recommendation is based on machine learning analysis of similar conditions and historical data."
    
    return explanation


@api_view(['POST'])
def get_enhanced_recommendations(request):
    """
    Get enhanced crop and fertilizer recommendations with detailed insights and explanations.
    
    POST Parameters:
        soil_nitrogen: Soil nitrogen level (kg/ha)
        soil_phosphorus: Soil phosphorus level (kg/ha)
        soil_potassium: Soil potassium level (kg/ha)
        soil_ph: Soil pH level
        temperature: Temperature (°C)
        humidity: Humidity (%)
        rainfall: Rainfall (mm)
        location: Location (state/region)
        season: Season (optional)
        
    Returns:
        JSON response with enhanced crop and fertilizer recommendations
    """
    try:
        # Load enhanced models
        load_enhanced_models()
        
        # Validate request data
        serializer = RecommendationRequestSerializer(data=request.data)
        if not serializer.is_valid():
            # If validation fails because of missing farm_id, we'll create a default one
            data_copy = request.data.copy()
            if 'farm_id' not in data_copy:
                data_copy['farm_id'] = 'default_farm'
            serializer = RecommendationRequestSerializer(data=data_copy)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract validated data
        soil_nitrogen = serializer.validated_data['soil_nitrogen']
        soil_phosphorus = serializer.validated_data['soil_phosphorus']
        soil_potassium = serializer.validated_data['soil_potassium']
        soil_ph = serializer.validated_data['soil_ph']
        temperature = serializer.validated_data['temperature']
        humidity = serializer.validated_data['humidity']
        rainfall = serializer.validated_data['rainfall']
        location = serializer.validated_data.get('location', '')
        season = serializer.validated_data.get('season', '')
        
        # Prepare sample data for crop prediction
        crop_sample_data = {
            'N': soil_nitrogen,
            'P': soil_phosphorus,
            'K': soil_potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': soil_ph,
            'rainfall': rainfall
        }
        
        # Crop prediction using enhanced model
        if crop_model_enhanced is None or crop_preprocessor_enhanced is None:
            return Response({'error': 'Enhanced crop recommendation system not available'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        try:
            # Preprocess the sample
            X_crop_sample = crop_preprocessor_enhanced.preprocess_single_sample(crop_sample_data)
            # Diagnostic log: capture preprocessed features
            try:
                print(f"[DIAG] crop_sample_data={crop_sample_data}")
                # If X_crop_sample is a numpy array or DataFrame, summarize it
                try:
                    if hasattr(X_crop_sample, 'shape'):
                        print(f"[DIAG] X_crop_sample ndarray shape={X_crop_sample.shape} values={str(X_crop_sample)}")
                    else:
                        print(f"[DIAG] X_crop_sample repr={repr(X_crop_sample)}")
                except Exception:
                    print(f"[DIAG] X_crop_sample repr={repr(X_crop_sample)}")
            except Exception as _e:
                print(f"[DIAG] Error logging X_crop_sample: {_e}")
            
            # Make prediction - handle different return patterns
            crop_prediction_result = crop_model_enhanced.predict(X_crop_sample)
            # Diagnostic log: raw model output
            try:
                print(f"[DIAG] crop_prediction_result repr={repr(crop_prediction_result)}")
            except Exception:
                pass
            
            # Handle case where predict returns only predictions, not probabilities
            if isinstance(crop_prediction_result, tuple) and len(crop_prediction_result) == 2:
                crop_predictions, crop_probabilities = crop_prediction_result
            else:
                # If only predictions are returned, use them and create default probabilities
                crop_predictions = crop_prediction_result
                # Create default probabilities (uniform distribution)
                unique_classes = len(np.unique(crop_predictions)) if hasattr(crop_predictions, '__len__') else 5
                crop_probabilities = np.full((1, unique_classes), 1.0/unique_classes)
            
            predicted_crop_class = crop_predictions[0] if hasattr(crop_predictions, '__len__') else crop_predictions
            # Compute confidence robustly
            try:
                crop_confidence = np.max(crop_probabilities) if hasattr(crop_probabilities, 'ndim') and crop_probabilities.ndim > 1 else (np.max(crop_probabilities) if hasattr(crop_probabilities, '__len__') else 0.85)
            except Exception:
                crop_confidence = 0.85

            # Diagnostic logging for prediction decode
            try:
                print(f"[DIAG] predicted_crop_class={predicted_crop_class}")
                print(f"[DIAG] crop_probabilities shape={getattr(crop_probabilities, 'shape', None)} values={repr(crop_probabilities)}")
            except Exception:
                pass
            
            # Decode the predicted class
            if hasattr(crop_preprocessor_enhanced.label_encoders, 'crop_encoder'):
                crop_name = crop_preprocessor_enhanced.label_encoders['crop_encoder'].inverse_transform([predicted_crop_class])[0]
            else:
                # Fallback to class names from metadata or default names
                metadata = load_model_metadata('crop')
                class_names = metadata.get('classes', ['rice', 'wheat', 'maize', 'cotton', 'sugarcane'])
                # Handle case where predicted_crop_class might be a tuple/list
                crop_index = predicted_crop_class[0] if isinstance(predicted_crop_class, (list, tuple)) else predicted_crop_class
                crop_name = class_names[int(crop_index)] if int(crop_index) < len(class_names) else 'wheat'
        except Exception as e:
            return Response({'error': f'Error predicting crop: {str(e)}'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Prepare sample data for fertilizer prediction
        fertilizer_sample_data = {
            'Nitrogen': soil_nitrogen,
            'Phosphorus': soil_phosphorus,
            'Potassium': soil_potassium,
            'pH': soil_ph,
            'Crop_Type_Encoded': float(predicted_crop_class[0]) if isinstance(predicted_crop_class, (list, tuple)) else float(predicted_crop_class),  # Use predicted crop class
            'Moisture': humidity  # Using humidity as a proxy for moisture
        }
        
        # Fertilizer prediction using enhanced model
        if fertilizer_model_enhanced is None or fertilizer_preprocessor_enhanced is None:
            return Response({'error': 'Enhanced fertilizer recommendation system not available'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        try:
            # Preprocess the sample
            X_fertilizer_sample = fertilizer_preprocessor_enhanced.preprocess_single_sample(fertilizer_sample_data)
            # Diagnostic log: capture fertilizer preprocessed features
            try:
                print(f"[DIAG] fertilizer_sample_data={fertilizer_sample_data}")
                try:
                    if hasattr(X_fertilizer_sample, 'shape'):
                        print(f"[DIAG] X_fertilizer_sample ndarray shape={X_fertilizer_sample.shape} values={str(X_fertilizer_sample)}")
                    else:
                        print(f"[DIAG] X_fertilizer_sample repr={repr(X_fertilizer_sample)}")
                except Exception:
                    print(f"[DIAG] X_fertilizer_sample repr={repr(X_fertilizer_sample)}")
            except Exception as _e:
                print(f"[DIAG] Error logging X_fertilizer_sample: {_e}")
            
            # Make prediction - handle different return patterns
            fertilizer_prediction_result = fertilizer_model_enhanced.predict(X_fertilizer_sample)
            # Diagnostic log: raw fertilizer model output
            try:
                print(f"[DIAG] fertilizer_prediction_result repr={repr(fertilizer_prediction_result)}")
            except Exception:
                pass
            
            # Handle case where predict returns only predictions, not probabilities
            if isinstance(fertilizer_prediction_result, tuple) and len(fertilizer_prediction_result) == 2:
                fertilizer_predictions, fertilizer_probabilities = fertilizer_prediction_result
            else:
                # If only predictions are returned, use them and create default probabilities
                fertilizer_predictions = fertilizer_prediction_result
                # Create default probabilities (uniform distribution)
                unique_classes = len(np.unique(fertilizer_predictions)) if hasattr(fertilizer_predictions, '__len__') else 5
                fertilizer_probabilities = np.full((1, unique_classes), 1.0/unique_classes)
            
            predicted_fertilizer_class = fertilizer_predictions[0] if hasattr(fertilizer_predictions, '__len__') else fertilizer_predictions
            try:
                fertilizer_confidence = np.max(fertilizer_probabilities) if hasattr(fertilizer_probabilities, 'ndim') and fertilizer_probabilities.ndim > 1 else (np.max(fertilizer_probabilities) if hasattr(fertilizer_probabilities, '__len__') else 0.85)
            except Exception:
                fertilizer_confidence = 0.85

            # Diagnostic logging for fertilizer decode
            try:
                print(f"[DIAG] predicted_fertilizer_class={predicted_fertilizer_class}")
                print(f"[DIAG] fertilizer_probabilities shape={getattr(fertilizer_probabilities, 'shape', None)} values={repr(fertilizer_probabilities)}")
            except Exception:
                pass
            
            # Decode the predicted class
            if hasattr(fertilizer_preprocessor_enhanced.label_encoders, 'fertilizer_encoder'):
                fertilizer_name = fertilizer_preprocessor_enhanced.label_encoders['fertilizer_encoder'].inverse_transform([predicted_fertilizer_class])[0]
            else:
                # Fallback to class names from metadata or default names
                metadata = load_model_metadata('fertilizer')
                class_names = metadata.get('classes', ['Urea', 'DAP', 'MOP', 'SSP', 'NPK 15-15-15'])
                # Handle case where predicted_fertilizer_class might be a tuple/list
                fertilizer_index = predicted_fertilizer_class[0] if isinstance(predicted_fertilizer_class, (list, tuple)) else predicted_fertilizer_class
                fertilizer_name = class_names[int(fertilizer_index)] if int(fertilizer_index) < len(class_names) else 'NPK 15-15-15'
        except Exception as e:
            return Response({'error': f'Error predicting fertilizer: {str(e)}'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Generate insights
        crop_metadata = load_model_metadata('crop')
        fertilizer_metadata = load_model_metadata('fertilizer')
        
        crop_insights = generate_crop_insights_enhanced(crop_name, crop_confidence, crop_sample_data, crop_metadata)
        fertilizer_insights = generate_fertilizer_insights_enhanced(fertilizer_name, fertilizer_confidence, fertilizer_sample_data, fertilizer_metadata)
        
        # Generate explanation using LLM when available
        explanation = generate_llm_explanation_enhanced(
            crop_name,
            fertilizer_name,
            crop_confidence,
            crop_sample_data,
            crop_insights,
            fertilizer_insights,
            location=location,
            season=season
        )
        
        # Calculate fertilizer quantity (simplified)
        base_quantity = 100
        if soil_nitrogen < 50:
            base_quantity += 30
        if soil_phosphorus < 30:
            base_quantity += 20
        if soil_potassium < 50:
            base_quantity += 25
        crop_factors = {
            'rice': 1.2, 'wheat': 1.0, 'maize': 1.1, 'cotton': 1.3, 'sugarcane': 1.5
        }
        factor = crop_factors.get(crop_name.lower(), 1.0)
        quantity = int(base_quantity * factor)
        
        # Prepare response
        recommendation = {
            'rank': 1,
            'crop': crop_name,
            'crop_confidence': float(crop_confidence),
            'fertilizer': fertilizer_name,
            'fertilizer_confidence': float(fertilizer_confidence),
            'quantity_kg_per_ha': float(quantity),
            'explanation': explanation,
            'crop_insights': crop_insights,
            'fertilizer_insights': fertilizer_insights
        }
        
        # Save recommendations to database for reports and charts
        try:
            crop_recommendation = CropRecommendation.objects.create(
                recommended_crop=crop_name,
                confidence_score=float(crop_confidence),
                soil_nitrogen=soil_nitrogen,
                soil_phosphorus=soil_phosphorus,
                soil_potassium=soil_potassium,
                soil_ph=soil_ph,
                temperature=temperature,
                humidity=humidity,
                rainfall=rainfall
            )
            
            # Save associated fertilizer recommendation
            fertilizer_record = FertilizerRecommendation.objects.create(
                crop_recommendation=crop_recommendation,
                recommended_fertilizer=fertilizer_name,
                quantity_kg_per_ha=float(quantity)
            )
            
            # Add database ID to response
            recommendation['id'] = crop_recommendation.id
        except Exception as e:
            print(f"Error saving recommendations to database: {e}")
            # Continue with recommendations even if saving fails
        
        return Response({
            'location': location,
            'season': season,
            'recommendation': recommendation
        })
    except Exception as e:
        # Log the error for debugging
        print(f"Error in get_enhanced_recommendations: {str(e)}")
        # Return a proper error response
        return Response({'error': f'Internal server error: {str(e)}'}, 
                       status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_recommendation_history(request):
    """
    Get recommendation history (dummy implementation since we're removing farm dependencies).
    
    Args:
        request: HTTP request object
        
    Returns:
        Response: JSON response with recommendation history
    """
    # Return empty history since we're not tracking farms anymore
    return Response({
        'history': [],
        'count': 0
    })
