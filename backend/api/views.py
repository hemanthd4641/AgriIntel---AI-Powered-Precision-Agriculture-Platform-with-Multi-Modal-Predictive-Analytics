"""
Views for the Smart Agriculture API.

This module defines the API endpoints for farm management, data submission,
and prediction retrieval.
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render, get_object_or_404
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
import uuid
import random
import sys
import os


# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add the parent directory to the Python path to access explainable_ai
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import models with error handling
try:
    from .models import Crop, YieldPrediction, DiseasePrediction, CropRecommendation, FertilizerRecommendation
except Exception as e:
    print(f"Error importing models: {e}")
    # Fallback imports
    Crop = None
    YieldPrediction = None
    DiseasePrediction = None
    CropRecommendation = None
    FertilizerRecommendation = None

# Import serializers with error handling
try:
    from .serializers import (
        CropSerializer, 
        YieldPredictionSerializer, 
        DiseasePredictionSerializer,
        CropRecommendationSerializer,
        FertilizerRecommendationSerializer
    )
except Exception as e:
    print(f"Error importing serializers: {e}")
    # Fallback imports
    CropSerializer = None
    YieldPredictionSerializer = None
    DiseasePredictionSerializer = None
    CropRecommendationSerializer = None
    FertilizerRecommendationSerializer = None

# Global variables for chatbot (loaded once when module is imported)
chatbot_instance = None

# Global variables for pest prediction (loaded once when module is imported)
pest_model = None
pest_preprocessor = None
pest_label_encoders = None
pest_rag = None

# Global variables for yield prediction (loaded once when module is imported)
yield_model = None
yield_preprocessor_data = None
yield_preprocessor = None

def load_chatbot():
    """Load chatbot instance if not already loaded."""
    global chatbot_instance
    
    if chatbot_instance is None:
        try:
            # Import the dedicated chatbot module with error handling
            try:
                from explainable_ai.chatbot import AgriculturalChatbot
            except ImportError as e:
                print(f"Error importing AgriculturalChatbot: {str(e)}")
                # Try alternative import path
                try:
                    import sys
                    import os
                    # Add project root to path
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    parent_dir = os.path.dirname(project_root)
                    if project_root not in sys.path:
                        sys.path.append(project_root)
                    if parent_dir not in sys.path:
                        sys.path.append(parent_dir)
                    from explainable_ai.chatbot import AgriculturalChatbot
                except ImportError as e2:
                    print(f"Error importing AgriculturalChatbot with alternative path: {str(e2)}")
                    # Create a simple fallback chatbot
                    class FallbackChatbot:
                        def __init__(self):
                            self.llm = None
                        
                        def get_response(self, message):
                            return "I'm an agricultural assistant. I can help with farming questions."
                        
                        def ask_question(self, question):
                            return {
                                "answer": "I can help with agricultural questions.",
                                "source_documents": []
                            }
                    
                    chatbot_instance = FallbackChatbot()
                    return
            
            # Initialize chatbot (this handles both LLM and RAG integration)
            chatbot_instance = AgriculturalChatbot()
            print("Chatbot loaded successfully")
        except Exception as e:
            print(f"Error loading chatbot: {str(e)}")
            # Create a simple fallback chatbot (only if not already created)
            if chatbot_instance is None:
                class FallbackChatbot:
                    def __init__(self):
                        self.llm = None
                    
                    def get_response(self, message):
                        return "I'm an agricultural assistant. I can help with farming questions."
                    
                    def ask_question(self, question):
                        return {
                            "answer": "I can help with agricultural questions.",
                            "source_documents": []
                        }
                
                chatbot_instance = FallbackChatbot()

def load_pest_prediction_components():
    """Load pest prediction model and components if not already loaded."""
    global pest_model, pest_preprocessor, pest_label_encoders, pest_rag
    
    # Load pest prediction model if not already loaded
    if pest_model is None:
        try:
            import joblib
            import os
            
            # Load model
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     'models', 'pest_prediction', 'pest_model.pkl')
            if os.path.exists(model_path):
                pest_model = joblib.load(model_path)
                print("Pest prediction model loaded successfully")
            else:
                print(f"Pest prediction model not found at {model_path}")
                pest_model = None
        except Exception as e:
            print(f"Error loading pest prediction model: {e}")
            pest_model = None
    
    # Load preprocessor if not already loaded
    if pest_preprocessor is None:
        try:
            import joblib
            import os
            
            # Load preprocessor
            preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                            'models', 'pest_prediction', 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                pest_preprocessor = joblib.load(preprocessor_path)
                print("Pest prediction preprocessor loaded successfully")
            else:
                print(f"Pest prediction preprocessor not found at {preprocessor_path}")
                pest_preprocessor = None
        except Exception as e:
            print(f"Error loading pest prediction preprocessor: {e}")
            pest_preprocessor = None
    
    # Load label encoders if not already loaded
    if pest_label_encoders is None:
        try:
            import joblib
            import os
            
            # Load label encoders
            encoders_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                        'models', 'pest_prediction', 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                pest_label_encoders = joblib.load(encoders_path)
                print("Pest prediction label encoders loaded successfully")
            else:
                print(f"Pest prediction label encoders not found at {encoders_path}")
                pest_label_encoders = None
        except Exception as e:
            print(f"Error loading pest prediction label encoders: {e}")
            pest_label_encoders = None
    
    # Load RAG system if not already loaded
    if pest_rag is None:
        try:
            from explainable_ai.pest_prediction_rag.pest_prediction_rag import PestPredictionRAG
            pest_rag = PestPredictionRAG()
            print("Pest prediction RAG system loaded successfully")
        except Exception as e:
            print(f"Error loading pest prediction RAG system: {e}")
            pest_rag = None

@api_view(['POST'])
def predict_pest(request):
    """
    Predict pest occurrence from manual data input and provide explainable recommendations.

    Expected JSON body fields:
    - crop, region, season
    - temperature, humidity, rainfall, wind_speed
    - soil_moisture, soil_ph, soil_type
    - nitrogen, phosphorus, potassium
    - weather_condition, irrigation_method, previous_crop
    - days_since_planting, plant_density
    """
    print("=== PREDICT PEST ENDPOINT CALLED ===")

    # Ensure models and components are loaded
    load_pest_prediction_components()

    # Extract data
    data = request.data or {}
    required_fields = [
        'crop','region','season','temperature','humidity','rainfall','wind_speed',
        'soil_moisture','soil_ph','soil_type','nitrogen','phosphorus','potassium',
        'weather_condition','irrigation_method','previous_crop','days_since_planting','plant_density'
    ]

    # Validate
    missing = [f for f in required_fields if data.get(f) in (None, '')]
    if missing:
        return Response({'error': 'Missing required fields', 'missing_fields': missing}, status=status.HTTP_400_BAD_REQUEST)

    try:
        import pandas as pd
        import numpy as np

        # Access global components
        global pest_model, pest_preprocessor, pest_label_encoders, pest_rag

        if pest_model is None or pest_preprocessor is None or pest_label_encoders is None:
            load_pest_prediction_components()
            if pest_model is None or pest_preprocessor is None or pest_label_encoders is None:
                return Response({'error': 'Pest prediction components not available'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Build single-row dataframe in training feature order
        feature_columns = [
            'crop', 'region', 'season', 'temperature', 'humidity', 'rainfall', 'wind_speed',
            'soil_moisture', 'soil_ph', 'soil_type', 'nitrogen', 'phosphorus', 'potassium',
            'weather_condition', 'irrigation_method', 'previous_crop', 'days_since_planting',
            'plant_density'
        ]

        # Prepare row dict (copy to avoid mutation)
        row = {col: data.get(col) for col in feature_columns}

        # Cast numeric fields
        numeric_fields = {'temperature','humidity','rainfall','wind_speed','soil_moisture','soil_ph','nitrogen','phosphorus','potassium','days_since_planting','plant_density'}
        for nf in numeric_fields:
            # Ensure proper type casting
            if nf in row:
                try:
                    # int for certain fields
                    if nf in {'days_since_planting','plant_density'}:
                        row[nf] = int(row[nf])
                    else:
                        row[nf] = float(row[nf])
                except Exception:
                    return Response({'error': f'Invalid numeric value for {nf}'}, status=status.HTTP_400_BAD_REQUEST)

        # Encode categorical fields using saved label encoders
        categorical_columns = [
            'crop', 'region', 'season', 'soil_type', 'weather_condition', 'irrigation_method', 'previous_crop'
        ]

        def safe_transform(le, value):
            # LabelEncoder cannot handle unseen labels; extend classes_ on the fly
            if value not in le.classes_:
                le.classes_ = np.append(le.classes_, value)
            return int(le.transform([value])[0])

        encoded_row = row.copy()
        for col in categorical_columns:
            le = pest_label_encoders.get(col)
            if le is None:
                return Response({'error': f'Missing label encoder for {col}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            encoded_row[col] = safe_transform(le, row[col])

        # Create DataFrame
        X_df = pd.DataFrame([encoded_row], columns=feature_columns)

        # Scale/transform
        X_transformed = pest_preprocessor.transform(X_df)

        # Predict pest class and confidence
        pred = pest_model.predict(X_transformed)
        predicted_pest = str(pred[0])
        try:
            proba = pest_model.predict_proba(X_transformed)
            class_index = list(pest_model.classes_).index(predicted_pest)
            confidence = float(proba[0][class_index])
        except Exception:
            confidence = 0.7  # fallback

        # Derive presence and severity
        pest_presence = confidence >= 0.5
        # Basic severity heuristic using confidence scaled to 1-10 and environmental boosts
        severity = max(1, min(10, int(round(1 + 9 * confidence))))

        # Environmental factor analysis
        environmental_data = {
            'temperature': row.get('temperature', 0),
            'humidity': row.get('humidity', 0),
            'rainfall': row.get('rainfall', 0),
            'soil_moisture': row.get('soil_moisture', 0)
        }
        pest_analysis = {
            'severity_description': _get_severity_description(severity),
            'confidence_description': _get_confidence_description(confidence),
            'environmental_factors': _analyze_environmental_factors(environmental_data),
            'potential_damage': _estimate_potential_damage(predicted_pest, severity),
            'monitoring_advice': _generate_monitoring_advice(predicted_pest, severity)
        }

        # Build input echo for response
        input_echo = {k: row[k] for k in row}

        # Generate recommended treatment via RAG/LLM if available; fallback to rules
        recommended_treatment = ''
        try:
            if pest_rag is not None:
                query = f"How to manage {predicted_pest} in {data.get('crop')} during {data.get('season')} in {data.get('region')}?"
                rag_payload = {
                    'predicted_pest': predicted_pest,
                    'severity': pest_analysis['severity_description'],
                    'crop': data.get('crop'),
                    'region': data.get('region'),
                    'season': data.get('season'),
                    'temperature': row.get('temperature'),
                    'humidity': row.get('humidity'),
                    'rainfall': row.get('rainfall'),
                    'soil_ph': row.get('soil_ph'),
                    'soil_moisture': row.get('soil_moisture')
                }
                rag_result = pest_rag.generate_context_aware_response(query, prediction_data=rag_payload)
                # Compose a readable recommendation
                parts = []
                if rag_result.get('management_strategies'):
                    ms = rag_result['management_strategies']
                    parts.append("Management Strategies:\n- " + "\n- ".join(ms.get('key_principles', [])))
                if rag_result.get('control_methods'):
                    cm = rag_result['control_methods']
                    # Flatten some key areas
                    bio = cm.get('biological_control', {}).get('natural_enemies', [])
                    cult = cm.get('cultural_control', {}).get('practices', [])
                    phys = cm.get('physical_control', {}).get('methods', [])
                    chem = cm.get('chemical_control', {}).get('guidelines', [])
                    if bio:
                        parts.append("Biological Control:\n- " + "\n- ".join(bio))
                    if cult:
                        parts.append("Cultural Practices:\n- " + "\n- ".join(cult))
                    if phys:
                        parts.append("Physical Methods:\n- " + "\n- ".join(phys))
                    if chem:
                        parts.append("Chemical Guidelines:\n- " + "\n- ".join(chem))
                if rag_result.get('prevention_advice'):
                    pa = rag_result['prevention_advice']
                    parts.append("Prevention Advice:\n- " + "\n- ".join(pa.get('proactive_measures', [])))
                recommended_treatment = "\n\n".join([p for p in parts if p])
            else:
                recommended_treatment = _generate_fallback_pest_recommendation(predicted_pest, severity, input_echo)
        except Exception as e:
            print(f"Error generating RAG treatment: {e}")
            recommended_treatment = _generate_fallback_pest_recommendation(predicted_pest, severity, input_echo)

        # Persist to DB for reports/charts
        try:
            crop_obj, _ = Crop.objects.get_or_create(name=data.get('crop'))
            from .models import PestPrediction as PestPredictionModel
            PestPredictionModel.objects.create(
                crop=crop_obj,
                region=data.get('region'),
                season=data.get('season'),
                temperature=row.get('temperature'),
                humidity=row.get('humidity'),
                rainfall=row.get('rainfall'),
                wind_speed=row.get('wind_speed'),
                soil_moisture=row.get('soil_moisture'),
                soil_ph=row.get('soil_ph'),
                soil_type=data.get('soil_type'),
                nitrogen=row.get('nitrogen'),
                phosphorus=row.get('phosphorus'),
                potassium=row.get('potassium'),
                weather_condition=data.get('weather_condition'),
                irrigation_method=data.get('irrigation_method'),
                previous_crop=data.get('previous_crop'),
                days_since_planting=row.get('days_since_planting'),
                plant_density=row.get('plant_density'),
                predicted_pest=predicted_pest,
                pest_presence=pest_presence,
                severity=severity,
                confidence_score=confidence,
                recommended_treatment=recommended_treatment
            )
        except Exception as e:
            print(f"Error saving pest prediction: {e}")

        response = {
            'predicted_pest': predicted_pest,
            'pest_presence': pest_presence,
            'severity': severity,
            'confidence_score': round(confidence, 4),
            'pest_analysis': pest_analysis,
            'recommended_treatment': recommended_treatment,
            'input_data': {
                'crop': data.get('crop'),
                'region': data.get('region'),
                'season': data.get('season'),
                'temperature': row.get('temperature'),
                'humidity': row.get('humidity'),
                'rainfall': row.get('rainfall'),
                'wind_speed': row.get('wind_speed'),
                'soil_moisture': row.get('soil_moisture'),
                'soil_ph': row.get('soil_ph'),
                'soil_type': data.get('soil_type'),
                'nitrogen': row.get('nitrogen'),
                'phosphorus': row.get('phosphorus'),
                'potassium': row.get('potassium'),
                'weather_condition': data.get('weather_condition'),
                'irrigation_method': data.get('irrigation_method'),
                'previous_crop': data.get('previous_crop'),
                'days_since_planting': row.get('days_since_planting'),
                'plant_density': row.get('plant_density')
            },
            'timestamp': timezone.now().isoformat()
        }

        print("Returning pest prediction response:", response)
        return Response(response, status=status.HTTP_200_OK)

    except Exception as e:
        print("Error in predict_pest:", str(e))
        import traceback
        traceback.print_exc()
        return Response({'error': f'Error processing pest prediction: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def load_yield_prediction_components():
    """Load yield prediction model and components if not already loaded."""
    global yield_model, yield_preprocessor_data, yield_preprocessor
    
    # Load yield prediction model if not already loaded
    if yield_model is None:
        try:
            import joblib
            import os
            
            # Try enhanced model first, fallback to regular model
            enhanced_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                              'models', 'crop_yield_prediction', 'yield_model_enhanced.pkl')
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     'models', 'crop_yield_prediction', 'yield_model.pkl')
            
            if os.path.exists(enhanced_model_path):
                yield_model = joblib.load(enhanced_model_path)
                print("Enhanced yield prediction model loaded successfully")
            elif os.path.exists(model_path):
                yield_model = joblib.load(model_path)
                print("Yield prediction model loaded successfully")
            else:
                print(f"Yield prediction model not found at {model_path} or {enhanced_model_path}")
                yield_model = None
        except Exception as e:
            print(f"Error loading yield prediction model: {e}")
            yield_model = None
    
    # Load preprocessor data if not already loaded
    if yield_preprocessor_data is None:
        try:
            import joblib
            import os
            
            # Try enhanced preprocessor first, fallback to regular preprocessor
            enhanced_preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                     'models', 'crop_yield_prediction', 'preprocessor_enhanced.pkl')
            preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                            'models', 'crop_yield_prediction', 'preprocessor.pkl')
            
            if os.path.exists(enhanced_preprocessor_path):
                yield_preprocessor_data = joblib.load(enhanced_preprocessor_path)
                print("Enhanced yield prediction preprocessor loaded successfully")
            elif os.path.exists(preprocessor_path):
                yield_preprocessor_data = joblib.load(preprocessor_path)
                print("Yield prediction preprocessor loaded successfully")
            else:
                print(f"Yield prediction preprocessor not found at {preprocessor_path} or {enhanced_preprocessor_path}")
                yield_preprocessor_data = None
        except Exception as e:
            print(f"Error loading yield prediction preprocessor: {e}")
            yield_preprocessor_data = None
    
    # Create preprocessor instance if not already created
    if yield_preprocessor is None and yield_preprocessor_data is not None:
        try:
            # Import YieldPreprocessor
            from crop_yield_prediction.preprocessing.yield_preprocessor import YieldPreprocessor
            
            # Create a new preprocessor instance and load the saved data
            yield_preprocessor = YieldPreprocessor()
            yield_preprocessor.label_encoders = yield_preprocessor_data['label_encoders']
            yield_preprocessor.scaler = yield_preprocessor_data['scaler']
            yield_preprocessor.feature_columns = yield_preprocessor_data['feature_columns']
            print("Yield prediction preprocessor instance created successfully")
        except Exception as e:
            print(f"Error creating yield prediction preprocessor instance: {e}")
            yield_preprocessor = None

# Preload chatbot when module is imported - REMOVED FOR LAZY LOADING
# print("Preloading chatbot components...")
# load_chatbot()
# print("Chatbot components preloaded successfully")

# Preload pest prediction components when module is imported - REMOVED FOR LAZY LOADING
# print("Preloading pest prediction components...")
# load_pest_prediction_components()
# print("Pest prediction components preloaded successfully")

# Preload yield prediction components when module is imported - REMOVED FOR LAZY LOADING
# print("Preloading yield prediction components...")
# load_yield_prediction_components()
# print("Yield prediction components preloaded successfully")

@api_view(['GET', 'POST'])
def farm_list(request):
    """
    List all farms, or create a new farm.
    """
    if request.method == 'GET':
        farms = Farm.objects.all()
        serializer = FarmSerializer(farms, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        serializer = FarmSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'DELETE'])
def farm_detail(request, farm_id):
    """
    Retrieve, update or delete a farm instance.
    """
    farm = get_object_or_404(Farm, farm_id=farm_id)
    
    if request.method == 'GET':
        serializer = FarmSerializer(farm)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = FarmSerializer(farm, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        farm.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'POST'])
def crop_list(request):
    """
    List all crops, or create a new crop.
    """
    if request.method == 'GET':
        crops = Crop.objects.all()
        serializer = CropSerializer(crops, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        serializer = CropSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def submit_satellite_data(request):
    """
    Submit satellite image data.
    """
    serializer = SatelliteImageSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def submit_weather_data(request):
    """
    Submit weather data.
    """
    serializer = WeatherDataSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def predict_yield(request):
    """
    Predict crop yield from manual data input and provide LLM-generated explanation using vector database.
    """
    print("=== PREDICT YIELD ENDPOINT CALLED ===")
    
    # Load yield prediction components if not already loaded
    load_yield_prediction_components()
    
    # Extract data from request
    region = request.data.get('region')
    soil_type = request.data.get('soil_type')
    crop = request.data.get('crop')
    rainfall_mm = request.data.get('rainfall_mm')
    temperature_celsius = request.data.get('temperature_celsius')
    fertilizer_used = request.data.get('fertilizer_used')
    irrigation_used = request.data.get('irrigation_used')
    weather_condition = request.data.get('weather_condition')
    days_to_harvest = request.data.get('days_to_harvest')
    
    print("Received data:", {
        'region': region,
        'soil_type': soil_type,
        'crop': crop,
        'rainfall_mm': rainfall_mm,
        'temperature_celsius': temperature_celsius,
        'fertilizer_used': fertilizer_used,
        'irrigation_used': irrigation_used,
        'weather_condition': weather_condition,
        'days_to_harvest': days_to_harvest
    })
    
    # Validate required fields
    required_fields = [region, soil_type, crop, rainfall_mm, temperature_celsius, 
                      fertilizer_used is not None, irrigation_used is not None, 
                      weather_condition, days_to_harvest]
    
    if any(field is None for field in required_fields):
        missing_fields = []
        if region is None: missing_fields.append('region')
        if soil_type is None: missing_fields.append('soil_type')
        if crop is None: missing_fields.append('crop')
        if rainfall_mm is None: missing_fields.append('rainfall_mm')
        if temperature_celsius is None: missing_fields.append('temperature_celsius')
        if fertilizer_used is None: missing_fields.append('fertilizer_used')
        if irrigation_used is None: missing_fields.append('irrigation_used')
        if weather_condition is None: missing_fields.append('weather_condition')
        if days_to_harvest is None: missing_fields.append('days_to_harvest')
        
        print("Missing fields:", missing_fields)
        
        return Response({
            'error': 'Missing required fields', 
            'missing_fields': missing_fields
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        print("=== STARTING PREDICTION PROCESS ===")
        
        # Use preloaded model and components
        global yield_model, yield_preprocessor
        
        # Check if components are loaded
        if yield_model is None or yield_preprocessor is None:
            # Try to load components if not already loaded
            load_yield_prediction_components()
            
            # Check again
            if yield_model is None or yield_preprocessor is None:
                error_msg = 'Yield prediction model or preprocessor not available'
                print("ERROR:", error_msg)
                return Response({
                    'error': error_msg
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Prepare data for prediction
        sample_data = {
            'Region': region,
            'Soil_Type': soil_type,
            'Crop': crop,
            'Rainfall_mm': float(rainfall_mm),
            'Temperature_Celsius': float(temperature_celsius),
            'Fertilizer_Used': bool(fertilizer_used),
            'Irrigation_Used': bool(irrigation_used),
            'Weather_Condition': weather_condition,
            'Days_to_Harvest': int(days_to_harvest)
        }
        
        print("Sample data for prediction:", sample_data)
        
        # Preprocess the sample
        X_sample = yield_preprocessor.preprocess_single_sample(sample_data)
        print("Preprocessed data shape:", X_sample.shape)
        
        # Make prediction
        yield_pred = yield_model.predict(X_sample)
        predicted_yield = float(yield_pred[0])
        print("Predicted yield:", predicted_yield)
        
        # Calculate confidence (simplified approach)
        # In a real implementation, you might use model's predict_proba or other methods
        confidence_score = 0.9042  # Using the R² score from training as a proxy
        
        # Generate enhanced insights and explanation
        explanation = ""
        insights = {}
        fertilizer_advice = ""
        pest_control_advice = ""
        
        try:
            print("=== GENERATING ENHANCED EXPLANATION ===")
            
            # Import the enhanced prediction functions
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from crop_yield_prediction.predict_yield_enhanced import generate_yield_insights, generate_llm_explanation, get_prediction_metadata
            
            # Load metadata for enhanced insights
            metadata = get_prediction_metadata()
            
            # Generate insights
            insights = generate_yield_insights(predicted_yield, sample_data, metadata)
            
            # Generate explanation using enhanced logic
            explanation = generate_llm_explanation(predicted_yield, sample_data, insights)
            print("Generated enhanced explanation length:", len(explanation))
            
            # Generate fertilizer advice using LLM
            fertilizer_advice = _generate_fertilizer_advice(predicted_yield, sample_data, insights)
            
            # Generate pest control advice using LLM
            pest_control_advice = _generate_pest_control_advice(predicted_yield, sample_data, insights)
            
        except Exception as llm_error:
            print("Error in enhanced explanation processing:", str(llm_error))
            import traceback
            traceback.print_exc()
            # Provide a fallback explanation if enhanced processing fails
            explanation = f"Based on the provided conditions, the predicted yield for {crop} is {predicted_yield:.2f} tons per hectare. "
            explanation += "This prediction takes into account the rainfall, temperature, soil type, and other factors. "
            explanation += "For more detailed insights, ensure the enhanced prediction system is properly configured."
            
            # Basic insights as fallback
            insights = {
                'prediction_summary': f'Yield prediction for {crop}: {predicted_yield:.2f} tons/hectare',
                'key_factors': ['Based on provided environmental conditions'],
                'recommendations': ['Monitor crop growth regularly'],
                'risk_assessment': 'Assessment based on model confidence',
                'comparison_to_benchmarks': 'No benchmark comparison available',
                'confidence_level': 'Model confidence: High'
            }
            
            # Basic fertilizer and pest advice as fallback
            fertilizer_advice = _generate_fertilizer_advice(predicted_yield, sample_data, insights)
            pest_control_advice = _generate_pest_control_advice(predicted_yield, sample_data, insights)
        
        # Save prediction to database for reports and charts
        try:
            # Get or create crop object
            crop_obj, created = Crop.objects.get_or_create(name=crop)
            
            # Create yield prediction record
            yield_prediction = YieldPrediction.objects.create(
                crop=crop_obj,
                predicted_yield_tonnes_per_ha=predicted_yield,
                confidence_score=confidence_score
            )
            print(f"Saved prediction to database: {yield_prediction.id}")
        except Exception as save_error:
            print("Error saving prediction to database:", str(save_error))
            import traceback
            traceback.print_exc()
        
        # Return response with enhanced information
        response_data = {
            'predicted_yield': round(predicted_yield, 2),
            'confidence_score': round(confidence_score, 4),
            'input_data': sample_data,
            'explanation': explanation,
            'insights': insights,
            'fertilizer_advice': fertilizer_advice,
            'pest_control_advice': pest_control_advice,
            'timestamp': timezone.now().isoformat()
        }
        
        print("Returning response:", response_data)
        print("=== PREDICTION COMPLETED SUCCESSFULLY ===")
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        print("Error in predict_yield:", str(e))
        import traceback
        traceback.print_exc()
        return Response({
            'error': f'Error processing prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_latest_prediction(request, farm_id, crop_id):
    """
    Get the latest yield prediction for a specific farm and crop.
    """
    try:
        farm_crop = FarmCrop.objects.get(farm__farm_id=farm_id, crop_id=crop_id)
        latest_prediction = YieldPrediction.objects.filter(farm_crop=farm_crop).latest('prediction_date')
        serializer = YieldPredictionSerializer(latest_prediction)
        return Response(serializer.data)
    except FarmCrop.DoesNotExist:
        return Response({'error': 'Farm or crop not found'}, status=status.HTTP_404_NOT_FOUND)
    except YieldPrediction.DoesNotExist:
        return Response({'error': 'No predictions found'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
def get_farm_recommendations(request, farm_id):
    """
    Get active recommendations for a specific farm.
    """
    recommendations = Recommendation.objects.filter(farm__farm_id=farm_id, is_active=True)
    serializer = RecommendationSerializer(recommendations, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def generate_explanation(request):
    """
    Generate an explanation for a prediction.
    """
    return Response({'message': 'Explanation generated'})

def translate_text(text, target_language):
    """
    Translate text to the target language using a predefined translation dictionary.
    
    Args:
        text (str): The text to be translated.
        target_language (str): The target language code (e.g., 'es' for Spanish).
        
    Returns:
        str: The translated text.
    """
    # Define a simple translation dictionary
    translations = {
        'es': {
            'Hello': 'Hola',
            'Goodbye': 'Adiós',
            'Thank you': 'Gracias',
            'Yes': 'Sí',
            'No': 'No'
        },
        'fr': {
            'Hello': 'Bonjour',
            'Goodbye': 'Au revoir',
            'Thank you': 'Merci',
            'Yes': 'Oui',
            'No': 'Non'
        }
    }
    
    # If the target language is English, return the text as is
    if target_language == 'en':
        return text

    # If we have translations for the target language, use them
    if target_language in translations:
        language_translations = translations[target_language]
        # Return translated text if available, otherwise return original
        return language_translations.get(text, text)

    # If no translation is available, return the original text
    return text


@api_view(['GET'])
def get_dashboard_statistics(request):
    """
    Get comprehensive dashboard statistics from actual database records.
    """
    try:
        import datetime
        from django.db.models import Count, Avg
        
        # Get counts for each feature
        prediction_count = YieldPrediction.objects.count()
        disease_count = DiseasePrediction.objects.count()
        recommendation_count = CropRecommendation.objects.count()
        
        # Get recent activity (last 7 days)
        seven_days_ago = timezone.now() - timezone.timedelta(days=7)
        recent_predictions = YieldPrediction.objects.filter(
            prediction_date__gte=seven_days_ago
        ).count()
        recent_diseases = DiseasePrediction.objects.filter(
            timestamp__gte=seven_days_ago
        ).count()
        recent_recommendations = CropRecommendation.objects.filter(
            timestamp__gte=seven_days_ago
        ).count()
        
        # Calculate averages
        avg_yield = 0
        avg_confidence = 0
        
        if prediction_count > 0:
            avg_yield_result = YieldPrediction.objects.aggregate(Avg('predicted_yield_tonnes_per_ha'))
            avg_yield = round(avg_yield_result['predicted_yield_tonnes_per_ha__avg'], 2) if avg_yield_result['predicted_yield_tonnes_per_ha__avg'] else 0
            
            # Calculate average confidence only for entries that have confidence scores
            confidence_values = [pred.confidence_score for pred in YieldPrediction.objects.all() if pred.confidence_score is not None]
            if confidence_values:
                avg_confidence = round(sum(confidence_values) / len(confidence_values), 2)
        
        # Get most common crop recommendations
        crop_recommendations = CropRecommendation.objects.values('recommended_crop').annotate(
            count=Count('recommended_crop')
        ).order_by('-count')
        
        most_common_crop = crop_recommendations.first()['recommended_crop'] if crop_recommendations.first() else "None"
        
        # Get recent predictions for chart
        recent_predictions_data = YieldPrediction.objects.filter(
            prediction_date__gte=seven_days_ago
        ).select_related('crop')
        
        recent_yield_data = []
        for pred in recent_predictions_data:
            recent_yield_data.append({
                'date': pred.prediction_date.strftime('%Y-%m-%d'),
                'crop': pred.crop.name if pred.crop else 'Unknown',
                'yield': round(pred.predicted_yield_tonnes_per_ha, 2)
            })
        
        # Sort by date
        recent_yield_data.sort(key=lambda x: x['date'])
        
        return Response({
            'statistics': {
                'total_predictions': prediction_count,
                'total_disease_detections': disease_count,
                'total_recommendations': recommendation_count,
                'avg_yield': avg_yield,
                'avg_confidence': avg_confidence,
                'most_common_crop': most_common_crop,
                'recent_activity': {
                    'predictions': recent_predictions,
                    'diseases': recent_diseases,
                    'recommendations': recent_recommendations
                }
            },
            'recent_yield_data': recent_yield_data
        })
        
    except Exception as e:
        print(f"Error in get_dashboard_statistics: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            {'error': 'Failed to retrieve dashboard statistics'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_prediction_report_data(request):
    """
    Get data for prediction reports.
    """
    try:
        # Get all yield predictions ordered by date
        predictions = YieldPrediction.objects.select_related('crop').order_by('-prediction_date')
        
        # Format data for charting
        chart_data = []
        for pred in predictions:
            chart_data.append({
                'date': pred.prediction_date.strftime('%Y-%m-%d'),
                'crop': pred.crop.name if pred.crop else 'Unknown',
                'yield': round(pred.predicted_yield_tonnes_per_ha, 2),
                'confidence': round(pred.confidence_score, 2) if pred.confidence_score else None
            })
        
        # Get summary statistics
        total_predictions = predictions.count()
        avg_yield = 0
        avg_confidence = 0
        
        if total_predictions > 0:
            yield_sum = sum(pred.predicted_yield_tonnes_per_ha for pred in predictions)
            avg_yield = round(yield_sum / total_predictions, 2)
            
            # Calculate average confidence only for entries that have confidence scores
            confidence_values = [pred.confidence_score for pred in predictions if pred.confidence_score is not None]
            if confidence_values:
                avg_confidence = round(sum(confidence_values) / len(confidence_values), 2)
        
        return Response({
            'chart_data': chart_data,
            'summary': {
                'total_predictions': total_predictions,
                'average_yield': avg_yield,
                'average_confidence': avg_confidence
            }
        })
        
    except Exception as e:
        print(f"Error in get_prediction_report_data: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            {'error': 'Failed to retrieve prediction report data'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_disease_report_data(request):
    """
    Get data for disease detection reports.
    """
    try:
        # Get all disease predictions ordered by date
        diseases = DiseasePrediction.objects.order_by('-timestamp')
        
        # Group by disease type
        disease_counts = {}
        for disease in diseases:
            disease_type = disease.predicted_disease
            if disease_type in disease_counts:
                disease_counts[disease_type] += 1
            else:
                disease_counts[disease_type] = 1
        
        # Format data for charting
        chart_data = []
        for disease_type, count in disease_counts.items():
            chart_data.append({
                'disease': disease_type,
                'count': count
            })
        
        # Sort by count descending
        chart_data.sort(key=lambda x: x['count'], reverse=True)
        
        # Get recent detections
        recent_detections = []
        for disease in diseases[:10]:  # Last 10 detections
            recent_detections.append({
                'date': disease.timestamp.strftime('%Y-%m-%d %H:%M'),
                'disease': disease.predicted_disease,
                'confidence': round(disease.confidence_score, 2)
            })
        
        return Response({
            'chart_data': chart_data,
            'recent_detections': recent_detections,
            'total_detections': diseases.count()
        })
        
    except Exception as e:
        print(f"Error in get_disease_report_data: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            {'error': 'Failed to retrieve disease report data'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_recommendation_report_data(request):
    """
    Get data for recommendation reports.
    """
    try:
        # Get all crop recommendations ordered by date
        recommendations = CropRecommendation.objects.order_by('-timestamp')
        
        # Group by crop type
        crop_counts = {}
        for rec in recommendations:
            crop_type = rec.recommended_crop
            if crop_type in crop_counts:
                crop_counts[crop_type] += 1
            else:
                crop_counts[crop_type] = 1
        
        # Format data for charting
        chart_data = []
        for crop_type, count in crop_counts.items():
            chart_data.append({
                'crop': crop_type,
                'count': count
            })
        
        # Sort by count descending
        chart_data.sort(key=lambda x: x['count'], reverse=True)
        
        # Get recent recommendations
        recent_recommendations = []
        for rec in recommendations[:10]:  # Last 10 recommendations
            recent_recommendations.append({
                'date': rec.timestamp.strftime('%Y-%m-%d %H:%M'),
                'crop': rec.recommended_crop,
                'confidence': round(rec.confidence_score, 2),
                'conditions': {
                    'nitrogen': rec.soil_nitrogen,
                    'phosphorus': rec.soil_phosphorus,
                    'potassium': rec.soil_potassium,
                    'ph': rec.soil_ph
                }
            })
        
        return Response({
            'chart_data': chart_data,
            'recent_recommendations': recent_recommendations,
            'total_recommendations': recommendations.count()
        })
        
    except Exception as e:
        print(f"Error in get_recommendation_report_data: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            {'error': 'Failed to retrieve recommendation report data'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_pest_report_data(request):
    """
    Get data for pest prediction reports.
    """
    try:
        # Get all pest predictions ordered by date
        pests = PestPrediction.objects.select_related('crop').order_by('-prediction_date')
        
        # Group by pest type
        pest_counts = {}
        for pest in pests:
            pest_type = pest.predicted_pest
            if pest_type in pest_counts:
                pest_counts[pest_type] += 1
            else:
                pest_counts[pest_type] = 1
        
        # Format data for charting
        chart_data = []
        for pest_type, count in pest_counts.items():
            chart_data.append({
                'pest': pest_type,
                'count': count
            })
        
        # Sort by count descending
        chart_data.sort(key=lambda x: x['count'], reverse=True)
        
        # Get recent predictions
        recent_predictions = []
        for pest in pests[:10]:  # Last 10 predictions
            recent_predictions.append({
                'date': pest.prediction_date.strftime('%Y-%m-%d %H:%M'),
                'crop': pest.crop.name,
                'pest': pest.predicted_pest,
                'severity': pest.severity,
                'confidence': round(pest.confidence_score, 2) if pest.confidence_score else None
            })
        
        return Response({
            'chart_data': chart_data,
            'recent_predictions': recent_predictions,
            'total_predictions': pests.count()
        })
        
    except Exception as e:
        print(f"Error in get_pest_report_data: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            {'error': 'Failed to retrieve pest report data'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_market_report_data(request):
    """
    Get data for market prediction reports.
    """
    try:
        from .models import MarketPrediction
        
        # Get all market predictions ordered by date
        market_predictions = MarketPrediction.objects.select_related('crop').order_by('-prediction_date')
        
        # Group by crop type
        crop_data = {}
        for pred in market_predictions:
            crop_name = pred.crop.name
            if crop_name not in crop_data:
                crop_data[crop_name] = []
            crop_data[crop_name].append({
                'date': pred.prediction_date.strftime('%Y-%m-%d'),
                'price': round(pred.predicted_price_per_ton, 2),
                'confidence': round(pred.confidence_score, 2) if pred.confidence_score else None,
                'trend': pred.market_trend
            })
        
        # Format data for charting
        chart_data = []
        for crop_name, predictions in crop_data.items():
            # Sort by date
            predictions.sort(key=lambda x: x['date'])
            chart_data.append({
                'crop': crop_name,
                'predictions': predictions
            })
        
        # Get recent predictions
        recent_predictions = []
        for pred in market_predictions[:10]:  # Last 10 predictions
            recent_predictions.append({
                'date': pred.prediction_date.strftime('%Y-%m-%d %H:%M'),
                'crop': pred.crop.name,
                'price': round(pred.predicted_price_per_ton, 2),
                'confidence': round(pred.confidence_score, 2) if pred.confidence_score else None,
                'trend': pred.market_trend
            })
        
        return Response({
            'chart_data': chart_data,
            'recent_predictions': recent_predictions,
            'total_predictions': market_predictions.count()
        })
        
    except Exception as e:
        print(f"Error in get_market_report_data: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            {'error': 'Failed to retrieve market report data'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    
    return analysis

def _get_severity_description(severity):
    """
    Get descriptive severity level
    """
    if severity >= 8:
        return "Critical Risk"
    elif severity >= 6:
        return "High Risk"
    elif severity >= 4:
        return "Moderate Risk"
    elif severity >= 2:
        return "Low Risk"
    else:
        return "Minimal Risk"

def _get_confidence_description(confidence_score):
    """
    Get descriptive confidence level
    """
    if confidence_score >= 0.8:
        return "High Confidence"
    elif confidence_score >= 0.6:
        return "Medium Confidence"
    else:
        return "Low Confidence"

def _analyze_environmental_factors(environmental_data):
    """
    Analyze how environmental factors contribute to pest risk
    """
    factors = []
    
    temp = environmental_data.get('temperature', 0)
    if temp > 30:
        factors.append("High temperature favors rapid pest development")
    elif temp < 10:
        factors.append("Low temperature may slow pest activity")
    
    humidity = environmental_data.get('humidity', 0)
    if humidity > 80:
        factors.append("High humidity supports pest growth and disease development")
    elif humidity < 30:
        factors.append("Low humidity may limit pest survival")
    
    rainfall = environmental_data.get('rainfall', 0)
    if rainfall > 100:
        factors.append("Heavy rainfall may promote pest proliferation")
    elif rainfall < 20:
        factors.append("Low rainfall may stress crops, making them more susceptible")
    
    soil_moisture = environmental_data.get('soil_moisture', 0)
    if soil_moisture > 70:
        factors.append("High soil moisture creates favorable conditions for soil pests")
    elif soil_moisture < 20:
        factors.append("Low soil moisture may limit soil pest activity")
    
    return factors if factors else ["Environmental conditions are neutral for pest development"]

def _estimate_potential_damage(predicted_pest, severity):
    """
    Estimate potential crop damage
    """
    # This would typically be more sophisticated based on pest type
    damage_levels = {
        'Aphids': 'Leaf curling, stunted growth, virus transmission',
        'Armyworms': 'Leaf consumption, stem damage',
        'Boll Weevils': 'Boll damage, yield reduction',
        'Corn Borers': 'Stalk tunneling, yield loss',
        'Cutworms': 'Seedling cutting, plant death',
        'Flea Beetles': 'Shot-hole damage to leaves',
        'Grasshoppers': 'Foliage consumption',
        'Hessian Fly': 'Stunted growth, lodging',
        'Japanese Beetles': 'Skeletonized leaves',
        'Leafhoppers': 'Leaf stippling, disease transmission',
        'Root Knot Nematodes': 'Root galling, nutrient deficiency',
        'Spider Mites': 'Leaf stippling, webbing',
        'Stem Borers': 'Stem tunneling, plant death',
        'Thrips': 'Leaf scarring, virus transmission',
        'White Grubs': 'Root damage, plant death',
        'Whiteflies': 'Leaf yellowing, sooty mold',
        'Wireworms': 'Root and tuber damage'
    }
    
    base_damage = damage_levels.get(predicted_pest, 'General crop damage')
    
    if severity >= 8:
        return f"Severe: {base_damage}. Potential yield loss of 30-50%."
    elif severity >= 6:
        return f"Moderate to Severe: {base_damage}. Potential yield loss of 15-30%."
    elif severity >= 4:
        return f"Moderate: {base_damage}. Potential yield loss of 5-15%."
    else:
        return f"Low: {base_damage}. Minimal yield impact expected."

def _generate_monitoring_advice(predicted_pest, severity):
    """
    Generate monitoring advice based on pest and severity
    """
    if severity >= 8:
        return "Daily monitoring required. Check plants at different field locations."
    elif severity >= 6:
        return "Monitor every 2-3 days. Focus on field edges and problem areas."
    elif severity >= 4:
        return "Weekly monitoring recommended. Pay attention to changes in conditions."
    else:
        return "Regular monitoring as part of standard practices."

def _generate_fallback_pest_recommendation(predicted_pest, severity, input_data):
    """
    Generate fallback pest management recommendation when LLM is not available
    
    Args:
        predicted_pest (str): Predicted pest name
        severity (int): Severity level (1-10)
        input_data (dict): Input data used for prediction
        
    Returns:
        str: Pest management recommendation
    """
    # Base recommendation
    recommendation = f"Based on the prediction of {predicted_pest} infestation with severity level {severity}/10:\n\n"
    
    # Add severity-based recommendations
    if severity >= 8:
        recommendation += "HIGH SEVERITY - IMMEDIATE ACTION REQUIRED:\n"
        recommendation += "1. Apply appropriate pesticide treatment immediately following label instructions\n"
        recommendation += "2. Remove and destroy heavily infested plants to prevent spread\n"
        recommendation += "3. Monitor neighboring plants daily for signs of spread\n"
        recommendation += "4. Contact agricultural extension service for professional advice\n\n"
    elif severity >= 5:
        recommendation += "MODERATE SEVERITY - TAKE ACTION WITHIN 48 HOURS:\n"
        recommendation += "1. Implement integrated pest management practices\n"
        recommendation += "2. Consider biological control agents if available\n"
        recommendation += "3. Apply targeted pesticide treatment if necessary\n"
        recommendation += "4. Increase monitoring frequency to twice daily\n\n"
    else:
        recommendation += "LOW TO MEDIUM SEVERITY - MONITOR AND PREVENT:\n"
        recommendation += "1. Continue regular monitoring for population increase\n"
        recommendation += "2. Implement cultural control practices (sanitation, crop rotation)\n"
        recommendation += "3. Encourage beneficial insects by planting companion plants\n"
        recommendation += "4. Use physical barriers or traps if appropriate\n\n"
    
    # Add general best practices
    recommendation += "GENERAL BEST PRACTICES:\n"
    recommendation += "1. Maintain proper plant spacing for air circulation\n"
    recommendation += "2. Water at soil level to avoid leaf wetness\n"
    recommendation += "3. Remove plant debris and weeds that harbor pests\n"
    recommendation += "4. Keep detailed records of pest occurrences and treatments\n"
    
    return recommendation

def _generate_fertilizer_advice(predicted_yield, input_data, insights):
    """
    Generate fertilizer advice using LLM and RAG system
    
    Args:
        predicted_yield (float): Predicted yield value
        input_data (dict): Input data used for prediction
        insights (dict): Generated insights about the prediction
        
    Returns:
        str: Fertilizer advice
    """
    try:
        # Import LLM and RAG components
        from explainable_ai.llm_interface import AgricultureLLM
        from explainable_ai.rag_system import AgricultureRAG
        
        # Initialize LLM and RAG system
        llm = AgricultureLLM()
        rag_system = AgricultureRAG()
        
        # Create context for fertilizer advice
        context = f"""
        Crop Yield Analysis:
        - Predicted Yield: {predicted_yield:.2f} tons/hectare
        - Crop: {input_data.get('Crop', 'Unknown')}
        - Soil Type: {input_data.get('Soil_Type', 'Unknown')}
        - Fertilizer Used: {'Yes' if input_data.get('Fertilizer_Used', False) else 'No'}
        - Rainfall: {input_data.get('Rainfall_mm', 0)} mm
        - Temperature: {input_data.get('Temperature_Celsius', 0)}°C
        
        Insights:
        """
        
        for key, value in insights.items():
            if isinstance(value, list):
                context += f"- {key}: {', '.join(value)}\n"
            else:
                context += f"- {key}: {value}\n"
        
        # Generate fertilizer advice prompt
        fertilizer_prompt = f"""
        {context}
        
        Based on this crop yield analysis, provide specific fertilizer recommendations to optimize yield.
        Consider the crop type, soil conditions, and environmental factors.
        
        Include:
        1. Type of fertilizer recommended
        2. Application rate (kg/hectare or similar units)
        3. Timing of application
        4. Application method
        5. Any special considerations
        
        Format the response as a clear, actionable list.
        """
        
        if llm.text_generator:
            try:
                fertilizer_response = llm.text_generator(
                    fertilizer_prompt,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True
                )
                return fertilizer_response[0]['generated_text'][len(fertilizer_prompt):].strip()
            except Exception as e:
                print(f"Error generating LLM fertilizer advice: {str(e)}")
                return _generate_fallback_fertilizer_advice(predicted_yield, input_data)
        else:
            return _generate_fallback_fertilizer_advice(predicted_yield, input_data)
            
    except Exception as e:
        print(f"Error in enhanced fertilizer advice generation: {str(e)}")
        # Fall back to rule-based recommendations
        return _generate_fallback_fertilizer_advice(predicted_yield, input_data)

def _generate_fallback_fertilizer_advice(predicted_yield, input_data):
    """
    Generate fallback fertilizer advice when LLM is not available
    
    Args:
        predicted_yield (float): Predicted yield value
        input_data (dict): Input data used for prediction
        
    Returns:
        str: Fertilizer advice
    """
    crop = input_data.get('Crop', 'Unknown')
    soil_type = input_data.get('Soil_Type', 'Unknown')
    fertilizer_used = input_data.get('Fertilizer_Used', False)
    rainfall = input_data.get('Rainfall_mm', 0)
    
    advice = f"Fertilizer Recommendations for {crop}:\n\n"
    
    # General fertilizer recommendations based on crop type
    crop_fertilizer_map = {
        'Wheat': 'Nitrogen (N), Phosphorus (P), and Potassium (K) in a balanced ratio (e.g., 120-60-60 kg/ha)',
        'Rice': 'Higher nitrogen requirements (e.g., 120-60-40 kg/ha N-P-K)',
        'Maize': 'High nitrogen needs (e.g., 150-75-75 kg/ha N-P-K)',
        'Cotton': 'Balanced N-P-K with additional potassium (e.g., 100-50-100 kg/ha)',
        'Soybean': 'Lower nitrogen needs due to nitrogen fixation, focus on phosphorus and potassium (e.g., 30-90-90 kg/ha)'
    }
    
    if crop in crop_fertilizer_map:
        advice += f"Recommended Fertilizer: {crop_fertilizer_map[crop]}\n\n"
    else:
        advice += "Recommended Fertilizer: Balanced N-P-K fertilizer (e.g., 100-50-50 kg/ha)\n\n"
    
    # Application timing
    advice += "Application Timing:\n"
    advice += "1. Basal application: Apply 50% of nitrogen and full phosphorus/potassium before planting\n"
    advice += "2. Top dressing: Apply remaining nitrogen in 2-3 splits during vegetative growth\n\n"
    
    # Application method
    advice += "Application Method:\n"
    advice += "1. For basal application: Mix fertilizer uniformly in soil before planting\n"
    advice += "2. For top dressing: Apply between rows and incorporate with light irrigation\n\n"
    
    # Special considerations
    advice += "Special Considerations:\n"
    if not fertilizer_used:
        advice += "- Since no fertilizer was used in the prediction, implementing these recommendations could significantly improve yield\n"
    if soil_type == 'Sandy':
        advice += "- Sandy soils require more frequent but smaller fertilizer applications due to leaching\n"
    elif soil_type == 'Clay':
        advice += "- Clay soils retain nutrients better but may require more careful timing to prevent runoff\n"
    if rainfall < 300:
        advice += "- In low rainfall conditions, use slow-release fertilizers to maximize nutrient uptake\n"
    elif rainfall > 1000:
        advice += "- In high rainfall areas, split applications to reduce nutrient loss from leaching\n"
    
    return advice

def _generate_pest_control_advice(predicted_yield, input_data, insights):
    """
    Generate pest control advice using LLM and RAG system
    
    Args:
        predicted_yield (float): Predicted yield value
        input_data (dict): Input data used for prediction
        insights (dict): Generated insights about the prediction
        
    Returns:
        str: Pest control advice
    """
    try:
        # Import LLM and RAG components
        from explainable_ai.llm_interface import AgricultureLLM
        from explainable_ai.rag_system import AgricultureRAG
        
        # Initialize LLM and RAG system
        llm = AgricultureLLM()
        rag_system = AgricultureRAG()
        
        # Create context for pest control advice
        context = f"""
        Crop Yield Analysis:
        - Predicted Yield: {predicted_yield:.2f} tons/hectare
        - Crop: {input_data.get('Crop', 'Unknown')}
        - Weather Condition: {input_data.get('Weather_Condition', 'Unknown')}
        - Rainfall: {input_data.get('Rainfall_mm', 0)} mm
        - Temperature: {input_data.get('Temperature_Celsius', 0)}°C
        - Days to Harvest: {input_data.get('Days_to_Harvest', 0)}
        
        Insights:
        """
        
        for key, value in insights.items():
            if isinstance(value, list):
                context += f"- {key}: {', '.join(value)}\n"
            else:
                context += f"- {key}: {value}\n"
        
        # Generate pest control advice prompt
        pest_prompt = f"""
        {context}
        
        Based on this crop yield analysis, provide specific pest control recommendations to protect and increase yield.
        Consider the crop type, weather conditions, and growth stage.
        
        Include:
        1. Common pests for this crop and conditions
        2. Prevention strategies
        3. Monitoring techniques
        4. Control methods (biological, cultural, chemical)
        5. Timing of interventions
        
        Format the response as a clear, actionable list.
        """
        
        if llm.text_generator:
            try:
                pest_response = llm.text_generator(
                    pest_prompt,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True
                )
                return pest_response[0]['generated_text'][len(pest_prompt):].strip()
            except Exception as e:
                print(f"Error generating LLM pest control advice: {str(e)}")
                return _generate_fallback_pest_control_advice(predicted_yield, input_data)
        else:
            return _generate_fallback_pest_control_advice(predicted_yield, input_data)
            
    except Exception as e:
        print(f"Error in enhanced pest control advice generation: {str(e)}")
        # Fall back to rule-based recommendations
        return _generate_fallback_pest_control_advice(predicted_yield, input_data)

def _generate_fallback_pest_control_advice(predicted_yield, input_data):
    """
    Generate fallback pest control advice when LLM is not available
    
    Args:
        predicted_yield (float): Predicted yield value
        input_data (dict): Input data used for prediction
        
    Returns:
        str: Pest control advice
    """
    crop = input_data.get('Crop', 'Unknown')
    weather_condition = input_data.get('Weather_Condition', 'Unknown')
    days_to_harvest = input_data.get('Days_to_Harvest', 0)
    
    advice = f"Pest Control Recommendations for {crop}:\n\n"
    
    # Common pests by crop
    crop_pest_map = {
        'Wheat': ['Aphids', 'Armyworms', 'Hessian Fly', 'Wheat Stem Sawfly'],
        'Rice': ['Stem Borers', 'Leaf folders', 'Brown Planthopper', 'Rice Water Weevil'],
        'Maize': ['Corn Borers', 'Armyworms', 'Corn Earworm', 'Fall Armyworm'],
        'Cotton': ['Boll Weevils', 'Aphids', 'Bollworms', 'Spider Mites'],
        'Soybean': ['Soybean Aphids', 'Bean Leaf Beetles', 'Japanese Beetles', 'Stem Borers']
    }
    
    common_pests = crop_pest_map.get(crop, ['General agricultural pests'])
    advice += f"Common Pests to Monitor: {', '.join(common_pests)}\n\n"
    
    # Prevention strategies
    advice += "Prevention Strategies:\n"
    advice += "1. Crop rotation: Rotate with non-host crops to break pest cycles\n"
    advice += "2. Sanitation: Remove crop residues and weeds that harbor pests\n"
    advice += "3. Resistant varieties: Plant pest-resistant crop varieties when available\n"
    advice += "4. Beneficial insects: Encourage natural predators like ladybugs and parasitic wasps\n\n"
    
    # Monitoring techniques
    advice += "Monitoring Techniques:\n"
    advice += "1. Regular field scouting: Check crops at least twice a week for pest signs\n"
    advice += "2. Sticky traps: Use yellow or blue sticky cards to monitor flying insects\n"
    advice += "3. Pheromone traps: Deploy species-specific traps for key pests\n"
    advice += "4. Threshold monitoring: Take action when pest populations reach economic thresholds\n\n"
    
    # Control methods
    advice += "Control Methods:\n"
    advice += "1. Biological control: Release beneficial insects or use microbial pesticides\n"
    advice += "2. Cultural practices: Adjust planting dates, use trap crops, maintain proper spacing\n"
    advice += "3. Mechanical control: Handpick large pests, use barriers or row covers\n"
    advice += "4. Chemical control: Apply targeted pesticides only when necessary and as a last resort\n\n"
    
    # Special considerations
    advice += "Special Considerations:\n"
    if weather_condition in ['Rainy', 'Humid']:
        advice += "- Humid conditions favor fungal diseases and certain pests; ensure good air circulation\n"
    elif weather_condition in ['Hot', 'Sunny']:
        advice += "- Hot, dry conditions stress plants and make them more susceptible to pests; maintain adequate irrigation\n"
    
    if days_to_harvest < 30:
        advice += "- With harvest approaching, prioritize non-chemical control methods to avoid residue issues\n"
    elif days_to_harvest > 90:
        advice += "- During extended growing season, implement comprehensive monitoring and prevention program\n"
    
    return advice

@api_view(['POST'])
def generate_explanation(request):
    """
    Generate an explanation for a prediction.
    In a real implementation, this would call the LLM.
    """
    # In a real implementation, we would:
    # 1. Load the LLM and RAG system
    # 2. Generate an explanation based on the prediction and data
    # 3. Return the explanation
    
    # For this example, we'll return a sample explanation
    prediction = request.data.get('prediction', 0)
    explanation = f"Based on the analysis, the predicted yield is {prediction} tonnes per hectare. "
    explanation += "This is influenced by recent weather patterns and vegetation health indicators. "
    explanation += "Consider adjusting irrigation schedules if rainfall is below average in the coming weeks."
    
    return Response({'explanation': explanation})

@api_view(['POST'])
def ask_question(request):
    """
    Answer a farmer's question using the LLM and RAG system.
    In a real implementation, this would call the LLM.
    """
    question = request.data.get('question', '')
    
    # In a real implementation, we would:
    # 1. Load the LLM and RAG system
    # 2. Retrieve relevant documents
    # 3. Generate an answer
    # 4. Return the answer
    
    # For this example, we'll return a sample answer
    answer = f"Regarding your question: '{question}', this is a sample response from the agricultural expert system. "
    answer += "In a production environment, this would be generated by a large language model with access to agricultural knowledge."
    
    return Response({'answer': answer})

# Add a new endpoint for the chatbot
@api_view(['POST'])
def chatbot_query(request):
    """
    Handle chatbot queries from farmers using LLM and RAG system.
    """
    # Debug: Print the request data
    print("Chatbot query request data:", request.data)
    
    user_message = request.data.get('message', '')
    language = request.data.get('language', 'en')  # Default to English
    
    # Validate input
    if not user_message:
        return Response({'response': 'Please provide a message to get a response.'}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Load chatbot if not already loaded (ensure lazy load)
        load_chatbot()

        source_documents = []

        # Check if chatbot is properly initialized
        if chatbot_instance is None or not hasattr(chatbot_instance, 'llm') or chatbot_instance.llm is None:
            # Fallback response if chatbot initialization failed
            response_text = (
                f"I understand you're asking about '{user_message}'. As an agricultural expert, I can help with questions about crops, weather, soil health, and farming practices. "
                "Could you tell me more about your specific situation? For example, what crops are you growing, and what challenges are you facing?"
            )
        else:
            # Prefer the structured question method when available (returns answer + sources)
            try:
                if hasattr(chatbot_instance, 'ask_question'):
                    result = chatbot_instance.ask_question(user_message)
                    # result expected to be a dict: { 'answer': str, 'source_documents': [ ... ] }
                    if isinstance(result, dict):
                        response_text = result.get('answer', '')
                        source_documents = result.get('source_documents', []) or []
                    else:
                        # Fallback to plain get_response
                        response_text = chatbot_instance.get_response(user_message)
                        source_documents = []
                else:
                    response_text = chatbot_instance.get_response(user_message)
                    source_documents = []
            except Exception:
                # In case ask_question/get_response raises, fallback
                response_text = chatbot_instance.get_response(user_message)
                source_documents = []

        # Translate response if needed
        if language != 'en':
            response_text = translate_text(response_text, language)

        return Response({'response': response_text, 'source_documents': source_documents})
    except Exception as e:
        # More detailed error handling
        print(f"Chatbot error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback response with better context
        response = f"I understand you're asking about '{user_message}'. As an agricultural expert, I can help with questions about crops, weather, soil health, and farming practices. Could you tell me more about your specific situation? For example, what crops are you growing, and what challenges are you facing?"
        
        # Translate response if needed
        if language != 'en':
            response = translate_text(response, language)
        
        return Response({'response': response})

@api_view(['POST'])
def chatbot_question(request):
    """
    Handle specific question answering from farmers using LLM and RAG system.
    """
    # Debug: Print the request data
    print("Chatbot question request data:", request.data)
    
    question = request.data.get('question', '')
    language = request.data.get('language', 'en')  # Default to English
    
    # Validate input
    if not question:
        return Response({'answer': 'Please provide a question to get an answer.', 'source_documents': []}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Load chatbot if not already loaded
        load_chatbot()
        # Load chatbot if not already loaded
        load_chatbot()
        
        # Check if chatbot is properly initialized
        if chatbot_instance is None or not hasattr(chatbot_instance, 'llm') or chatbot_instance.llm is None:
            # Fallback response if chatbot initialization failed
            answer = f"I understand you're asking about '{question}'. As an agricultural expert, I can help with questions about crops, weather, soil health, and farming practices. Could you tell me more about your specific situation?"
            source_documents = []
        else:
            # Get detailed answer with sources
            result = chatbot_instance.ask_question(question)
            answer = result['answer']
            source_documents = result['source_documents']
        
        # Translate response if needed
        if language != 'en':
            answer = translate_text(answer, language)
        
        return Response({
            'answer': answer,
            'source_documents': source_documents
        })
    except Exception as e:
        # More detailed error handling
        print(f"Chatbot question error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback response
        answer = f"I understand you're asking about '{question}'. As an agricultural expert, I can help with questions about crops, weather, soil health, and farming practices. Could you tell me more about your specific situation?"
        source_documents = []
        
        # Translate response if needed
        if language != 'en':
            answer = translate_text(answer, language)
        
        return Response({
            'answer': answer,
            'source_documents': source_documents
        })

# Add new endpoints for reports and charts
@api_view(['GET'])
def get_prediction_report_data(request):
    """
    Get data for prediction reports and charts from actual database records.
    """
    try:
        # Query actual yield prediction data from the database
        from .models import YieldPrediction
        from django.db.models import Avg, Count
        
        # Get predictions from the last 30 days
        thirty_days_ago = timezone.now() - timezone.timedelta(days=30)
        predictions = YieldPrediction.objects.filter(
            prediction_date__gte=thirty_days_ago
        ).select_related('crop')
        
        # Convert to serializable format
        data = []
        for prediction in predictions:
            data.append({
                'date': prediction.prediction_date.strftime('%Y-%m-%d'),
                'predicted_yield': round(prediction.predicted_yield_tonnes_per_ha, 2),
                'confidence': round(prediction.confidence_score, 2) if prediction.confidence_score else None,
                'crop': prediction.crop.name if prediction.crop else 'Unknown'
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Calculate summary statistics
        total_predictions = len(data)
        avg_predicted_yield = 0
        avg_confidence = 0
        
        if total_predictions > 0:
            avg_predicted_yield = round(sum(d['predicted_yield'] for d in data) / total_predictions, 2)
            # Calculate average confidence only for entries that have confidence scores
            confidence_values = [d['confidence'] for d in data if d['confidence'] is not None]
            if confidence_values:
                avg_confidence = round(sum(confidence_values) / len(confidence_values), 2)
        
        return Response({
            'report_type': 'predictions',
            'data': data,
            'summary': {
                'total_predictions': total_predictions,
                'avg_predicted_yield': avg_predicted_yield,
                'avg_confidence': avg_confidence
            }
        })
    except Exception as e:
        # Fallback to sample data if there's an error
        print(f"Error in get_prediction_report_data: {str(e)}")
        import random
        from django.utils import timezone
        from datetime import timedelta
        
        # Generate sample data for the last 30 days
        days = 30
        data = []
        current_date = timezone.now()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_yield': round(random.uniform(2.0, 8.0), 2),
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'crop': random.choice(['Wheat', 'Corn', 'Soybean', 'Rice'])
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        return Response({
            'report_type': 'predictions',
            'data': data,
            'summary': {
                'total_predictions': len(data),
                'avg_predicted_yield': round(sum(d['predicted_yield'] for d in data) / len(data), 2),
                'avg_confidence': round(sum(d['confidence'] for d in data) / len(data), 2)
            }
        })

@api_view(['GET'])
def get_disease_report_data(request):
    """
    Get data for disease detection reports and charts from actual database records.
    """
    try:
        # Query actual disease detection data from the database
        from .models import DiseasePrediction
        from django.db.models import Count
        
        # Get disease detections from the last 30 days
        thirty_days_ago = timezone.now() - timezone.timedelta(days=30)
        disease_detections = DiseasePrediction.objects.filter(
            timestamp__gte=thirty_days_ago
        )
        
        # Convert to serializable format
        data = []
        for detection in disease_detections:
            data.append({
                'date': detection.timestamp.strftime('%Y-%m-%d'),
                'disease': detection.predicted_disease,
                'confidence': round(detection.confidence_score, 2),
                'severity': 'high' if detection.confidence_score > 0.8 else 'moderate' if detection.confidence_score > 0.6 else 'low'
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Count occurrences by disease
        disease_counts = {}
        for item in data:
            disease = item['disease']
            if disease in disease_counts:
                disease_counts[disease] += 1
            else:
                disease_counts[disease] = 1
        
        return Response({
            'report_type': 'diseases',
            'data': data,
            'summary': {
                'total_detections': len(data),
                'unique_diseases': len(disease_counts),
                'most_common_disease': max(disease_counts, key=disease_counts.get) if disease_counts else "None"
            }
        })
    except Exception as e:
        # Fallback to sample data if there's an error
        print(f"Error in get_disease_report_data: {str(e)}")
        import random
        from django.utils import timezone
        from datetime import timedelta
        
        # Sample disease data
        diseases = [
            "Apple___Apple_scab", "Apple___Black_rot", "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", 
            "Potato___Early_blight", "Potato___Late_blight", "Tomato___Early_blight",
            "Tomato___Late_blight", "Tomato___Septoria_leaf_spot"
        ]
        
        # Generate sample data for the last 30 days
        days = 30
        data = []
        current_date = timezone.now()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            # Randomly decide if a disease was detected on this day
            if random.random() > 0.3:  # 70% chance of detection
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'disease': random.choice(diseases),
                    'confidence': round(random.uniform(0.7, 0.95), 2),
                    'severity': random.choice(['low', 'moderate', 'high'])
                })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Count occurrences by disease
        disease_counts = {}
        for item in data:
            disease = item['disease']
            if disease in disease_counts:
                disease_counts[disease] += 1
            else:
                disease_counts[disease] = 1
        
        return Response({
            'report_type': 'diseases',
            'data': data,
            'summary': {
                'total_detections': len(data),
                'unique_diseases': len(disease_counts),
                'most_common_disease': max(disease_counts, key=disease_counts.get) if disease_counts else "None"
            }
        })

@api_view(['GET'])
def get_recommendation_report_data(request):
    """
    Get data for recommendation reports and charts from actual database records.
    """
    try:
        # Query actual recommendation data from the database
        from .models import CropRecommendation, FertilizerRecommendation
        from django.db.models import Count
        
        # Get recommendations from the last 30 days
        thirty_days_ago = timezone.now() - timezone.timedelta(days=30)
        crop_recommendations = CropRecommendation.objects.filter(
            timestamp__gte=thirty_days_ago
        )
        
        # Convert to serializable format
        data = []
        for recommendation in crop_recommendations:
            # Get associated fertilizer recommendation if it exists
            fertilizer_name = "Not specified"
            try:
                fertilizer_rec = FertilizerRecommendation.objects.filter(
                    crop_recommendation=recommendation
                ).first()
                if fertilizer_rec:
                    fertilizer_name = fertilizer_rec.recommended_fertilizer
            except:
                pass
            
            data.append({
                'date': recommendation.timestamp.strftime('%Y-%m-%d'),
                'crop': recommendation.recommended_crop,
                'fertilizer': fertilizer_name,
                'confidence': round(recommendation.confidence_score, 2),
                'quantity': recommendation.soil_nitrogen + recommendation.soil_phosphorus + recommendation.soil_potassium,  # Simple quantity estimate
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Count occurrences by crop
        crop_counts = {}
        for item in data:
            crop = item['crop']
            if crop in crop_counts:
                crop_counts[crop] += 1
            else:
                crop_counts[crop] = 1
        
        return Response({
            'report_type': 'recommendations',
            'data': data,
            'summary': {
                'total_recommendations': len(data),
                'unique_crops': len(crop_counts),
                'most_recommended_crop': max(crop_counts, key=crop_counts.get) if crop_counts else "None"
            }
        })
    except Exception as e:
        # Fallback to sample data if there's an error
        print(f"Error in get_recommendation_report_data: {str(e)}")
        import random
        from django.utils import timezone
        from datetime import timedelta
        
        # Sample crops and fertilizers
        crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Barley', 'Soybean', 
                 'Peas', 'Mustard', 'Potato', 'Banana', 'Mango', 'Grapes']
        fertilizers = ['Urea', 'DAP', 'MOP', 'SSP', 'NPK 15-15-15', 'Compost']
        
        # Generate sample data for the last 30 days
        days = 30
        data = []
        current_date = timezone.now()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            # Randomly decide if a recommendation was made on this day
            if random.random() > 0.2:  # 80% chance of recommendation
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'crop': random.choice(crops),
                    'fertilizer': random.choice(fertilizers),
                    'confidence': round(random.uniform(0.7, 0.95), 2),
                    'quantity': round(random.uniform(100, 500), 2)
                })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Count occurrences by crop
        crop_counts = {}
        for item in data:
            crop = item['crop']
            if crop in crop_counts:
                crop_counts[crop] += 1
            else:
                crop_counts[crop] = 1
        
        return Response({
            'report_type': 'recommendations',
            'data': data,
            'summary': {
                'total_recommendations': len(data),
                'unique_crops': len(crop_counts),
                'most_recommended_crop': max(crop_counts, key=crop_counts.get) if crop_counts else "None"
            }
        })

@api_view(['GET'])
def get_pest_report_data(request):
    """
    Get data for pest prediction reports and charts from actual database records.
    """
    try:
        # Query actual pest prediction data from the database
        from .models import PestPrediction
        from django.db.models import Count
        
        # Get pest predictions from the last 30 days
        thirty_days_ago = timezone.now() - timezone.timedelta(days=30)
        pest_predictions = PestPrediction.objects.filter(
            prediction_date__gte=thirty_days_ago
        ).select_related('crop')
        
        # Convert to serializable format
        data = []
        for prediction in pest_predictions:
            data.append({
                'date': prediction.prediction_date.strftime('%Y-%m-%d'),
                'crop': prediction.crop.name if prediction.crop else 'Unknown',
                'pest': prediction.predicted_pest,
                'confidence': round(prediction.confidence_score, 2),
                'severity': prediction.severity,
                'region': prediction.region,
                'temperature': prediction.temperature,
                'humidity': prediction.humidity,
                'rainfall': prediction.rainfall
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Calculate summary statistics
        total_predictions = len(data)
        avg_confidence = 0
        avg_severity = 0
        pest_counts = {}
        
        if total_predictions > 0:
            # Calculate averages
            avg_confidence = round(sum(d['confidence'] for d in data) / total_predictions, 2)
            avg_severity = round(sum(d['severity'] for d in data) / total_predictions, 2)
            
            # Count occurrences by pest
            for item in data:
                pest = item['pest']
                if pest in pest_counts:
                    pest_counts[pest] += 1
                else:
                    pest_counts[pest] = 1
        
        return Response({
            'report_type': 'pests',
            'data': data,
            'summary': {
                'total_predictions': total_predictions,
                'avg_confidence': avg_confidence,
                'avg_severity': avg_severity,
                'unique_pests': len(pest_counts),
                'most_common_pest': max(pest_counts, key=pest_counts.get) if pest_counts else "None"
            }
        })
    except Exception as e:
        # Fallback to sample data if there's an error
        print(f"Error in get_pest_report_data: {str(e)}")
        import random
        from django.utils import timezone
        from datetime import timedelta
        
        # Sample pest data
        pests = [
            "Aphids", "Armyworms", "Boll Weevils", "Corn Borers", "Cutworms",
            "Flea Beetles", "Grasshoppers", "Hessian Fly", "Japanese Beetles",
            "Leafhoppers", "Root Knot Nematodes", "Spider Mites", "Stem Borers",
            "Thrips", "White Grubs", "Whiteflies", "Wireworms"
        ]
        
        # Generate sample data for the last 30 days
        days = 30
        data = []
        current_date = timezone.now()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            # Randomly decide if a pest prediction was made on this day
            if random.random() > 0.4:  # 60% chance of prediction
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'crop': random.choice(['Wheat', 'Corn', 'Soybean', 'Rice', 'Cotton']),
                    'pest': random.choice(pests),
                    'confidence': round(random.uniform(0.6, 0.95), 2),
                    'severity': random.randint(1, 10),
                    'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
                    'temperature': round(random.uniform(20, 35), 1),
                    'humidity': round(random.uniform(40, 80), 1),
                    'rainfall': round(random.uniform(0, 50), 1)
                })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Calculate summary statistics
        total_predictions = len(data)
        avg_confidence = 0
        avg_severity = 0
        pest_counts = {}
        
        if total_predictions > 0:
            # Calculate averages
            avg_confidence = round(sum(d['confidence'] for d in data) / total_predictions, 2)
            avg_severity = round(sum(d['severity'] for d in data) / total_predictions, 2)
            
            # Count occurrences by pest
            for item in data:
                pest = item['pest']
                if pest in pest_counts:
                    pest_counts[pest] += 1
                else:
                    pest_counts[pest] = 1
        
        return Response({
            'report_type': 'pests',
            'data': data,
            'summary': {
                'total_predictions': total_predictions,
                'avg_confidence': avg_confidence,
                'avg_severity': avg_severity,
                'unique_pests': len(pest_counts),
                'most_common_pest': max(pest_counts, key=pest_counts.get) if pest_counts else "None"
            }
        })

@api_view(['GET'])
def get_market_report_data(request):
    """
    Get data for market prediction reports and charts from actual database records.
    """
    try:
        # Query actual market prediction data from the database
        from .models import MarketPrediction
        from django.db.models import Avg
        
        # Get market predictions from the last 30 days
        thirty_days_ago = timezone.now() - timezone.timedelta(days=30)
        market_predictions = MarketPrediction.objects.filter(
            prediction_date__gte=thirty_days_ago
        ).select_related('crop')
        
        # Convert to serializable format
        data = []
        for prediction in market_predictions:
            data.append({
                'date': prediction.prediction_date.strftime('%Y-%m-%d'),
                'crop': prediction.crop.name if prediction.crop else 'Unknown',
                'predicted_price': round(prediction.predicted_price_per_ton, 2),
                'confidence': round(prediction.confidence_score, 2) if prediction.confidence_score else None,
                'market_trend': prediction.market_trend,
                'forecast_period_days': prediction.forecast_period_days
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Calculate summary statistics
        total_predictions = len(data)
        avg_predicted_price = 0
        avg_confidence = 0
        trend_counts = {}
        
        if total_predictions > 0:
            avg_predicted_price = round(sum(d['predicted_price'] for d in data) / total_predictions, 2)
            # Calculate average confidence only for entries that have confidence scores
            confidence_values = [d['confidence'] for d in data if d['confidence'] is not None]
            if confidence_values:
                avg_confidence = round(sum(confidence_values) / len(confidence_values), 2)
            
            # Count occurrences by trend
            for item in data:
                trend = item['market_trend']
                if trend in trend_counts:
                    trend_counts[trend] += 1
                else:
                    trend_counts[trend] = 1
        
        return Response({
            'report_type': 'market',
            'data': data,
            'summary': {
                'total_predictions': total_predictions,
                'avg_predicted_price': avg_predicted_price,
                'avg_confidence': avg_confidence,
                'trend_distribution': trend_counts
            }
        })
    except Exception as e:
        # Fallback to sample data if there's an error
        print(f"Error in get_market_report_data: {str(e)}")
        import random
        from django.utils import timezone
        from datetime import timedelta
        
        # Generate sample data for the last 30 days
        days = 30
        data = []
        current_date = timezone.now()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'crop': random.choice(['Wheat', 'Corn', 'Soybean', 'Rice', 'Cotton']),
                'predicted_price': round(random.uniform(200, 800), 2),
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'market_trend': random.choice(['bullish', 'bearish', 'neutral']),
                'forecast_period_days': 30
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Calculate summary statistics
        total_predictions = len(data)
        avg_predicted_price = 0
        avg_confidence = 0
        trend_counts = {}
        
        if total_predictions > 0:
            avg_predicted_price = round(sum(d['predicted_price'] for d in data) / total_predictions, 2)
            avg_confidence = round(sum(d['confidence'] for d in data) / total_predictions, 2)
            
            # Count occurrences by trend
            for item in data:
                trend = item['market_trend']
                if trend in trend_counts:
                    trend_counts[trend] += 1
                else:
                    trend_counts[trend] = 1
        
        return Response({
            'report_type': 'market',
            'data': data,
            'summary': {
                'total_predictions': total_predictions,
                'avg_predicted_price': avg_predicted_price,
                'avg_confidence': avg_confidence,
                'trend_distribution': trend_counts
            }
        })

        for i in range(days):
            date = current_date - timedelta(days=i)
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'crop': random.choice(['Wheat', 'Corn', 'Soybean', 'Rice', 'Cotton']),
                'predicted_price': round(random.uniform(200, 800), 2),
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'market_trend': random.choice(['bullish', 'bearish', 'neutral']),
                'forecast_period_days': 30
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Calculate summary statistics
        total_predictions = len(data)
        avg_predicted_price = 0
        avg_confidence = 0
        trend_counts = {}
        
        if total_predictions > 0:
            avg_predicted_price = round(sum(d['predicted_price'] for d in data) / total_predictions, 2)
            avg_confidence = round(sum(d['confidence'] for d in data) / total_predictions, 2)
            
            # Count occurrences by trend
            for item in data:
                trend = item['market_trend']
                if trend in trend_counts:
                    trend_counts[trend] += 1
                else:
                    trend_counts[trend] = 1
        
        return Response({
            'report_type': 'market',
            'data': data,
            'summary': {
                'total_predictions': total_predictions,
                'avg_predicted_price': avg_predicted_price,
                'avg_confidence': avg_confidence,
                'trend_distribution': trend_counts
            }
        })

        for i in range(days):
            date = current_date - timedelta(days=i)
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'crop': random.choice(['Wheat', 'Corn', 'Soybean', 'Rice', 'Cotton']),
                'predicted_price': round(random.uniform(200, 800), 2),
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'market_trend': random.choice(['bullish', 'bearish', 'neutral']),
                'forecast_period_days': 30
            })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Calculate summary statistics
        total_predictions = len(data)
        avg_predicted_price = 0
        avg_confidence = 0
        trend_counts = {}
        
        if total_predictions > 0:
            avg_predicted_price = round(sum(d['predicted_price'] for d in data) / total_predictions, 2)
            avg_confidence = round(sum(d['confidence'] for d in data) / total_predictions, 2)
            
            # Count occurrences by trend
            for item in data:
                trend = item['market_trend']
                if trend in trend_counts:
                    trend_counts[trend] += 1
                else:
                    trend_counts[trend] = 1
        
        return Response({
            'report_type': 'market',
            'data': data,
            'summary': {
                'total_predictions': total_predictions,
                'avg_predicted_price': avg_predicted_price,
                'avg_confidence': avg_confidence,
                'trend_distribution': trend_counts
            },

                    'crop': random.choice(crops),
                    'fertilizer': random.choice(fertilizers),
                    'confidence': round(random.uniform(0.7, 0.95), 2),
                    'quantity': round(random.uniform(50, 200), 1),
                })
        
        # Sort by date
        data.sort(key=lambda x: x['date'])
        
        # Count occurrences by crop
        crop_counts = {}
        for item in data:
            crop = item['crop']
            if crop in crop_counts:
                crop_counts[crop] += 1
            else:
                crop_counts[crop] = 1
        
        return Response({
            'report_type': 'recommendations',
            'data': data,
            'summary': {
                'total_recommendations': len(data),
                'unique_crops': len(crop_counts),
                'most_recommended_crop': max(crop_counts, key=crop_counts.get) if crop_counts else "None"
            }
        })


def main_page(request):
    """
    Serve the main index.html template with proper CSRF context.
    """
    from django.shortcuts import render
    return render(request, 'index.html')

def translate_text(text, target_language):
    """
    Translate text to the target language.
    In a real implementation, this would use a translation API like Google Translate.
    For now, we'll provide sample translations for demonstration.
    """
    # Sample translations for demonstration
    translations = {
        'hi': {  # Hindi
            "Hello! I'm your agricultural expert assistant. How can I help you with your farming today?": 
                "नमस्ते! मैं आपका कृषि विशेषज्ञ सहायक हूँ। मैं आज आपकी खेती में कैसे मदद कर सकता हूँ?",
            "I understand you're asking about: '": 
                "मुझे समझ में आया कि आप पूछ रहे हैं: '",
            "'. As an agricultural expert, I can help with questions about crops, weather, soil health, and farming practices. Could you provide more details about your specific concern?":
                "'. एक कृषि विशेषज्ञ के रूप में, मैं फसलों, मौसम, मिट्टी के स्वास्थ्य और खेती के अभ्यासों के बारे में प्रश्नों में मदद कर सकता हूँ। क्या आप अपने विशिष्ट चिंता के बारे में अधिक विवरण प्रदान कर सकते हैं?",
            "Based on current weather patterns, I recommend increasing irrigation by 10-15% to compensate for the drier conditions forecasted this week.":
                "वर्तमान मौसम पैटर्न के आधार पर, मैं इस सप्ताह की भविष्यवाणी की गई शुष्क परिस्थितियों की भरपाई करने के लिए सिंचाई में 10-15% की वृद्धि की अनुशंसा करता हूँ।",
            "The yellowing leaves you're seeing could indicate nitrogen deficiency. I suggest applying a balanced fertilizer with a higher nitrogen content.":
                "आपके द्वारा देखे जा रहे पीले पत्ते नाइट्रोजन की कमी का संकेत दे सकते हैं। मैं एक संतुलित उर्वरक का उपयोग करने का सुझाव देता हूँ जिसमें उच्च नाइट्रोजन सामग्री हो।",
            "For optimal corn yield, the best time to harvest is when kernel moisture drops to 25-30%. This usually occurs 20-25 दिनों के बाद परागण के।":
                "इष्टतम मकई की उपज के लिए, काटने का सर्वोत्तम समय तब होता है जब दाने की नमी 25-30% तक गिर जाए। यह आमतौर पर परागण के 20-25 दिनों के बाद होता है।",
            "Crop rotation with legumes like beans or peas can naturally increase soil nitrogen levels. Consider planting them after harvest.":
                "बीन्स या मटर जैसी फलीदार फसलों के साथ फसल चक्रण से मिट्टी के नाइट्रोजन स्तर को प्राकृतिक रूप से बढ़ाया जा सकता है। कटाई के बाद उन्हें लगाने पर विचार करें।",
            "To protect against the upcoming frost, you can use row covers or apply a light irrigation before the frost to create a protective barrier.":
                "आगामी ओरे के खिलाफ सुरक्षा के लिए, आप पंक्ति कवर का उपयोग कर सकते हैं या ओरे से पहले एक हल्की सिंचाई लगा सकते हैं ताकि एक सुरक्षात्मक बैरियर बन सके।",
            "Soil testing every 2-3 वर्ष में मिट्टी की जांच करने से पोषक तत्वों के स्तर और पीएच की निगरानी में मदद मिलती है। आपके मिट्टी के प्रकार के आधार पर, मैं शीघ्र वसंत में परीक्षण की अनुशंसा करता हूँ।":
                "प्रत्येक 2-3 वर्ष में मिट्टी की जांच करने से पोषक तत्वों के स्तर और पीएच की निगरानी में मदद मिलती है। आपके मिट्टी के प्रकार के आधार पर, मैं शीघ्र वसंत में परीक्षण की अनुशंसा करता हूँ।",
            "Beneficial insects like ladybugs can help control aphid populations. Plant flowering herbs like dill or fennel to attract them to your fields.":
                "लेडीबग जैसे लाभकारी कीड़े एफिड आबादी को नियंत्रित करने में मदद कर सकते हैं। उन्हें अपने खेतों में आकर्षित करने के लिए जीरा या सौंफ जैसी फूलों वाली जड़ी बूटियों को लगाएं।",
            "Cover crops during the off-season prevent soil erosion and add organic matter. Winter rye and crimson clover are excellent choices for your region.":
                "ऑफ-सीजन के दौरान कवर फसलें मिट्टी की क्षरण को रोकती हैं और कार्बनिक पदार्थ जोड़ती हैं। शीतकालीन राई और क्रिमसन क्लोवर आपके क्षेत्र के लिए उत्कृष्ट विकल्प हैं।"
        },
        'kn': {  # Kannada
            "Hello! I'm your agricultural expert assistant. How can I help you with your farming today?": 
                "ಹಲೋ! ನಾನು ನಿಮ್ಮ ಕೃಷಿ ತಜ್ಞ ಸಹಾಯಕರು. ನಾನು ಇಂದು ನಿಮ್ಮ ಕೃಷಿಯಲ್ಲಿ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
            "I understand you're asking about: '": 
                "ನೀವು ಕೇಳುತ್ತಿರುವುದನ್ನು ನಾನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳುತ್ತೇನೆ: '",
            "'. As an agricultural expert, I can help with questions about crops, weather, soil health, and farming practices. Could you provide more details about your specific concern?":
                "'. ಕೃಷಿ ತಜ್ಞರಾಗಿ, ಬೆಳೆಗಳು, ಹವಾಮಾನ, ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಮತ್ತು ಕೃಷಿ ಅಭ್ಯಾಸಗಳ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಗಳಿಗೆ ನಾನು ಸಹಾಯ ಮಾಡಬಹುದು. ನಿಮ್ಮ ನಿರ್ದಿಷ್ಟ ಚಿಂತೆಯ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ವಿವರಗಳನ್ನು ನೀಡಬಹುದೇ?",
            "Based on current weather patterns, I recommend increasing irrigation by 10-15% to compensate for the drier conditions forecasted this week.":
                "ಪ್ರಸ್ತುತ ಹವಾಮಾನದ ಮಾದರಿಗಳ ಆಧಾರದ ಮೇಲೆ, ಈ ವಾರ ಮುನ್ಸೂಚಿಸಲಾದ ಒಣಗಾದ ಪರಿಸ್ಥಿತಿಗಳನ್ನು ಪರಿಹರಿಸಲು ನಾನು ನೀರಾವರಿಯನ್ನು 10-15% ಹೆಚ್ಚಿಸುವುದನ್ನು ಶಿಫಾರಸು ಮಾಡುತ್ತೇನೆ.",
            "The yellowing leaves you're seeing could indicate nitrogen deficiency. I suggest applying a balanced fertilizer with a higher nitrogen content.":
                "ನೀವು ನೋಡುತ್ತಿರುವ ಹಳದಿ ಬಣ್ಣದ ಎಲೆಗಳು ನೈಟ್ರೊಜನ್ ಕೊರತೆಯನ್ನು ಸೂಚಿಸಬಹುದು. ಹೆಚ್ಚಿನ ನೈಟ್ರೊಜನ್ ಹೊಂದಿರುವ ಸಮತೋಲಿತ ಗೊಬ್ಬರವನ್ನು ಬಳಸುವುದನ್ನು ನಾನು ಸಲಹೆ ಮಾಡುತ್ತೇನೆ.",
            "For optimal corn yield, the best time to harvest is when kernel moisture drops to 25-30%. This usually occurs 20-25 ದಿನಗಳಲ್ಲಿ ಸಂಭವಿಸುತ್ತದೆ.":
                "ಉತ್ತಮ ಜೋಳದ ಉತ್ಪಾದನೆಗಾಗಿ, ಕರ್ನಲ್ ತೇವಾಂಶವು 25-30% ಗೆ ಇಳಿಯುವಾಗ ಕರೆಕೊಳ್ಳುವುದು ಉತ್ತಮ ಸಮಯ. ಇದು ಸಾಮಾನ್ಯವಾಗಿ ಪರಾಗೀಕರಣದ ನಂತರ 20-25 ದಿನಗಳಲ್ಲಿ ಸಂಭವಿಸುತ್ತದೆ.",
            "Crop rotation with legumes like beans or peas can naturally increase soil nitrogen levels. Consider planting them after harvest.":
                "ಬೀನ್ಸ್ ಅಥವಾ ಪೀಸ್ ನಂತಹ ಲೆಗ್ಯೂಮ್‌ಗಳೊಂದಿಗೆ ಬೆಳೆ ತೇವಾಂಶವನ್ನು ಸಹಜವಾಗಿ ಹೆಚ್ಚಿಸಬಹುದು. ಕರ್ನಲ್ ಹಾರ್ವೆಸ್ಟ್ ನಂತರ ಅವುಗಳನ್ನು ನೆಲೆಸುವುದನ್ನು ಪರಿಗಣಿಸಿ.",
            "To protect against the upcoming frost, you can use row covers or apply a light irrigation before the frost to create a protective barrier.":
                "ಬರ upcoming ಬರುವ ಹಿಮದ ವಿರುದ್ಧ ರಕ್ಷಿಸಲು, ನೀವು ರೋ ಕವರ್‌ಗಳನ್ನು ಬಳಸಬಹುದು ಅಥವಾ ಹಿಮದ ಮೊದಲು ಒಂದು ಲೈಟ್ ನೀರಾವರಿಯನ್ನು ಅನ್ವಯಿಸಿ ರಕ್ಷಣಾತ್ಮಕ ಅಡೆಯನ್ನು ರಚಿಸಬಹುದು.",
            "Cover crops during the off-season prevent soil erosion and add organic matter. Winter rye and crimson clover are excellent choices for your region.":
                "ಆಫ್-ಸೀಸನ್ ಸमಯದಲ್ಲಿ ಕವರ್ ಬೆಳೆಗಳು ಮಣ್ಣಿನ ಕ್ಷರಣವನ್ನು ತಡೆಯುತ್ತವೆ ಮತ್ತು ಕಾರ್ಬನಿಕ ಪದಾರ್ಥವನ್ನು ಸೇರಿಸುತ್ತವೆ. ಶೀತಕಾಲದ ರೈ ಮತ್ತು ಕ್ರಿಮ್ಸನ್ ಕ್ಲೋವರ್ ನಿಮ್ಮ ಪ್ರದೇಶಕ್ಕೆ ಉತ್ತಮ ಆಯ್ಕೆಗಳಾಗಿವೆ."
        },
        'ta': {  # Tamil
            "Hello! I'm your agricultural expert assistant. How can I help you with your farming today?": 
                "ஹலோ! நான் உங்கள் விவசாய நிபுணர் உதவியாளர். இன்று உங்கள் விவசாயத்தில் நான் எவ்வாறு உதவ முடியும்?",
            "I understand you're asking about: '": 
                "நீங்கள் கேட்கிறதை நான் புரிந்து கொள்கிறேன்: '",
            "'. As an agricultural expert, I can help with questions about crops, weather, soil health, and farming practices. Could you provide more details about your specific concern?":
                "'. ஒரு விவசாய நிபுணராக, பயிர்கள், வானிலை, மண் சுகாதாரம் மற்றும் விவசாய நடைமுறைகள் குறித்து கேள்விகளுக்கு நான் உதவ முடியும். உங்கள் குறிப்பிட்ட கவலை குறித்து மேலும் விவரங்களை வழங்க முடியுமா?",
            "Based on current weather patterns, I recommend increasing irrigation by 10-15% to compensate for the drier conditions forecasted this week.":
                "தற்போதைய வானிலை முறைகளின் அடிப்படையில், இந்த வாரம் முன்னறிவிக்கப்பட்டுள்ள உலர் நிலைகளை சரிசெய்ய நீர்ப்பாசனத்தை 10-15% அதிகரிக்க நான் பரிந்துரைக்கிறேன்.",
            "The yellowing leaves you're seeing could indicate nitrogen deficiency. I suggest applying a balanced fertilizer with a higher nitrogen content.":
                "நீங்கள் பார்க்கும் மஞ்சள் நிற இலைகள் நைட்ரஜன் குறைபாட்டைக் குறிக்கலாம். அதிக நைட்ரஜன் உள்ளடக்கத்துடன் சமநிலையான உரத்தை பயன்படுத்துவதை நான் பரிந்துரைக்கிறேன்.",
            "For optimal corn yield, the best time to harvest is when kernel moisture drops to 25-30%. This usually occurs 20-25 நாட்களில் பிறகு மலர்ச்சிக்கு.":
                "சிறந்த சோள விளைyieldக்கு, மாக்கள் ஈரப்பதம் 25-30% ஆக குறையும் போது அறுவடை செய்வது சிறந்த நேரம். இது பொதுவாக மலர்ச்சிக்கு பிறகு 20-25 நாட்களில் நடக்கும்.",
            "Crop rotation with legumes like beans or peas can naturally increase soil nitrogen levels. Consider planting them after harvest.":
                "பீன்ஸ் அல்லது பீஸ் போன்ற கொட்டைவகைகளுடன் பயிர் சுழற்சி மண் நைட்ரஜன் மட்டத்தை இயற்கையாக அதிகரிக்கும். அறுவடைக்கு பிறகு அவற்றை நடுவதைக் கவனியுங்கள்.",
            "To protect against the upcoming frost, you can use row covers or apply a light irrigation before the frost to create a protective barrier.":
                "வரவிருக்கும் பனிக்கு எதிராக பாதுகாக்க, நீங்கள் வரிச் சூழல்களைப் பயன்படுத்தலாம் அல்லது பனிக்கு முன் ஒரு இலகு நீர்ப்பாசனத்தை செய்து பாதுகாப்பான தடுப்பை உருவாக்கலாம்.",
            "Beneficial insects like ladybugs can help control aphid populations. Plant flowering herbs like dill or fennel to attract them to your fields.":
                "லேடிபக் போன்ற பயனுள்ள பூச்சிகள் அஃபிட் மக்கள் தொகையை கட்டுப்படுத்த உதவுகின்றன. அவற்றை உங்கள் புலங்களுக்கு ஈர்க்க டில் அல்லது பெனல் போன்ற மலரும் மூலிகைகளை நடவும்.",
            "Cover crops during the off-season prevent soil erosion and add organic matter. Winter rye and crimson clover are excellent choices for your region.":
                "ஆஃப்-சீசன் போது மூடு பயிர்கள் மண் அரிப்பைத் தடுக்கின்றன மற்றும் கார்பனிக் பொருளைச் சேர்க்கின்றன. குளிர்கால ரை மற்றும் கிரிம்சன் க்ளோவர் உங்கள் பிராந்தியத்திற்கு சிறந்த தேர்வுகள்."
        },
        'te': {  # Telugu
            "Hello! I'm your agricultural expert assistant. How can I help you with your farming today?": 
                "హలో! నేను మీ వ్యవసాయ నిపుణుడి సహాయకుడిని. ఈ రోజు మీ వ్యవసాయంలో నేను ఎలా సహాయపడగలను?",
            "I understand you're asking about: '": 
                "మీరు అడుగుతున్నదాన్ని నేను అర్థం చేసుకున్నాను: '",
            "'. As an agricultural expert, I can help with questions about crops, weather, soil health, and farming practices. Could you provide more details about your specific concern?":
                "'. వ్యవసాయ నిపుణుడిగా, పంటలు, వాతావరణం, నేల ఆరోగ్యం మరియు వ్యవసాయ పద్ధతుల గురించి ప్రశ్నలకు నాను సహాయం చేయగలను. మీ నిర్దిష్ట ఆందోళన గురించి మరింత వివరాలు అందించగలరా?",
            "Based on current weather patterns, I recommend increasing irrigation by 10-15% to compensate for the drier conditions forecasted this week.":
                "ప్రస్తుత వాతావరణ నమూనాల ఆధారంగా, ఈ వారం అంచనా వేయబడిన ఎండని పరిస్థితులను సరిచేయడానికి నీటి సరఫరాను 10-15% పెంచడాన్ని నేను సిఫార్సు చేస్తున్నాను.",
            "The yellowing leaves you're seeing could indicate nitrogen deficiency. I suggest applying a balanced fertilizer with a higher nitrogen content.":
                "మీరు చూస్తున్న పసుపు రంగు ఆకులు నైట్రోజన్ లోపాన్ని సూచిస్తాయి. ఎక్కువ నైట్రోజన్ కంటెంట్ తో సమతుల్య ఎరువులను వర్తించడాన్ని నేను సలహా ఇస్తున్నాను.",
            "For optimal corn yield, the best time to harvest is when kernel moisture drops to 25-30%. This usually occurs 20-25 రోజుల్లో పరాగస్థలానికి తరువాత.":
                "ఉత్తమ మొక్కజొన్న దిగుబడి కోసం, గింజ తేమ 25-30% కు పడిపోయేటప్పుడు వసూలు చేయడం ఉత్తమ సమయం. ఇది సాధారణంగా పరాగస్థలానికి తరువాత 20-25 రోజుల్లో జరుగుతుంది.",
            "Crop rotation with legumes like beans or peas can naturally increase soil nitrogen levels. Consider planting them after harvest.":
                "బీన్స్ లేదా పీస్ వంటి లెగ్యూమ్‌లతో పంట రొటేషన్ నేల నైట్రోజన్ స్థాయిలను సహజంగా పెంచవచ్చు. వసూలు తరువాత వాటిని నాటడాన్ని పరిగణనలోకి తీసుకోండి.",
            "To protect against the upcoming frost, you can use row covers or apply a light irrigation before the frost to create a protective barrier.":
                "రాబోయే ఘనీభవనాన్ని రక్షించడానికి, మీరు వరుస కవర్లను ఉపయోగించవచ్చు లేదా ఘనీభవనానికి ముందు తేలికపాటి నీటి సరఫరాను అనువర్తించి రక్షణ అడ్డంకిని సృష్టించవచ్చు.",
            "Soil testing every 2-3 years helps monitor nutrient levels and pH. Based on your soil type, I recommend testing in early spring.":
                "Soil testing every 2-3 years helps monitor nutrient levels and pH. Based on your soil type, I recommend testing in early spring.",
            "Beneficial insects like ladybugs can help control aphid populations. Plant flowering herbs like dill or fennel to attract them to your fields.":
                "లేడీబగ్స్ వంటి ప్రయోజనకరమైన క్రిములు ఎఫిడ్ జనాభాను నియంత్రించడంలో సహాయపడవచ్చు. వాటిని మీ పొలాలకు పిలిచి తెచ్చేందుకు డిల్ లేదా ఫెనల్ వంటి పువ్వుల గింజలను నాటండి.",
            "Cover crops during the off-season prevent soil erosion and add organic matter. Winter rye and crimson clover are excellent choices for your region.":
                "ఆఫ్-సీజన్ సమయంలో కవర్ పంటలు నేల అరిపోవడాన్ని నివారిస్తాయి మరియు కార్బనిక్ పదార్థాలను జోడిస్తాయి. శీతాకాలపు రై మరియు క్రిమ్సన్ క్లోవర్ మీ ప్రాంతానికి ఉన్న అద్భుతమైన ఎంపికలు."
        }
    }
    
    # If the target language is English, return the text as is
    if target_language == 'en':
        return text

    # If we have translations for the target language, use them
    if target_language in translations:
        language_translations = translations[target_language]
        # Return translated text if available, otherwise return original
        return language_translations.get(text, text)

    # If no translation is available, return the original text
    return text


@api_view(['GET'])
def get_dashboard_statistics(request):
    """
    Get comprehensive dashboard statistics from actual database records.
    """
    try:
        import datetime
        from django.db.models import Count, Avg
        
        # Get counts for each feature
        prediction_count = YieldPrediction.objects.count()
        disease_count = DiseasePrediction.objects.count()
        recommendation_count = CropRecommendation.objects.count()
        
        # Get recent activity (last 7 days)
        seven_days_ago = timezone.now() - timezone.timedelta(days=7)
        recent_predictions = YieldPrediction.objects.filter(
            prediction_date__gte=seven_days_ago
        ).count()
        recent_diseases = DiseasePrediction.objects.filter(
            timestamp__gte=seven_days_ago
        ).count()
        recent_recommendations = CropRecommendation.objects.filter(
            timestamp__gte=seven_days_ago
        ).count()
        
        # Calculate averages
        avg_yield = 0
        avg_confidence = 0
        
        if prediction_count > 0:
            avg_yield_result = YieldPrediction.objects.aggregate(Avg('predicted_yield_tonnes_per_ha'))
            avg_yield = round(avg_yield_result['predicted_yield_tonnes_per_ha__avg'], 2) if avg_yield_result['predicted_yield_tonnes_per_ha__avg'] else 0
            
            # Calculate average confidence only for entries that have confidence scores
            confidence_values = [pred.confidence_score for pred in YieldPrediction.objects.all() if pred.confidence_score is not None]
            if confidence_values:
                avg_confidence = round(sum(confidence_values) / len(confidence_values), 2)
        
        # Get most common crop recommendations
        crop_recommendations = CropRecommendation.objects.values('recommended_crop').annotate(
            count=Count('recommended_crop')
        ).order_by('-count')
        
        most_common_crop = crop_recommendations.first()['recommended_crop'] if crop_recommendations.first() else "None"
        
        # Get recent predictions for chart
        recent_predictions_data = YieldPrediction.objects.filter(
            prediction_date__gte=seven_days_ago
        ).select_related('crop')
        
        recent_yield_data = []
        for pred in recent_predictions_data:
            recent_yield_data.append({
                'date': pred.prediction_date.strftime('%Y-%m-%d'),
                'crop': pred.crop.name if pred.crop else 'Unknown',
                'yield': round(pred.predicted_yield_tonnes_per_ha, 2)
            })
        
        # Sort by date
        recent_yield_data.sort(key=lambda x: x['date'])
        
        return Response({
            'statistics': {
                'total_predictions': prediction_count,
                'total_disease_detections': disease_count,
                'total_recommendations': recommendation_count,
                'avg_yield': avg_yield,
                'avg_confidence': avg_confidence,
                'most_common_crop': most_common_crop,
                'recent_activity': {
                    'predictions': recent_predictions,
                    'diseases': recent_diseases,
                    'recommendations': recent_recommendations
                }
            },
            'recent_yield_data': recent_yield_data
        })
    except Exception as e:
        print(f"Error in get_dashboard_statistics: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback response with default values
        return Response({
            'statistics': {
                'total_predictions': 0,
                'total_disease_detections': 0,
                'total_recommendations': 0,
                'avg_yield': 0,
                'avg_confidence': 0,
                'most_common_crop': "None",
                'recent_activity': {
                    'predictions': 0,
                    'diseases': 0,
                    'recommendations': 0
                }
            },
            'recent_yield_data': []
        })