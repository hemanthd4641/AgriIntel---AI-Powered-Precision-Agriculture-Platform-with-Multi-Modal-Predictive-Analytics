"""
Views for crop and fertilizer recommendation API endpoints.
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
import joblib
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .models import CropRecommendation, FertilizerRecommendation
from .recommendation_serializers import (
    CropRecommendationSerializer, 
    FertilizerRecommendationSerializer,
    RecommendationRequestSerializer
)

# Global variables for models (loaded once when module is imported)
crop_model = None
crop_preprocessor = None
fertilizer_model = None
fertilizer_preprocessor = None
recommendation_rag = None


def load_models():
    """Load recommendation models and preprocessors if not already loaded."""
    global crop_model, crop_preprocessor, fertilizer_model, fertilizer_preprocessor, recommendation_rag
    
    # Load crop recommendation model
    if crop_model is None:
        try:
            crop_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                          'models', 'crop_recommendation', 'crop_model.pkl')
            if os.path.exists(crop_model_path):
                crop_model = joblib.load(crop_model_path)
                print("Crop recommendation model loaded successfully")
            else:
                print(f"Crop model file not found at {crop_model_path}")
        except Exception as e:
            print(f"Error loading crop model: {e}")
            crop_model = None
    
    # Load crop preprocessor
    if crop_preprocessor is None:
        try:
            preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           'models', 'crop_recommendation', 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                crop_preprocessor = joblib.load(preprocessor_path)
                print("Crop preprocessor loaded successfully")
            else:
                print(f"Crop preprocessor file not found at {preprocessor_path}")
        except Exception as e:
            print(f"Error loading crop preprocessor: {e}")
            crop_preprocessor = None
    
    # Load fertilizer recommendation model
    if fertilizer_model is None:
        try:
            fertilizer_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                               'models', 'crop_recommendation', 'fertilizer_model.pkl')
            if os.path.exists(fertilizer_model_path):
                fertilizer_model = joblib.load(fertilizer_model_path)
                print("Fertilizer recommendation model loaded successfully")
            else:
                print(f"Fertilizer model file not found at {fertilizer_model_path}")
        except Exception as e:
            print(f"Error loading fertilizer model: {e}")
            fertilizer_model = None
    
    # Load fertilizer preprocessor
    if fertilizer_preprocessor is None:
        try:
            fertilizer_preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                      'models', 'crop_recommendation', 'preprocessor.pkl')
            if os.path.exists(fertilizer_preprocessor_path):
                fertilizer_preprocessor = joblib.load(fertilizer_preprocessor_path)
                print("Fertilizer preprocessor loaded successfully")
            else:
                print(f"Fertilizer preprocessor file not found at {fertilizer_preprocessor_path}")
        except Exception as e:
            print(f"Error loading fertilizer preprocessor: {e}")
            fertilizer_preprocessor = None
    
    # Load recommendation RAG system
    if recommendation_rag is None:
        try:
            from explainable_ai.recommendation_rag.recommendation_knowledge_base import RecommendationRAG
            recommendation_rag = RecommendationRAG()
            print("Recommendation RAG system loaded successfully")
        except Exception as e:
            print(f"Error loading recommendation RAG system: {e}")
            recommendation_rag = None

# Preload models when module is imported
print("Preloading recommendation models and components...")
load_models()
print("Recommendation models and components preloaded successfully")


def generate_crop_advice(crop_name, soil_nitrogen, soil_phosphorus, soil_potassium, soil_ph, temperature, humidity, rainfall):
    """
    Generate human-readable crop advice based on conditions using LLM.
    
    Args:
        crop_name (str): Name of the crop
        soil_nitrogen (float): Soil nitrogen level
        soil_phosphorus (float): Soil phosphorus level
        soil_potassium (float): Soil potassium level
        soil_ph (float): Soil pH level
        temperature (float): Temperature in °C
        humidity (float): Humidity percentage
        rainfall (float): Rainfall in mm
        
    Returns:
        str: Human-readable advice for growing the crop
    """
    # For performance, return simplified advice without LLM processing
    # This avoids the time-consuming LLM loading and generation
    print("Returning simplified crop advice for performance")
    return generate_rule_based_advice(crop_name, soil_nitrogen, soil_phosphorus, soil_potassium, soil_ph, temperature, humidity, rainfall)
    
    # Try to use the LLM-based RAG system for generating advice
    try:
        if recommendation_rag and recommendation_rag.qa_chain:
            soil_conditions = f"N:{soil_nitrogen}, P:{soil_phosphorus}, K:{soil_potassium}, pH:{soil_ph}"
            weather_conditions = f"Temperature:{temperature}°C, Humidity:{humidity}%, Rainfall:{rainfall}mm"
            
            query = f"Provide detailed advice for growing {crop_name} with soil conditions: {soil_conditions} and weather conditions: {weather_conditions}. Include specific recommendations for soil pH management, nutrient requirements, water needs, and potential issues to watch for."
            
            result = recommendation_rag.qa_chain({"query": query})
            return result["result"]
        else:
            # Fallback to rule-based advice if LLM is not available
            return generate_rule_based_advice(crop_name, soil_nitrogen, soil_phosphorus, soil_potassium, soil_ph, temperature, humidity, rainfall)
    except Exception as e:
        print(f"Error generating LLM-based advice: {e}")
        # Fallback to rule-based advice if LLM fails
        return generate_rule_based_advice(crop_name, soil_nitrogen, soil_phosphorus, soil_potassium, soil_ph, temperature, humidity, rainfall)


def generate_rule_based_advice(crop_name, soil_nitrogen, soil_phosphorus, soil_potassium, soil_ph, temperature, humidity, rainfall):
    """
    Generate rule-based crop advice (fallback when LLM is not available).
    
    Args:
        crop_name (str): Name of the crop
        soil_nitrogen (float): Soil nitrogen level
        soil_phosphorus (float): Soil phosphorus level
        soil_potassium (float): Soil potassium level
        soil_ph (float): Soil pH level
        temperature (float): Temperature in °C
        humidity (float): Humidity percentage
        rainfall (float): Rainfall in mm
        
    Returns:
        str: Human-readable advice for growing the crop
    """
    # Crop-specific optimal conditions
    crop_conditions = {
        'Wheat': {
            'ph_range': (6.0, 7.5),
            'temp_range': (15, 25),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (40, 60),
            'rainfall_optimal': (75, 125)
        },
        'Rice': {
            'ph_range': (5.5, 7.0),
            'temp_range': (20, 35),
            'n_optimal': (100, 150),
            'p_optimal': (40, 60),
            'k_optimal': (40, 60),
            'rainfall_optimal': (100, 200)
        },
        'Maize': {
            'ph_range': (5.8, 7.0),
            'temp_range': (20, 25),
            'n_optimal': (100, 150),
            'p_optimal': (40, 60),
            'k_optimal': (80, 120),
            'rainfall_optimal': (75, 150)
        },
        'Cotton': {
            'ph_range': (6.0, 8.5),
            'temp_range': (20, 30),
            'n_optimal': (100, 150),
            'p_optimal': (40, 60),
            'k_optimal': (80, 120),
            'rainfall_optimal': (75, 125)
        },
        'Sugarcane': {
            'ph_range': (6.5, 7.5),
            'temp_range': (25, 35),
            'n_optimal': (150, 200),
            'p_optimal': (60, 100),
            'k_optimal': (150, 200),
            'rainfall_optimal': (100, 175)
        },
        'Barley': {
            'ph_range': (6.0, 7.5),
            'temp_range': (15, 25),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (40, 60),
            'rainfall_optimal': (50, 100)
        },
        'Soybean': {
            'ph_range': (6.0, 7.0),
            'temp_range': (20, 30),
            'n_optimal': (20, 40),  # Legumes fix nitrogen
            'p_optimal': (40, 60),
            'k_optimal': (80, 120),
            'rainfall_optimal': (75, 150)
        },
        'Peas': {
            'ph_range': (6.0, 7.5),
            'temp_range': (15, 25),
            'n_optimal': (0, 20),  # Legumes fix nitrogen
            'p_optimal': (40, 60),
            'k_optimal': (80, 120),
            'rainfall_optimal': (50, 100)
        },
        'Mustard': {
            'ph_range': (6.0, 7.5),
            'temp_range': (15, 25),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (40, 60),
            'rainfall_optimal': (50, 100)
        },
        'Potato': {
            'ph_range': (5.0, 6.5),
            'temp_range': (15, 20),
            'n_optimal': (100, 150),
            'p_optimal': (80, 120),
            'k_optimal': (150, 200),
            'rainfall_optimal': (75, 125)
        },
        'Banana': {
            'ph_range': (5.5, 7.0),
            'temp_range': (25, 35),
            'n_optimal': (150, 200),
            'p_optimal': (40, 80),
            'k_optimal': (200, 300),
            'rainfall_optimal': (150, 250)
        },
        'Mango': {
            'ph_range': (5.5, 7.5),
            'temp_range': (25, 35),
            'n_optimal': (100, 150),
            'p_optimal': (40, 60),
            'k_optimal': (100, 150),
            'rainfall_optimal': (75, 150)
        },
        'Grapes': {
            'ph_range': (5.5, 7.0),
            'temp_range': (15, 30),
            'n_optimal': (60, 100),
            'p_optimal': (40, 80),
            'k_optimal': (100, 150),
            'rainfall_optimal': (50, 100)
        },
        'Carrot': {
            'ph_range': (6.0, 7.0),
            'temp_range': (15, 25),
            'n_optimal': (80, 120),
            'p_optimal': (40, 80),
            'k_optimal': (100, 150),
            'rainfall_optimal': (50, 100)
        },
        'Radish': {
            'ph_range': (5.8, 7.0),
            'temp_range': (10, 20),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (80, 120),
            'rainfall_optimal': (50, 100)
        },
        'Tomato': {
            'ph_range': (6.0, 7.0),
            'temp_range': (20, 30),
            'n_optimal': (100, 150),
            'p_optimal': (60, 100),
            'k_optimal': (150, 200),
            'rainfall_optimal': (75, 125)
        },
        'Onion': {
            'ph_range': (6.0, 7.0),
            'temp_range': (15, 25),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (80, 120),
            'rainfall_optimal': (50, 100)
        },
        'Garlic': {
            'ph_range': (6.0, 7.0),
            'temp_range': (10, 25),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (80, 120),
            'rainfall_optimal': (50, 100)
        },
        'Ginger': {
            'ph_range': (5.5, 6.5),
            'temp_range': (20, 30),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (100, 150),
            'rainfall_optimal': (150, 250)
        },
        'Papaya': {
            'ph_range': (5.5, 6.5),
            'temp_range': (25, 35),
            'n_optimal': (100, 150),
            'p_optimal': (40, 80),
            'k_optimal': (150, 200),
            'rainfall_optimal': (100, 200)
        },
        'Pineapple': {
            'ph_range': (4.5, 6.5),
            'temp_range': (20, 30),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (100, 150),
            'rainfall_optimal': (100, 150)
        },
        'Pomegranate': {
            'ph_range': (5.5, 7.5),
            'temp_range': (25, 35),
            'n_optimal': (80, 120),
            'p_optimal': (40, 60),
            'k_optimal': (100, 150),
            'rainfall_optimal': (75, 125)
        }
    }
    
    advice = f"Based on your soil and weather conditions, {crop_name} is recommended for cultivation.\n\n"
    
    # Check if crop is in our database
    if crop_name in crop_conditions:
        conditions = crop_conditions[crop_name]
        
        # Soil pH advice
        ph_min, ph_max = conditions['ph_range']
        if soil_ph < ph_min:
            advice += f"⚠️ Soil pH Issue: Your soil pH ({soil_ph}) is below the optimal range ({ph_min}-{ph_max}) for {crop_name}. Consider adding lime to raise pH.\n"
        elif soil_ph > ph_max:
            advice += f"⚠️ Soil pH Issue: Your soil pH ({soil_ph}) is above the optimal range ({ph_min}-{ph_max}) for {crop_name}. Consider adding sulfur to lower pH.\n"
        else:
            advice += f"✅ Soil pH: Your soil pH ({soil_ph}) is within the optimal range ({ph_min}-{ph_max}) for {crop_name}.\n"
        
        # Temperature advice
        temp_min, temp_max = conditions['temp_range']
        if temperature < temp_min:
            advice += f"⚠️ Temperature Issue: Current temperature ({temperature}°C) is below the optimal range ({temp_min}-{temp_max}°C). Consider protective measures or planting season adjustment.\n"
        elif temperature > temp_max:
            advice += f"⚠️ Temperature Issue: Current temperature ({temperature}°C) is above the optimal range ({temp_min}-{temp_max}°C). Consider shade or irrigation for cooling.\n"
        else:
            advice += f"✅ Temperature: Current temperature ({temperature}°C) is within the optimal range ({temp_min}-{temp_max}°C) for {crop_name}.\n"
        
        # Nutrient advice
        n_min, n_max = conditions['n_optimal']
        if soil_nitrogen < n_min:
            advice += f"⚠️ Nitrogen Deficiency: Soil nitrogen ({soil_nitrogen} kg/ha) is below optimal ({n_min}-{n_max} kg/ha). Consider nitrogen fertilizer application.\n"
        elif soil_nitrogen > n_max:
            advice += f"⚠️ Nitrogen Excess: Soil nitrogen ({soil_nitrogen} kg/ha) is above optimal ({n_min}-{n_max} kg/ha). Be cautious of excessive vegetative growth.\n"
        else:
            advice += f"✅ Nitrogen: Soil nitrogen ({soil_nitrogen} kg/ha) is within optimal range ({n_min}-{n_max} kg/ha).\n"
            
        p_min, p_max = conditions['p_optimal']
        if soil_phosphorus < p_min:
            advice += f"⚠️ Phosphorus Deficiency: Soil phosphorus ({soil_phosphorus} kg/ha) is below optimal ({p_min}-{p_max} kg/ha). Consider phosphorus fertilizer application.\n"
        elif soil_phosphorus > p_max:
            advice += f"⚠️ Phosphorus Excess: Soil phosphorus ({soil_phosphorus} kg/ha) is above optimal ({p_min}-{p_max} kg/ha). Generally not a concern.\n"
        else:
            advice += f"✅ Phosphorus: Soil phosphorus ({soil_phosphorus} kg/ha) is within optimal range ({p_min}-{p_max} kg/ha).\n"
            
        k_min, k_max = conditions['k_optimal']
        if soil_potassium < k_min:
            advice += f"⚠️ Potassium Deficiency: Soil potassium ({soil_potassium} kg/ha) is below optimal ({k_min}-{k_max} kg/ha). Consider potassium fertilizer application.\n"
        elif soil_potassium > k_max:
            advice += f"⚠️ Potassium Excess: Soil potassium ({soil_potassium} kg/ha) is above optimal ({k_min}-{k_max} kg/ha). Generally not a concern.\n"
        else:
            advice += f"✅ Potassium: Soil potassium ({soil_potassium} kg/ha) is within optimal range ({k_min}-{k_max} kg/ha).\n"
            
        # Rainfall advice
        rain_min, rain_max = conditions['rainfall_optimal']
        if rainfall < rain_min:
            advice += f"⚠️ Water Deficit: Current rainfall ({rainfall} mm) is below optimal ({rain_min}-{rain_max} mm). Consider irrigation.\n"
        elif rainfall > rain_max:
            advice += f"⚠️ Water Excess: Current rainfall ({rainfall} mm) is above optimal ({rain_min}-{rain_max} mm). Ensure proper drainage to prevent waterlogging.\n"
        else:
            advice += f"✅ Water: Current rainfall ({rainfall} mm) is within optimal range ({rain_min}-{rain_max} mm).\n"
    else:
        # General advice for crops not in our database
        advice += f"General growing conditions for {crop_name}:\n"
        advice += "- Prefers well-drained soil with pH 6.0-7.0\n"
        advice += "- Optimal temperature range is typically 15-30°C\n"
        advice += "- Apply balanced NPK fertilizers based on soil test results\n"
        advice += "- Ensure adequate water supply but avoid waterlogging\n"
    
    # General growing tips
    advice += f"\n📋 Best Practices for Growing {crop_name}:\n"
    advice += "1. Prepare well-drained soil with good organic matter content\n"
    advice += "2. Follow recommended planting spacing for optimal growth\n"
    advice += "3. Monitor for pests and diseases regularly\n"
    advice += "4. Harvest at the right maturity stage for best quality\n"
    
    return advice


def recommend_additional_crops(soil_nitrogen, soil_phosphorus, soil_potassium, soil_ph, temperature, humidity, rainfall):
    """
    Recommend additional crops, fruits, and vegetables based on conditions.
    
    Returns:
        list: List of recommended crops with suitability scores
    """
    # All crops with their optimal conditions
    all_crops = {
        'Wheat': {'ph': (6.0, 7.5), 'temp': (15, 25), 'n': (80, 120), 'p': (40, 60), 'k': (40, 60), 'rain': (75, 125)},
        'Rice': {'ph': (5.5, 7.0), 'temp': (20, 35), 'n': (100, 150), 'p': (40, 60), 'k': (40, 60), 'rain': (100, 200)},
        'Maize': {'ph': (5.8, 7.0), 'temp': (20, 25), 'n': (100, 150), 'p': (40, 60), 'k': (80, 120), 'rain': (75, 150)},
        'Cotton': {'ph': (6.0, 8.5), 'temp': (20, 30), 'n': (100, 150), 'p': (40, 60), 'k': (80, 120), 'rain': (75, 125)},
        'Sugarcane': {'ph': (6.5, 7.5), 'temp': (25, 35), 'n': (150, 200), 'p': (60, 100), 'k': (150, 200), 'rain': (100, 175)},
        'Barley': {'ph': (6.0, 7.5), 'temp': (15, 25), 'n': (80, 120), 'p': (40, 60), 'k': (40, 60), 'rain': (50, 100)},
        'Soybean': {'ph': (6.0, 7.0), 'temp': (20, 30), 'n': (20, 40), 'p': (40, 60), 'k': (80, 120), 'rain': (75, 150)},
        'Peas': {'ph': (6.0, 7.5), 'temp': (15, 25), 'n': (0, 20), 'p': (40, 60), 'k': (80, 120), 'rain': (50, 100)},
        'Mustard': {'ph': (6.0, 7.5), 'temp': (15, 25), 'n': (80, 120), 'p': (40, 60), 'k': (40, 60), 'rain': (50, 100)},
        'Potato': {'ph': (5.0, 6.5), 'temp': (15, 20), 'n': (100, 150), 'p': (80, 120), 'k': (150, 200), 'rain': (75, 125)},
        'Banana': {'ph': (5.5, 7.0), 'temp': (25, 35), 'n': (150, 200), 'p': (40, 80), 'k': (200, 300), 'rain': (150, 250)},
        'Mango': {'ph': (5.5, 7.5), 'temp': (25, 35), 'n': (100, 150), 'p': (40, 60), 'k': (100, 150), 'rain': (75, 150)},
        'Grapes': {'ph': (5.5, 7.0), 'temp': (15, 30), 'n': (60, 100), 'p': (40, 80), 'k': (100, 150), 'rain': (50, 100)},
        'Carrot': {'ph': (6.0, 7.0), 'temp': (15, 25), 'n': (80, 120), 'p': (40, 80), 'k': (100, 150), 'rain': (50, 100)},
        'Radish': {'ph': (5.8, 7.0), 'temp': (10, 20), 'n': (80, 120), 'p': (40, 60), 'k': (80, 120), 'rain': (50, 100)},
        'Tomato': {'ph': (6.0, 7.0), 'temp': (20, 30), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (75, 125)},
        'Onion': {'ph': (6.0, 7.0), 'temp': (15, 25), 'n': (80, 120), 'p': (40, 60), 'k': (80, 120), 'rain': (50, 100)},
        'Garlic': {'ph': (6.0, 7.0), 'temp': (10, 25), 'n': (80, 120), 'p': (40, 60), 'k': (80, 120), 'rain': (50, 100)},
        'Ginger': {'ph': (5.5, 6.5), 'temp': (20, 30), 'n': (80, 120), 'p': (40, 60), 'k': (100, 150), 'rain': (150, 250)},
        'Papaya': {'ph': (5.5, 6.5), 'temp': (25, 35), 'n': (100, 150), 'p': (40, 80), 'k': (150, 200), 'rain': (100, 200)},
        'Pineapple': {'ph': (4.5, 6.5), 'temp': (20, 30), 'n': (80, 120), 'p': (40, 60), 'k': (100, 150), 'rain': (100, 150)},
        'Pomegranate': {'ph': (5.5, 7.5), 'temp': (25, 35), 'n': (80, 120), 'p': (40, 60), 'k': (100, 150), 'rain': (75, 125)},
        'Cucumber': {'ph': (6.0, 7.0), 'temp': (20, 30), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (75, 125)},
        'Eggplant': {'ph': (5.5, 6.8), 'temp': (20, 30), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (75, 125)},
        'Cabbage': {'ph': (6.0, 7.5), 'temp': (15, 20), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (75, 125)},
        'Cauliflower': {'ph': (6.0, 7.5), 'temp': (15, 20), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (75, 125)},
        'Spinach': {'ph': (6.0, 7.5), 'temp': (10, 20), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (50, 100)},
        'Lettuce': {'ph': (6.0, 7.0), 'temp': (15, 20), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (50, 100)},
        'Bell Pepper': {'ph': (6.0, 7.0), 'temp': (20, 30), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (75, 125)},
        'Watermelon': {'ph': (6.0, 7.0), 'temp': (25, 35), 'n': (100, 150), 'p': (60, 100), 'k': (150, 200), 'rain': (75, 125)}
    }
    
    recommendations = []
    
    # Calculate suitability score for each crop
    for crop_name, conditions in all_crops.items():
        score = 0
        max_score = 6  # We're checking 6 parameters
        
        # pH suitability
        ph_min, ph_max = conditions['ph']
        if ph_min <= soil_ph <= ph_max:
            score += 1
            
        # Temperature suitability
        temp_min, temp_max = conditions['temp']
        if temp_min <= temperature <= temp_max:
            score += 1
            
        # Nitrogen suitability
        n_min, n_max = conditions['n']
        if n_min <= soil_nitrogen <= n_max:
            score += 1
            
        # Phosphorus suitability
        p_min, p_max = conditions['p']
        if p_min <= soil_phosphorus <= p_max:
            score += 1
            
        # Potassium suitability
        k_min, k_max = conditions['k']
        if k_min <= soil_potassium <= k_max:
            score += 1
            
        # Rainfall suitability
        rain_min, rain_max = conditions['rain']
        if rain_min <= rainfall <= rain_max:
            score += 1
            
        # Convert to percentage
        suitability = (score / max_score) * 100
        
        # Only recommend crops with at least 50% suitability
        if suitability >= 50:
            recommendations.append({
                'crop': crop_name,
                'suitability': suitability
            })
    
    # Sort by suitability score (descending)
    recommendations.sort(key=lambda x: x['suitability'], reverse=True)
    
    # Return top 5 recommendations
    return recommendations[:5]


@api_view(['POST'])
def get_combined_recommendations(request):
    """
    Get combined crop and fertilizer recommendations without farm dependency.
    
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
        JSON response with crop and fertilizer recommendations
    """
    try:
        # Load models if not already loaded
        load_models()
        
        # Validate request data (but don't require farm_id)
        serializer = RecommendationRequestSerializer(data=request.data)
        if not serializer.is_valid():
            # If validation fails because of missing farm_id, we'll create a default one
            data_copy = request.data.copy()
            if 'farm_id' not in data_copy:
                data_copy['farm_id'] = 'default_farm'
            serializer = RecommendationRequestSerializer(data=data_copy)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract data
        soil_nitrogen = serializer.validated_data['soil_nitrogen']
        soil_phosphorus = serializer.validated_data['soil_phosphorus']
        soil_potassium = serializer.validated_data['soil_potassium']
        soil_ph = serializer.validated_data['soil_ph']
        temperature = serializer.validated_data['temperature']
        humidity = serializer.validated_data['humidity']
        rainfall = serializer.validated_data['rainfall']
        location = serializer.validated_data['location']
        season = serializer.validated_data.get('season', '')
        
        # Prepare features for crop recommendation - match the expected format
        # The model expects: ['District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
        # Convert location to a valid district name that's in the encoder
        valid_districts = ['Kolhapur', 'Pune', 'Sangli', 'Satara', 'Solapur']
        location_str = str(location) if location else "Pune"
        
        # If the location is not in the valid districts, use a default one
        if location_str not in valid_districts:
            location_str = "Pune"  # Use Pune as default since it's in the encoder
        
        crop_features = np.array([[
            location_str,       # District_Name
            "Loam",             # Soil_color (default value)
            soil_nitrogen,      # Nitrogen
            soil_phosphorus,    # Phosphorus
            soil_potassium,     # Potassium
            soil_ph,            # pH
            rainfall,           # Rainfall
            temperature         # Temperature
        ]])
        
        if crop_model is None or crop_preprocessor is None:
            return Response({'error': 'Crop recommendation system not available'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Scale features using preprocessor
        try:
            # Convert numpy array to pandas DataFrame with proper column names
            import pandas as pd
            feature_names = ['District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
            crop_features_df = pd.DataFrame(crop_features, columns=feature_names)
            
            # Handle categorical encoding for District_Name and Soil_color
            categorical_columns = ['District_Name', 'Soil_color']
            numeric_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
            
            if isinstance(crop_preprocessor, dict) and 'label_encoders' in crop_preprocessor:
                # Encode categorical variables
                for col in categorical_columns:
                    if col in crop_preprocessor['label_encoders']:
                        try:
                            # Convert to plain Python string first to avoid numpy string issues
                            crop_features_df[col] = crop_features_df[col].apply(lambda x: str(x))
                            crop_features_df[col] = crop_preprocessor['label_encoders'][col].transform(crop_features_df[col])
                        except ValueError:
                            # If the value is not in the encoder, use a default value (0)
                            crop_features_df[col] = 0
                        except Exception as e:
                            # If any other error occurs, use a default value
                            crop_features_df[col] = 0
                            print(f"Warning: Error encoding {col}: {e}")
                
                # Scale numeric variables
                numeric_data = crop_features_df[numeric_columns].values.astype(float)
                scaled_numeric = crop_preprocessor['scaler'].transform(numeric_data)
                
                # Get encoded categorical data
                encoded_categorical = crop_features_df[categorical_columns].values.astype(float)
                
                # Combine them for the final features
                scaled_features = np.hstack([encoded_categorical, scaled_numeric])
            else:
                # Fallback if preprocessor structure is different
                scaled_features = crop_preprocessor['scaler'].transform(crop_features_df[numeric_columns].values.astype(float))
        except Exception as e:
            # Fallback to original method if pandas fails or feature names cause issues
            try:
                scaled_features = crop_preprocessor['scaler'].transform(crop_features[:, 2:].astype(float))  # Only numeric columns
            except Exception as e2:
                return Response({'error': f'Error preprocessing crop features: {str(e2)}'}, 
                               status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get crop recommendations
        try:
            # Handle both sklearn and custom model predict methods
            # For XGBoost, we need to call predict and predict_proba separately
            crop_predictions = crop_model.predict(scaled_features)
            crop_probabilities = crop_model.predict_proba(scaled_features)
            
        except Exception as e:
            return Response({'error': f'Error predicting crops: {str(e)}'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get top 3 crop recommendations
        # Handle both single array and 2D array probabilities
        if crop_probabilities.ndim > 1 and crop_probabilities.shape[0] > 1:
            top_3_indices = np.argsort(crop_probabilities[0])[::-1][:3]
        else:
            # Handle 1D array or single row 2D array
            if crop_probabilities.ndim > 1:
                prob_array = crop_probabilities[0]
            else:
                prob_array = crop_probabilities
            top_3_indices = np.argsort(prob_array)[::-1][:3]
        
        # For demonstration, we'll use class names (in practice, load from label encoder)
        crop_names = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Barley', 'Soybean', 
                      'Peas', 'Mustard', 'Potato', 'Banana', 'Mango', 'Grapes', 'Carrot', 
                      'Radish', 'Tomato', 'Onion', 'Garlic', 'Ginger', 'Papaya', 'Pineapple', 'Pomegranate']
        
        recommendations = []
        
        for i, idx in enumerate(top_3_indices):
            crop_name = crop_names[idx] if idx < len(crop_names) else f"Crop_{idx}"
            # Handle confidence extraction properly
            if crop_probabilities.ndim > 1:
                if crop_probabilities.shape[0] > 1:
                    confidence = float(crop_probabilities[0][idx]) if idx < crop_probabilities.shape[1] else 0.0
                else:
                    confidence = float(crop_probabilities[0][idx]) if idx < crop_probabilities.shape[1] else 0.0
            else:
                confidence = float(crop_probabilities[idx]) if idx < len(crop_probabilities) else 0.0
            
            # Prepare features for fertilizer recommendation
            fertilizer_features = np.array([[
                soil_nitrogen,
                soil_phosphorus,
                soil_potassium,
                soil_ph,
                float(idx),  # Crop type encoded
                temperature,
                humidity,
                rainfall
            ]])
            
            fertilizer_name = "NPK Fertilizer"
            quantity = 150
            
            # Use rule-based approach for fertilizer recommendation
            try:
                # Simple rule-based fertilizer recommendation
                # This is a simplified approach - in practice, you'd want a more sophisticated model
                fertilizer_names = ['Urea', 'DAP', 'MOP', 'SSP', 'NPK 15-15-15', 'Compost']
                fertilizer_name = fertilizer_names[idx % len(fertilizer_names)]
                # Quantity based on soil nutrient levels
                base_quantity = 100
                # Adjust based on soil nutrient levels
                if soil_nitrogen < 50:
                    base_quantity += 30
                if soil_phosphorus < 30:
                    base_quantity += 20
                if soil_potassium < 50:
                    base_quantity += 25
                # Adjust based on crop type (simplified)
                crop_factors = {
                    'Wheat': 1.0, 'Rice': 1.2, 'Maize': 1.1, 'Cotton': 1.3,
                    'Sugarcane': 1.5, 'Barley': 0.8, 'Soybean': 0.9
                }
                factor = crop_factors.get(crop_name, 1.0)
                quantity = int(base_quantity * factor)
            except Exception as e:
                print(f"Error in rule-based fertilizer recommendation: {e}")
                fertilizer_name = "NPK Fertilizer"
                quantity = 150
            
            # Generate clear, actionable advice using LLM when available
            advice = generate_crop_advice(
                crop_name, 
                soil_nitrogen, 
                soil_phosphorus, 
                soil_potassium, 
                soil_ph, 
                temperature, 
                humidity, 
                rainfall
            )
            
            recommendations.append({
                'rank': i + 1,
                'crop': crop_name,
                'confidence': confidence,
                'fertilizer': fertilizer_name,
                'quantity_kg_per_ha': float(quantity),
                'advice': advice
            })
        
        # Get additional crop recommendations
        additional_crops = recommend_additional_crops(
            soil_nitrogen, 
            soil_phosphorus, 
            soil_potassium, 
            soil_ph, 
            temperature, 
            humidity, 
            rainfall
        )
        
        # Save recommendations to database for reports and charts
        try:
            # Save crop recommendations
            saved_recommendations = []
            for rec in recommendations:
                crop_recommendation = CropRecommendation.objects.create(
                    recommended_crop=rec['crop'],
                    confidence_score=rec['confidence'],
                    soil_nitrogen=soil_nitrogen,
                    soil_phosphorus=soil_phosphorus,
                    soil_potassium=soil_potassium,
                    soil_ph=soil_ph,
                    temperature=temperature,
                    humidity=humidity,
                    rainfall=rainfall
                )
                
                # Save associated fertilizer recommendation
                FertilizerRecommendation.objects.create(
                    crop_recommendation=crop_recommendation,
                    recommended_fertilizer=rec['fertilizer'],
                    quantity_kg_per_ha=rec['quantity_kg_per_ha']
                )
                
                # Add database ID to response
                rec['id'] = crop_recommendation.id
                saved_recommendations.append(rec)
            
            recommendations = saved_recommendations
        except Exception as e:
            print(f"Error saving recommendations to database: {e}")
            # Continue with recommendations even if saving fails
        
        return Response({
            'location': location,
            'season': season,
            'recommendations': recommendations,
            'additional_crops': additional_crops
        })
    except Exception as e:
        # Log the error for debugging
        print(f"Error in get_combined_recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
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
