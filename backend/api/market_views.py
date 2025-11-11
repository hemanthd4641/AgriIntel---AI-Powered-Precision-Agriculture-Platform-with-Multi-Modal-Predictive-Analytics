"""
Market Prediction Views for the Smart Agriculture API.

This module defines the API endpoints for market price prediction functionality.
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import the market prediction module
try:
    from market_prediction.predict_market import predict_market_price, load_market_model, load_llm_components
    MARKET_PREDICTION_AVAILABLE = True
except ImportError:
    MARKET_PREDICTION_AVAILABLE = False
    print("Market prediction module not available")

from .models import Crop, MarketPrediction
from .serializers import MarketPredictionSerializer

# Preload model and LLM components when the module is imported
if MARKET_PREDICTION_AVAILABLE:
    print("Preloading market prediction model and LLM components...")
    load_market_model()
    load_llm_components()
    print("Market prediction model and LLM components preloaded successfully")

@api_view(['POST'])
def predict_market_price_view(request):
    """
    Predict market price for a crop based on various factors
    """
    print("=== PREDICT MARKET PRICE ENDPOINT CALLED ===")
    print(f"Request method: {request.method}")
    print(f"Request content type: {request.content_type}")
    print(f"Request data: {request.data}")
    
    if not MARKET_PREDICTION_AVAILABLE:
        print("Market prediction module not available")
        return Response({
            'error': 'Market prediction module not available. Please train the model first.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # Extract data from request
    crop_name = request.data.get('crop')
    region = request.data.get('region')
    season = request.data.get('season')
    yield_prediction = request.data.get('yield_prediction')
    
    print("Extracted data:", {
        'crop': crop_name,
        'region': region,
        'season': season,
        'yield_prediction': yield_prediction
    })
    
    # Validate required fields
    required_fields = [crop_name, region, season, yield_prediction]
    
    if any(field is None for field in required_fields):
        missing_fields = []
        if crop_name is None: missing_fields.append('crop')
        if region is None: missing_fields.append('region')
        if season is None: missing_fields.append('season')
        if yield_prediction is None: missing_fields.append('yield_prediction')
        
        print("Missing fields:", missing_fields)
        
        return Response({
            'error': 'Missing required fields', 
            'missing_fields': missing_fields
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        print("=== STARTING MARKET PREDICTION PROCESS ===")
        
        # Get or create crop object
        crop, created = Crop.objects.get_or_create(name=crop_name)
        print(f"Crop object: {crop}, created: {created}")
        
        # Safely convert values to appropriate types with defaults
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
                
        def safe_int(value, default=0):
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        # Optional parameters with defaults
        global_demand = request.data.get('global_demand', 'medium')
        weather_impact = request.data.get('weather_impact', 'normal')
        economic_condition = request.data.get('economic_condition', 'stable')
        supply_index = request.data.get('supply_index', 60.0)
        demand_index = request.data.get('demand_index', 60.0)
        inventory_level = request.data.get('inventory_level', 50.0)
        export_demand = request.data.get('export_demand', 60.0)
        production_cost = request.data.get('production_cost', 200.0)
        days_to_harvest = request.data.get('days_to_harvest', 90)
        fertilizer_usage = request.data.get('fertilizer_usage', 'medium')
        irrigation_usage = request.data.get('irrigation_usage', 'medium')
        
        print("All parameters:", {
            'crop': crop_name,
            'region': region,
            'season': season,
            'yield_prediction': yield_prediction,
            'global_demand': global_demand,
            'weather_impact': weather_impact,
            'economic_condition': economic_condition,
            'supply_index': supply_index,
            'demand_index': demand_index,
            'inventory_level': inventory_level,
            'export_demand': export_demand,
            'production_cost': production_cost,
            'days_to_harvest': days_to_harvest,
            'fertilizer_usage': fertilizer_usage,
            'irrigation_usage': irrigation_usage
        })
        
        # Predict price using the enhanced market prediction module
        print("Calling predict_market_price function...")
        prediction_result = predict_market_price(
            crop=crop_name,
            region=region,
            season=season,
            yield_prediction=safe_float(yield_prediction),
            global_demand=global_demand,
            weather_impact=weather_impact,
            economic_condition=economic_condition,
            supply_index=safe_float(supply_index, 60.0),
            demand_index=safe_float(demand_index, 60.0),
            inventory_level=safe_float(inventory_level, 50.0),
            export_demand=safe_float(export_demand, 60.0),
            production_cost=safe_float(production_cost, 200.0),
            days_to_harvest=safe_int(days_to_harvest, 90),
            fertilizer_usage=fertilizer_usage,
            irrigation_usage=irrigation_usage
        )
        
        print("Prediction result:", prediction_result)
        
        if 'error' in prediction_result:
            print("Error in prediction result:", prediction_result['error'])
            return Response({
                'error': prediction_result['error']
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        predicted_price = prediction_result['predicted_price']
        market_trend = prediction_result['market_trend']
        confidence_score = prediction_result['confidence_score']
        enhanced_insights = prediction_result.get('enhanced_insights', {})
        
        print("Saving prediction to database...")
        # Save prediction to database
        market_prediction = MarketPrediction.objects.create(
            crop=crop,
            predicted_price_per_ton=predicted_price,
            confidence_score=confidence_score,
            market_trend=market_trend,
            forecast_period_days=30
        )
        
        print("Serializing response...")
        # Serialize and return response
        serializer = MarketPredictionSerializer(market_prediction)
        response_data = serializer.data
        
        # Add enhanced insights to response
        response_data['enhanced_insights'] = enhanced_insights
        response_data['explanation'] = _generate_explanation(
            crop_name, predicted_price, market_trend, global_demand, weather_impact,
            supply_index, demand_index, inventory_level, export_demand
        )
        
        # Add additional market intelligence
        response_data['market_intelligence'] = {
            'price_outlook': f"Based on current trends, {crop_name} prices are expected to {'rise' if market_trend == 'bullish' else 'fall' if market_trend == 'bearish' else 'remain stable'} over the next 30 days.",
            'timing_advice': _generate_timing_advice(market_trend, confidence_score),
            'risk_factors': _generate_risk_factors(global_demand, weather_impact, supply_index, demand_index),
            'historical_context': _generate_historical_context(crop_name, predicted_price)
        }
        
        print("Returning response:", response_data)
        print("=== MARKET PREDICTION COMPLETED SUCCESSFULLY ===")
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        print("Error in predict_market_price:", str(e))
        import traceback
        traceback.print_exc()
        return Response({
            'error': f'Error processing market prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def _generate_explanation(crop, predicted_price, market_trend, global_demand, weather_impact,
                         supply_index, demand_index, inventory_level, export_demand):
    """
    Generate explanation for the market prediction
    """
    trend_descriptions = {
        'bullish': 'prices are expected to rise',
        'bearish': 'prices are expected to fall',
        'neutral': 'prices are expected to remain stable'
    }
    
    explanation = f"Based on current market conditions, {crop} {trend_descriptions.get(market_trend, 'prices are expected to remain stable')}. "
    
    # Safely convert values to float for calculations
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    supply_index_float = safe_float(supply_index)
    demand_index_float = safe_float(demand_index)
    inventory_level_float = safe_float(inventory_level)
    
    # Add factors based on input data
    if demand_index_float != 0 and (supply_index_float / demand_index_float) < 0.8:
        explanation += "Supply constraints relative to demand are driving prices upward. "
    elif demand_index_float != 0 and (supply_index_float / demand_index_float) > 1.2:
        explanation += "Excess supply relative to demand is putting downward pressure on prices. "
    
    if inventory_level_float < 30:
        explanation += "Low inventory levels are supporting higher prices. "
    elif inventory_level_float > 70:
        explanation += "High inventory levels are putting pressure on prices. "
    
    if global_demand == 'high':
        explanation += "Strong global demand is supporting higher prices. "
    elif global_demand == 'low':
        explanation += "Weak global demand is putting pressure on prices. "
    
    if weather_impact == 'poor':
        explanation += "Adverse weather conditions are expected to reduce supply and increase prices. "
    elif weather_impact == 'excellent':
        explanation += "Favorable weather conditions are expected to increase supply and moderate prices. "
    
    explanation += f"The predicted price of ${predicted_price:.2f} per ton reflects these combined factors."
    
    return explanation

def _generate_timing_advice(market_trend, confidence_score):
    """
    Generate timing advice based on market trend and confidence
    """
    advice = ""
    if market_trend == 'bullish':
        advice = "Consider holding inventory for higher prices."
        if confidence_score > 0.8:
            advice += " Strong confidence in upward trend - may be optimal time to delay sales."
    elif market_trend == 'bearish':
        advice = "Consider forward contracting to lock in prices."
        if confidence_score > 0.8:
            advice += " Strong confidence in downward trend - may be optimal time to sell early."
    else:
        advice = "Monitor market conditions closely for opportunities."
        if confidence_score < 0.6:
            advice += " Lower confidence - consider hedging strategies to protect against volatility."
    
    return advice

def _generate_risk_factors(global_demand, weather_impact, supply_index, demand_index):
    """
    Generate risk factors based on input data
    """
    risks = []
    
    # Safely convert values to float for calculations
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    supply_index_float = safe_float(supply_index)
    demand_index_float = safe_float(demand_index)
    
    # Add factors based on input data
    if demand_index_float != 0 and (supply_index_float / demand_index_float) < 0.8:
        risks.append("Supply constraints relative to demand may drive prices upward")
    elif demand_index_float != 0 and (supply_index_float / demand_index_float) > 1.2:
        risks.append("Excess supply relative to demand may put downward pressure on prices")
    
    if global_demand == 'high':
        risks.append("Strong global demand supports higher prices")
    elif global_demand == 'low':
        risks.append("Weak global demand puts pressure on prices")
    
    if weather_impact == 'poor':
        risks.append("Adverse weather conditions may reduce supply and increase prices")
    elif weather_impact == 'excellent':
        risks.append("Favorable weather conditions may increase supply and moderate prices")
    
    return risks if risks else ["No significant risk factors identified at this time"]

def _generate_historical_context(crop, predicted_price):
    """
    Generate historical context for the prediction
    """
    # This would typically query a database of historical prices
    # For now, we'll provide a generic message
    return f"Historical data shows {crop} prices typically fluctuate based on seasonal patterns and global market conditions. The current prediction of ${predicted_price:.2f}/ton should be considered in the context of these broader trends."

@api_view(['GET'])
def get_market_predictions(request):
    """
    Get recent market predictions
    """
    try:
        # Get the 10 most recent predictions
        predictions = MarketPrediction.objects.select_related('crop').order_by('-prediction_date')[:10]
        serializer = MarketPredictionSerializer(predictions, many=True)
        return Response(serializer.data)
    except Exception as e:
        return Response({
            'error': f'Error retrieving market predictions: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_crop_market_history(request, crop_id):
    """
    Get market prediction history for a specific crop
    """
    try:
        predictions = MarketPrediction.objects.filter(crop_id=crop_id).select_related('crop').order_by('-prediction_date')
        serializer = MarketPredictionSerializer(predictions, many=True)
        return Response(serializer.data)
    except Exception as e:
        return Response({
            'error': f'Error retrieving crop market history: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)