"""
Market prediction script for real-time price forecasting
"""

import joblib
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from django.utils import timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global variables for model, preprocessor, LLM, and RAG (loaded once)
market_model = None
market_preprocessor = None
agriculture_llm = None
market_rag = None

# Try to import LLM and RAG components
try:
    from explainable_ai.llm_interface import AgricultureLLM
    from explainable_ai.market_prediction_rag.market_prediction_rag import MarketPredictionRAG
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM components not available, using fallback methods")

from market_prediction.preprocessing.market_preprocessor import MarketPreprocessor

def load_market_model():
    """
    Load the trained market prediction model and preprocessor (cached)
    
    Returns:
        tuple: (model, preprocessor) or (None, None) if not found
    """
    global market_model, market_preprocessor
    
    # Return cached model and preprocessor if already loaded
    if market_model is not None and market_preprocessor is not None:
        return market_model, market_preprocessor
    
    try:
        # Define paths
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'market_prediction')
        model_path = os.path.join(models_dir, 'market_model.pkl')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            print("Market prediction model not found. Please train the model first.")
            print(f"Expected model path: {model_path}")
            print(f"Expected preprocessor path: {preprocessor_path}")
            return None, None
        
        # Load model and preprocessor
        market_model = joblib.load(model_path)
        market_preprocessor = MarketPreprocessor()
        market_preprocessor.load_preprocessor(preprocessor_path)
        
        print("Market prediction model loaded successfully")
        return market_model, market_preprocessor
        
    except Exception as e:
        print(f"Error loading market prediction model: {str(e)}")
        return None, None

def load_llm_components():
    """
    Load LLM and RAG components (cached)
    """
    global agriculture_llm, market_rag
    
    # Return cached components if already loaded
    if agriculture_llm is not None and market_rag is not None:
        return agriculture_llm, market_rag
    
    try:
        if LLM_AVAILABLE:
            agriculture_llm = AgricultureLLM()
            market_rag = MarketPredictionRAG()
            print("LLM and RAG components loaded successfully")
        else:
            print("LLM components not available")
    except Exception as e:
        print(f"Error loading LLM components: {str(e)}")
        agriculture_llm = None
        market_rag = None
    
    return agriculture_llm, market_rag

def predict_market_price(crop, region, season, yield_prediction, global_demand='medium', weather_impact='normal', 
                        economic_condition='stable', supply_index=60.0, demand_index=60.0, inventory_level=50.0,
                        export_demand=60.0, production_cost=200.0, days_to_harvest=90, fertilizer_usage='medium',
                        irrigation_usage='medium'):
    """
    Predict market price for a crop based on various factors
    
    Args:
        crop (str): Crop name
        region (str): Geographic region
        season (str): Growing season
        yield_prediction (float): Predicted yield in tons/hectare
        global_demand (str): Global demand level (low/medium/high)
        weather_impact (str): Weather impact on crop (poor/normal/excellent)
        economic_condition (str): Economic condition (recession/stable/growth)
        supply_index (float): Supply index (0-100)
        demand_index (float): Demand index (0-100)
        inventory_level (float): Inventory level (0-100)
        export_demand (float): Export demand (0-100)
        production_cost (float): Production cost per ton
        days_to_harvest (int): Days until harvest
        fertilizer_usage (str): Fertilizer usage (low/medium/high)
        irrigation_usage (str): Irrigation usage (low/medium/high)
        
    Returns:
        dict: Prediction results including price, trend, and confidence
    """
    # Load model and preprocessor
    model, preprocessor = load_market_model()
    
    try:
        # Prepare input data with all features
        sample_data = {
            'crop': crop,
            'region': region,
            'season': season,
            'yield_prediction': float(yield_prediction),
            'global_demand': global_demand,
            'weather_impact': weather_impact,
            'economic_condition': economic_condition,
            'supply_index': float(supply_index),
            'demand_index': float(demand_index),
            'inventory_level': float(inventory_level),
            'export_demand': float(export_demand),
            'production_cost': float(production_cost),
            'days_to_harvest': int(days_to_harvest),
            'fertilizer_usage': fertilizer_usage,
            'irrigation_usage': irrigation_usage,
            'year': timezone.now().year,
            'month': timezone.now().month,
            'day_of_year': timezone.now().timetuple().tm_yday
        }
        
        predicted_price = None
        if model is not None and preprocessor is not None:
            # Preprocess the sample and predict using trained model
            X_sample = preprocessor.preprocess_single_sample(sample_data)
            predicted_price = model.predict(X_sample)[0]
        else:
            # Fallback: estimate price with a transparent heuristic when model isn't trained
            predicted_price = _fallback_price_estimate(
                production_cost=float(production_cost),
                supply_index=float(supply_index),
                demand_index=float(demand_index),
                inventory_level=float(inventory_level),
                export_demand=float(export_demand),
                yield_prediction=float(yield_prediction)
            )
        
        # Determine market trend
        market_trend = _determine_market_trend(
            yield_prediction, global_demand, weather_impact, 
            supply_index, demand_index, inventory_level, export_demand
        )
        
        # Calculate confidence score
        confidence_score = _calculate_confidence_score(
            yield_prediction, global_demand, weather_impact, 
            supply_index, demand_index, inventory_level, export_demand
        )
        
        # Generate enhanced insights using LLM and RAG if available
        enhanced_insights = _generate_enhanced_insights(
            crop, predicted_price, market_trend, sample_data, confidence_score
        )
        
        return {
            'predicted_price': round(predicted_price, 2),
            'market_trend': market_trend,
            'confidence_score': round(confidence_score, 4),
            'input_data': sample_data,
            'enhanced_insights': enhanced_insights
        }
        
    except Exception as e:
        print(f"Error in market price prediction: {str(e)}")
        # As a last resort, return a safe heuristic with low confidence
        try:
            predicted_price = _fallback_price_estimate(
                production_cost=float(production_cost),
                supply_index=float(supply_index),
                demand_index=float(demand_index),
                inventory_level=float(inventory_level),
                export_demand=float(export_demand),
                yield_prediction=float(yield_prediction)
            )
            market_trend = _determine_market_trend(
                yield_prediction, global_demand, weather_impact,
                supply_index, demand_index, inventory_level, export_demand
            )
            confidence_score = 0.45
            enhanced_insights = _generate_enhanced_insights(
                crop, predicted_price, market_trend,
                {
                    'yield_prediction': yield_prediction,
                    'global_demand': global_demand,
                    'weather_impact': weather_impact,
                    'economic_condition': economic_condition,
                    'supply_index': supply_index,
                    'demand_index': demand_index,
                    'inventory_level': inventory_level,
                    'export_demand': export_demand
                },
                confidence_score
            )
            return {
                'predicted_price': round(predicted_price, 2),
                'market_trend': market_trend,
                'confidence_score': round(confidence_score, 4),
                'input_data': sample_data,
                'enhanced_insights': enhanced_insights
            }
        except Exception:
            return {
                'error': f'Error in market price prediction: {str(e)}',
                'predicted_price': None,
                'market_trend': 'unknown',
                'confidence_score': 0.0
            }

def _fallback_price_estimate(production_cost: float, supply_index: float, demand_index: float,
                             inventory_level: float, export_demand: float, yield_prediction: float) -> float:
    """Heuristic price estimate when the trained model isn't available.
    The formula blends production cost and simple market signals. Scaled for USD/ton outputs.
    """
    # Baseline markup over cost
    price = production_cost * 1.35
    
    # Demand vs supply adjustment
    if demand_index <= 0:
        demand_index = 1.0
    ratio = supply_index / demand_index
    price *= (1.0 + (0.25 if ratio < 0.8 else -0.2 if ratio > 1.2 else 0.0))
    
    # Inventory dampener / booster
    if inventory_level < 30:
        price *= 1.1
    elif inventory_level > 70:
        price *= 0.92
    
    # Export demand uplift
    if export_demand > 70:
        price *= 1.05
    elif export_demand < 40:
        price *= 0.97
    
    # Yield scarcity effect (lower yield => higher price)
    if yield_prediction < 2.0:
        price *= 1.15
    elif yield_prediction > 6.0:
        price *= 0.95
    
    # Keep in reasonable bounds
    return max(50.0, min(price, 2000.0))

def _determine_market_trend(yield_prediction, global_demand, weather_impact, 
                           supply_index, demand_index, inventory_level, export_demand):
    """
    Determine market trend based on multiple factors
    """
    # Simple logic to determine trend
    bullish_factors = 0
    bearish_factors = 0
    
    # Low yield is bullish for prices
    if yield_prediction < 2.0:
        bullish_factors += 2
    elif yield_prediction > 6.0:
        bearish_factors += 2
    
    # High demand is bullish
    if global_demand == 'high':
        bullish_factors += 2
    elif global_demand == 'low':
        bearish_factors += 2
    
    # Poor weather is bullish (reduces supply)
    if weather_impact == 'poor':
        bullish_factors += 2
    elif weather_impact == 'excellent':
        bearish_factors += 1
    
    # Supply-demand balance
    supply_demand_ratio = supply_index / demand_index if demand_index > 0 else 1
    if supply_demand_ratio < 0.8:
        bullish_factors += 2
    elif supply_demand_ratio > 1.2:
        bearish_factors += 2
    
    # Inventory levels
    if inventory_level < 30:
        bullish_factors += 1
    elif inventory_level > 70:
        bearish_factors += 1
    
    # Export demand
    if export_demand > 70:
        bullish_factors += 1
    elif export_demand < 40:
        bearish_factors += 1
    
    if bullish_factors > bearish_factors:
        return 'bullish'
    elif bearish_factors > bullish_factors:
        return 'bearish'
    else:
        return 'neutral'

def _calculate_confidence_score(yield_prediction, global_demand, weather_impact, 
                               supply_index, demand_index, inventory_level, export_demand):
    """
    Calculate confidence score for the prediction based on multiple factors
    """
    # Base confidence
    confidence = 0.75
    
    # Adjust based on data completeness and quality
    factors_present = 0
    total_factors = 7
    
    if yield_prediction is not None:
        factors_present += 1
    if global_demand != 'medium':
        factors_present += 1
    if weather_impact != 'normal':
        factors_present += 1
    if supply_index != 60.0:
        factors_present += 1
    if demand_index != 60.0:
        factors_present += 1
    if inventory_level != 50.0:
        factors_present += 1
    if export_demand != 60.0:
        factors_present += 1
    
    # Adjust confidence based on factors present
    confidence += (factors_present / total_factors) * 0.2
    
    # Ensure confidence is within bounds
    return max(0.1, min(0.95, confidence))

def _generate_enhanced_insights(crop, predicted_price, market_trend, input_data, confidence_score):
    """
    Generate enhanced insights using LLM and RAG if available (with caching)
    
    Args:
        crop (str): Crop name
        predicted_price (float): Predicted price per ton
        market_trend (str): Market trend
        input_data (dict): Input data used for prediction
        confidence_score (float): Confidence score
        
    Returns:
        dict: Enhanced insights and recommendations
    """
    # Initialize fallback insights
    insights = {
        'prediction_summary': f'{crop} market price prediction: ${predicted_price}/ton',
        'market_trend': market_trend,
        'key_factors': [],
        'recommendations': [],
        'risk_assessment': 'Moderate',
        'confidence_level': 'High' if confidence_score > 0.8 else 'Medium' if confidence_score > 0.6 else 'Low'
    }
    
    # If LLM is not available, use rule-based insights
    if not LLM_AVAILABLE:
        return _generate_rule_based_insights(crop, predicted_price, market_trend, input_data, confidence_score, insights)
    
    try:
        # Load LLM and RAG components
        llm, rag_system = load_llm_components()
        
        if llm is None or rag_system is None:
            print("LLM or RAG system not available")
            return _generate_rule_based_insights(crop, predicted_price, market_trend, input_data, confidence_score, insights)
        
        # Generate comprehensive market analysis using LLM
        analysis_prompt = f"""
        Provide a comprehensive market analysis for {crop} with a predicted price of ${predicted_price}/ton and a {market_trend} market trend.
        
        Consider these factors:
        - Yield prediction: {input_data.get('yield_prediction')} tons/hectare
        - Global demand: {input_data.get('global_demand')}
        - Weather impact: {input_data.get('weather_impact')}
        - Economic condition: {input_data.get('economic_condition')}
        - Supply index: {input_data.get('supply_index')}
        - Demand index: {input_data.get('demand_index')}
        - Inventory level: {input_data.get('inventory_level')}
        - Export demand: {input_data.get('export_demand')}
        
        Please provide:
        1. A detailed market analysis explaining the price prediction
        2. Key factors driving this trend
        3. Risk assessment for farmers
        4. Market outlook for the next 30 days
        
        Format the response in clear sections with actionable insights for farmers.
        """
        
        if llm.text_generator:
            try:
                analysis_response = llm.text_generator(
                    analysis_prompt,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True
                )
                insights['market_analysis'] = analysis_response[0]['generated_text'][len(analysis_prompt):].strip()
            except Exception as e:
                print(f"Error generating LLM market analysis: {str(e)}")
                insights['market_analysis'] = "Comprehensive market analysis not available at this time."
        else:
            insights['market_analysis'] = "LLM not available for comprehensive market analysis."
        
        # Generate detailed recommendations using RAG
        recommendation_query = f"What are the best market strategies for {crop} at ${predicted_price}/ton with a {market_trend} trend?"
        try:
            rag_context = rag_system.generate_context_aware_response(recommendation_query, {
                'crop': crop,
                'predicted_price': predicted_price,
                'market_trend': market_trend,
                'confidence_score': confidence_score,
                'yield_prediction': input_data.get('yield_prediction'),
                'global_demand': input_data.get('global_demand'),
                'weather_impact': input_data.get('weather_impact')
            })
            
            insights['llm_recommendations'] = rag_context
        except Exception as e:
            print(f"Error generating RAG recommendations: {str(e)}")
            insights['llm_recommendations'] = "Detailed recommendations not available at this time."
            
    except Exception as e:
        print(f"Error in enhanced insights generation: {str(e)}")
        # Fall back to rule-based insights
        return _generate_rule_based_insights(crop, predicted_price, market_trend, input_data, confidence_score, insights)
    
    return insights

def _generate_rule_based_insights(crop, predicted_price, market_trend, input_data, confidence_score, insights):
    """
    Generate rule-based insights as fallback when LLM is not available
    """
    # Add key factors based on input data
    if input_data.get('yield_prediction', 0) < 2.0:
        insights['key_factors'].append('Low yield predictions are driving prices upward')
    elif input_data.get('yield_prediction', 0) > 6.0:
        insights['key_factors'].append('High yield predictions are putting downward pressure on prices')
    
    if input_data.get('global_demand') == 'high':
        insights['key_factors'].append('Strong global demand is supporting higher prices')
    elif input_data.get('global_demand') == 'low':
        insights['key_factors'].append('Weak global demand is putting pressure on prices')
    
    if input_data.get('weather_impact') == 'poor':
        insights['key_factors'].append('Adverse weather conditions are expected to reduce supply and increase prices')
    elif input_data.get('weather_impact') == 'excellent':
        insights['key_factors'].append('Favorable weather conditions are expected to increase supply and moderate prices')
    
    # Add recommendations based on market trend
    if market_trend == 'bullish':
        insights['recommendations'].append('Consider holding inventory for higher prices')
        insights['recommendations'].append('Plan for increased revenue potential')
        insights['recommendations'].append('Consider forward contracting a portion of your crop to lock in current high prices')
    elif market_trend == 'bearish':
        insights['recommendations'].append('Consider forward contracting to lock in prices')
        insights['recommendations'].append('Explore value-added processing options')
        insights['recommendations'].append('Focus on cost reduction strategies to maintain profitability')
    else:
        insights['recommendations'].append('Monitor market conditions for opportunities')
        insights['recommendations'].append('Maintain balanced inventory levels')
        insights['recommendations'].append('Consider hedging strategies to protect against price volatility')
    
    # Add risk assessment
    if confidence_score < 0.5:
        insights['risk_assessment'] = 'High'
        insights['confidence_level'] = 'Low'
    elif confidence_score < 0.7:
        insights['risk_assessment'] = 'Moderate'
        insights['confidence_level'] = 'Medium'
    
    return insights

# Example usage
if __name__ == "__main__":
    # Example prediction with enhanced features
    result = predict_market_price(
        crop='Wheat',
        region='North',
        season='Summer',
        yield_prediction=3.5,
        global_demand='medium',
        weather_impact='normal',
        economic_condition='stable',
        supply_index=60.0,
        demand_index=60.0,
        inventory_level=50.0,
        export_demand=60.0,
        production_cost=150.0,
        days_to_harvest=90,
        fertilizer_usage='medium',
        irrigation_usage='medium'
    )
    
    print("Market Prediction Result:")
    print(f"Predicted Price: ${result.get('predicted_price', 'N/A')}/ton")
    print(f"Market Trend: {result.get('market_trend', 'N/A')}")
    print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
    
    if 'enhanced_insights' in result:
        insights = result['enhanced_insights']
        print(f"\nPrediction Summary: {insights.get('prediction_summary', 'N/A')}")
        print(f"Market Trend: {insights.get('market_trend', 'N/A')}")
        print(f"Confidence Level: {insights.get('confidence_level', 'N/A')}")
        print(f"Risk Assessment: {insights.get('risk_assessment', 'N/A')}")
        
        if 'llm_explanation' in insights:
            print(f"\nLLM Explanation: {insights['llm_explanation']}")
        
        if 'llm_recommendations' in insights:
            print(f"\nLLM Recommendations: {insights['llm_recommendations']}")