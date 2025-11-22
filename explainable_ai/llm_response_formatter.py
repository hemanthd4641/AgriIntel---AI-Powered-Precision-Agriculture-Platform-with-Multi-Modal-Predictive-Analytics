"""
Unified LLM Response Formatter for Smart Agriculture Project

This module provides a consistent formatting system for LLM-generated responses
across all agricultural features including yield prediction, disease detection,
crop recommendations, market analysis, and pest management.
"""

class LLMResponseFormatter:
    """Unified formatter for LLM responses across all agricultural features"""
    
    @staticmethod
    def format_yield_prediction_response(prediction_data, insights, explanation):
        """
        Format yield prediction response with consistent structure
        
        Args:
            prediction_data (dict): Input data used for prediction
            insights (dict): Generated insights about the prediction
            explanation (str): LLM-generated explanation
            
        Returns:
            dict: Formatted response
        """
        return {
            "type": "yield_prediction",
            "prediction": {
                "value": prediction_data.get("predicted_yield"),
                "unit": "tons/hectare",
                "crop": prediction_data.get("Crop", "Unknown")
            },
            "analysis": {
                "summary": insights.get("prediction_summary", ""),
                "confidence": insights.get("confidence_level", ""),
                "benchmark_comparison": insights.get("comparison_to_benchmarks", "")
            },
            "factors": {
                "key_factors": insights.get("key_factors", []),
                "detailed_factors": insights.get("detailed_factors", {}),
                "nutrient_analysis": insights.get("nutrient_analysis", {}),
                "water_management": insights.get("water_management", {}),
                "timing_advice": insights.get("timing_advice", {})
            },
            "recommendations": {
                "immediate_actions": insights.get("recommendations", []),
                "detailed_explanation": explanation
            },
            "metadata": {
                "timestamp": prediction_data.get("timestamp", ""),
                "feature_version": "enhanced"
            }
        }
    
    @staticmethod
    def format_disease_detection_response(disease_name, severity, advice_data):
        """
        Format disease detection response with consistent structure
        
        Args:
            disease_name (str): Name of the detected disease
            severity (str): Severity level
            advice_data (dict): Disease advice data
            
        Returns:
            dict: Formatted response
        """
        return {
            "type": "disease_detection",
            "diagnosis": {
                "disease": disease_name,
                "severity": severity,
                "confidence": advice_data.get("confidence", "N/A")
            },
            "management": {
                "comprehensive_advice": advice_data.get("comprehensive_advice", ""),
                "treatment_options": advice_data.get("treatment_options", []),
                "prevention_strategies": advice_data.get("prevention_strategies", []),
                "monitoring_guidance": advice_data.get("monitoring_guidance", {})
            },
            "related_information": {
                "documents": advice_data.get("related_documents", []),
                "additional_resources": []
            },
            "metadata": {
                "feature_version": "enhanced"
            }
        }
    
    @staticmethod
    def format_crop_recommendation_response(crop_name, advice_data, fertilizer_data):
        """
        Format crop recommendation response with consistent structure
        
        Args:
            crop_name (str): Name of the recommended crop
            advice_data (dict): Crop advice data
            fertilizer_data (dict): Fertilizer advice data
            
        Returns:
            dict: Formatted response
        """
        return {
            "type": "crop_recommendation",
            "recommendation": {
                "crop": crop_name,
                "confidence": advice_data.get("confidence", "N/A")
            },
            "cultivation_guide": {
                "comprehensive_advice": advice_data.get("comprehensive_advice", ""),
                "planting_recommendations": advice_data.get("planting_recommendations", {}),
                "fertilization_schedule": advice_data.get("fertilization_schedule", {}),
                "irrigation_guidance": advice_data.get("irrigation_guidance", {}),
                "pest_management": advice_data.get("pest_management", {}),
                "harvest_guidance": advice_data.get("harvest_guidance", {})
            },
            "fertilizer_advice": {
                "product": fertilizer_data.get("fertilizer_type", "N/A"),
                "application_methods": fertilizer_data.get("application_methods", {}),
                "timing_guidance": fertilizer_data.get("timing_guidance", {}),
                "dosage_recommendations": fertilizer_data.get("dosage_recommendations", {}),
                "safety_precautions": fertilizer_data.get("safety_precautions", [])
            },
            "metadata": {
                "feature_version": "enhanced"
            }
        }
    
    @staticmethod
    def format_market_prediction_response(prediction_data, analysis_data):
        """
        Format market prediction response with consistent structure
        
        Args:
            prediction_data (dict): Market prediction data
            analysis_data (dict): Market analysis data
            
        Returns:
            dict: Formatted response
        """
        return {
            "type": "market_prediction",
            "prediction": {
                "crop": prediction_data.get("crop", "Unknown"),
                "price": prediction_data.get("predicted_price", 0),
                "unit": "$/ton",
                "trend": prediction_data.get("market_trend", "neutral")
            },
            "market_analysis": {
                "comprehensive_analysis": analysis_data.get("comprehensive_analysis", ""),
                "price_outlook": analysis_data.get("market_analysis", {}).get("price_outlook", ""),
                "supply_demand_factors": analysis_data.get("market_analysis", {}).get("supply_demand_factors", []),
                "external_influences": analysis_data.get("market_analysis", {}).get("external_influences", []),
                "market_indicators": analysis_data.get("market_analysis", {}).get("market_indicators", {})
            },
            "strategies": {
                "timing_advice": analysis_data.get("timing_advice", {}),
                "risk_assessment": analysis_data.get("risk_assessment", {})
            },
            "supporting_data": {
                "related_documents": analysis_data.get("related_documents", []),
                "additional_context": analysis_data.get("context", "")
            },
            "metadata": {
                "confidence": prediction_data.get("confidence_score", 0),
                "feature_version": "enhanced"
            }
        }
    
    @staticmethod
    def format_pest_prediction_response(prediction_data, management_data):
        """
        Format pest prediction response with consistent structure
        
        Args:
            prediction_data (dict): Pest prediction data
            management_data (dict): Pest management data
            
        Returns:
            dict: Formatted response
        """
        return {
            "type": "pest_prediction",
            "prediction": {
                "pest": prediction_data.get("predicted_pest", "Unknown"),
                "severity": prediction_data.get("severity", "Moderate"),
                "confidence": prediction_data.get("confidence_score", 0)
            },
            "management_plan": {
                "comprehensive_analysis": management_data.get("comprehensive_analysis", ""),
                "integrated_strategies": management_data.get("management_strategies", {}),
                "control_methods": management_data.get("control_methods", {}),
                "prevention_advice": management_data.get("prevention_advice", {})
            },
            "supporting_information": {
                "related_documents": management_data.get("related_documents", []),
                "context": management_data.get("context", "")
            },
            "metadata": {
                "feature_version": "enhanced"
            }
        }
    
    @staticmethod
    def format_chatbot_response(user_query, response_data, source_documents=None):
        """
        Format chatbot response with consistent structure
        
        Args:
            user_query (str): Original user query
            response_data (dict): Response data from chatbot
            source_documents (list): Source documents used
            
        Returns:
            dict: Formatted response
        """
        return {
            "type": "chatbot_response",
            "query": user_query,
            "response": {
                "answer": response_data.get("answer", response_data.get("response", "")),
                "confidence": response_data.get("confidence", "N/A")
            },
            "supporting_information": {
                "source_documents": source_documents or response_data.get("source_documents", []),
                "related_topics": []
            },
            "metadata": {
                "feature_version": "enhanced"
            }
        }
    
    @staticmethod
    def format_error_response(error_message, feature_type, details=None):
        """
        Format error response with consistent structure
        
        Args:
            error_message (str): Error message
            feature_type (str): Type of feature that encountered error
            details (dict): Additional error details
            
        Returns:
            dict: Formatted error response
        """
        return {
            "type": "error",
            "error": {
                "message": error_message,
                "feature_type": feature_type,
                "details": details or {}
            },
            "recommendations": {
                "immediate_actions": [
                    "Check input data for accuracy",
                    "Verify system components are properly configured",
                    "Try the request again after a brief moment"
                ],
                "support_contacts": [
                    "Contact system administrator for persistent issues",
                    "Check system logs for detailed error information"
                ]
            },
            "metadata": {
                "timestamp": "",
                "feature_version": "enhanced"
            }
        }

# Example usage
if __name__ == "__main__":
    # Example of formatting a yield prediction response
    formatter = LLMResponseFormatter()
    
    sample_prediction_data = {
        "predicted_yield": 4.2,
        "Crop": "Wheat",
        "timestamp": "2023-06-15T10:30:00Z"
    }
    
    sample_insights = {
        "prediction_summary": "Good yield prediction of 4.20 tons/hectare for Wheat.",
        "confidence_level": "High confidence - model explains variance well",
        "comparison_to_benchmarks": "Prediction is within 20% of typical Wheat yield benchmark of 3.5 tons/hectare",
        "key_factors": [
            "Positive: Rainfall is within optimal range for crop growth",
            "Positive: Temperature is within optimal range for crop development"
        ],
        "recommendations": [
            "Monitor crop closely as harvest approaches",
            "Ensure adequate storage facilities are available"
        ]
    }
    
    sample_explanation = "Based on optimal growing conditions..."
    
    formatted_response = formatter.format_yield_prediction_response(
        sample_prediction_data, 
        sample_insights, 
        sample_explanation
    )
    
    print("Formatted Yield Prediction Response:")
    print(formatted_response)