"""
Knowledge Base Package for Smart Agriculture Project

This package contains enhanced knowledge bases for various agricultural domains.
"""

# Import the main classes for easy access
from .crop_yield_knowledge_base import EnhancedYieldPredictionKB
from .plant_disease_knowledge_base import EnhancedPlantDiseaseKB
from .crop_recommendation_knowledge_base import EnhancedCropRecommendationKB
from .market_prediction_knowledge_base import EnhancedMarketPredictionKB
from .unified_knowledge_base import UnifiedAgriculturalKB

__all__ = [
    'EnhancedYieldPredictionKB',
    'EnhancedPlantDiseaseKB',
    'EnhancedCropRecommendationKB',
    'EnhancedMarketPredictionKB',
    'UnifiedAgriculturalKB'
]