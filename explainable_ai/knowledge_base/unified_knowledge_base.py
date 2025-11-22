"""
Unified Agricultural Knowledge Base

This module provides a unified interface to access all agricultural knowledge bases
including crop yield, plant diseases, crop recommendations, pests, and weeds.
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from explainable_ai.knowledge_base.crop_yield_knowledge_base import EnhancedYieldPredictionKB
from explainable_ai.knowledge_base.plant_disease_knowledge_base import EnhancedPlantDiseaseKB
from explainable_ai.knowledge_base.crop_recommendation_knowledge_base import EnhancedCropRecommendationKB
from explainable_ai.knowledge_base.market_prediction_knowledge_base import EnhancedMarketPredictionKB
from explainable_ai.knowledge_base.pest_prediction_knowledge_base import EnhancedPestPredictionKB

class UnifiedAgriculturalKB:
    """Unified interface for all agricultural knowledge bases"""
    
    def __init__(self, knowledge_base_path='explainable_ai/knowledge_base'):
        """
        Initialize the unified agricultural knowledge base
        
        Args:
            knowledge_base_path: Path to the knowledge base directory
        """
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize all specialized knowledge bases
        try:
            self.yield_kb = EnhancedYieldPredictionKB(
                knowledge_base_path=os.path.join(knowledge_base_path, 'crop_yield')
            )
        except Exception as e:
            print(f"Warning: Could not initialize yield knowledge base: {str(e)}")
            self.yield_kb = None
        
        try:
            self.disease_kb = EnhancedPlantDiseaseKB(
                knowledge_base_path=os.path.join(knowledge_base_path, 'plant_disease')
            )
        except Exception as e:
            print(f"Warning: Could not initialize disease knowledge base: {str(e)}")
            self.disease_kb = None
        
        try:
            self.recommendation_kb = EnhancedCropRecommendationKB(
                knowledge_base_path=os.path.join(knowledge_base_path, 'crop_recommendation')
            )
        except Exception as e:
            print(f"Warning: Could not initialize recommendation knowledge base: {str(e)}")
            self.recommendation_kb = None
        
        try:
            self.market_prediction_kb = EnhancedMarketPredictionKB(
                knowledge_base_path=os.path.join(knowledge_base_path, 'market_prediction')
            )
        except Exception as e:
            print(f"Warning: Could not initialize market prediction knowledge base: {str(e)}")
            self.market_prediction_kb = None
        
        try:
            self.pest_prediction_kb = EnhancedPestPredictionKB(
                knowledge_base_path=os.path.join(knowledge_base_path, 'pest_prediction')
            )
        except Exception as e:
            print(f"Warning: Could not initialize pest prediction knowledge base: {str(e)}")
            self.pest_prediction_kb = None
        
        print("Unified agricultural knowledge base initialized with available specialized modules")
    
    def query_knowledge_base(self, query, domain=None, k=3):
        """
        Query the appropriate knowledge base based on the domain or search all
        
        Args:
            query: Query string
            domain: Specific domain to search (yield, disease, recommendation, market_prediction, pest_prediction, or None for all)
            k: Number of documents to retrieve per knowledge base
            
        Returns:
            list: List of relevant documents with similarity scores
        """
        results = []
        
        # Search in specific domain or all domains
        if (domain == 'yield' or domain is None) and self.yield_kb is not None:
            yield_results = self.yield_kb.retrieve_relevant_documents(query, k)
            for result in yield_results:
                result['domain'] = 'crop_yield'
            results.extend(yield_results)
        
        if (domain == 'disease' or domain is None) and self.disease_kb is not None:
            disease_results = self.disease_kb.retrieve_relevant_documents(query, k)
            for result in disease_results:
                result['domain'] = 'plant_disease'
            results.extend(disease_results)
        
        if (domain == 'recommendation' or domain is None) and self.recommendation_kb is not None:
            recommendation_results = self.recommendation_kb.retrieve_relevant_documents(query, k)
            for result in recommendation_results:
                result['domain'] = 'crop_recommendation'
            results.extend(recommendation_results)
        
        if (domain == 'market_prediction' or domain is None) and self.market_prediction_kb is not None:
            market_results = self.market_prediction_kb.retrieve_relevant_documents(query, k)
            for result in market_results:
                result['domain'] = 'market_prediction'
            results.extend(market_results)
        
        if (domain == 'pest_prediction' or domain is None) and self.pest_prediction_kb is not None:
            pest_results = self.pest_prediction_kb.retrieve_relevant_documents(query, k)
            for result in pest_results:
                result['domain'] = 'pest_prediction'
            results.extend(pest_results)
        
        # Sort results by similarity score (higher is better)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return results[:k]
    
    def add_domain_specific_information(self, domain, **kwargs):
        """
        Add new information to a specific domain knowledge base
        
        Args:
            domain: Domain to add information to (yield, disease, recommendation, market_prediction, pest_prediction)
            **kwargs: Domain-specific parameters
        """
        if domain == 'yield' and self.yield_kb is not None:
            # Add yield prediction record
            prediction_data = kwargs.get('prediction_data', {})
            actual_yield = kwargs.get('actual_yield')
            self.yield_kb.add_yield_prediction_record(prediction_data, actual_yield)
            
        elif domain == 'disease' and self.disease_kb is not None:
            # Add disease information
            disease_name = kwargs.get('disease_name')
            symptoms = kwargs.get('symptoms')
            management_strategies = kwargs.get('management_strategies')
            if disease_name and symptoms and management_strategies:
                self.disease_kb.add_disease_information(disease_name, symptoms, management_strategies)
                
        elif domain == 'recommendation' and self.recommendation_kb is not None:
            # Add recommendation information
            crop_name = kwargs.get('crop_name')
            soil_conditions = kwargs.get('soil_conditions')
            weather_conditions = kwargs.get('weather_conditions')
            fertilizer_recommendation = kwargs.get('fertilizer_recommendation')
            if crop_name and soil_conditions and weather_conditions and fertilizer_recommendation:
                self.recommendation_kb.add_recommendation_information(
                    crop_name, soil_conditions, weather_conditions, fertilizer_recommendation)
        
        elif domain == 'market_prediction' and self.market_prediction_kb is not None:
            # Add market prediction information
            title = kwargs.get('title')
            content = kwargs.get('content')
            metadata = kwargs.get('metadata', {})
            if title and content:
                self.market_prediction_kb.add_market_information(title, content, metadata)
        
        elif domain == 'pest_prediction' and self.pest_prediction_kb is not None:
            # Add pest prediction information
            title = kwargs.get('title')
            content = kwargs.get('content')
            metadata = kwargs.get('metadata', {})
            if title and content:
                self.pest_prediction_kb.add_pest_information(title, content, metadata)
    
    def save_all_knowledge_bases(self, base_path=None):
        """
        Save all knowledge bases to disk
        
        Args:
            base_path: Base path to save knowledge bases (default: knowledge_base_path)
        """
        if base_path is None:
            base_path = self.knowledge_base_path
            
        if self.yield_kb is not None:
            self.yield_kb.save_knowledge_base(os.path.join(base_path, 'crop_yield'))
        if self.disease_kb is not None:
            self.disease_kb.save_knowledge_base(os.path.join(base_path, 'plant_disease'))
        if self.recommendation_kb is not None:
            self.recommendation_kb.save_knowledge_base(os.path.join(base_path, 'crop_recommendation'))
        if self.market_prediction_kb is not None:
            self.market_prediction_kb.save_knowledge_base(os.path.join(base_path, 'market_prediction'))
        if self.pest_prediction_kb is not None:
            self.pest_prediction_kb.save_knowledge_base(os.path.join(base_path, 'pest_prediction'))
        
        print("All available knowledge bases saved successfully")
    
    def load_all_knowledge_bases(self, base_path=None):
        """
        Load all knowledge bases from disk
        
        Args:
            base_path: Base path to load knowledge bases from (default: knowledge_base_path)
        """
        if base_path is None:
            base_path = self.knowledge_base_path
            
        if self.yield_kb is not None:
            self.yield_kb.load_knowledge_base_from_disk(os.path.join(base_path, 'crop_yield'))
        if self.disease_kb is not None:
            self.disease_kb.load_knowledge_base_from_disk(os.path.join(base_path, 'plant_disease'))
        if self.recommendation_kb is not None:
            self.recommendation_kb.load_knowledge_base_from_disk(os.path.join(base_path, 'crop_recommendation'))
        if self.market_prediction_kb is not None:
            self.market_prediction_kb.load_knowledge_base_from_disk(os.path.join(base_path, 'market_prediction'))
        if self.pest_prediction_kb is not None:
            self.pest_prediction_kb.load_knowledge_base_from_disk(os.path.join(base_path, 'pest_prediction'))
        
        print("All available knowledge bases loaded successfully")

# Example usage
if __name__ == "__main__":
    # Initialize unified knowledge base
    unified_kb = UnifiedAgriculturalKB()
    
    # Example queries
    queries = [
        "How to improve wheat yield?",
        "What are the symptoms of tomato late blight?",
        "What fertilizer is best for rice cultivation?",
        "How do weather conditions affect market prices?",
        "How to identify and manage aphid infestations?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = unified_kb.query_knowledge_base(query, k=2)
        for i, result in enumerate(results):
            print(f"  {i+1}. [{result['domain']}] {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
            print(f"     Content: {result['content'][:150]}...")