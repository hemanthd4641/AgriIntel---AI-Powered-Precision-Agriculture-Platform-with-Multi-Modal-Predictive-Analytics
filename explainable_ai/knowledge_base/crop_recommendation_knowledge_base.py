"""
Enhanced Crop Recommendation Knowledge Base

This module creates a comprehensive knowledge base specifically for crop and fertilizer recommendations
using datasets and FAISS vector database with enhanced data loading capabilities.
"""

import numpy as np
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
# Handle text splitter import with fallbacks
try:
    from langchain_text_splitters.character import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# Handle Document import with fallbacks
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

# Handle the HuggingFaceEmbeddings import with fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class EnhancedCropRecommendationKB:
    """Enhanced knowledge base for crop and fertilizer recommendations"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', knowledge_base_path='explainable_ai/knowledge_base/crop_recommendation'):
        """
        Initialize the enhanced crop recommendation knowledge base
        
        Args:
            model_name: Name of the sentence transformer model for embeddings
            knowledge_base_path: Path to the knowledge base directory
        """
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize FAISS vector store
        self.vector_store = None
        
        # Load knowledge base
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """
        Load comprehensive crop recommendation knowledge from multiple sources
        """
        documents = []
        
        # Load crop recommendation dataset - fix the path to be relative to the project root
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        dataset_path = os.path.join(project_root, "datasets", "crop_recommendation", "Crop and fertilizer dataset.csv")
        
        if os.path.exists(dataset_path):
            try:
                print("Loading crop recommendation dataset...")
                # Load a sample of the dataset to avoid memory issues
                df = pd.read_csv(dataset_path, nrows=5000)  # Load only first 5,000 rows for demo
                print(f"Loaded {len(df)} rows from dataset")
                
                # Create knowledge documents from dataset patterns
                # Group by crop to find common patterns
                for crop in df['Crop'].unique():
                    crop_data = df[df['Crop'] == crop]
                    
                    # Calculate average values for this crop
                    avg_nitrogen = crop_data['Nitrogen'].mean()
                    avg_phosphorus = crop_data['Phosphorus'].mean()
                    avg_potassium = crop_data['Potassium'].mean()
                    avg_ph = crop_data['pH'].mean()
                    avg_rainfall = crop_data['Rainfall'].mean()
                    avg_temperature = crop_data['Temperature'].mean()
                    
                    # Get most common fertilizer for this crop
                    common_fertilizer = crop_data['Fertilizer'].mode().iloc[0] if not crop_data['Fertilizer'].mode().empty else "Not specified"
                    
                    # Create knowledge document
                    content = f"""
                    Crop: {crop}
                    Optimal Growing Conditions:
                    - Nitrogen Requirement: {avg_nitrogen:.1f} ppm
                    - Phosphorus Requirement: {avg_phosphorus:.1f} ppm
                    - Potassium Requirement: {avg_potassium:.1f} ppm
                    - Optimal pH Range: {avg_ph-0.5:.1f} - {avg_ph+0.5:.1f}
                    - Average Rainfall: {avg_rainfall:.1f} mm
                    - Temperature Range: {avg_temperature-3:.1f} - {avg_temperature+3:.1f}°C
                    
                    Recommended Fertilizer: {common_fertilizer}
                    
                    Additional Considerations:
                    - Soil testing is recommended before planting
                    - Adjust fertilizer rates based on soil test results
                    - Consider crop rotation to maintain soil health
                    - Monitor for nutrient deficiencies during growth
                    - Apply fertilizers at appropriate growth stages
                    """
                    
                    document = Document(
                        page_content=content,
                        metadata={
                            "title": f"{crop} Recommendation Knowledge",
                            "crop": crop,
                            "type": "crop_recommendation"
                        }
                    )
                    documents.append(document)
                
                print(f"Loaded recommendation knowledge for {len(df['Crop'].unique())} crops")
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
        else:
            print(f"Dataset not found at {dataset_path}")
        
        # Add comprehensive crop recommendation knowledge
        crop_knowledge = [
            {
                "title": "Wheat Cultivation",
                "content": """Wheat cultivation requires specific conditions for optimal growth and yield.
                Soil Requirements:
                - Well-drained soil with pH 6.0-7.5
                - Good fertility with adequate nitrogen, phosphorus, and potassium
                - Proper soil structure for root development
                
                Climate Requirements:
                - Optimal temperature for growth is 15-25°C
                - Moderate rainfall or irrigation (400-600mm during growing season)
                - Cool temperatures during grain filling for quality
                
                Fertilization:
                - Apply nitrogen fertilizer in split doses - half at planting and half at tillering stage
                - Phosphorus and potassium applied before planting
                - Consider sulfur and micronutrient needs based on soil tests
                
                Management Practices:
                - Proper seeding rate and depth
                - Weed control during early growth stages
                - Disease monitoring, especially for rust diseases
                - Timely harvesting at proper moisture content"""
            },
            {
                "title": "Rice Cultivation",
                "content": """Rice cultivation has specific requirements for water and nutrients.
                Soil Requirements:
                - Clayey soil with pH 5.5-7.0 for best results
                - Good water retention capacity
                - Adequate organic matter content
                
                Climate Requirements:
                - Requires 100-150 cm of water during growing season
                - Optimal temperature range is 20-35°C
                - High humidity during grain development
                
                Fertilization:
                - Apply basal fertilizer before transplanting
                - Top-dress with nitrogen during tillering
                - Consider potassium for straw strength and disease resistance
                - Micronutrients like zinc may be needed in some soils
                
                Management Practices:
                - Proper water management through irrigation scheduling
                - Weed control in early growth stages
                - Pest monitoring for stem borers and leaf folders
                - Timely harvesting to prevent shattering losses"""
            },
            {
                "title": "Maize Cultivation",
                "content": """Maize cultivation requires warm temperatures and adequate nutrients.
                Soil Requirements:
                - Well-drained fertile soil with pH 5.8-7.0
                - Good organic matter content
                - Adequate depth for root development
                
                Climate Requirements:
                - Thrives in warm weather with 20-25°C temperature
                - Requires 500-800mm of water during growing season
                - Sensitive to frost during germination and early growth
                
                Fertilization:
                - Apply nitrogen fertilizer in three splits: at planting, knee-high stage, and tasseling
                - Phosphorus applied before planting for root development
                - Potassium for stalk strength and disease resistance
                - Consider zinc as it's commonly deficient for maize
                
                Management Practices:
                - Proper plant population for hybrid varieties
                - Weed control during early growth stages
                - Pest monitoring for corn borers and armyworms
                - Timely harvesting at proper moisture content"""
            },
            {
                "title": "Cotton Cultivation",
                "content": """Cotton cultivation requires warm weather and good soil conditions.
                Soil Requirements:
                - Deep, well-drained soil with pH 6.0-8.5
                - Good fertility with adequate nitrogen and potassium
                - Proper drainage to prevent waterlogging
                
                Climate Requirements:
                - Requires warm weather with 20-30°C temperature
                - Needs 500-1000mm of water during growing season
                - Long frost-free period for full maturity
                
                Fertilization:
                - Apply nitrogen and potassium fertilizers in splits during growing season
                - Phosphorus applied before planting for root development
                - Consider magnesium and sulfur in some soils
                - Micronutrients like boron may be needed for boll development
                
                Management Practices:
                - Proper plant spacing for air circulation
                - Weed control during early growth stages
                - Pest monitoring for boll weevils and aphids
                - Timely defoliation for harvest management"""
            },
            {
                "title": "Sugarcane Cultivation",
                "content": """Sugarcane cultivation requires specific conditions for high sugar content.
                Soil Requirements:
                - Deep, well-drained soil with pH 6.5-7.5
                - High organic matter content
                - Good water retention capacity
                
                Climate Requirements:
                - Prefers warm climate with 25-35°C temperature
                - Requires 1200-1500mm of water during growing season
                - Needs high sunlight for sugar development
                
                Fertilization:
                - Apply organic manure along with NPK fertilizers before planting
                - Nitrogen in splits during tillering and grand growth stages
                - Potassium for sugar content and disease resistance
                - Consider micronutrients like zinc and boron
                
                Management Practices:
                - Proper ratoon management for successive crops
                - Weed control during early growth stages
                - Pest monitoring for borers and whiteflies
                - Timely harvesting for optimal sugar content"""
            }
        ]
        
        # Add crop knowledge documents
        for doc in crop_knowledge:
            document = Document(
                page_content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "crop": doc["title"].replace(" Cultivation", ""),
                    "type": "crop_recommendation"
                }
            )
            documents.append(document)
        
        # Add comprehensive fertilizer knowledge
        fertilizer_knowledge = [
            {
                "title": "Organic Fertilizers",
                "content": """Organic fertilizers improve soil structure and provide slow-release nutrients.
                Benefits:
                - Improve water retention and soil microbial activity
                - Enhance soil structure and aggregation
                - Provide a wide range of micronutrients
                - Reduce nutrient leaching and runoff
                
                Application:
                - Apply 5-10 tons per hectare before planting
                - Incorporate into soil during tillage operations
                - Consider timing with crop nutrient demand
                - Monitor decomposition rates for nutrient availability
                
                Common Types:
                - Composted animal manures (cattle, poultry, sheep)
                - Plant-based composts and green manures
                - Fish emulsions and seaweed extracts
                - Bone meal and blood meal for specific nutrients"""
            },
            {
                "title": "Fertilizer Application Timing",
                "content": """Proper fertilizer application timing maximizes nutrient use efficiency.
                General Principles:
                - Apply basal fertilizer before planting for early growth
                - Top-dress with nitrogen during active growth stages
                - Avoid fertilizer application during flowering to prevent flower drop
                - Consider weather conditions and soil moisture levels
                
                Crop-Specific Timing:
                - Cereals: Basal application + tillering/top-dress
                - Legumes: Less nitrogen needed due to nitrogen fixation
                - Vegetables: Multiple applications during growth cycle
                - Root crops: Emphasis on potassium for tuber development
                
                Factors to Consider:
                - Soil temperature and moisture
                - Crop growth stage and nutrient demand
                - Weather forecasts to avoid leaching losses
                - Equipment availability and labor constraints"""
            },
            {
                "title": "Soil Testing",
                "content": """Regular soil testing helps determine nutrient deficiencies and pH levels.
                Benefits:
                - Determine existing nutrient levels and pH
                - Adjust fertilizer application rates for efficiency
                - Identify nutrient imbalances and toxicities
                - Monitor changes in soil properties over time
                
                Testing Procedures:
                - Test soil every 2-3 years for routine monitoring
                - Collect samples from multiple locations in the field
                - Sample at proper depth for the crop being grown
                - Submit samples to certified laboratories for analysis
                
                Interpretation:
                - Compare results to crop nutrient requirements
                - Consider soil test ratings (low, medium, high)
                - Adjust fertilizer recommendations based on yield goals
                - Account for nutrient interactions and antagonisms"""
            },
            {
                "title": "Water Management",
                "content": """Proper irrigation scheduling improves nutrient uptake and crop yields.
                Principles:
                - Apply fertilizers when soil moisture is adequate for nutrient movement
                - Avoid fertilizer application during waterlogging conditions
                - Coordinate irrigation with nutrient application timing
                - Monitor soil moisture to optimize water use efficiency
                
                Methods:
                - Drip irrigation for efficient water and nutrient use
                - Sprinkler systems for uniform water distribution
                - Furrow irrigation for row crops with proper management
                - Flood irrigation with careful timing and duration
                
                Benefits:
                - Improved nutrient uptake efficiency
                - Reduced nutrient losses through leaching
                - Better crop growth and yield potential
                - Conservation of water resources"""
            },
            {
                "title": "Integrated Nutrient Management",
                "content": """Combine organic and inorganic fertilizers for sustainable crop production.
                Principles:
                - Use crop residues, green manure, and biofertilizers along with chemical fertilizers
                - Consider nutrient contributions from all sources
                - Balance nutrient supply with crop demand
                - Maintain soil health for long-term productivity
                
                Practices:
                - Incorporate legumes in rotation for nitrogen fixation
                - Use organic amendments to improve soil properties
                - Apply biofertilizers to enhance nutrient availability
                - Monitor and adjust based on crop response
                
                Benefits:
                - Reduced dependence on chemical fertilizers
                - Improved soil health and structure
                - Enhanced nutrient use efficiency
                - Sustainable agricultural practices"""
            },
            {
                "title": "Micronutrient Management",
                "content": """Deficiency of micronutrients affects crop growth and yield.
                Common Deficiencies:
                - Zinc: Stunted growth, interveinal chlorosis
                - Iron: Interveinal chlorosis in young leaves
                - Manganese: Interveinal chlorosis, poor growth
                - Boron: Poor fruit/seed development, growing point death
                
                Management:
                - Apply micronutrient fertilizers based on soil test results
                - Foliar application is effective for quick correction
                - Consider chelated forms for better availability
                - Monitor crop symptoms for early detection
                
                Application Methods:
                - Soil application for long-term corrections
                - Foliar sprays for quick fixes and supplements
                - Seed treatments for early season needs
                - Blending with other fertilizers for uniform distribution"""
            },
            {
                "title": "Fertilizer Efficiency",
                "content": """Use fertigation and other methods to improve fertilizer use efficiency.
                Methods:
                - Fertigation (fertilizer through irrigation) for efficient nutrient use
                - Apply fertilizers in bands near plant roots
                - Incorporate fertilizers into soil to prevent losses
                - Use controlled-release fertilizers for extended availability
                
                Factors Affecting Efficiency:
                - Soil pH and nutrient interactions
                - Moisture and temperature conditions
                - Application timing relative to crop demand
                - Method and placement of application
                
                Benefits:
                - Reduced fertilizer losses to environment
                - Improved crop response and yield
                - Better return on fertilizer investment
                - Reduced environmental impact"""
            }
        ]
        
        # Add fertilizer knowledge documents
        for doc in fertilizer_knowledge:
            document = Document(
                page_content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "type": "fertilizer_recommendation"
                }
            )
            documents.append(document)
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        try:
            self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            # Create a simple fallback vector store
            self.vector_store = self._create_fallback_vector_store(split_documents)
        
        print(f"Loaded {len(split_documents)} document chunks into the crop recommendation knowledge base")
    
    def _create_fallback_vector_store(self, documents):
        """
        Create a fallback vector store when normal embedding fails
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS: Simple vector store with basic embeddings
        """
        try:
            # Use a simple embedding method as fallback
            embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
            return FAISS.from_documents(documents, embeddings)
        except Exception as e:
            print(f"Error in fallback vector store creation: {str(e)}")
            # If all else fails, create a minimal vector store
            class SimpleVectorStore:
                def __init__(self, documents):
                    self.documents = documents
                
                def similarity_search_with_score(self, query, k=3):
                    # Return the first k documents as a simple fallback
                    return [(doc, 0.5) for doc in self.documents[:k]]
            
            return SimpleVectorStore(documents)
    
    def add_recommendation_information(self, crop_name, soil_conditions, weather_conditions, fertilizer_recommendation):
        """
        Add new recommendation information to the knowledge base
        
        Args:
            crop_name: Name of the crop
            soil_conditions: Description of soil conditions
            weather_conditions: Description of weather conditions
            fertilizer_recommendation: Recommended fertilizer application
        """
        # Create a document from the recommendation information
        content = f"""
        Crop: {crop_name}
        Soil Conditions: {soil_conditions}
        Weather Conditions: {weather_conditions}
        Fertilizer Recommendation: {fertilizer_recommendation}
        """
        
        document = Document(
            page_content=content,
            metadata={
                "title": f"{crop_name} Recommendation",
                "crop": crop_name,
                "type": "crop_recommendation"
            }
        )
        
        # Split document into chunks
        split_documents = self.text_splitter.split_documents([document])
        
        # Add to vector store
        if self.vector_store is not None:
            try:
                if hasattr(self.vector_store, 'add_documents'):
                    self.vector_store.add_documents(split_documents)
                    print(f"Added recommendation information with {len(split_documents)} chunks to the knowledge base")
                else:
                    # For fallback vector store, just add to documents list
                    if hasattr(self.vector_store, 'documents'):
                        self.vector_store.documents.extend(split_documents)
                        print(f"Added recommendation information with {len(split_documents)} chunks to the fallback knowledge base")
            except Exception as e:
                print(f"Error adding document to vector store: {str(e)}")
    
    def retrieve_relevant_documents(self, query, k=3):
        """
        Retrieve relevant documents based on a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            list: List of relevant documents with similarity scores
        """
        try:
            if self.vector_store is None:
                print("Knowledge base not loaded")
                return []
            
            # Search for similar documents
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                docs = self.vector_store.similarity_search_with_score(query, k=k)
            else:
                # For fallback vector store
                docs = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            results = []
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": float(score)
                })
            
            return results
        except Exception as e:
            print(f"Error in retrieve_relevant_documents: {str(e)}")
            return []

    def save_knowledge_base(self, path=None):
        """
        Save the knowledge base to disk
        
        Args:
            path: Path to save the knowledge base (default: knowledge_base_path)
        """
        if path is None:
            path = self.knowledge_base_path
            
        if self.vector_store is not None and hasattr(self.vector_store, 'save_local'):
            try:
                os.makedirs(path, exist_ok=True)
                self.vector_store.save_local(path)
                print(f"Knowledge base saved to {path}")
            except Exception as e:
                print(f"Error saving knowledge base: {str(e)}")
        else:
            print("Vector store does not support saving or is not initialized")
    
    def load_knowledge_base_from_disk(self, path=None):
        """
        Load the knowledge base from disk
        
        Args:
            path: Path to load the knowledge base from (default: knowledge_base_path)
        """
        if path is None:
            path = self.knowledge_base_path
            
        if os.path.exists(path):
            try:
                self.vector_store = FAISS.load_local(path, self.embeddings)
                print(f"Knowledge base loaded from {path}")
            except Exception as e:
                print(f"Error loading knowledge base: {str(e)}")
        else:
            print(f"Knowledge base not found at {path}")

# Example usage
if __name__ == "__main__":
    # Initialize crop recommendation knowledge base
    kb = EnhancedCropRecommendationKB()
    
    # Example query
    query = "What are the best practices for growing wheat?"
    results = kb.retrieve_relevant_documents(query, k=2)
    
    print(f"Query: {query}")
    print("Retrieved documents:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
        print(f"   Content: {result['content'][:200]}...")
