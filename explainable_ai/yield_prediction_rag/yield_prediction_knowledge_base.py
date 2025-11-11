"""
Yield Prediction Knowledge Base for Smart Agriculture Project

This module creates a knowledge base specifically for crop yield prediction
using the dataset and FAISS vector database.
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

# Handle the HuggingFaceEmbeddings import with fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Handle Document import with fallbacks
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

class YieldPredictionRAG:
    """RAG system for crop yield prediction knowledge"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', knowledge_base_path='explainable_ai/yield_prediction_rag/knowledge_base'):
        """
        Initialize the yield prediction RAG system
        
        Args:
            model_name: Name of the sentence transformer model for embeddings
            knowledge_base_path: Path to the knowledge base directory
        """
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize sentence transformer model for embeddings
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not initialize embedding model {model_name}. Error: {str(e)}")
            self.embedding_model = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize FAISS vector store
        self.vector_store = None
        
        # Load knowledge base
        self.load_yield_prediction_knowledge()
    
    def load_yield_prediction_knowledge(self):
        """
        Load crop yield prediction knowledge from dataset into the RAG system
        """
        documents = []
        
        # Load crop yield prediction dataset
        # Fix the dataset path to be relative to the project root
        import os
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                   "datasets", "crop_yield_prediction", "crop_yield.csv")
        print(f"Looking for dataset at: {dataset_path}")
        if os.path.exists(dataset_path):
            try:
                print("Loading dataset...")
                # Load a small sample of the dataset to avoid memory issues
                # Use sampling to get a representative subset
                df_full = pd.read_csv(dataset_path, nrows=5000)  # Load only first 5000 rows for demo
                print(f"Loaded {len(df_full)} rows from dataset")
                
                # Sample to get a smaller, more manageable dataset
                df = df_full.sample(n=min(1000, len(df_full)), random_state=42)
                print(f"Sampled {len(df)} rows for processing")
                
                # Create knowledge documents from dataset patterns
                # Group by crop to find common patterns
                for crop in df['Crop'].unique()[:10]:  # Limit to first 10 crops to avoid too many documents
                    crop_data = df[df['Crop'] == crop]
                    
                    # Calculate average values for this crop
                    avg_rainfall = crop_data['Rainfall_mm'].mean()
                    avg_temperature = crop_data['Temperature_Celsius'].mean()
                    avg_days_to_harvest = crop_data['Days_to_Harvest'].mean()
                    avg_yield = crop_data['Yield_tons_per_hectare'].mean()
                    
                    # Calculate success factors
                    high_yield_threshold = crop_data['Yield_tons_per_hectare'].quantile(0.75)
                    high_yield_data = crop_data[crop_data['Yield_tons_per_hectare'] >= high_yield_threshold]
                    
                    success_rate_fertilizer = high_yield_data['Fertilizer_Used'].mean() if len(high_yield_data) > 0 else 0
                    success_rate_irrigation = high_yield_data['Irrigation_Used'].mean() if len(high_yield_data) > 0 else 0
                    
                    # Create knowledge document
                    content = f"""
                    Crop: {crop}
                    Optimal Growing Conditions:
                    - Average Rainfall: {avg_rainfall:.1f} mm
                    - Average Temperature: {avg_temperature:.1f}°C
                    - Average Days to Harvest: {avg_days_to_harvest:.0f} days
                    - Expected Yield: {avg_yield:.2f} tons/hectare
                    
                    Success Factors:
                    - Fertilizer Usage Success Rate: {success_rate_fertilizer*100:.1f}%
                    - Irrigation Usage Success Rate: {success_rate_irrigation*100:.1f}%
                    
                    High yield predictions for {crop} are typically associated with:
                    - Adequate rainfall ({avg_rainfall*.8:.1f} - {avg_rainfall*1.2:.1f} mm)
                    - Optimal temperature range ({avg_temperature-3:.1f} - {avg_temperature+3:.1f}°C)
                    - Proper timing of harvest around {avg_days_to_harvest:.0f} days
                    - {crop} responds well to fertilizer: {'Yes' if success_rate_fertilizer > 0.5 else 'No'}
                    - {crop} responds well to irrigation: {'Yes' if success_rate_irrigation > 0.5 else 'No'}
                    """
                    
                    document = Document(
                        page_content=content,
                        metadata={
                            "title": f"{crop} Yield Prediction Knowledge",
                            "crop": crop,
                            "type": "yield_prediction"
                        }
                    )
                    documents.append(document)
                
                print(f"Loaded yield prediction knowledge for {min(10, len(df['Crop'].unique()))} crops")
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
                # Create some sample documents if dataset loading fails
                documents.extend(self._create_sample_documents())
        else:
            print(f"Dataset not found at {dataset_path}")
            # Create some sample documents if dataset is not found
            documents.extend(self._create_sample_documents())
        
        # Add general agricultural knowledge about yield prediction
        general_knowledge = [
            {
                "title": "Factors Affecting Crop Yield",
                "content": "Crop yield is influenced by multiple factors including weather conditions, soil quality, water availability, and farming practices. Temperature and rainfall are critical factors that directly impact plant growth. Proper fertilization and irrigation can significantly improve yields."
            },
            {
                "title": "Soil Requirements for High Yield",
                "content": "Soil fertility is fundamental to achieving high crop yields. Key soil properties include pH level, organic matter content, nutrient availability (nitrogen, phosphorus, potassium), and proper drainage. Soil testing should be conducted regularly to monitor these factors and adjust fertilization programs accordingly."
            },
            {
                "title": "Water Management for Crop Production",
                "content": "Water is essential for all plant processes. Both water deficiency and excess can reduce crop yields. Irrigation scheduling should be based on crop water requirements, soil moisture levels, and weather forecasts. Drip irrigation and other water-efficient technologies can optimize water use while maintaining high yields."
            },
            {
                "title": "Fertilization and Crop Nutrition",
                "content": "Plants require adequate nutrition to achieve their yield potential. Macronutrients (nitrogen, phosphorus, potassium) and micronutrients (iron, zinc, manganese) must be supplied in appropriate amounts. Soil testing and plant tissue analysis guide fertilization decisions. Split applications of nutrients can improve uptake efficiency and reduce losses."
            },
            {
                "title": "Pest and Disease Management",
                "content": "Pests and diseases can cause significant yield losses if not properly managed. Integrated pest management (IPM) combines preventive, cultural, biological, and chemical control methods. Regular scouting and early detection are crucial for effective control. Resistant varieties and proper timing of control measures maximize effectiveness while minimizing environmental impact."
            }
        ]
        
        # Add general knowledge documents
        for doc in general_knowledge:
            document = Document(
                page_content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "type": "general_agriculture"
                }
            )
            documents.append(document)
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        try:
            if self.embedding_model:
                embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
                self.vector_store = FAISS.from_documents(split_documents, embeddings)
            else:
                # Create a simple fallback vector store with basic embeddings
                print("Using fallback embedding method")
                self.vector_store = self._create_fallback_vector_store(split_documents)
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            # Create a simple fallback vector store
            self.vector_store = self._create_fallback_vector_store(split_documents)
        
        print(f"Loaded {len(split_documents)} document chunks into the yield prediction knowledge base")
    
    def _create_sample_documents(self):
        """
        Create sample documents when dataset is not available
        
        Returns:
            list: List of sample Document objects
        """
        sample_docs = [
            {
                "title": "Cotton Yield Prediction Knowledge",
                "content": "Cotton typically requires 800-1200mm of rainfall and temperatures between 20-30°C for optimal growth. Fertilizer usage increases yield by approximately 25-30% when properly applied. Irrigation can supplement rainfall to maintain consistent moisture levels. Expected yield ranges from 2-4 tons/hectare under good conditions."
            },
            {
                "title": "Rice Yield Prediction Knowledge",
                "content": "Rice cultivation requires substantial water, typically 1200-1800mm of rainfall or equivalent irrigation. Temperature requirements range from 20-35°C. Fertilizer application is crucial for high yields, with nitrogen being particularly important. Expected yield ranges from 4-8 tons/hectare depending on variety and conditions."
            },
            {
                "title": "Wheat Yield Prediction Knowledge",
                "content": "Wheat grows best in temperate climates with 300-600mm of rainfall during the growing season. Optimal temperatures range from 15-25°C. Proper fertilization with nitrogen, phosphorus, and potassium is essential for high yields. Expected yield ranges from 3-6 tons/hectare depending on variety and management practices."
            }
        ]
        
        documents = []
        for doc in sample_docs:
            document = Document(
                page_content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "type": "yield_prediction"
                }
            )
            documents.append(document)
        
        return documents
    
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
            # This is a very basic implementation that just stores documents
            # without proper embeddings, but allows the system to function
            class SimpleVectorStore:
                def __init__(self, documents):
                    self.documents = documents
                
                def similarity_search_with_score(self, query, k=3):
                    # Return the first k documents as a simple fallback
                    return [(doc, 0.5) for doc in self.documents[:k]]
            
            return SimpleVectorStore(documents)
    
    def add_yield_prediction_record(self, prediction_data, actual_yield=None):
        """
        Add a new yield prediction record to the knowledge base
        
        Args:
            prediction_data: Dictionary with prediction input data
            actual_yield: Actual yield if known (for learning purposes)
        """
        # Create a document from the prediction
        content = f"""
        Yield Prediction Record:
        - Region: {prediction_data.get('Region', 'Unknown')}
        - Soil Type: {prediction_data.get('Soil_Type', 'Unknown')}
        - Crop: {prediction_data.get('Crop', 'Unknown')}
        - Rainfall: {prediction_data.get('Rainfall_mm', 'Unknown')} mm
        - Temperature: {prediction_data.get('Temperature_Celsius', 'Unknown')}°C
        - Fertilizer Used: {prediction_data.get('Fertilizer_Used', 'Unknown')}
        - Irrigation Used: {prediction_data.get('Irrigation_Used', 'Unknown')}
        - Weather Condition: {prediction_data.get('Weather_Condition', 'Unknown')}
        - Days to Harvest: {prediction_data.get('Days_to_Harvest', 'Unknown')}
        """
        
        if actual_yield:
            content += f"- Actual Yield: {actual_yield} tons/hectare\n"
        
        document = Document(
            page_content=content,
            metadata={
                "title": "Yield Prediction Record",
                "type": "prediction_record",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )
        
        # Split document into chunks
        split_documents = self.text_splitter.split_documents([document])
        
        # Add to vector store
        if self.vector_store is not None:
            try:
                if hasattr(self.vector_store, 'add_documents'):
                    self.vector_store.add_documents(split_documents)
                    print(f"Added yield prediction record with {len(split_documents)} chunks to the knowledge base")
                else:
                    # For fallback vector store, just add to documents list
                    if hasattr(self.vector_store, 'documents'):
                        self.vector_store.documents.extend(split_documents)
                        print(f"Added yield prediction record with {len(split_documents)} chunks to the fallback knowledge base")
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
                embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
                self.vector_store = FAISS.load_local(path, embeddings)
                print(f"Knowledge base loaded from {path}")
            except Exception as e:
                print(f"Error loading knowledge base: {str(e)}")
        else:
            print(f"Knowledge base not found at {path}")

# Example usage
if __name__ == "__main__":
    # Initialize yield prediction RAG system
    rag_system = YieldPredictionRAG()
    
    # Example query
    query = "What factors affect cotton yield?"
    results = rag_system.retrieve_relevant_documents(query, k=2)
    
    print(f"Query: {query}")
    print("Retrieved documents:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
        print(f"   Content: {result['content'][:200]}...")