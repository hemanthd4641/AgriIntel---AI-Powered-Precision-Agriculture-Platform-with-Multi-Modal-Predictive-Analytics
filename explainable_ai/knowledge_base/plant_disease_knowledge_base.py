"""
Enhanced Plant Disease Knowledge Base

This module creates a comprehensive knowledge base specifically for plant disease identification
and management using datasets and FAISS vector database with enhanced data loading capabilities.
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

class EnhancedPlantDiseaseKB:
    """Enhanced knowledge base for plant disease identification and management"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', knowledge_base_path='explainable_ai/knowledge_base/plant_disease'):
        """
        Initialize the enhanced plant disease knowledge base
        
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
        Load comprehensive plant disease knowledge from multiple sources
        """
        documents = []
        
        # Load plant disease dataset information - fix the path to be relative to the project root
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        dataset_info_path = os.path.join(project_root, "datasets", "plant_disease", "dataset_info.csv")
        
        if os.path.exists(dataset_info_path):
            try:
                print("Loading plant disease dataset information...")
                df = pd.read_csv(dataset_info_path)
                print(f"Loaded information for {len(df)} plant disease categories")
                
                # Create knowledge documents from dataset information
                for _, row in df.iterrows():
                    class_name = row['class_name']
                    description = row['description']
                    image_count = row['image_count']
                    
                    # Create knowledge document
                    content = f"""
                    Plant Disease: {class_name}
                    Description: {description}
                    Available Reference Images: {image_count}
                    
                    Key Identification Features:
                    - Visual symptoms on leaves, stems, fruits, or roots
                    - Environmental conditions that favor development
                    - Host plants commonly affected
                    - Life cycle and transmission methods
                    
                    Management Strategies:
                    - Cultural practices to prevent occurrence
                    - Biological control options
                    - Chemical treatment recommendations
                    - Timing of interventions for maximum effectiveness
                    - Safety precautions when applying treatments
                    """
                    
                    document = Document(
                        page_content=content,
                        metadata={
                            "title": f"{class_name} Disease Information",
                            "disease": class_name,
                            "type": "plant_disease"
                        }
                    )
                    documents.append(document)
                
                print(f"Loaded disease information for {len(df)} categories")
            except Exception as e:
                print(f"Error loading dataset info: {str(e)}")
        else:
            print(f"Dataset info not found at {dataset_info_path}")
        
        # Add comprehensive plant disease knowledge
        disease_knowledge = [
            {
                "title": "Apple Scab",
                "content": """Apple Scab is a fungal disease that affects apple trees, caused by Venturia inaequalis.
                Symptoms include olive-green to black spots on leaves and fruit. Severely infected leaves may drop prematurely.
                Management strategies:
                - Apply fungicides during the growing season, especially during wet periods
                - Remove and destroy infected leaves to reduce inoculum
                - Plant resistant apple varieties when possible
                - Ensure good air circulation through proper pruning
                - Apply protective fungicides before rainy periods in spring"""
            },
            {
                "title": "Black Rot",
                "content": """Black Rot is a fungal disease affecting apples, caused by Botryosphaeria obtusa.
                Characterized by dark, sunken lesions on fruit that often develop small black dots (fungal fruiting bodies).
                Management strategies:
                - Pruning infected branches during dormant season
                - Removing and destroying mummified fruit and dead bark
                - Applying fungicides during bloom and fruit development
                - Maintaining good orchard hygiene to reduce inoculum sources
                - Ensuring proper tree nutrition to improve resistance"""
            },
            {
                "title": "Cedar Apple Rust",
                "content": """Cedar Apple Rust is a disease requiring both apple and cedar trees to complete its lifecycle.
                Symptoms include yellow-orange spots on apple leaves that develop into tube-like structures.
                Management strategies:
                - Removing nearby cedar trees if possible (within 1 mile)
                - Applying fungicides during the critical infection period in spring
                - Planting resistant apple varieties
                - Monitoring cedar trees for galls and removing them in late winter
                - Proper timing of fungicide applications based on weather conditions"""
            },
            {
                "title": "Corn Common Rust",
                "content": """Corn Common Rust is a fungal disease caused by Puccinia sorghi.
                Causes reddish-brown pustules on leaves that rupture to release spores.
                Management strategies:
                - Planting resistant corn varieties
                - Crop rotation to reduce inoculum buildup
                - Fungicide application if necessary during critical growth stages
                - Early planting to avoid peak rust development periods
                - Monitoring fields regularly for early detection"""
            },
            {
                "title": "Corn Northern Leaf Blight",
                "content": """Corn Northern Leaf Blight is a fungal disease caused by Exserohilum turcicum.
                Causes long, elliptical lesions on leaves that are gray-green to tan with dark borders.
                Management strategies:
                - Crop rotation with non-host crops
                - Residue management through tillage or decomposition
                - Fungicide application in severe cases during critical growth stages
                - Planting resistant hybrids when available
                - Proper nitrogen management to maintain plant health"""
            },
            {
                "title": "Grape Black Rot",
                "content": """Grape Black Rot is a fungal disease caused by Guignardia bidwellii.
                Causes black spots on fruit and leaves, with infected berries becoming mummified.
                Management strategies:
                - Pruning to improve air circulation and reduce humidity
                - Sanitation by removing infected fruit and leaves
                - Fungicide application during bloom and fruit development
                - Proper vineyard floor management to reduce inoculum
                - Timing applications based on weather and growth stage"""
            },
            {
                "title": "Potato Early Blight",
                "content": """Potato Early Blight is a fungal disease caused by Alternaria solani.
                Causes dark spots with concentric rings on leaves, often with yellow halos.
                Management strategies:
                - Crop rotation with non-host crops for 2-3 years
                - Mulching to prevent soil-borne spores from splashing
                - Fungicide application when disease pressure is high
                - Proper irrigation management to reduce leaf wetness
                - Removing infected plant debris after harvest"""
            },
            {
                "title": "Tomato Bacterial Spot",
                "content": """Tomato Bacterial Spot is a bacterial disease caused by Xanthomonas species.
                Causes dark spots on leaves, stems, and fruit that may have yellow halos.
                Management strategies:
                - Using disease-free seeds and transplants
                - Crop rotation with non-host crops
                - Copper-based bactericides when conditions favor disease
                - Avoiding overhead irrigation and working in wet fields
                - Proper sanitation of tools and equipment"""
            },
            {
                "title": "Tomato Early Blight",
                "content": """Tomato Early Blight is a fungal disease caused by Alternaria solani.
                Causes dark spots with target-like patterns on leaves, starting with lower leaves.
                Management strategies:
                - Crop rotation with non-host crops
                - Mulching to prevent soil-borne spores from splashing
                - Fungicide application when necessary
                - Proper spacing to improve air circulation
                - Removing infected leaves and debris"""
            },
            {
                "title": "Tomato Late Blight",
                "content": """Tomato Late Blight is a serious fungal disease caused by Phytophthora infestans.
                Causes water-soaked lesions on leaves and fruit, often with white fungal growth on undersides.
                Management strategies:
                - Immediate removal of infected plants to prevent spread
                - Fungicide application as a preventive measure in high-risk periods
                - Proper spacing and staking to improve air circulation
                - Avoiding overhead irrigation
                - Monitoring weather conditions for disease-favorable periods"""
            },
            {
                "title": "Tomato Leaf Mold",
                "content": """Tomato Leaf Mold is a fungal disease caused by Passalora fulva.
                Causes yellow spots on upper leaf surfaces and white mold on lower surfaces.
                Management strategies:
                - Improving air circulation through proper spacing and pruning
                - Reducing humidity in greenhouse environments
                - Applying fungicides when conditions favor disease
                - Removing infected leaves promptly
                - Avoiding wetting foliage during irrigation"""
            },
            {
                "title": "Tomato Septoria Leaf Spot",
                "content": """Tomato Septoria Leaf Spot is a fungal disease caused by Septoria lycopersici.
                Causes small, circular spots with dark borders and light centers on leaves.
                Management strategies:
                - Crop rotation with non-host crops
                - Mulching to prevent soil-borne spores from splashing
                - Fungicide application when necessary
                - Proper spacing to improve air circulation
                - Removing infected leaves and debris"""
            },
            {
                "title": "Tomato Spider Mites",
                "content": """Tomato Spider Mites are tiny pests that cause stippling and webbing on leaves.
                Damage appears as yellow or bronze speckling on leaf surfaces.
                Management strategies:
                - Releasing beneficial insects like predatory mites
                - Applying miticides if necessary (target early infestations)
                - Maintaining adequate irrigation to reduce plant stress
                - Regular monitoring, especially during hot, dry conditions
                - Removing heavily infested plant parts"""
            }
        ]
        
        # Add disease knowledge documents
        for doc in disease_knowledge:
            document = Document(
                page_content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "disease": doc["title"],
                    "type": "plant_disease"
                }
            )
            documents.append(document)
        
        # Add general disease management knowledge
        general_knowledge = [
            {
                "title": "Integrated Disease Management",
                "content": """Integrated Disease Management (IDM) combines multiple strategies to effectively control plant diseases while minimizing environmental impact.
                Key components include:
                - Prevention through cultural practices
                - Monitoring and early detection
                - Accurate diagnosis and identification
                - Selection of appropriate control methods
                - Evaluation of management effectiveness
                
                Prevention strategies:
                - Use of disease-resistant varieties
                - Proper site selection and preparation
                - Crop rotation to break disease cycles
                - Sanitation to reduce inoculum sources
                - Proper nutrition to maintain plant health"""
            },
            {
                "title": "Disease Diagnosis and Identification",
                "content": """Accurate disease diagnosis is crucial for effective management.
                Key steps include:
                - Careful observation of symptoms on different plant parts
                - Consideration of environmental conditions
                - Examination of disease patterns in the field
                - Laboratory testing when necessary
                - Comparison with known disease descriptions
                
                Important factors to consider:
                - Time of symptom appearance
                - Weather conditions preceding symptoms
                - History of the disease in the area
                - Cultural practices that may contribute
                - Presence of insect vectors or other pests"""
            },
            {
                "title": "Chemical Disease Control",
                "content": """Chemical control involves the use of fungicides, bactericides, and other pesticides to manage plant diseases.
                Best practices include:
                - Accurate disease identification before application
                - Selection of appropriate products for the target disease
                - Proper timing based on disease life cycle and weather
                - Following label instructions for rates and intervals
                - Rotating chemical classes to prevent resistance
                
                Important considerations:
                - Pre-harvest intervals for food safety
                - Environmental impact on beneficial organisms
                - Resistance management strategies
                - Compatibility with other pesticides
                - Worker safety during application"""
            }
        ]
        
        # Add general knowledge documents
        for doc in general_knowledge:
            document = Document(
                page_content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "type": "general_disease_management"
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
        
        print(f"Loaded {len(split_documents)} document chunks into the plant disease knowledge base")
    
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
    
    def _load_disease_data(self):
        """Load plant disease information from CSV"""
        try:
            # Updated to reflect Colab training approach
            # Dataset info file is not needed when using Colab trained models
            # Disease data is loaded directly from the model predictions
            self.disease_data = {}
            print("Plant disease data will be loaded from Colab trained model predictions")
        except Exception as e:
            print(f"Warning: Could not load disease data: {e}")
            self.disease_data = {}
    
    def add_disease_information(self, disease_name, symptoms, management_strategies):
        """
        Add new disease information to the knowledge base
        
        Args:
            disease_name: Name of the disease
            symptoms: Description of disease symptoms
            management_strategies: Recommended management approaches
        """
        # Create a document from the disease information
        content = f"""
        Plant Disease: {disease_name}
        Symptoms: {symptoms}
        Management Strategies: {management_strategies}
        """
        
        document = Document(
            page_content=content,
            metadata={
                "title": f"{disease_name} Information",
                "disease": disease_name,
                "type": "plant_disease"
            }
        )
        
        # Split document into chunks
        split_documents = self.text_splitter.split_documents([document])
        
        # Add to vector store
        if self.vector_store is not None:
            try:
                if hasattr(self.vector_store, 'add_documents'):
                    self.vector_store.add_documents(split_documents)
                    print(f"Added disease information with {len(split_documents)} chunks to the knowledge base")
                else:
                    # For fallback vector store, just add to documents list
                    if hasattr(self.vector_store, 'documents'):
                        self.vector_store.documents.extend(split_documents)
                        print(f"Added disease information with {len(split_documents)} chunks to the fallback knowledge base")
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
    # Initialize plant disease knowledge base
    kb = EnhancedPlantDiseaseKB()
    
    # Example query
    query = "How to manage tomato late blight?"
    results = kb.retrieve_relevant_documents(query, k=2)
    
    print(f"Query: {query}")
    print("Retrieved documents:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
        print(f"   Content: {result['content'][:200]}...")