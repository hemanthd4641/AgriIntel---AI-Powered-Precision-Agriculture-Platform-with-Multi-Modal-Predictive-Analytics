"""
Enhanced Pest Prediction Knowledge Base

This module implements a specialized knowledge base for pest prediction
using FAISS vector database for retrieval-augmented generation.
"""

import os
import json
import numpy as np
import pandas as pd
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


class EnhancedPestPredictionKB:
    """Enhanced knowledge base for pest prediction information"""
    
    def __init__(self, knowledge_base_path='explainable_ai/knowledge_base/pest_prediction', model_name='all-MiniLM-L6-v2'):
        """
        Initialize the enhanced pest prediction knowledge base
        
        Args:
            knowledge_base_path: Path to store knowledge base files
            model_name: Name of the sentence transformer model for embeddings
        """
        self.knowledge_base_path = knowledge_base_path
        self.model_name = model_name
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path)
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
        
        # Load knowledge base from disk or create new one
        self.load_knowledge_base_from_disk()
        
        # If no knowledge base loaded, initialize with default content
        if self.vector_store is None:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """
        Initialize the knowledge base with default pest prediction information
        """
        print("Initializing pest prediction knowledge base with default content...")
        
        # Pest prediction knowledge documents
        pest_documents = [
            {
                "title": "Pest Identification and Management",
                "content": "Accurate pest identification is the first step in effective pest management. Different pests require different control strategies. Visual inspection of plants, monitoring pest populations, and understanding pest life cycles are essential for successful management. Integrated Pest Management (IPM) combines biological, cultural, physical, and chemical control methods for sustainable pest control. Early detection and intervention are key to preventing crop damage and yield losses."
            },
            {
                "title": "Common Agricultural Pests",
                "content": "Aphids are small, soft-bodied insects that feed on plant sap and can transmit plant viruses. Armyworms are caterpillars that can defoliate crops rapidly. Boll weevils attack cotton plants by feeding on and laying eggs in cotton buds and flowers. Corn borers tunnel into corn stalks and ears, reducing yield and quality. Cutworms sever young plants at the soil surface. Flea beetles create small holes in leaves and can transmit bacterial diseases. Grasshoppers consume large amounts of plant material and can cause extensive crop damage. Hessian fly larvae feed on wheat and other grasses, stunting plant growth."
            },
            {
                "title": "Environmental Factors Affecting Pest Populations",
                "content": "Temperature, humidity, rainfall, and wind all influence pest development, reproduction, and survival. Warm temperatures generally accelerate pest life cycles and increase reproduction rates. High humidity favors fungal diseases and some pests. Rainfall can wash away pesticides and create conditions favorable for certain pests. Wind can disperse pests and their natural enemies. Understanding these relationships helps predict pest outbreaks and optimize control timing."
            },
            {
                "title": "Soil Conditions and Pest Dynamics",
                "content": "Soil type, moisture, pH, and nutrient levels affect both crop health and pest populations. Sandy soils may favor certain root pests while clay soils may harbor others. Soil moisture levels influence the survival of soil-dwelling pests and their natural enemies. Soil pH affects the availability of nutrients and can influence plant resistance to pests. Organic matter content affects soil structure and the habitat for beneficial organisms that control pests."
            },
            {
                "title": "Crop Rotation and Pest Management",
                "content": "Crop rotation is a fundamental pest management strategy that breaks pest life cycles and reduces pest populations. Different crops have different pest complexes, so rotating crops can reduce the buildup of specific pests. Legumes in rotation can improve soil nitrogen and support beneficial insects. Non-host crops in rotation can starve specialized pests. Planning rotations requires understanding pest-host relationships and crop growth characteristics."
            },
            {
                "title": "Biological Control Methods",
                "content": "Biological control uses natural enemies to manage pest populations. Predators like ladybugs, lacewings, and spiders consume pest insects. Parasitoids like wasps lay eggs in or on pest insects, killing them when the larvae emerge. Pathogens including bacteria, fungi, and viruses can control pest populations. Conservation of natural enemies through habitat management enhances biological control. Augmentative release of beneficial insects can supplement natural populations."
            },
            {
                "title": "Cultural Control Practices",
                "content": "Cultural practices modify the growing environment to reduce pest problems. Sanitation removes pest habitats and food sources. Proper planting dates can avoid peak pest activity periods. Plant spacing affects air circulation and pest movement. Tillage can destroy pest overwintering sites and reduce populations. Trap crops attract pests away from main crops. Companion planting can repel pests or attract beneficial insects."
            },
            {
                "title": "Chemical Control Considerations",
                "content": "Chemical control should be used judiciously as part of an integrated approach. Selective pesticides target specific pests while preserving beneficial insects. Broad-spectrum pesticides can disrupt natural enemy populations and lead to pest resurgence. Resistance management requires rotating pesticide classes and using appropriate rates. Timing applications to pest life stages maximizes effectiveness and minimizes environmental impact. Following label instructions ensures safety and efficacy."
            },
            {
                "title": "Monitoring and Thresholds",
                "content": "Regular monitoring detects pest presence and population levels before economic damage occurs. Scouting methods include visual inspection, sticky traps, pheromone traps, and sweep nets. Economic thresholds determine when control actions are justified based on pest density and crop value. Action thresholds trigger control measures to prevent pest populations from reaching economic levels. Record keeping tracks pest populations, control measures, and their effectiveness."
            },
            {
                "title": "Pest Prediction Models",
                "content": "Pest prediction models use weather data, crop development stages, and historical pest patterns to forecast pest activity. Degree-day models predict pest development based on temperature accumulation. Moisture models predict conditions favorable for pest development or disease infection. Phenological models link pest activity to crop growth stages. Combining multiple models improves prediction accuracy. Real-time data collection and model updating enhance predictive capability."
            }
        ]
        
        # Create Document objects
        documents = []
        for doc in pest_documents:
            document = Document(
                page_content=doc["content"],
                metadata={"title": doc["title"]}
            )
            documents.append(document)
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(split_documents, self.embedding_model)
        print(f"Initialized knowledge base with {len(split_documents)} document chunks")
    
    def add_pest_information(self, title, content, metadata=None):
        """
        Add new pest information to the knowledge base
        
        Args:
            title: Title of the information
            content: Content of the information
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}
        metadata["title"] = title
        
        # Create Document object
        document = Document(
            page_content=content,
            metadata=metadata
        )
        
        # Split document into chunks
        split_documents = self.text_splitter.split_documents([document])
        
        # Add to vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(split_documents, self.embedding_model)
        else:
            self.vector_store.add_documents(split_documents)
        
        print(f"Added {len(split_documents)} chunks to pest prediction knowledge base")
    
    def retrieve_relevant_documents(self, query, k=3):
        """
        Retrieve relevant documents from the knowledge base
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            list: List of relevant documents with similarity scores
        """
        if self.vector_store is None:
            return []
        
        # Search for relevant documents
        search_results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        results = []
        for doc, score in search_results:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity': float(score)
            })
        
        return results
    
    def save_knowledge_base(self, base_path=None):
        """
        Save the knowledge base to disk
        
        Args:
            base_path: Base path to save knowledge base (default: knowledge_base_path)
        """
        if base_path is None:
            base_path = self.knowledge_base_path
            
        if self.vector_store is None:
            print("No knowledge base to save")
            return
            
        # Create directory if it doesn't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        # Save vector store
        vector_store_path = os.path.join(base_path, 'pest_prediction_faiss')
        self.vector_store.save_local(vector_store_path)
        print(f"Pest prediction knowledge base saved to {vector_store_path}")
    
    def load_knowledge_base_from_disk(self, base_path=None):
        """
        Load the knowledge base from disk
        
        Args:
            base_path: Base path to load knowledge base from (default: knowledge_base_path)
        """
        if base_path is None:
            base_path = self.knowledge_base_path
            
        # Check if vector store exists
        vector_store_path = os.path.join(base_path, 'pest_prediction_faiss')
        if os.path.exists(vector_store_path):
            try:
                self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
                print(f"Pest prediction knowledge base loaded from {vector_store_path}")
            except Exception as e:
                print(f"Error loading pest prediction knowledge base: {str(e)}")
                self.vector_store = None
        else:
            print("No existing pest prediction knowledge base found")
            self.vector_store = None

# Example usage
if __name__ == "__main__":
    # Initialize pest prediction knowledge base
    pest_kb = EnhancedPestPredictionKB()
    
    # Example query
    query = "How to identify and manage aphid infestations?"
    results = pest_kb.retrieve_relevant_documents(query, k=2)
    
    print(f"\nQuery: {query}")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
        print(f"     Content: {result['content'][:150]}...")