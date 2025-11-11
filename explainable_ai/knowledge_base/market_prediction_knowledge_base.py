"""
Enhanced Market Prediction Knowledge Base

This module implements a specialized knowledge base for market prediction
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

class EnhancedMarketPredictionKB:
    """Enhanced knowledge base for market prediction information"""
    
    def __init__(self, knowledge_base_path='explainable_ai/knowledge_base/market_prediction', model_name='all-MiniLM-L6-v2'):
        """
        Initialize the enhanced market prediction knowledge base
        
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
        Initialize the knowledge base with default market prediction information
        """
        print("Initializing market prediction knowledge base with default content...")
        
        # Market prediction knowledge documents
        market_documents = [
            {
                "title": "Market Price Factors",
                "content": "Agricultural commodity prices are influenced by multiple factors including supply and demand dynamics, weather conditions, global economic trends, government policies, and geopolitical events. Supply factors include crop yields, planting acreage, and inventory levels. Demand factors encompass population growth, dietary changes, industrial usage, and export/import patterns. Weather conditions significantly impact both supply through crop production and demand through consumption patterns."
            },
            {
                "title": "Supply and Demand Analysis",
                "content": "Supply and demand analysis is fundamental to market price prediction. Supply factors include planted acreage, yield expectations, carryover stocks, and production costs. Demand factors encompass domestic consumption, export demand, industrial usage, and substitution effects. The balance between supply and demand determines price direction. When supply exceeds demand, prices tend to fall. When demand exceeds supply, prices tend to rise. Understanding these dynamics helps predict price movements."
            },
            {
                "title": "Weather Impact on Markets",
                "content": "Weather conditions significantly influence agricultural markets through their impact on crop production. Drought conditions reduce yields and increase prices. Excessive rainfall can damage crops and delay harvests. Temperature extremes affect crop development and quality. Seasonal weather patterns influence planting and harvesting decisions. Long-term climate trends affect regional production capabilities. Weather forecasts are crucial for short-term price predictions. Historical weather data helps understand production variability."
            },
            {
                "title": "Global Market Trends",
                "content": "Global agricultural markets are interconnected through trade flows and price relationships. Exchange rates affect export competitiveness and import costs. Economic growth rates influence demand for agricultural products. Energy prices impact production costs and biofuel demand. Trade policies including tariffs and quotas affect market access. Geopolitical events can disrupt supply chains and trade flows. Global stock levels provide insight into supply-demand balance. Monitoring these factors helps predict market direction."
            },
            {
                "title": "Technical Analysis Principles",
                "content": "Technical analysis examines price movements and trading patterns to predict future prices. Support and resistance levels indicate potential price reversal points. Moving averages smooth price data to identify trends. Chart patterns such as head and shoulders or triangles suggest future price movements. Volume analysis confirms price movements. Momentum indicators measure price change velocity. Trend lines connect price highs or lows to show direction. Combining multiple technical tools improves prediction accuracy."
            },
            {
                "title": "Fundamental Analysis Methods",
                "content": "Fundamental analysis evaluates economic factors that influence market prices. Production forecasts based on acreage and yield estimates determine supply. Consumption projections based on population and income growth determine demand. Inventory levels show supply-demand balance. Cost of production affects price support levels. Government policies including subsidies and trade measures impact markets. Financial factors such as interest rates and inflation affect commodity investments. Seasonal patterns reflect regular supply-demand cycles."
            },
            {
                "title": "Risk Management Strategies",
                "content": "Effective risk management protects against adverse price movements. Futures contracts allow price locking for future delivery. Options provide price protection while maintaining upside potential. Forward contracts with local elevators secure prices. Crop insurance protects against production losses. Diversification across crops and markets reduces concentration risk. Storage strategies take advantage of seasonal price patterns. Marketing plans spread sales across time periods. Financial planning ensures adequate working capital."
            },
            {
                "title": "Market Timing Considerations",
                "content": "Market timing involves selling at favorable price levels. Seasonal patterns reflect regular supply-demand relationships. Harvest periods typically experience lower prices due to increased supply. Planting seasons may see higher prices due to usage and carryout concerns. Weather events can create temporary price volatility. News events cause immediate market reactions. Technical indicators help identify entry and exit points. Combining fundamental and technical analysis improves timing decisions."
            },
            {
                "title": "Price Volatility Factors",
                "content": "Price volatility reflects uncertainty in supply-demand relationships. Weather uncertainty increases during growing seasons. Production estimates are revised as growing conditions change. Global economic uncertainty affects demand expectations. Policy changes create market uncertainty. Speculative activity increases short-term volatility. Thin markets with limited participants show greater price swings. Information flow affects price discovery. Understanding volatility helps manage price risk."
            },
            {
                "title": "Data Sources for Prediction",
                "content": "Reliable data sources are essential for accurate market predictions. Government reports provide official production and inventory estimates. Weather services offer forecasts and historical data. Trade organizations track commercial activity. Financial markets reflect investor sentiment. News services provide current events affecting markets. Satellite imagery monitors crop conditions globally. Economic indicators show demand trends. Combining multiple data sources improves prediction accuracy."
            }
        ]
        
        # Create Document objects
        documents = []
        for doc in market_documents:
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
    
    def add_market_information(self, title, content, metadata=None):
        """
        Add new market information to the knowledge base
        
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
        
        print(f"Added {len(split_documents)} chunks to market prediction knowledge base")
    
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
        vector_store_path = os.path.join(base_path, 'market_prediction_faiss')
        self.vector_store.save_local(vector_store_path)
        print(f"Market prediction knowledge base saved to {vector_store_path}")
    
    def load_knowledge_base_from_disk(self, base_path=None):
        """
        Load the knowledge base from disk
        
        Args:
            base_path: Base path to load knowledge base from (default: knowledge_base_path)
        """
        if base_path is None:
            base_path = self.knowledge_base_path
            
        # Check if vector store exists
        vector_store_path = os.path.join(base_path, 'market_prediction_faiss')
        if os.path.exists(vector_store_path):
            try:
                self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
                print(f"Market prediction knowledge base loaded from {vector_store_path}")
            except Exception as e:
                print(f"Error loading market prediction knowledge base: {str(e)}")
                self.vector_store = None
        else:
            print("No existing market prediction knowledge base found")
            self.vector_store = None

# Example usage
if __name__ == "__main__":
    # Initialize market prediction knowledge base
    market_kb = EnhancedMarketPredictionKB()
    
    # Example query
    query = "How does weather affect crop prices?"
    results = market_kb.retrieve_relevant_documents(query, k=2)
    
    print(f"\nQuery: {query}")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
        print(f"     Content: {result['content'][:150]}...")