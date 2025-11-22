"""
Unified RAG System for Smart Agriculture Project

This module implements a unified Retrieval-Augmented Generation system
that combines multiple agricultural knowledge domains.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Handle all imports with proper fallbacks
try:
    from langchain_text_splitters.character import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain_community.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            # Fallback if neither is available
            RecursiveCharacterTextSplitter = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain.vectorstores import FAISS
    except ImportError:
        # Fallback if neither is available
        FAISS = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            # Fallback if none are available
            HuggingFaceEmbeddings = None

# Import Document class
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        try:
            from langchain.schema import Document
        except ImportError:
            # Fallback Document class
            class Document:
                def __init__(self, page_content, metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}

# Import SentenceTransformer for fallback
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Import the unified knowledge base
from explainable_ai.knowledge_base.unified_knowledge_base import UnifiedAgriculturalKB

class AgricultureRAG:
    """RAG system for agricultural knowledge retrieval"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', knowledge_base_path='explainable_ai/knowledge_base'):
        """
        Initialize the RAG system
        
        Args:
            model_name: Name of the sentence transformer model for embeddings
            knowledge_base_path: Path to the knowledge base directory
        """
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize unified agricultural knowledge base
        try:
            self.unified_kb = UnifiedAgriculturalKB(knowledge_base_path)
            print("Using enhanced unified agricultural knowledge base")
        except Exception as e:
            print(f"Error initializing unified knowledge base: {str(e)}")
            print("Falling back to basic knowledge base")
            self.unified_kb = None
            self._init_basic_knowledge_base()
    
    def _init_basic_knowledge_base(self):
        """
        Initialize basic knowledge base as fallback
        """
        # Initialize sentence transformer model for embeddings
        self.embedding_model = SentenceTransformer(self.model_name)
        
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
        Load agricultural knowledge documents into the RAG system
        """
        # Only load if we're using the basic knowledge base
        if self.unified_kb is not None:
            return
            
        documents = []
        
        # Load comprehensive agricultural knowledge documents
        sample_documents = [
            {
                "title": "Crop Rotation Benefits",
                "content": "Crop rotation is a farming practice where different crops are grown in the same field across different seasons or years. This practice helps maintain soil fertility, reduces pest and disease buildup, and can improve overall crop yields. Common rotation patterns include legumes followed by cereals, which helps fix nitrogen in the soil. Benefits include breaking pest and disease cycles, improving soil structure, and optimizing nutrient use."
            },
            {
                "title": "Soil Health Management",
                "content": "Maintaining soil health is crucial for sustainable agriculture. Key practices include adding organic matter through composting, cover cropping during off-seasons, and minimizing soil disturbance through reduced tillage. Soil testing should be conducted regularly to monitor pH levels, nutrient content, and organic matter. Healthy soil supports better water retention, root development, and nutrient uptake."
            },
            {
                "title": "Irrigation Best Practices",
                "content": "Efficient irrigation is essential for optimal crop growth while conserving water resources. Drip irrigation systems deliver water directly to plant roots, reducing evaporation and runoff. Timing irrigation during early morning or late evening minimizes water loss. Monitoring soil moisture levels helps determine when irrigation is needed. Different crops have varying water requirements throughout their growth stages."
            },
            {
                "title": "Pest Management Strategies",
                "content": "Integrated Pest Management (IPM) combines biological, cultural, physical, and chemical tools to manage pests effectively. Strategies include crop rotation, beneficial insect habitats, pheromone traps, and selective pesticide use. Regular scouting and identification of pests are essential for timely intervention. Prevention is more effective and economical than treatment after infestation occurs."
            },
            {
                "title": "Fertilizer Application Guidelines",
                "content": "Proper fertilizer application requires understanding crop nutrient needs and soil nutrient levels. Soil testing guides fertilizer selection and application rates. Split applications of nitrogen during the growing season can improve uptake efficiency. Organic fertilizers release nutrients slowly and improve soil structure. Balanced fertilization prevents nutrient deficiencies and excesses that can harm crops."
            },
            {
                "title": "Weather Impact on Crops",
                "content": "Weather conditions significantly affect crop growth and development. Temperature influences germination, flowering, and fruit development. Adequate rainfall or irrigation is essential, but excessive moisture can lead to root diseases. Wind can cause physical damage and increase evapotranspiration. Frost can damage or kill crops, especially during sensitive growth stages. Monitoring weather forecasts helps farmers prepare protective measures."
            },
            {
                "title": "Harvest Timing and Storage",
                "content": "Optimal harvest timing maximizes yield and quality. Grain crops should be harvested when moisture content is appropriate for storage to prevent spoilage. Fruits and vegetables should be harvested at peak ripeness for best quality. Proper storage conditions, including temperature and humidity control, extend shelf life and maintain nutritional value. Post-harvest handling affects marketability and reduces losses."
            },
            {
                "title": "Plant Disease Identification",
                "content": "Early identification of plant diseases is crucial for effective management. Common symptoms include leaf spots, wilting, discoloration, stunted growth, and unusual formations. Fungal diseases often show circular spots with different colored centers. Bacterial infections may cause water-soaked lesions. Viral diseases typically result in mottled or distorted growth. Accurate diagnosis guides appropriate treatment strategies."
            },
            {
                "title": "Weed Control Methods",
                "content": "Effective weed control requires understanding weed life cycles and growth patterns. Methods include mechanical cultivation, mulching, herbicide application, and crop competition. Pre-emergent herbicides prevent weed seed germination. Post-emergent herbicides target actively growing weeds. Cultural practices like proper planting density and timing can suppress weed growth. Regular monitoring helps identify weed problems before they become severe."
            },
            {
                "title": "Crop Selection for Climate",
                "content": "Selecting appropriate crops for local climate conditions improves success rates. Consider factors like growing season length, temperature ranges, rainfall patterns, and frost dates. Drought-tolerant varieties perform better in dry regions. Cool-season crops thrive in moderate temperatures. Heat-tolerant varieties are essential in hot climates. Matching crops to climate reduces risk and improves yields."
            }
        ]
        
        # Create Document objects
        for doc in sample_documents:
            document = Document(
                page_content=doc["content"],
                metadata={"title": doc["title"]}
            )
            documents.append(document)
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vector_store = FAISS.from_documents(split_documents, embeddings)
        
        print(f"Loaded {len(split_documents)} document chunks into the knowledge base")
    
    def add_document(self, title, content):
        """
        Add a new document to the knowledge base
        
        Args:
            title: Title of the document
            content: Content of the document
        """
        # If we're using the unified knowledge base, add to the appropriate domain
        if self.unified_kb is not None:
            # Determine which domain this document belongs to based on content
            if any(keyword in title.lower() or keyword in content.lower() for keyword in ['yield', 'harvest', 'production']):
                # Add to yield knowledge base
                self.unified_kb.add_domain_specific_information(
                    'yield',
                    prediction_data={'Crop': title, 'Notes': content[:100]},
                    actual_yield=None
                )
            elif any(keyword in title.lower() or keyword in content.lower() for keyword in ['disease', 'blight', 'rust', 'spot']):
                # Add to disease knowledge base
                self.unified_kb.add_domain_specific_information(
                    'disease',
                    disease_name=title,
                    symptoms=content[:200],
                    management_strategies=content[200:]
                )
            elif any(keyword in title.lower() or keyword in content.lower() for keyword in ['crop', 'fertilizer', 'planting']):
                # Add to recommendation knowledge base
                self.unified_kb.add_domain_specific_information(
                    'recommendation',
                    crop_name=title,
                    soil_conditions="General conditions",
                    weather_conditions="General conditions",
                    fertilizer_recommendation=content[:200]
                )
            elif any(keyword in title.lower() or keyword in content.lower() for keyword in ['pest', 'weed', 'insect', 'bug']):
                # Add to pest/weed knowledge base
                self.unified_kb.add_domain_specific_information(
                    'pest_weed',
                    category_name=title,
                    description=content[:200],
                    management_strategies=content[200:]
                )
            else:
                # Add to general knowledge - we'll add to yield KB as default
                self.unified_kb.add_domain_specific_information(
                    'yield',
                    prediction_data={'Topic': title, 'Information': content[:100]},
                    actual_yield=None
                )
            return
        
        # Create Document object
        document = Document(
            page_content=content,
            metadata={"title": title}
        )
        
        # Split document into chunks
        split_documents = self.text_splitter.split_documents([document])
        
        # Add to vector store
        self.vector_store.add_documents(split_documents)
        
        print(f"Added document '{title}' with {len(split_documents)} chunks to the knowledge base")
    
    def retrieve_relevant_documents(self, query, k=5):
        """
        Retrieve relevant documents based on a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            list: List of relevant documents with similarity scores
        """
        try:
            # Use unified knowledge base if available
            if self.unified_kb is not None:
                results = self.unified_kb.query_knowledge_base(query, k=k)
                return results
            
            # Fallback to basic knowledge base
            if self.vector_store is None:
                print("Knowledge base not loaded")
                return []
            
            # Search for similar documents (fetch more candidates to allow reranking)
            fetch_k = max(k * 3, k + 5)
            try:
                docs = self.vector_store.similarity_search_with_score(query, k=fetch_k)
            except Exception as e:
                print(f"Vector store similarity search error: {e}")
                return []

            # If we have no embedding model available for reranking, just return the raw results
            try:
                query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
                # normalize
                q_norm = np.linalg.norm(query_emb)
                if q_norm > 0:
                    query_emb = query_emb / q_norm
                reranked = []
                for doc, score in docs:
                    text = doc.page_content or ''
                    # Compute embedding for the candidate chunk (fast for small k)
                    try:
                        doc_emb = self.embedding_model.encode([text], convert_to_numpy=True)
                        dnorm = np.linalg.norm(doc_emb)
                        if dnorm > 0:
                            doc_emb = doc_emb / dnorm
                        # cosine similarity
                        sim = float(np.dot(query_emb.reshape(-1), doc_emb.reshape(-1)))
                    except Exception:
                        sim = float(score)

                    reranked.append({
                        "content": text,
                        "metadata": doc.metadata,
                        "orig_score": float(score),
                        "rerank_score": sim
                    })

                # Sort by rerank_score descending
                reranked.sort(key=lambda x: x.get('rerank_score', x.get('orig_score', 0)), reverse=True)
                return reranked[:k]
            except Exception as e:
                # If reranking fails, fall back to original scores
                print(f"Reranking failed: {e}")
                results = []
                for doc, score in docs[:k]:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": float(score)
                    })
                return results
        except Exception as e:
            print(f"Error in retrieve_relevant_documents: {str(e)}")
            return []

    def query_knowledge_base(self, query, domain=None, k=5):
        """
        Query the knowledge base with domain-specific filtering
        
        Args:
            query: Query string
            domain: Specific domain to search (yield, disease, recommendation, pest_weed, or None for all)
            k: Number of documents to retrieve
            
        Returns:
            list: List of relevant documents with similarity scores
        """
        # Use unified knowledge base if available
        if self.unified_kb is not None:
            return self.unified_kb.query_knowledge_base(query, domain=domain, k=k)
        
        # Fallback to basic knowledge base
        return self.retrieve_relevant_documents(query, k=k)

    def save_knowledge_base(self, path=None):
        """
        Save the knowledge base to disk
        
        Args:
            path: Path to save the knowledge base (default: knowledge_base_path)
        """
        # If we're using the unified knowledge base, save all domains
        if self.unified_kb is not None:
            self.unified_kb.save_all_knowledge_bases(path)
            return
            
        if path is None:
            path = self.knowledge_base_path
            
        if self.vector_store is not None:
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            print(f"Knowledge base saved to {path}")
    
    def load_knowledge_base_from_disk(self, path=None):
        """
        Load the knowledge base from disk
        
        Args:
            path: Path to load the knowledge base from (default: knowledge_base_path)
        """
        # If we're using the unified knowledge base, load all domains
        if self.unified_kb is not None:
            self.unified_kb.load_all_knowledge_bases(path)
            return
            
        if path is None:
            path = self.knowledge_base_path
            
        if os.path.exists(path):
            embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            self.vector_store = FAISS.load_local(path, embeddings)
            print(f"Knowledge base loaded from {path}")
        else:
            print(f"Knowledge base not found at {path}")

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag_system = AgricultureRAG()
    
    # Example queries
    queries = [
        "How does weather affect crop growth?",
        "What are the symptoms of plant disease?",
        "How to control weeds in farming?",
        "Best practices for soil health management"
    ]
    
    print("=== Testing RAG System ===")
    for query in queries:
        print(f"\nQuery: {query}")
        results = rag_system.retrieve_relevant_documents(query, k=2)
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
            print(f"   Content: {result['content'][:200]}...")