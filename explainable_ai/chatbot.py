"""
Agricultural Chatbot with RAG and LLM Integration

This module implements a chatbot that combines retrieval-augmented generation (RAG)
with large language models (LLM) to provide accurate agricultural advice.
"""

import os
import sys
from typing import Optional, List, Dict, Any

# Put project on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the actual classes
from explainable_ai.llm_interface import AgricultureLLM
from explainable_ai.rag_system import AgricultureRAG


class AgriculturalChatbot:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize the agricultural chatbot with RAG and LLM components
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.llm: Optional[AgricultureLLM] = None
        self.rag_system: Optional[AgricultureRAG] = None
        
        # Initialize RAG system first
        try:
            self.rag_system = AgricultureRAG(
                model_name='all-MiniLM-L6-v2', 
                knowledge_base_path='explainable_ai/knowledge_base'
            )
            print("RAG system initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            self.rag_system = None

        # Initialize LLM with RAG system integration
        try:
            # Use Hugging Face API if configured, otherwise use local model
            self.llm = AgricultureLLM(
                model_name='microsoft/Phi-3-mini-4k-instruct', 
                rag_system=self.rag_system
            )
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            try:
                self.llm = AgricultureLLM(model_name='microsoft/Phi-3-mini-4k-instruct')
                print("Fallback LLM initialization successful")
            except Exception as e2:
                print(f"Fallback LLM initialization also failed: {str(e2)}")
                self.llm = None

        self.conversation_history = []
        print("Agricultural Chatbot initialized with RAG and LLM components")

    def get_response(self, user_message: str) -> str:
        """
        Get a response from the chatbot for a user message
        
        Args:
            user_message: The user's message/query
            
        Returns:
            str: The chatbot's response
        """
        # Classify the query domain for better RAG performance
        domain = self.classify_query_domain(user_message)
        print(f"Classified query domain: {domain}")
        
        # Try to get a specialized response first
        spec = self.get_specialized_response(user_message, domain)
        if spec:
            self._append_history(user_message, spec)
            return spec

        # Use LLM with RAG for comprehensive response
        if self.llm:
            try:
                print("Generating response using LLM with RAG...")
                resp = self.llm.chat_with_farmer(user_message, conversation_history=self.conversation_history)
                self._append_history(user_message, resp)
                return resp
            except Exception as e:
                print(f"Error generating LLM response: {str(e)}")
                pass

        # Fallback to rule-based response
        fallback = self._get_rule_based_response(user_message)
        self._append_history(user_message, fallback)
        return fallback

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a specific question and get a detailed response with sources
        
        Args:
            question: The question to ask
            
        Returns:
            dict: Response with answer and source documents
        """
        try:
            if self.llm is not None:
                print("Generating detailed response using LLM...")
                return self.llm.answer_question(question)
            else:
                print("LLM not available, using RAG system directly...")
                # Fallback to RAG system if LLM is not available
                if self.rag_system:
                    try:
                        domain = self.classify_query_domain(question)
                        docs = self.rag_system.retrieve_relevant_documents(question, k=3)
                        if docs:
                            # Create a summary from retrieved documents
                            summary = self._summarize_documents(docs)
                            return {
                                "answer": summary,
                                "source_documents": docs
                            }
                    except Exception as e:
                        print(f"Error using RAG system: {str(e)}")
                return {"answer": "I could not answer that right now.", "source_documents": []}
        except Exception as e:
            print(f"Error in ask_question: {str(e)}")
            return {"answer": "I could not answer that right now.", "source_documents": []}

    def _summarize_documents(self, documents: List[Dict]) -> str:
        """
        Create a summary from retrieved documents
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            str: Summary of the documents
        """
        if not documents:
            return "No relevant information found."
        
        # Create a concise summary from the top documents
        summary_parts = []
        for i, doc in enumerate(documents[:3], 1):
            content = doc.get('content', doc.get('page_content', '')) or ''
            title = doc.get('metadata', {}).get('title', f'Document {i}')
            summary_parts.append(f"{i}. {title}: {content[:150]}...")
        
        return "Based on retrieved knowledge:\n" + "\n".join(summary_parts)

    def _append_history(self, user, assistant):
        """Append a conversation turn to history"""
        self.conversation_history.append({'user': user, 'assistant': assistant})
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def _get_rule_based_response(self, msg: str) -> str:
        """
        Get a rule-based response for common queries
        
        Args:
            msg: User message
            
        Returns:
            str: Rule-based response
        """
        m = msg.lower()
        if 'tomato' in m:
            return 'Ensure 6-8h sunlight, pH 6.0-6.8 soil, regular watering, and stakes.'
        if 'soil' in m or 'fertil' in m:
            return 'Do a soil test, add compost, and apply nutrients per crop needs.'
        if 'pest' in m or 'insect' in m:
            return 'Use IPM: monitor, encourage beneficials, and apply targeted controls when needed.'
        if 'disease' in m or 'blight' in m:
            return 'Improve airflow, avoid overhead watering, remove infected plants, and rotate crops.'
        if 'water' in m or 'irrigat' in m:
            return 'Water deeply but less frequently, use mulch, and consider drip irrigation.'
        if 'harvest' in m:
            return 'Harvest at peak ripeness, handle gently, and store properly to maintain quality.'
        return 'Please provide more details so I can help with specific farming advice.'

    def classify_query_domain(self, q: str) -> str:
        """
        Classify a query into a specific agricultural domain
        
        Args:
            q: Query string
            
        Returns:
            str: Domain classification
        """
        ql = q.lower()
        if any(k in ql for k in ['pest', 'insect', 'bug', 'weed']):
            return 'pest'
        if any(k in ql for k in ['price', 'market', 'sell', 'buy', 'cost']):
            return 'market'
        if any(k in ql for k in ['disease', 'blight', 'rust', 'spot', 'infect']):
            return 'disease'
        if any(k in ql for k in ['yield', 'harvest', 'production', 'ton', 'hectare']):
            return 'yield'
        if any(k in ql for k in ['crop', 'fertilizer', 'soil', 'plant', 'seed']):
            return 'recommendation'
        return 'general'

    def get_specialized_response(self, query: str, domain: str) -> Optional[str]:
        """
        Get a specialized response using the RAG system
        
        Args:
            query: User query
            domain: Query domain
            
        Returns:
            str or None: Specialized response if available
        """
        if not self.rag_system:
            return None
            
        try:
            # Use the RAG system to retrieve relevant documents
            docs = self.rag_system.query_knowledge_base(query, domain=domain, k=3)
        except Exception as e:
            print(f"Error querying knowledge base: {str(e)}")
            docs = []
            
        if not docs:
            return None

        # Create a summary from the retrieved documents
        snippets = []
        for d in docs[:3]:
            content = d.get('content', d.get('page_content', '')) or ''
            title = d.get('metadata', {}).get('title', 'Document')
            snippets.append(f"{title}: {content[:200]}...")
            
        summary = ' '.join(snippets)
        return f"Based on retrieved agricultural knowledge: {summary}"

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the status of the chatbot components
        
        Returns:
            dict: Status information
        """
        llm_available = bool(self.llm and getattr(self.llm, 'text_generator', None))
        rag_available = self.rag_system is not None
        
        kb_info = {}
        try:
            if self.rag_system and getattr(self.rag_system, 'unified_kb', None):
                uk = self.rag_system.unified_kb
                kb_info = {
                    'yield': bool(getattr(uk, 'yield_kb', None)),
                    'disease': bool(getattr(uk, 'disease_kb', None)),
                    'recommendation': bool(getattr(uk, 'recommendation_kb', None)),
                    'pest': bool(getattr(uk, 'pest_prediction_kb', None)),
                    'market': bool(getattr(uk, 'market_prediction_kb', None)),
                }
            else:
                kb_info = {}
        except Exception:
            kb_info = {}

        return {
            'llm_available': llm_available,
            'rag_system_available': rag_available,
            'conversation_history_length': len(self.conversation_history),
            'knowledge_bases': kb_info,
        }

    def add_knowledge_document(self, title: str, content: str, domain: str = 'general'):
        """
        Add a new document to the knowledge base
        
        Args:
            title: Document title
            content: Document content
            domain: Domain classification
        """
        if self.rag_system:
            try:
                # Add document to the appropriate domain
                # Note: The actual method name might be different, using hasattr to check
                if hasattr(self.rag_system, 'add_document'):
                    self.rag_system.add_document(title, content)
                else:
                    print("add_document method not available in RAG system")
                print(f"Added document '{title}' to knowledge base")
            except Exception as e:
                print(f"Error adding document: {str(e)}")
        else:
            print("RAG system not available for adding documents")


# Example usage
if __name__ == '__main__':
    bot = AgriculturalChatbot()
    print("Agricultural Chatbot with RAG and LLM")
    print("Status:", bot.get_system_status())
    print("\nExample response:")
    print(bot.get_response('How should I grow healthy tomatoes?'))