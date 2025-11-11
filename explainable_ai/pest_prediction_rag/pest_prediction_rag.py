"""
Pest Prediction RAG System for Smart Agriculture Project

This module implements a Retrieval-Augmented Generation system using FAISS
for providing context-aware pest prediction explanations and recommendations.
"""

import numpy as np
import pandas as pd
import faiss
import json
import os
import re
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

# Import the unified knowledge base
from explainable_ai.knowledge_base.unified_knowledge_base import UnifiedAgriculturalKB

class PestPredictionRAG:
    """RAG system for pest prediction knowledge retrieval"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', knowledge_base_path='explainable_ai/knowledge_base'):
        """
        Initialize the RAG system for pest prediction
        
        Args:
            model_name: Name of the sentence transformer model for embeddings
            knowledge_base_path: Path to the knowledge base directory
        """
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize unified agricultural knowledge base
        try:
            self.unified_kb = UnifiedAgriculturalKB(knowledge_base_path)
            print("Using enhanced unified agricultural knowledge base for pest prediction")
        except Exception as e:
            print(f"Error initializing unified knowledge base: {str(e)}")
            print("Falling back to basic pest prediction knowledge base")
            self.unified_kb = None
            self._init_basic_knowledge_base()
    
    def _init_basic_knowledge_base(self):
        """
        Initialize basic pest prediction knowledge base as fallback
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
        Load pest prediction knowledge documents into the RAG system
        """
        # Only load if we're using the basic knowledge base
        if self.unified_kb is not None:
            return
            
        documents = []
        
        # Load comprehensive pest prediction knowledge documents
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
        for doc in pest_documents:
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
        
        print(f"Loaded {len(split_documents)} pest prediction document chunks into the knowledge base")
    
    def add_document(self, title, content):
        """
        Add a new document to the pest prediction knowledge base
        
        Args:
            title: Title of the document
            content: Content of the document
        """
        # If we're using the unified knowledge base, add to the pest prediction domain
        if self.unified_kb is not None:
            self.unified_kb.add_domain_specific_information(
                'pest_prediction',
                title=title,
                content=content
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
    
    def query_knowledge_base(self, query, k=3):
        """
        Query the pest prediction knowledge base
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            list: List of relevant documents with similarity scores
        """
        # If we're using the unified knowledge base, query the pest prediction domain
        if self.unified_kb is not None:
            return self.unified_kb.query_knowledge_base(query, domain='pest_prediction', k=k)
        
        # Otherwise, use the basic knowledge base
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
    
    def generate_context_aware_response(self, query, prediction_data=None):
        """
        Generate a comprehensive context-aware response with detailed pest management strategies
        
        Args:
            query: User query
            prediction_data: Pest prediction data for context
            
        Returns:
            dict: Comprehensive pest management analysis including strategies, control methods, and prevention advice
        """
        try:
            # Retrieve relevant documents
            relevant_docs = self.query_knowledge_base(query, k=5)
            
            # Create context from retrieved documents
            context = "Relevant pest management information:\n"
            for i, doc in enumerate(relevant_docs, 1):
                content = doc.get('content', doc.get('page_content', ''))[:500]
                context += f"{i}. {content}\n"
            
            # Add prediction data to context if provided
            if prediction_data:
                context += f"\nPest Prediction Data:\n"
                for key, value in prediction_data.items():
                    context += f"- {key}: {value}\n"
            
            # Generate comprehensive pest management analysis using LLM if available
            comprehensive_analysis = self._generate_comprehensive_pest_analysis(context, prediction_data)
            
            # Extract key components
            management_strategies = self._extract_management_strategies(comprehensive_analysis, prediction_data)
            control_methods = self._extract_control_methods(comprehensive_analysis, prediction_data)
            prevention_advice = self._extract_prevention_advice(comprehensive_analysis, prediction_data)
            
            return {
                "query": query,
                "context": context,
                "comprehensive_analysis": comprehensive_analysis,
                "management_strategies": management_strategies,
                "control_methods": control_methods,
                "prevention_advice": prevention_advice,
                "related_documents": relevant_docs
            }
            
        except Exception as e:
            print(f"Error in generate_context_aware_response: {e}")
            # Fallback response
            return {
                "query": query,
                "context": "Context not available",
                "comprehensive_analysis": self._generate_fallback_pest_analysis(query, prediction_data),
                "management_strategies": self._extract_management_strategies("", prediction_data),
                "control_methods": self._extract_control_methods("", prediction_data),
                "prevention_advice": self._extract_prevention_advice("", prediction_data),
                "related_documents": []
            }
    
    def _generate_comprehensive_pest_analysis(self, context, prediction_data):
        """
        Generate comprehensive pest analysis using LLM.
        
        Args:
            context (str): Context information from knowledge base
            prediction_data (dict): Pest prediction data
            
        Returns:
            str: Comprehensive pest analysis
        """
        try:
            # Try to import LLM components
            from explainable_ai.llm_interface import AgricultureLLM
            
            # Initialize LLM
            llm = AgricultureLLM()
            
            # Create detailed prompt for comprehensive analysis
            detailed_prompt = f"""
            You are an expert agricultural pest management specialist. Based on the following context and pest data, 
            provide a comprehensive pest management analysis with detailed recommendations.
            
            Context Information:
            {context}
            
            Pest Prediction Data:
            {prediction_data}
            
            Please provide:
            1. Pest Identification: Accurate identification and life cycle information
            2. Damage Assessment: Potential crop damage and economic impact
            3. Management Strategies: Integrated approaches combining multiple methods
            4. Control Methods: Biological, cultural, physical, and chemical options
            5. Prevention Advice: Proactive measures to avoid infestations
            6. Monitoring Techniques: Methods for tracking pest populations
            7. Timing Recommendations: Optimal intervention periods
            8. Resistance Management: Strategies to prevent pesticide resistance
            9. Environmental Considerations: Minimizing impact on non-target organisms
            10. Economic Thresholds: When intervention becomes cost-effective
            
            Format your response in clear sections with actionable pest management advice.
            """
            
            # Generate analysis using LLM
            if llm.text_generator:
                try:
                    response = llm.text_generator(
                        detailed_prompt,
                        max_new_tokens=1000,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9
                    )
                    # Extract generated text
                    if isinstance(response, list) and len(response) > 0:
                        if isinstance(response[0], dict) and 'generated_text' in response[0]:
                            generated_text = response[0]['generated_text']
                            # Remove the prompt from the response
                            analysis = generated_text[len(detailed_prompt):].strip()
                            if analysis:
                                return analysis
                except Exception as e:
                    print(f"Error generating LLM analysis: {str(e)}")
            
        except ImportError as e:
            print(f"LLM components not available: {str(e)}")
        except Exception as e:
            print(f"Error in LLM analysis generation: {str(e)}")
        
        # Fallback to rule-based analysis if LLM fails
        return self._generate_fallback_pest_analysis(context, prediction_data)
    
    def _generate_fallback_pest_analysis(self, context, prediction_data):
        """
        Generate fallback pest analysis when LLM is not available.
        
        Args:
            context (str): Context information
            prediction_data (dict): Pest prediction data
            
        Returns:
            str: Fallback pest analysis
        """
        pest_name = prediction_data.get('predicted_pest', 'Unknown pest') if prediction_data else 'Unknown pest'
        severity = prediction_data.get('severity', 'Moderate') if prediction_data else 'Moderate'
        crop = prediction_data.get('crop', 'Unknown crop') if prediction_data else 'Unknown crop'
        
        analysis = f"Comprehensive Pest Management Plan for {pest_name} in {crop}\n\n"
        analysis += f"Severity Level: {severity}\n\n"
        
        # Pest Identification
        analysis += "1. PEST IDENTIFICATION:\n"
        analysis += f"   - Confirm {pest_name} through visual inspection of characteristic damage\n"
        analysis += f"   - Understand life cycle stages for optimal intervention timing\n"
        analysis += f"   - Monitor for presence of beneficial insects and natural enemies\n\n"
        
        # Damage Assessment
        analysis += "2. DAMAGE ASSESSMENT:\n"
        analysis += f"   - {pest_name} can cause {'severe' if severity.lower() == 'high' else 'moderate' if severity.lower() == 'moderate' else 'minor'} damage to {crop}\n"
        analysis += f"   - Economic impact depends on infestation level and crop stage\n"
        analysis += f"   - Early detection prevents significant yield losses\n\n"
        
        # Management Strategies
        analysis += "3. MANAGEMENT STRATEGIES:\n"
        analysis += f"   - Implement Integrated Pest Management (IPM) approach\n"
        analysis += f"   - Combine multiple control methods for sustainable management\n"
        analysis += f"   - Monitor regularly and intervene when thresholds are exceeded\n\n"
        
        # Control Methods
        analysis += "4. CONTROL METHODS:\n"
        analysis += f"   - Biological: Encourage natural enemies like ladybugs and parasitoids\n"
        analysis += f"   - Cultural: Practice crop rotation and sanitation\n"
        analysis += f"   - Physical: Use barriers and traps to exclude pests\n"
        analysis += f"   - Chemical: Apply selective pesticides when necessary\n\n"
        
        # Prevention Advice
        analysis += "5. PREVENTION ADVICE:\n"
        analysis += f"   - Select resistant varieties when available\n"
        analysis += f"   - Maintain proper plant nutrition and irrigation\n"
        analysis += f"   - Remove crop debris and weeds that harbor pests\n"
        analysis += f"   - Establish beneficial insect habitats\n\n"
        
        return analysis
    
    def _extract_management_strategies(self, analysis_text, prediction_data):
        """
        Extract management strategies from comprehensive analysis.
        
        Args:
            analysis_text (str): Comprehensive analysis text
            prediction_data (dict): Pest prediction data
            
        Returns:
            dict: Management strategies components
        """
        pest_name = prediction_data.get('predicted_pest', 'Unknown pest') if prediction_data else 'Unknown pest'
        severity = prediction_data.get('severity', 'Moderate') if prediction_data else 'Moderate'
        
        return {
            "integrated_approach": f"Use Integrated Pest Management (IPM) for {pest_name} control",
            "key_principles": [
                "Combine biological, cultural, physical, and chemical control methods",
                "Monitor pest populations regularly to detect early infestations",
                "Intervene when pest populations reach economic thresholds",
                "Preserve beneficial insects and natural enemies whenever possible"
            ],
            "severity_based_strategies": {
                "low": f"Monitor {pest_name} populations weekly and implement preventive measures",
                "moderate": f"Increase monitoring frequency for {pest_name} and prepare control measures",
                "high": f"Immediate action required for {pest_name} control with multiple approaches"
            }.get(severity.lower(), f"Implement appropriate {pest_name} management based on current severity level"),
            "long_term_planning": [
                "Develop crop rotation plans to break pest cycles",
                "Establish beneficial insect habitats around fields",
                "Maintain soil health to support plant resistance",
                "Keep detailed records of pest occurrences and control measures"
            ]
        }
    
    def _extract_control_methods(self, analysis_text, prediction_data):
        """
        Extract control methods from comprehensive analysis.
        
        Args:
            analysis_text (str): Comprehensive analysis text
            prediction_data (dict): Pest prediction data
            
        Returns:
            dict: Control methods components
        """
        pest_name = prediction_data.get('predicted_pest', 'Unknown pest') if prediction_data else 'Unknown pest'
        
        return {
            "biological_control": {
                "natural_enemies": [
                    "Ladybugs and lacewings for aphid control",
                    "Parasitic wasps for caterpillar management",
                    "Predatory mites for spider mite control"
                ],
                "conservation": [
                    "Plant diverse flowering plants to support beneficial insects",
                    "Reduce pesticide use to preserve natural enemies",
                    "Provide overwintering sites for beneficial insects"
                ]
            },
            "cultural_control": {
                "practices": [
                    "Crop rotation to break pest life cycles",
                    "Sanitation to remove pest habitats and food sources",
                    "Proper planting dates to avoid peak pest activity",
                    "Optimal plant spacing for air circulation"
                ],
                "soil_management": [
                    "Maintain proper soil moisture levels",
                    "Add organic matter to improve soil health",
                    "Test soil pH and adjust as needed"
                ]
            },
            "physical_control": {
                "methods": [
                    "Use row covers to exclude pests",
                    "Install sticky traps for monitoring and control",
                    "Apply mulch to prevent soil-dwelling pests",
                    "Handpick large pests when populations are low"
                ],
                "barriers": [
                    "Copper strips for slug and snail control",
                    "Fine mesh screens for small insects",
                    "Trunk bands for tree pests"
                ]
            },
            "chemical_control": {
                "guidelines": [
                    "Use selective pesticides to preserve beneficial insects",
                    "Apply at optimal life stages for maximum effectiveness",
                    "Rotate pesticide classes to prevent resistance",
                    "Follow label instructions for safety and efficacy"
                ],
                "application_timing": f"Apply control measures for {pest_name} at vulnerable life stages"
            }
        }
    
    def _extract_prevention_advice(self, analysis_text, prediction_data):
        """
        Extract prevention advice from comprehensive analysis.
        
        Args:
            analysis_text (str): Comprehensive analysis text
            prediction_data (dict): Pest prediction data
            
        Returns:
            dict: Prevention advice components
        """
        crop = prediction_data.get('crop', 'Unknown crop') if prediction_data else 'Unknown crop'
        
        return {
            "proactive_measures": [
                f"Select pest-resistant varieties of {crop} when available",
                "Maintain optimal plant nutrition and irrigation practices",
                "Remove crop debris and weeds that harbor pests",
                "Establish beneficial insect habitats around fields"
            ],
            "monitoring_strategies": [
                "Scout fields regularly for early pest detection",
                "Use pheromone traps to monitor pest flight activity",
                "Track weather conditions that favor pest development",
                "Record pest populations and damage levels"
            ],
            "seasonal_considerations": [
                "Prepare for overwintering pest emergence in spring",
                "Monitor for summer pest population buildups",
                "Implement fall cleanup to reduce pest survival",
                "Plan winter activities to break pest cycles"
            ],
            "record_keeping": [
                "Document pest occurrences and population levels",
                "Record effectiveness of control measures",
                "Track environmental conditions affecting pest development",
                "Maintain spray records and resistance management plans"
            ]
        }

# Example usage
if __name__ == "__main__":
    # Initialize pest prediction RAG system
    rag_system = PestPredictionRAG()
    
    # Example query
    query = "How to manage aphid infestations in wheat?"
    results = rag_system.query_knowledge_base(query, k=2)
    
    print(f"\nQuery: {query}")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
        print(f"     Content: {result['content'][:150]}...")