"""
Market Prediction RAG System for Smart Agriculture Project

This module implements a Retrieval-Augmented Generation system using FAISS
for providing context-aware market prediction explanations and recommendations.
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

class MarketPredictionRAG:
    """RAG system for market prediction knowledge retrieval"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', knowledge_base_path='explainable_ai/knowledge_base'):
        """
        Initialize the RAG system for market prediction
        
        Args:
            model_name: Name of the sentence transformer model for embeddings
            knowledge_base_path: Path to the knowledge base directory
        """
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize unified agricultural knowledge base
        try:
            self.unified_kb = UnifiedAgriculturalKB(knowledge_base_path)
            print("Using enhanced unified agricultural knowledge base for market prediction")
        except Exception as e:
            print(f"Error initializing unified knowledge base: {str(e)}")
            print("Falling back to basic market prediction knowledge base")
            self.unified_kb = None
            self._init_basic_knowledge_base()
    
    def _init_basic_knowledge_base(self):
        """
        Initialize basic market prediction knowledge base as fallback
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
        Load market prediction knowledge documents into the RAG system
        """
        # Only load if we're using the basic knowledge base
        if self.unified_kb is not None:
            return
            
        documents = []
        
        # Load comprehensive market prediction knowledge documents
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
        for doc in market_documents:
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
        
        print(f"Loaded {len(split_documents)} market prediction document chunks into the knowledge base")
    
    def add_document(self, title, content):
        """
        Add a new document to the market prediction knowledge base
        
        Args:
            title: Title of the document
            content: Content of the document
        """
        # If we're using the unified knowledge base, add to the market prediction domain
        if self.unified_kb is not None:
            self.unified_kb.add_domain_specific_information(
                'market_prediction',
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
        Query the market prediction knowledge base
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            list: List of relevant documents with similarity scores
        """
        # If we're using the unified knowledge base, query the market prediction domain
        if self.unified_kb is not None:
            return self.unified_kb.query_knowledge_base(query, domain='market_prediction', k=k)
        
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
        Generate a comprehensive context-aware response with detailed market analysis and advice
        
        Args:
            query: User query
            prediction_data: Market prediction data for context
            
        Returns:
            dict: Comprehensive market analysis including analysis, timing advice, and risk assessment
        """
        try:
            # Retrieve relevant documents
            relevant_docs = self.query_knowledge_base(query, k=5)
            
            # Create context from retrieved documents
            context = "Relevant market information:\n"
            for i, doc in enumerate(relevant_docs, 1):
                content = doc.get('content', doc.get('page_content', ''))[:500]
                context += f"{i}. {content}\n"
            
            # Add prediction data to context if provided
            if prediction_data:
                context += f"\nMarket Prediction Data:\n"
                for key, value in prediction_data.items():
                    context += f"- {key}: {value}\n"
            
            # Generate comprehensive market analysis using LLM if available
            comprehensive_analysis = self._generate_comprehensive_market_analysis(context, prediction_data)
            
            # Extract key components
            market_analysis = self._extract_market_analysis(comprehensive_analysis, prediction_data)
            timing_advice = self._extract_timing_advice(comprehensive_analysis, prediction_data)
            risk_assessment = self._extract_risk_assessment(comprehensive_analysis, prediction_data)
            
            return {
                "query": query,
                "context": context,
                "comprehensive_analysis": comprehensive_analysis,
                "market_analysis": market_analysis,
                "timing_advice": timing_advice,
                "risk_assessment": risk_assessment,
                "related_documents": relevant_docs
            }
            
        except Exception as e:
            print(f"Error in generate_context_aware_response: {e}")
            # Fallback response
            return {
                "query": query,
                "context": "Context not available",
                "comprehensive_analysis": self._generate_fallback_market_analysis(query, prediction_data),
                "market_analysis": self._extract_market_analysis("", prediction_data),
                "timing_advice": self._extract_timing_advice("", prediction_data),
                "risk_assessment": self._extract_risk_assessment("", prediction_data),
                "related_documents": []
            }
    
    def _generate_comprehensive_market_analysis(self, context, prediction_data):
        """
        Generate comprehensive market analysis using LLM.
        
        Args:
            context (str): Context information from knowledge base
            prediction_data (dict): Market prediction data
            
        Returns:
            str: Comprehensive market analysis
        """
        try:
            # Try to import LLM components
            from explainable_ai.llm_interface import AgricultureLLM
            
            # Initialize LLM
            llm = AgricultureLLM()
            
            # Create detailed prompt for comprehensive analysis
            detailed_prompt = f"""
            You are an expert agricultural market analyst. Based on the following context and market data, 
            provide a comprehensive market analysis with detailed recommendations.
            
            Context Information:
            {context}
            
            Market Prediction Data:
            {prediction_data}
            
            Please provide:
            1. Market Analysis: Detailed interpretation of current market conditions
            2. Price Outlook: Short-term and medium-term price projections
            3. Supply-Demand Dynamics: Analysis of key factors affecting supply and demand
            4. External Influences: Impact of weather, policies, and global events
            5. Timing Advice: Optimal buying/selling strategies and market entry points
            6. Risk Assessment: Potential risks and mitigation strategies
            7. Investment Recommendations: Suggestions for hedging and portfolio management
            8. Regional Variations: Geographic differences in market trends
            9. Seasonal Patterns: Impact of seasonal factors on price movements
            10. Long-term Outlook: Future market trends and structural changes
            
            Format your response in clear sections with actionable market intelligence.
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
        return self._generate_fallback_market_analysis(context, prediction_data)
    
    def _generate_fallback_market_analysis(self, context, prediction_data):
        """
        Generate fallback market analysis when LLM is not available.
        
        Args:
            context (str): Context information
            prediction_data (dict): Market prediction data
            
        Returns:
            str: Fallback market analysis
        """
        crop = prediction_data.get('crop', 'Unknown crop') if prediction_data else 'Unknown crop'
        predicted_price = prediction_data.get('predicted_price', 'N/A') if prediction_data else 'N/A'
        market_trend = prediction_data.get('market_trend', 'N/A') if prediction_data else 'N/A'
        
        analysis = f"Comprehensive Market Analysis for {crop}\n\n"
        analysis += f"Predicted Price: ${predicted_price}/ton\n"
        analysis += f"Market Trend: {market_trend}\n\n"
        
        # Market Analysis
        analysis += "1. MARKET ANALYSIS:\n"
        if market_trend.lower() == 'bullish':
            analysis += f"   - {crop} prices are expected to rise due to favorable market conditions\n"
        elif market_trend.lower() == 'bearish':
            analysis += f"   - {crop} prices are expected to decline due to oversupply or weak demand\n"
        else:
            analysis += f"   - {crop} prices are expected to remain stable with minor fluctuations\n"
        analysis += "   - Monitor supply chain developments and policy changes\n\n"
        
        # Price Outlook
        analysis += "2. PRICE OUTLOOK:\n"
        analysis += f"   - Short-term: Expect {'increasing' if market_trend.lower() == 'bullish' else 'decreasing' if market_trend.lower() == 'bearish' else 'stable'} price trend\n"
        analysis += f"   - Medium-term: Market fundamentals will drive price direction\n"
        analysis += "   - Long-term: Global demand and production trends will shape market\n\n"
        
        # Supply-Demand Dynamics
        analysis += "3. SUPPLY-DEMAND DYNAMICS:\n"
        analysis += "   - Monitor production forecasts and inventory levels\n"
        analysis += "   - Track export/import data and trade policies\n"
        analysis += "   - Analyze consumption patterns and substitution effects\n\n"
        
        # Timing Advice
        analysis += "4. TIMING ADVICE:\n"
        analysis += "   - Consider forward contracting to lock in favorable prices\n"
        analysis += "   - Monitor seasonal patterns for optimal marketing windows\n"
        analysis += "   - Watch for weather events that may impact supply\n\n"
        
        # Risk Assessment
        analysis += "5. RISK ASSESSMENT:\n"
        analysis += "   - Price volatility may create opportunities and risks\n"
        analysis += "   - Policy changes could impact market access and profitability\n"
        analysis += "   - Weather events may disrupt supply chains\n\n"
        
        return analysis
    
    def _extract_market_analysis(self, analysis_text, prediction_data):
        """
        Extract market analysis from comprehensive analysis.
        
        Args:
            analysis_text (str): Comprehensive analysis text
            prediction_data (dict): Market prediction data
            
        Returns:
            dict: Market analysis components
        """
        crop = prediction_data.get('crop', 'Unknown crop') if prediction_data else 'Unknown crop'
        predicted_price = prediction_data.get('predicted_price', 'N/A') if prediction_data else 'N/A'
        market_trend = prediction_data.get('market_trend', 'N/A') if prediction_data else 'N/A'
        
        return {
            "price_outlook": f"Based on current trends, {crop} prices are expected to {'rise' if market_trend.lower() == 'bullish' else 'fall' if market_trend.lower() == 'bearish' else 'remain stable'} over the next 30 days.",
            "supply_demand_factors": [
                "Monitor production forecasts and yield expectations",
                "Track export/import data and trade policies",
                "Analyze consumption patterns and industrial demand",
                "Watch inventory levels and carryover stocks"
            ],
            "external_influences": [
                "Weather conditions affecting production",
                "Government policies and subsidies",
                "Global economic trends and exchange rates",
                "Energy prices impacting production costs"
            ],
            "market_indicators": {
                "current_price_level": predicted_price,
                "trend_direction": market_trend,
                "volatility_expectation": "Moderate" if market_trend != 'neutral' else "Low"
            }
        }
    
    def _extract_timing_advice(self, analysis_text, prediction_data):
        """
        Extract timing advice from comprehensive analysis.
        
        Args:
            analysis_text (str): Comprehensive analysis text
            prediction_data (dict): Market prediction data
            
        Returns:
            dict: Timing advice components
        """
        market_trend = prediction_data.get('market_trend', 'N/A') if prediction_data else 'N/A'
        confidence_score = prediction_data.get('confidence_score', 0.5) if prediction_data else 0.5
        
        advice = ""
        if market_trend.lower() == 'bullish':
            advice = "Consider holding inventory for higher prices."
            if confidence_score > 0.8:
                advice += " Strong confidence in upward trend - may be optimal time to delay sales."
        elif market_trend.lower() == 'bearish':
            advice = "Consider forward contracting to lock in prices."
            if confidence_score > 0.8:
                advice += " Strong confidence in downward trend - may be optimal time to sell early."
        else:
            advice = "Monitor market conditions closely for opportunities."
            if confidence_score < 0.6:
                advice += " Lower confidence - consider hedging strategies to protect against volatility."
        
        return {
            "primary_advice": advice,
            "optimal_timing": {
                "buying_opportunities": "Monitor for price dips during oversupply periods",
                "selling_opportunities": "Look for price peaks during supply shortages",
                "contracting_strategies": "Consider futures and options for price protection"
            },
            "seasonal_considerations": [
                "Harvest periods typically see lower prices due to increased supply",
                "Planting seasons may see higher prices due to usage concerns",
                "Weather events can create temporary price volatility"
            ]
        }
    
    def _extract_risk_assessment(self, analysis_text, prediction_data):
        """
        Extract risk assessment from comprehensive analysis.
        
        Args:
            analysis_text (str): Comprehensive analysis text
            prediction_data (dict): Market prediction data
            
        Returns:
            dict: Risk assessment components
        """
        supply_index = prediction_data.get('supply_index', 60.0) if prediction_data else 60.0
        demand_index = prediction_data.get('demand_index', 60.0) if prediction_data else 60.0
        global_demand = prediction_data.get('global_demand', 'medium') if prediction_data else 'medium'
        weather_impact = prediction_data.get('weather_impact', 'normal') if prediction_data else 'normal'
        
        risks = []
        
        # Add factors based on input data
        if demand_index != 0 and (supply_index / demand_index) < 0.8:
            risks.append("Supply constraints relative to demand may drive prices upward")
        elif demand_index != 0 and (supply_index / demand_index) > 1.2:
            risks.append("Excess supply relative to demand may put downward pressure on prices")
        
        if global_demand == 'high':
            risks.append("Strong global demand supports higher prices")
        elif global_demand == 'low':
            risks.append("Weak global demand puts pressure on prices")
        
        if weather_impact == 'poor':
            risks.append("Adverse weather conditions may reduce supply and increase prices")
        elif weather_impact == 'excellent':
            risks.append("Favorable weather conditions may increase supply and moderate prices")
        
        return {
            "identified_risks": risks if risks else ["No significant risk factors identified at this time"],
            "risk_mitigation_strategies": [
                "Use futures contracts to lock in favorable prices",
                "Diversify across multiple crops and markets",
                "Maintain adequate working capital for operational flexibility",
                "Consider crop insurance to protect against production losses"
            ],
            "market_volatility": "Monitor for sudden price swings due to news events",
            "policy_risks": "Watch for changes in trade policies and government programs"
        }

# Example usage
if __name__ == "__main__":
    # Initialize market prediction RAG system
    rag_system = MarketPredictionRAG()
    
    # Example query
    query = "How does weather affect crop prices?"
    results = rag_system.query_knowledge_base(query, k=2)
    
    print(f"\nQuery: {query}")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['metadata']['title']} (Similarity: {result['similarity']:.4f})")
        print(f"     Content: {result['content'][:150]}...")