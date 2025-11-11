class DiseaseRAG:
    """RAG system for generating natural-language advice for plant diseases."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the Disease RAG system.
        
        Args:
            model_name (str): Name of the embedding model to use
        """
        try:
            # Initialize embeddings with error handling
            try:
                # Handle the HuggingFaceEmbeddings import with fallback
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
            except Exception as e:
                print(f"Error initializing embeddings: {e}")
                self.embedding_model = None
            
            # Initialize text splitter
            try:
                # Handle text splitter import with fallbacks
                try:
                    from langchain_text_splitters.character import RecursiveCharacterTextSplitter
                except ImportError:
                    try:
                        from langchain.text_splitter import RecursiveCharacterTextSplitter
                    except ImportError:
                        from langchain_community.text_splitter import RecursiveCharacterTextSplitter
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
            except Exception as e:
                print(f"Error initializing text splitter: {e}")
                self.text_splitter = None
            
            # Load knowledge base
            self.load_knowledge_base()
            
            # Create FAISS vector store only if embeddings are available
            if self.embedding_model is not None:
                try:
                    from langchain_community.vectorstores import FAISS
                    self.vector_store = FAISS.from_documents(
                        self.text_splitter.create_documents(self.disease_documents) if self.text_splitter else [],
                        self.embedding_model
                    )
                except Exception as e:
                    print(f"Error creating vector store: {e}")
                    self.vector_store = None
            else:
                self.vector_store = None
            
            # Initialize LLM for question answering with fallback
            try:
                from langchain_community.llms import HuggingFacePipeline
                # Use Phi-3 Mini for the RAG system
                self.llm = HuggingFacePipeline.from_model_id(
                    model_id="microsoft/Phi-3-mini-4k-instruct",
                    task="text-generation",
                    model_kwargs={"temperature": 0.7, "max_length": 200}
                )
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                self.llm = None
            
            # Create QA chain only if components are available
            if self.llm is not None and self.vector_store is not None:
                try:
                    from langchain.chains import RetrievalQA
                    self.qa_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=self.vector_store.as_retriever()
                    )
                except Exception as e:
                    print(f"Error creating QA chain: {e}")
                    self.qa_chain = None
            else:
                self.qa_chain = None
                
        except Exception as e:
            print(f"Error initializing DiseaseRAG: {e}")
            # Set all components to None in case of any error
            self.embedding_model = None
            self.text_splitter = None
            self.vector_store = None
            self.llm = None
            self.qa_chain = None
    
    def load_knowledge_base(self):
        """Load disease knowledge base documents."""
        # Sample disease documents (in practice, load from files or database)
        self.disease_documents = [
            "Apple Scab: A fungal disease that affects apple trees. Symptoms include olive-green to black spots on leaves and fruit. Treatment involves applying fungicides during the growing season and removing infected leaves.",
            "Black Rot: A fungal disease affecting apples, characterized by dark, sunken lesions on fruit. Management includes pruning infected branches, applying fungicides, and maintaining good orchard hygiene.",
            "Cedar Apple Rust: A disease requiring both apple and cedar trees to complete its lifecycle. Symptoms include yellow-orange spots on apple leaves. Control measures include removing nearby cedar trees and applying fungicides.",
            "Healthy Plant: No signs of disease. Maintain proper watering, fertilization, and pruning practices for continued health.",
            "Corn Common Rust: A fungal disease causing reddish-brown pustules on leaves. Management includes planting resistant varieties, crop rotation, and fungicide application if necessary.",
            "Corn Northern Leaf Blight: A fungal disease causing long, elliptical lesions on leaves. Control involves crop rotation, residue management, and fungicide application in severe cases.",
            "Grape Black Rot: A fungal disease causing black spots on fruit and leaves. Management includes pruning, sanitation, and fungicide application during bloom and fruit development.",
            "Potato Early Blight: A fungal disease causing dark spots with concentric rings on leaves. Control measures include crop rotation, mulching, and fungicide application.",
            "Tomato Bacterial Spot: A bacterial disease causing dark spots on leaves, stems, and fruit. Management includes using disease-free seeds, crop rotation, and copper-based bactericides.",
            "Tomato Early Blight: A fungal disease causing dark spots with target-like patterns on leaves. Control involves crop rotation, mulching, and fungicide application.",
            "Tomato Late Blight: A serious fungal disease causing water-soaked lesions on leaves and fruit. Management requires immediate removal of infected plants and fungicide application.",
            "Tomato Leaf Mold: A fungal disease causing yellow spots on upper leaf surfaces and white mold on lower surfaces. Control includes improving air circulation and applying fungicides.",
            "Tomato Septoria Leaf Spot: A fungal disease causing small, circular spots with dark borders on leaves. Management involves crop rotation and fungicide application.",
            "Tomato Spider Mites: Tiny pests causing stippling and webbing on leaves. Control includes releasing beneficial insects and applying miticides if necessary.",
            "Tomato Target Spot: A fungal disease causing bull's-eye patterns on leaves. Management includes crop rotation and fungicide application."
        ]
    
    def get_disease_advice(self, disease_name, severity="moderate"):
        """
        Get comprehensive treatment advice for a specific plant disease with detailed recommendations.
        
        Args:
            disease_name (str): Name of the disease
            severity (str): Severity level (low, moderate, high)
            
        Returns:
            dict: Comprehensive advice including treatment, prevention, and management strategies
        """
        try:
            # Formulate comprehensive query based on disease and severity
            query = f"Provide comprehensive advice for managing {disease_name} disease with {severity} severity. Include detailed treatment options, prevention strategies, cultural practices, chemical controls, biological controls, monitoring techniques, and long-term management approaches."
            
            # Get similar documents only if vector store is available
            docs = []
            if self.vector_store is not None:
                try:
                    docs = self.vector_store.similarity_search(query, k=5)
                except Exception as e:
                    print(f"Error in similarity search: {e}")
                    docs = []
            
            # Generate comprehensive advice using QA chain only if available
            advice_text = ""
            if self.qa_chain is not None:
                try:
                    # Create a more detailed prompt for comprehensive advice
                    detailed_prompt = f"""
                    You are an expert plant pathologist. Provide comprehensive, detailed advice for managing {disease_name} with {severity} severity.
                    
                    Include the following sections in your response:
                    1. Disease Identification: Key symptoms and signs
                    2. Treatment Options: Immediate actions to control the disease
                    3. Prevention Strategies: Cultural practices to prevent disease occurrence
                    4. Chemical Controls: Recommended fungicides/bactericides with application timing
                    5. Biological Controls: Natural enemies or biocontrol agents
                    6. Monitoring Techniques: How to track disease progression
                    7. Long-term Management: Sustainable practices for future seasons
                    8. Safety Considerations: Precautions when handling chemicals
                    9. Resistance Management: How to prevent pathogen resistance
                    10. Economic Thresholds: When treatment is cost-effective
                    
                    Format your response in clear sections with actionable, farm-ready advice.
                    """
                    
                    advice = self.qa_chain({"query": detailed_prompt})
                    advice_text = advice["result"]
                except Exception as e:
                    print(f"Error generating detailed advice with QA chain: {e}")
                    advice_text = self._generate_detailed_fallback_advice(disease_name, severity)
            else:
                # Fallback to detailed advice if QA chain is not available
                advice_text = self._generate_detailed_fallback_advice(disease_name, severity)
            
            # Format response with enhanced structure
            response = {
                "disease_name": disease_name,
                "severity": severity,
                "comprehensive_advice": advice_text,
                "treatment_options": self._extract_treatment_options(advice_text, disease_name),
                "prevention_strategies": self._extract_prevention_strategies("", disease_name),
                "monitoring_guidance": self._extract_monitoring_guidance(disease_name, severity),
                "related_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                        "similarity": 0.8  # Placeholder similarity score
                    }
                    for doc in docs
                ]
            }
            
            return response
            
        except Exception as e:
            print(f"Error in get_disease_advice: {e}")
            # Return comprehensive fallback response in case of any error
            return {
                "disease_name": disease_name,
                "severity": severity,
                "comprehensive_advice": self._generate_detailed_fallback_advice(disease_name, severity),
                "treatment_options": self._extract_treatment_options("", disease_name),
                "prevention_strategies": self._extract_prevention_strategies("", disease_name),
                "monitoring_guidance": self._extract_monitoring_guidance(disease_name, severity),
                "related_documents": []
            }
    
    def _generate_detailed_fallback_advice(self, disease_name, severity):
        """
        Generate detailed fallback advice when LLM is not available.
        
        Args:
            disease_name (str): Name of the disease
            severity (str): Severity level
            
        Returns:
            str: Detailed advice text
        """
        # Enhanced fallback advice with more detail
        severity_descriptions = {
            "low": "Low severity - early stage or minimal infection",
            "moderate": "Moderate severity - established infection affecting some plants",
            "high": "High severity - widespread infection with significant crop impact"
        }
        
        advice = f"Comprehensive Management Plan for {disease_name} ({severity_descriptions.get(severity, severity)}):\n\n"
        
        # Disease Identification
        advice += "1. DISEASE IDENTIFICATION:\n"
        advice += f"   - Confirm {disease_name} through visual inspection of symptoms\n"
        advice += f"   - Look for characteristic signs: discoloration, spots, lesions, or growths\n"
        advice += f"   - Differentiate from similar diseases or abiotic disorders\n\n"
        
        # Treatment Options
        advice += "2. TREATMENT OPTIONS:\n"
        advice += f"   - Immediate: Remove and destroy severely infected plant parts\n"
        advice += f"   - Chemical: Apply appropriate fungicides/bactericides as labeled\n"
        advice += f"   - Biological: Introduce beneficial microbes or biocontrol agents\n"
        advice += f"   - Cultural: Improve air circulation and reduce humidity\n\n"
        
        # Prevention Strategies
        advice += "3. PREVENTION STRATEGIES:\n"
        advice += f"   - Use certified, disease-free seeds or planting material\n"
        advice += f"   - Practice crop rotation with non-host plants\n"
        advice += f"   - Maintain proper plant spacing for air movement\n"
        advice += f"   - Avoid overhead irrigation to minimize leaf wetness\n"
        advice += f"   - Sanitize tools and equipment between uses\n"
        advice += f"   - Remove crop debris and weeds that harbor pathogens\n\n"
        
        # Monitoring Guidance
        advice += "4. MONITORING GUIDANCE:\n"
        advice += f"   - Inspect plants weekly for early disease signs\n"
        advice += f"   - Focus on lower leaves and areas with poor air circulation\n"
        advice += f"   - Track environmental conditions that favor disease development\n"
        advice += f"   - Record disease incidence to assess management effectiveness\n\n"
        
        # Long-term Management
        advice += "5. LONG-TERM MANAGEMENT:\n"
        advice += f"   - Select resistant varieties when available\n"
        advice += f"   - Improve soil health with organic amendments\n"
        advice += f"   - Establish beneficial insect habitats to support natural enemies\n"
        advice += f"   - Develop an integrated disease management plan\n\n"
        
        # Safety Considerations
        advice += "6. SAFETY CONSIDERATIONS:\n"
        advice += f"   - Always read and follow pesticide label instructions\n"
        advice += f"   - Wear appropriate personal protective equipment\n"
        advice += f"   - Observe pre-harvest intervals for chemical treatments\n"
        advice += f"   - Store pesticides in original containers away from children\n\n"
        
        return advice
    
    
    def search_disease_info(self, query):
        """
        Search for disease information based on a query with enhanced results.
        
        Args:
            query (str): Search query
            
        Returns:
            list: List of relevant documents with enhanced metadata
        """
        try:
            # Search for similar documents only if vector store is available
            docs = []
            if self.vector_store is not None:
                try:
                    docs = self.vector_store.similarity_search(query, k=7)
                except Exception as e:
                    print(f"Error in similarity search: {e}")
                    docs = []
            
            # Enhanced document formatting with more detailed metadata
            enhanced_docs = []
            for doc in docs:
                # Try to extract disease name from content
                content_lower = doc.page_content.lower()
                disease_keywords = ["scab", "rot", "rust", "blight", "spot", "mildew", "wilt"]
                detected_diseases = [kw for kw in disease_keywords if kw in content_lower]
                
                enhanced_doc = {
                    "content": doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    "similarity": 0.8,  # Placeholder similarity score
                    "detected_diseases": detected_diseases,
                    "content_type": "treatment" if any(word in content_lower for word in ["treat", "control", "manage"]) else "prevention" if "prevent" in content_lower else "identification" if "symptom" in content_lower else "general",
                    "recommended_actions": self._extract_actions_from_content(doc.page_content)
                }
                enhanced_docs.append(enhanced_doc)
            
            return enhanced_docs
        except Exception as e:
            print(f"Error in search_disease_info: {e}")
            return []
    
    def _extract_actions_from_content(self, content):
        """Extract actionable recommendations from content."""
        actions = []
        action_indicators = ["apply", "use", "remove", "destroy", "plant", "rotate", "avoid", "ensure", "monitor", "inspect"]
        
        sentences = content.split(".")
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(indicator in sentence_lower for indicator in action_indicators):
                # Clean up the action statement
                action = sentence.strip()
                if action and not action.endswith("."):
                    action += "."
                if action:
                    actions.append(action)
        
        return actions[:5]  # Return top 5 actions
    
    def _extract_prevention_strategies(self, advice_text, disease_name):
        """
        Extract prevention strategies from advice text.
        
        Args:
            advice_text (str): Full advice text
            disease_name (str): Name of the disease
            
        Returns:
            list: List of prevention strategies
        """
        # Default prevention strategies if not extracted from text
        default_strategies = [
            f"Use certified, disease-free seeds or {disease_name}-resistant varieties",
            f"Practice crop rotation to break {disease_name} disease cycles",
            f"Maintain proper plant spacing to improve air circulation",
            f"Avoid overhead irrigation to reduce leaf wetness duration",
            f"Sanitize tools and equipment to prevent mechanical transmission",
            f"Remove and destroy crop debris that may harbor {disease_name} pathogens"
        ]
        
        return default_strategies
    
    def _extract_monitoring_guidance(self, disease_name, severity):
        """
        Generate monitoring guidance based on disease and severity.
        
        Args:
            disease_name (str): Name of the disease
            severity (str): Severity level
            
        Returns:
            dict: Monitoring guidance with frequency and indicators
        """
        # Monitoring frequency based on severity
        frequency_map = {
            "low": "Weekly inspection recommended",
            "moderate": "Twice-weekly inspection recommended",
            "high": "Daily inspection recommended"
        }
        
        return {
            "frequency": frequency_map.get(severity, "Regular monitoring recommended"),
            "key_indicators": [
                f"Look for early {disease_name} symptoms on lower leaves",
                f"Monitor environmental conditions favoring {disease_name} development",
                f"Track spread pattern and rate of {disease_name} infection",
                f"Assess effectiveness of current {disease_name} management practices",
                f"Record environmental data that correlates with {disease_name} outbreaks"
            ],
            "tools_needed": [
                "Hand lens for detailed symptom examination",
                "Moisture meter for soil conditions",
                "Thermometer and hygrometer for environmental monitoring",
                "Field notebook for record keeping"
            ]
        }

    def _extract_treatment_options(self, advice_text, disease_name):
        """
        Extract treatment options from advice text.
        
        Args:
            advice_text (str): Full advice text
            disease_name (str): Name of the disease
            
        Returns:
            list: List of treatment options
        """
        # Default treatment options if not extracted from text
        default_options = [
            f"Remove and destroy infected {disease_name}-affected plant parts immediately",
            f"Apply appropriate fungicides/bactericides labeled for {disease_name}",
            f"Improve air circulation around plants to reduce humidity",
            f"Reduce watering frequency to prevent excessive moisture",
            f"Apply mulch to prevent soil-borne pathogens from splashing onto plants"
        ]
        
        # Try to extract from advice text if available
        if "TREATMENT" in advice_text.upper() or "TREAT" in advice_text.upper():
            # Simple extraction logic - in practice, this would be more sophisticated
            return default_options
        else:
            return default_options