"""
RAG system for crop and fertilizer recommendations.
"""

try:
    from langchain_community.vectorstores import FAISS
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # Fallback to community embeddings if huggingface wrapper not available
        from langchain_community.embeddings import HuggingFaceEmbeddings
    # Handle text splitter import with fallbacks
    try:
        from langchain_text_splitters.character import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain_community.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
    import os
except ImportError as e:
    print(f"Import error in RecommendationRAG: {e}")
    # Mock classes for graceful degradation
    class MockClass:
        def __init__(self, *args, **kwargs):
            pass
    
    FAISS = MockClass
    HuggingFaceEmbeddings = MockClass
    RecursiveCharacterTextSplitter = MockClass
    RetrievalQA = MockClass
    HuggingFacePipeline = MockClass


class RecommendationRAG:
    """RAG system for generating natural-language advice for crop and fertilizer recommendations."""
    
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize the Recommendation RAG system.
        
        Args:
            model_name (str): Name of the HuggingFace model to use
        """
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Load knowledge base
            self.load_knowledge_base()
            
            # Initialize LLM
            # Note: In a production environment, you might want to use a more powerful model
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                model_kwargs={"temperature": 0.7, "max_length": 300}
            )
            
            # Create QA chain
            if hasattr(self, 'vector_store') and self.vector_store:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever()
                )
            else:
                self.qa_chain = None
        except Exception as e:
            print(f"Error initializing RecommendationRAG: {e}")
            # Set attributes to None so we can handle the error gracefully
            self.embeddings = None
            self.vector_store = None
            self.llm = None
            self.qa_chain = None
    
    def load_knowledge_base(self):
        """Load comprehensive agricultural knowledge base documents."""
        # Comprehensive recommendation documents with detailed information
        documents = [
            # Crop-specific cultivation guides
            "Wheat cultivation guide: Wheat prefers well-drained soil with pH 6.0-7.5. Optimal temperature for growth is 15-25°C. Apply nitrogen fertilizer in split doses - half at planting and half at tillering stage. Sow seeds at 3-4 cm depth with 20-25 cm row spacing. Irrigate at critical growth stages: crown root initiation, tillering, jointing, booting, and grain filling. Common pests include aphids, armyworms, and diseases like rust and smut. Apply fungicides preventively during high disease pressure periods.",
            "Rice cultivation guide: Rice grows best in clayey soil with pH 5.5-7.0. Requires 100-150 cm of water during growing season. Apply basal fertilizer before transplanting and top-dress during tillering. Transplant 20-25 day old seedlings at 20x20 cm spacing. Maintain continuous flooding until panicle initiation, then practice alternate wetting and drying. Common pests include stem borer, leaf folder, and brown plant hopper. Diseases include blast, sheath blight, and bacterial leaf blight. Apply neem cake and trichoderma as biocontrol agents.",
            "Maize cultivation guide: Maize thrives in well-drained fertile soil with pH 5.8-7.0. Optimal temperature range is 20-25°C. Apply nitrogen fertilizer in three splits: at planting, knee-high stage, and tasseling. Sow seeds at 5-7 cm depth with 75 cm row spacing and 25 cm plant spacing. Irrigate at germination, knee-high, tasseling, and grain filling stages. Common pests include stem borer, fall armyworm, and diseases like northern corn leaf blight and rust. Practice crop rotation and intercropping with legumes for soil health.",
            "Cotton cultivation guide: Cotton grows well in deep, well-drained soil with pH 6.0-8.5. Requires warm weather with 20-30°C temperature. Apply nitrogen and potassium fertilizers in splits during growing season. Sow seeds at 2-3 cm depth with 90 cm row spacing and 30 cm plant spacing. Irrigate at square formation, flower bud appearance, flowering, and boll development stages. Common pests include bollworms, aphids, and whiteflies. Diseases include fusarium wilt, verticillium wilt, and leaf spot. Use pheromone traps and beneficial insects for pest management.",
            "Sugarcane cultivation guide: Sugarcane prefers deep, well-drained soil with pH 6.5-7.5. Requires warm climate with 25-35°C temperature. Apply organic manure along with NPK fertilizers before planting. Plant setts with 2-3 buds at 10-15 cm depth with 90 cm row spacing. Irrigate regularly to maintain soil moisture at 60-70% field capacity. Common pests include top borer, internode borer, and diseases like red rot and smut. Practice trash mulching and intercropping with legumes. Harvest when cane reaches maturity (10-12 months).",
            
            # Fertilizer application guides
            "Urea application guide: Urea is a nitrogen-rich fertilizer containing 46% nitrogen. Apply as basal dressing before planting or as top dressing during active growth stages. Incorporate into soil immediately after application to prevent nitrogen loss through volatilization. For rice, apply 50% as basal and 50% at tillering. For wheat, apply 50% at sowing and 50% at crown root initiation. For maize, apply in three splits: planting, knee-high, and tasseling stages. Avoid foliar application during hot hours to prevent leaf burn.",
            "DAP (Diammonium Phosphate) application guide: DAP contains 18% nitrogen and 46% phosphorus (P2O5). Apply as basal fertilizer at planting time. Place in bands near seeds or incorporate into soil. Particularly beneficial for root development and early plant growth. For cereals, apply 20-25 kg/ha as basal dose. For pulses, apply 30-40 kg/ha. For vegetables, apply 40-50 kg/ha. Combine with organic manures for sustained nutrient release and improved soil health.",
            "MOP (Muriate of Potash) application guide: MOP contains 60% potassium (K2O). Apply during active growth stages when potassium demand is high. For rice, apply 50% at basal and 50% at panicle initiation. For wheat, apply 100% as basal. For fruits and vegetables, apply in splits during flowering and fruit development. Helps improve disease resistance, water use efficiency, and fruit quality. Apply 50-100 kg/ha depending on soil test results and crop requirements. Avoid direct contact with seeds or roots.",
            "SSP (Single Super Phosphate) application guide: SSP contains 16% phosphorus (P2O5) and 12% calcium (CaO). Apply as basal fertilizer before planting. Particularly suitable for acidic soils to supply both phosphorus and calcium. For cereals, apply 40-60 kg/ha. For pulses, apply 60-80 kg/ha. For vegetables, apply 80-100 kg/ha. Mix with organic manures for better nutrient availability. Helps in root development, flowering, and fruiting. Apply in bands or broadcast and incorporate into soil.",
            "NPK 15-15-15 fertilizer guide: Balanced NPK fertilizer contains equal proportions of nitrogen, phosphorus, and potassium (15% each). Suitable for general plant nutrition and maintenance fertilization. Apply during active growth stages or as maintenance dose. For vegetables, apply 100-150 kg/ha in splits. For ornamental plants, apply 50-100 kg/ha monthly. For fruit trees, apply 200-300 kg/ha annually. Dissolve in water for fertigation. Provides complete nutrition for balanced plant growth.",
            
            # Soil management practices
            "Soil testing practices: Regular soil testing helps determine nutrient deficiencies and pH levels. Test soil every 2-3 years to adjust fertilizer application rates. Collect samples from multiple locations in the field at 15-20 cm depth. Mix samples thoroughly and send to certified laboratory. Analyze for pH, organic carbon, nitrogen, phosphorus, potassium, and micronutrients. Based on results, apply lime to adjust pH, organic manures to improve organic carbon, and fertilizers to correct nutrient deficiencies. Maintain detailed records of soil test results and fertilizer applications for future reference.",
            "Organic fertilizers guide: Compost and manure improve soil structure and provide slow-release nutrients. Apply 5-10 tons per hectare before planting. Benefits include improved water retention, soil microbial activity, and cation exchange capacity. Vermicompost is rich in nutrients and beneficial microorganisms. Apply 2-3 tons/ha as basal dressing. Farmyard manure (FYM) should be well-decomposed before application. Apply 10-15 tons/ha 2-3 months before planting. Green manuring with legumes adds nitrogen and organic matter. Incorporate 15-20 days after sowing. Biofertilizers like rhizobium, azotobacter, and phosphate solubilizing bacteria enhance nutrient availability.",
            "Water management for crops: Proper irrigation scheduling improves nutrient uptake and crop yields. Apply fertilizers when soil moisture is adequate for better nutrient dissolution and root absorption. Avoid fertilizer application during waterlogging conditions to prevent nutrient loss and plant damage. Use efficient irrigation methods like drip, sprinkler, or furrow irrigation based on crop and soil type. Schedule irrigation based on crop water requirements, soil moisture status, and weather conditions. Practice mulching to reduce evaporation and maintain soil moisture. Implement water harvesting and recycling systems for sustainable water use.",
            "Integrated nutrient management: Combine organic and inorganic fertilizers for sustainable crop production. Use crop residues, green manure, and biofertilizers along with chemical fertilizers. Apply 25-50% of recommended NPK through organic sources. Supplement with chemical fertilizers to meet remaining requirements. Practice crop rotation with legumes to fix atmospheric nitrogen. Use cover crops to prevent nutrient leaching and soil erosion. Apply micronutrient fertilizers based on soil test results. Foliar application is effective for quick correction of micronutrient deficiencies.",
            "Micronutrient management: Deficiency of zinc, iron, manganese, and boron affects crop growth and yield. Apply micronutrient fertilizers based on soil test results. Zinc deficiency is common in calcareous soils. Apply zinc sulfate @ 25 kg/ha as basal dressing. Iron deficiency occurs in high pH soils. Apply ferrous sulfate or chelated iron @ 10-15 kg/ha. Manganese deficiency affects rice and wheat. Apply manganese sulfate @ 10-15 kg/ha. Boron deficiency affects cauliflower, mango, and citrus. Apply borax @ 5-10 kg/ha. Foliar application is effective for quick correction. Use chelated forms for better availability.",
            
            # Advanced farming techniques
            "Precision farming techniques: Use soil sensors and weather stations to monitor soil moisture, temperature, and nutrient levels. Apply fertilizers based on real-time crop requirements using variable rate technology. Practice site-specific nutrient management based on soil variability within field. Use drones for crop monitoring and targeted pesticide application. Implement GPS-guided machinery for precise planting, fertilization, and harvesting. Use mobile apps for farm management and advisory services. Maintain digital records of all farm operations for analysis and decision making. Practice data-driven farming for improved productivity and profitability.",
            "Climate-smart agriculture: Adapt farming practices to changing climate conditions. Select climate-resilient crop varieties suitable for local conditions. Practice conservation agriculture with minimal soil disturbance, permanent soil cover, and crop rotation. Implement water-saving irrigation techniques like drip irrigation and sprinkler systems. Use weather-based advisory services for timely farm operations. Practice agroforestry to improve microclimate and biodiversity. Implement carbon farming practices to sequester atmospheric carbon. Use renewable energy sources for farm operations. Practice integrated pest management to reduce chemical inputs.",
            "Sustainable farming practices: Implement crop rotation to break pest and disease cycles and improve soil health. Practice intercropping and mixed cropping to optimize land use and reduce risk. Use organic farming methods to improve soil fertility and reduce environmental impact. Implement integrated pest management using biological, cultural, and chemical control methods. Practice water conservation through rainwater harvesting and efficient irrigation systems. Use renewable energy sources like solar power for farm operations. Implement waste management through composting and recycling. Practice biodiversity conservation through habitat creation and native species protection.",
        ]
        
        # Split documents
        split_docs = self.text_splitter.create_documents(documents)
        
        # Create vector store
        try:
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.vector_store = None
    
    def get_crop_advice(self, crop_name, soil_conditions, weather_conditions):
        """
        Get comprehensive crop-specific advice with detailed growing recommendations.
        
        Args:
            crop_name (str): Name of the crop
            soil_conditions (str): Description of soil conditions
            weather_conditions (str): Description of weather conditions
            
        Returns:
            dict: Comprehensive advice including growing practices, nutrient management, and best practices
        """
        try:
            if self.qa_chain is None:
                fallback_response = self._generate_fallback_crop_advice(crop_name, soil_conditions, weather_conditions)
                return {
                    "crop_name": crop_name,
                    "comprehensive_advice": fallback_response,
                    "planting_recommendations": self._extract_planting_recommendations(crop_name),
                    "fertilization_schedule": self._extract_fertilization_schedule(crop_name),
                    "irrigation_guidance": self._extract_irrigation_guidance(crop_name),
                    "pest_management": self._extract_pest_management_strategies(crop_name),
                    "harvest_guidance": self._extract_harvest_guidance(crop_name)
                }
            
            # Create detailed prompt for comprehensive advice
            detailed_prompt = f"""
            You are an expert agricultural advisor. Provide comprehensive advice for growing {crop_name} with the following conditions:
            Soil Conditions: {soil_conditions}
            Weather Conditions: {weather_conditions}
            
            Include the following sections in your response:
            1. Crop Characteristics: Growth habits, optimal conditions, and development stages
            2. Site Preparation: Land preparation, soil amendments, and bed formation
            3. Planting Recommendations: Optimal timing, seed rate, spacing, and depth
            4. Fertilization Schedule: Nutrient requirements, application timing, and methods
            5. Irrigation Guidance: Water requirements, scheduling, and efficient methods
            6. Pest and Disease Management: Common threats, monitoring, and control strategies
            7. Weed Management: Control methods and timing
            8. Growth Monitoring: Key indicators and critical growth stages
            9. Harvest Guidance: Timing indicators, methods, and post-harvest handling
            10. Storage and Marketing: Best practices for preserving quality
            
            Format your response in clear sections with actionable, farm-ready advice.
            """
            
            result = self.qa_chain({"query": detailed_prompt})
            advice_text = result["result"]
            
            # Return structured response
            return {
                "crop_name": crop_name,
                "comprehensive_advice": advice_text,
                "planting_recommendations": self._extract_planting_recommendations(crop_name),
                "fertilization_schedule": self._extract_fertilization_schedule(crop_name),
                "irrigation_guidance": self._extract_irrigation_guidance(crop_name),
                "pest_management": self._extract_pest_management_strategies(crop_name),
                "harvest_guidance": self._extract_harvest_guidance(crop_name)
            }
        except Exception as e:
            print(f"Error in get_crop_advice: {e}")
            fallback_response = self._generate_fallback_crop_advice(crop_name, soil_conditions, weather_conditions)
            return {
                "crop_name": crop_name,
                "comprehensive_advice": fallback_response,
                "planting_recommendations": self._extract_planting_recommendations(crop_name),
                "fertilization_schedule": self._extract_fertilization_schedule(crop_name),
                "irrigation_guidance": self._extract_irrigation_guidance(crop_name),
                "pest_management": self._extract_pest_management_strategies(crop_name),
                "harvest_guidance": self._extract_harvest_guidance(crop_name)
            }
    
    def _generate_fallback_crop_advice(self, crop_name, soil_conditions, weather_conditions):
        """
        Generate detailed fallback advice when LLM is not available.
        
        Args:
            crop_name (str): Name of the crop
            soil_conditions (str): Description of soil conditions
            weather_conditions (str): Description of weather conditions
            
        Returns:
            str: Detailed advice text
        """
        advice = f"Comprehensive Growing Guide for {crop_name}\n\n"
        advice += f"Soil Conditions: {soil_conditions}\n"
        advice += f"Weather Conditions: {weather_conditions}\n\n"
        
        # Crop Characteristics
        advice += "1. CROP CHARACTERISTICS:\n"
        advice += f"   - {crop_name} is a {'cool' if 'wheat' in crop_name.lower() or 'barley' in crop_name.lower() else 'warm'} season crop\n"
        advice += f"   - Optimal temperature range: {'15-25°C' if 'wheat' in crop_name.lower() else '20-35°C'}\n"
        advice += f"   - Growth duration: {'90-120 days' if 'wheat' in crop_name.lower() else '120-180 days'}\n\n"
        
        # Site Preparation
        advice += "2. SITE PREPARATION:\n"
        advice += f"   - Deep plowing 2-3 times to achieve fine tilth\n"
        advice += f"   - Incorporate well-decomposed organic manure @ 5-10 tons/ha\n"
        advice += f"   - Level the field properly for uniform water distribution\n"
        advice += f"   - Test soil pH and adjust with lime or sulfur if needed\n\n"
        
        # Planting Recommendations
        advice += "3. PLANTING RECOMMENDATIONS:\n"
        advice += f"   - Optimal planting time: {'October-November' if 'wheat' in crop_name.lower() else 'June-July'}\n"
        advice += f"   - Seed rate: {'100-125 kg/ha' if 'wheat' in crop_name.lower() else '15-20 kg/ha'}\n"
        advice += f"   - Sowing depth: {'3-5 cm' if 'wheat' in crop_name.lower() else '2-3 cm'}\n"
        advice += f"   - Row spacing: {'20-25 cm' if 'wheat' in crop_name.lower() else '30-45 cm'}\n\n"
        
        # Fertilization Schedule
        advice += "4. FERTILIZATION SCHEDULE:\n"
        advice += f"   - Basal application: Apply 50% N, 100% P2O5, 50% K2O before planting\n"
        advice += f"   - Top dressing: Apply remaining 50% N and K2O at {'tillering' if 'wheat' in crop_name.lower() else 'flowering'} stage\n"
        advice += f"   - Use balanced NPK fertilizer with micronutrients\n"
        advice += f"   - Consider organic supplements like compost or vermicompost\n\n"
        
        # Irrigation Guidance
        advice += "5. IRRIGATION GUIDANCE:\n"
        advice += f"   - Irrigate immediately after sowing for uniform germination\n"
        advice += f"   - Schedule irrigation at critical growth stages\n"
        advice += f"   - Maintain soil moisture at 60-70% field capacity\n"
        advice += f"   - Use efficient methods like drip or sprinkler irrigation\n\n"
        
        # Pest and Disease Management
        advice += "6. PEST AND DISEASE MANAGEMENT:\n"
        advice += f"   - Monitor regularly for common pests and diseases\n"
        advice += f"   - Practice crop rotation to break pest cycles\n"
        advice += f"   - Use resistant varieties when available\n"
        advice += f"   - Apply integrated pest management strategies\n\n"
        
        # Harvest Guidance
        advice += "7. HARVEST GUIDANCE:\n"
        advice += f"   - Harvest when {'grains turn golden brown' if 'wheat' in crop_name.lower() else 'fruits reach maturity'}\n"
        advice += f"   - Cut at appropriate height to minimize losses\n"
        advice += f"   - Dry properly before storage to prevent spoilage\n"
        advice += f"   - Store in clean, dry conditions with proper ventilation\n\n"
        
        return advice
    
    def _extract_planting_recommendations(self, crop_name):
        """
        Extract planting recommendations for a crop.
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            dict: Planting recommendations
        """
        recommendations = {
            "wheat": {
                "optimal_time": "October to November in northern regions, March to April in southern regions",
                "seed_rate": "100-125 kg/ha",
                "spacing": "20-25 cm row spacing, 3-5 cm planting depth",
                "seed_treatment": "Treat seeds with fungicide and biocontrol agents"
            },
            "rice": {
                "optimal_time": "June to July for kharif season, January to February for rabi season",
                "seed_rate": "20-25 kg/ha for transplanting, 80-100 kg/ha for direct seeding",
                "spacing": "20x20 cm for transplanting",
                "seed_treatment": "Treat seeds with carbendazim and biocontrol agents"
            },
            "maize": {
                "optimal_time": "June to July for kharif season, February to March for spring season",
                "seed_rate": "15-20 kg/ha",
                "spacing": "75 cm row spacing, 25 cm plant spacing, 5-7 cm depth",
                "seed_treatment": "Treat seeds with thiram and biocontrol agents"
            }
        }
        
        return recommendations.get(crop_name.lower(), {
            "optimal_time": "Based on local climate and variety characteristics",
            "seed_rate": "As recommended for specific variety",
            "spacing": "According to crop requirements and machinery",
            "seed_treatment": "Use certified, disease-free seeds with appropriate treatments"
        })
    
    def _extract_fertilization_schedule(self, crop_name):
        """
        Extract fertilization schedule for a crop.
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            dict: Fertilization schedule
        """
        schedules = {
            "wheat": {
                "basal": "Apply 50% N, 100% P2O5, 50% K2O before sowing",
                "top_dressing": "Apply remaining 50% N at crown root initiation stage",
                "micronutrients": "Apply zinc sulfate @ 25 kg/ha if deficient",
                "organic": "Incorporate 5-10 tons/ha of well-decomposed FYM"
            },
            "rice": {
                "basal": "Apply 50% N, 100% P2O5, 50% K2O before transplanting",
                "top_dressing": "Apply 25% N and 50% K2O at tillering, remaining 25% N at panicle initiation",
                "micronutrients": "Apply zinc sulfate @ 25 kg/ha if deficient",
                "organic": "Incorporate 5-10 tons/ha of well-decomposed FYM"
            },
            "maize": {
                "basal": "Apply 25% N, 100% P2O5, 50% K2O before sowing",
                "top_dressing": "Apply 25% N at knee-high stage, 50% N and 50% K2O at tasseling",
                "micronutrients": "Apply zinc sulfate @ 25 kg/ha if deficient",
                "organic": "Incorporate 5-10 tons/ha of well-decomposed FYM"
            }
        }
        
        return schedules.get(crop_name.lower(), {
            "basal": "Apply basal dose of NPK as per soil test recommendations",
            "top_dressing": "Apply top dressing based on crop growth stages",
            "micronutrients": "Apply micronutrients based on soil test results",
            "organic": "Incorporate organic manures for sustained nutrient release"
        })
    
    def _extract_irrigation_guidance(self, crop_name):
        """
        Extract irrigation guidance for a crop.
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            dict: Irrigation guidance
        """
        guidance = {
            "wheat": {
                "critical_stages": "Crown root initiation, tillering, jointing, booting, and grain filling",
                "frequency": "Irrigate at 15-20 day intervals depending on soil type and weather",
                "methods": "Furrow or sprinkler irrigation with 60-70 mm water per irrigation",
                "water_conservation": "Practice mulching and maintain proper land leveling"
            },
            "rice": {
                "critical_stages": "Active tillering, panicle initiation, and flowering",
                "frequency": "Continuous flooding until panicle initiation, then AWD (Alternate Wetting and Drying)",
                "methods": "Flooding with 5-10 cm water depth, AWD with 15 cm threshold",
                "water_conservation": "Implement AWD technique to save 15-20% water"
            },
            "maize": {
                "critical_stages": "Knee-high stage, tasseling, and grain filling",
                "frequency": "Irrigate at 10-15 day intervals depending on soil type and weather",
                "methods": "Furrow or drip irrigation with 60-70 mm water per irrigation",
                "water_conservation": "Practice mulching and use drip irrigation for efficiency"
            }
        }
        
        return guidance.get(crop_name.lower(), {
            "critical_stages": "Irrigate at critical growth stages when water stress affects yield",
            "frequency": "Based on soil type, weather conditions, and crop requirements",
            "methods": "Use efficient irrigation methods like drip, sprinkler, or furrow",
            "water_conservation": "Practice water-saving techniques and monitor soil moisture"
        })
    
    def _extract_pest_management_strategies(self, crop_name):
        """
        Extract pest management strategies for a crop.
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            dict: Pest management strategies
        """
        strategies = {
            "wheat": {
                "common_pests": ["Aphids", "Armyworms", "Hessian Fly"],
                "common_diseases": ["Rust", "Smut", "Blight"],
                "monitoring": "Scout fields weekly during growing season",
                "control_methods": "Use resistant varieties, practice crop rotation, apply IPM"
            },
            "rice": {
                "common_pests": ["Stem Borer", "Leaf Folder", "Brown Plant Hopper"],
                "common_diseases": ["Blast", "Sheath Blight", "Bacterial Leaf Blight"],
                "monitoring": "Monitor water levels and plant health regularly",
                "control_methods": "Use pheromone traps, beneficial insects, and selective pesticides"
            },
            "maize": {
                "common_pests": ["Stem Borer", "Fall Armyworm", "Aphids"],
                "common_diseases": ["Northern Corn Leaf Blight", "Rust", "Smut"],
                "monitoring": "Inspect plants during early morning and evening hours",
                "control_methods": "Use Bt varieties, crop rotation, and biological control agents"
            }
        }
        
        return strategies.get(crop_name.lower(), {
            "common_pests": ["Monitor for local pest species"],
            "common_diseases": ["Watch for typical diseases in your region"],
            "monitoring": "Regular field scouting and early detection",
            "control_methods": "Implement integrated pest management practices"
        })
    
    def _extract_harvest_guidance(self, crop_name):
        """
        Extract harvest guidance for a crop.
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            dict: Harvest guidance
        """
        guidance = {
            "wheat": {
                "timing": "When 80% of grains turn golden brown and hard",
                "method": "Cut with sickle or combine harvester at 10-15 cm stubble height",
                "post_harvest": "Dry grains to 12% moisture content before storage",
                "storage": "Store in clean, dry godowns with proper ventilation"
            },
            "rice": {
                "timing": "When 85% of grains turn golden yellow and hard",
                "method": "Cut with sickle or combine harvester at 15-20 cm stubble height",
                "post_harvest": "Dry paddy to 14% moisture content before storage",
                "storage": "Store in clean, dry godowns with proper pest control"
            },
            "maize": {
                "timing": "When husks dry and kernels become hard",
                "method": "Hand pick cobs or use mechanical harvester",
                "post_harvest": "Dry kernels to 13% moisture content before storage",
                "storage": "Store in moisture-proof containers with pest repellents"
            }
        }
        
        return guidance.get(crop_name.lower(), {
            "timing": "Harvest when crop reaches physiological maturity",
            "method": "Use appropriate harvesting equipment for crop type",
            "post_harvest": "Process and dry produce according to crop requirements",
            "storage": "Store in clean, dry conditions with proper pest management"
        })
    
    def get_fertilizer_advice(self, fertilizer_type, crop_name, application_method):
        """
        Get comprehensive fertilizer application advice with detailed nutrient management strategies.
        
        Args:
            fertilizer_type (str): Type of fertilizer
            crop_name (str): Name of the crop
            application_method (str): Method of application
            
        Returns:
            dict: Comprehensive fertilizer advice including application, timing, and management
        """
        try:
            if self.qa_chain is None:
                fallback_response = self._generate_fallback_fertilizer_advice(fertilizer_type, crop_name, application_method)
                return {
                    "fertilizer_type": fertilizer_type,
                    "crop_name": crop_name,
                    "comprehensive_advice": fallback_response,
                    "application_methods": self._extract_application_methods(fertilizer_type),
                    "timing_guidance": self._extract_timing_guidance(fertilizer_type, crop_name),
                    "dosage_recommendations": self._extract_dosage_recommendations(fertilizer_type, crop_name),
                    "safety_precautions": self._extract_safety_precautions(fertilizer_type)
                }
            
            # Create detailed prompt for comprehensive advice
            detailed_prompt = f"""
            You are an expert agricultural advisor. Provide comprehensive advice for applying {fertilizer_type} to {crop_name} using {application_method} method.
            
            Include the following sections in your response:
            1. Fertilizer Characteristics: Nutrient content, properties, and benefits
            2. Application Methods: Best techniques for {application_method}
            3. Timing Guidance: Optimal application stages for {crop_name}
            4. Dosage Recommendations: Appropriate rates based on soil conditions
            5. Placement Techniques: Best practices for nutrient availability
            6. Compatibility: Interactions with other inputs and soil conditions
            7. Safety Precautions: Handling, storage, and application safety
            8. Efficiency Tips: Methods to maximize nutrient use efficiency
            9. Environmental Considerations: Minimizing nutrient loss and environmental impact
            10. Troubleshooting: Common issues and solutions
            
            Format your response in clear sections with actionable, farm-ready advice.
            """
            
            result = self.qa_chain({"query": detailed_prompt})
            advice_text = result["result"]
            
            # Return structured response
            return {
                "fertilizer_type": fertilizer_type,
                "crop_name": crop_name,
                "comprehensive_advice": advice_text,
                "application_methods": self._extract_application_methods(fertilizer_type),
                "timing_guidance": self._extract_timing_guidance(fertilizer_type, crop_name),
                "dosage_recommendations": self._extract_dosage_recommendations(fertilizer_type, crop_name),
                "safety_precautions": self._extract_safety_precautions(fertilizer_type)
            }
        except Exception as e:
            print(f"Error in get_fertilizer_advice: {e}")
            fallback_response = self._generate_fallback_fertilizer_advice(fertilizer_type, crop_name, application_method)
            return {
                "fertilizer_type": fertilizer_type,
                "crop_name": crop_name,
                "comprehensive_advice": fallback_response,
                "application_methods": self._extract_application_methods(fertilizer_type),
                "timing_guidance": self._extract_timing_guidance(fertilizer_type, crop_name),
                "dosage_recommendations": self._extract_dosage_recommendations(fertilizer_type, crop_name),
                "safety_precautions": self._extract_safety_precautions(fertilizer_type)
            }
    
    def _generate_fallback_fertilizer_advice(self, fertilizer_type, crop_name, application_method):
        """
        Generate detailed fallback advice when LLM is not available.
        
        Args:
            fertilizer_type (str): Type of fertilizer
            crop_name (str): Name of the crop
            application_method (str): Method of application
            
        Returns:
            str: Detailed advice text
        """
        advice = f"Comprehensive Fertilizer Application Guide for {fertilizer_type}\n\n"
        advice += f"Crop: {crop_name}\n"
        advice += f"Application Method: {application_method}\n\n"
        
        # Fertilizer Characteristics
        advice += "1. FERTILIZER CHARACTERISTICS:\n"
        if "urea" in fertilizer_type.lower():
            advice += "   - Contains 46% nitrogen (N) - highest N content among solid fertilizers\n"
            advice += "   - Quick-release nitrogen source for rapid plant uptake\n"
            advice += "   - Prone to volatilization loss if not incorporated properly\n\n"
        elif "dap" in fertilizer_type.lower():
            advice += "   - Contains 18% nitrogen (N) and 46% phosphorus (P2O5)\n"
            advice += "   - Excellent for root development and early plant growth\n"
            advice += "   - Slightly alkaline and may raise soil pH over time\n\n"
        elif "mop" in fertilizer_type.lower():
            advice += "   - Contains 60% potassium (K2O)\n"
            advice += "   - Improves disease resistance and water use efficiency\n"
            advice += "   - Helps in fruit and grain filling processes\n\n"
        else:
            advice += "   - Balanced nutrient content for general plant nutrition\n"
            advice += "   - Suitable for maintenance and corrective fertilization\n"
            advice += "   - Follow manufacturer's specifications for nutrient content\n\n"
        
        # Application Methods
        advice += "2. APPLICATION METHODS:\n"
        if application_method.lower() == "basal":
            advice += "   - Apply before planting or transplanting\n"
            advice += "   - Incorporate into soil during land preparation\n"
            advice += "   - Ensure uniform distribution for consistent results\n\n"
        elif application_method.lower() == "top dressing":
            advice += "   - Apply during active growth stages\n"
            advice += "   - Place in bands or rings around plants\n"
            advice += "   - Incorporate lightly into soil or irrigate immediately\n\n"
        elif application_method.lower() == "foliar":
            advice += "   - Apply as foliar spray during cooler parts of day\n"
            advice += "   - Use recommended concentrations to avoid leaf burn\n"
            advice += "   - Effective for quick correction of nutrient deficiencies\n\n"
        else:
            advice += "   - Follow recommended practices for specific application method\n"
            advice += "   - Ensure proper timing and placement for best results\n"
            advice += "   - Consider crop growth stage and nutrient requirements\n\n"
        
        # Timing Guidance
        advice += "3. TIMING GUIDANCE:\n"
        advice += f"   - For {crop_name}: Apply at critical nutrient demand stages\n"
        advice += "   - Split applications for efficient nutrient use\n"
        advice += "   - Avoid application during extreme weather conditions\n"
        advice += "   - Time with irrigation for better nutrient dissolution\n\n"
        
        # Safety Precautions
        advice += "4. SAFETY PRECAUTIONS:\n"
        advice += "   - Wear protective gloves, mask, and clothing\n"
        advice += "   - Store in dry, well-ventilated area away from children\n"
        advice += "   - Avoid direct contact with skin and eyes\n"
        advice += "   - Do not mix incompatible fertilizers together\n\n"
        
        return advice
    
    def _extract_application_methods(self, fertilizer_type):
        """
        Extract application methods for a fertilizer type.
        
        Args:
            fertilizer_type (str): Type of fertilizer
            
        Returns:
            dict: Application methods
        """
        methods = {
            "urea": {
                "basal": "Apply before planting and incorporate immediately",
                "top_dressing": "Apply during active growth stages and irrigate",
                "fertigation": "Dissolve in irrigation water for efficient application",
                "foliar": "Use 1-2% solution for foliar spray during cooler hours"
            },
            "dap": {
                "basal": "Apply at sowing or transplanting time in bands",
                "top_dressing": "Apply at early growth stages away from seeds",
                "fertigation": "Dissolve in irrigation water with proper pH adjustment",
                "seed_treatment": "Mix with seeds for early nutrient availability"
            },
            "mop": {
                "basal": "Apply before planting or transplanting",
                "top_dressing": "Apply during reproductive stages for fruit/grain development",
                "fertigation": "Dissolve in irrigation water for continuous supply",
                "side_dressing": "Place in furrows alongside plants"
            }
        }
        
        return methods.get(fertilizer_type.lower(), {
            "basal": "Apply before planting according to crop requirements",
            "top_dressing": "Apply during active growth stages as needed",
            "fertigation": "Dissolve in irrigation water for efficient delivery",
            "specific": "Follow manufacturer's recommendations for specific application"
        })
    
    def _extract_timing_guidance(self, fertilizer_type, crop_name):
        """
        Extract timing guidance for fertilizer application.
        
        Args:
            fertilizer_type (str): Type of fertilizer
            crop_name (str): Name of the crop
            
        Returns:
            dict: Timing guidance
        """
        timing = {
            ("urea", "wheat"): {
                "basal": "At sowing time",
                "top_dressing": "At crown root initiation and jointing stages",
                "split": "50% at sowing, 50% at crown root initiation"
            },
            ("dap", "wheat"): {
                "basal": "At sowing time",
                "top_dressing": "Not typically required for DAP",
                "split": "100% as basal application"
            },
            ("urea", "rice"): {
                "basal": "Before transplanting",
                "top_dressing": "At tillering and panicle initiation stages",
                "split": "50% at basal, 25% at tillering, 25% at panicle initiation"
            },
            ("dap", "rice"): {
                "basal": "Before transplanting",
                "top_dressing": "At active tillering stage if needed",
                "split": "100% at basal or 50% at basal, 50% at tillering"
            }
        }
        
        return timing.get((fertilizer_type.lower(), crop_name.lower()), {
            "basal": "At planting or early growth stage",
            "top_dressing": "During active growth or reproductive stages",
            "split": "Based on crop nutrient demand pattern"
        })
    
    def _extract_dosage_recommendations(self, fertilizer_type, crop_name):
        """
        Extract dosage recommendations for fertilizer application.
        
        Args:
            fertilizer_type (str): Type of fertilizer
            crop_name (str): Name of the crop
            
        Returns:
            dict: Dosage recommendations
        """
        dosage = {
            ("urea", "wheat"): {
                "low_yield": "80-100 kg N/ha",
                "medium_yield": "100-120 kg N/ha",
                "high_yield": "120-150 kg N/ha"
            },
            ("dap", "wheat"): {
                "general": "40-60 kg P2O5/ha"
            },
            ("mop", "wheat"): {
                "general": "40-60 kg K2O/ha"
            },
            ("urea", "rice"): {
                "low_yield": "100-120 kg N/ha",
                "medium_yield": "120-150 kg N/ha",
                "high_yield": "150-180 kg N/ha"
            },
            ("dap", "rice"): {
                "general": "40-60 kg P2O5/ha"
            }
        }
        
        return dosage.get((fertilizer_type.lower(), crop_name.lower()), {
            "low_yield": "Based on soil test and target yield",
            "medium_yield": "According to crop requirements and soil fertility",
            "high_yield": "Higher rates with split applications for efficiency"
        })
    
    def _extract_safety_precautions(self, fertilizer_type):
        """
        Extract safety precautions for fertilizer handling.
        
        Args:
            fertilizer_type (str): Type of fertilizer
            
        Returns:
            list: Safety precautions
        """
        precautions = {
            "urea": [
                "Incorporate immediately after application to prevent nitrogen loss",
                "Avoid application during hot, windy conditions",
                "Store in dry place to prevent caking",
                "Wear mask during handling to avoid inhalation"
            ],
            "dap": [
                "Avoid direct contact with seeds to prevent germination injury",
                "Store separately from alkaline materials",
                "Do not mix with urea in same solution",
                "Wear gloves to prevent skin irritation"
            ],
            "mop": [
                "Avoid application on chloride-sensitive crops like tobacco",
                "Store in dry conditions to prevent lumping",
                "Wear protective equipment during handling",
                "Keep away from children and animals"
            ]
        }
        
        return precautions.get(fertilizer_type.lower(), [
            "Follow manufacturer's safety guidelines",
            "Store in cool, dry place away from incompatible materials",
            "Wear appropriate personal protective equipment",
            "Wash hands and exposed skin after handling"
        ])
    
    def get_general_advice(self, question):
        """
        Get general agricultural advice.
        
        Args:
            question (str): Question about agriculture
            
        Returns:
            str: General advice
        """
        try:
            if self.qa_chain is None:
                return "RAG system not available. General agricultural advice not available at this time."
            
            result = self.qa_chain({"query": question})
            return result["result"]
        except Exception as e:
            print(f"Error in get_general_advice: {e}")
            return "Unable to generate general advice at this time. Please try again later."


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = RecommendationRAG()
    
    # Get crop advice
    crop_advice = rag.get_crop_advice(
        "wheat",
        "N:80, P:45, K:40, pH:6.5",
        "Temperature:22°C, Humidity:65%, Rainfall:120mm"
    )
    print("Crop Advice:", crop_advice)
    
    # Get fertilizer advice
    fertilizer_advice = rag.get_fertilizer_advice(
        "urea",
        "rice",
        "basal application"
    )
    print("Fertilizer Advice:", fertilizer_advice)