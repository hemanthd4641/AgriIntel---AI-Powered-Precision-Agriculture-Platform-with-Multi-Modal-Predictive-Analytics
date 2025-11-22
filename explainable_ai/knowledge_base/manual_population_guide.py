"""
Manual Population Guide for Agricultural Knowledge Bases

This script demonstrates how to manually populate the vector databases with 
comprehensive agricultural information for crop yield, plant diseases, 
crop recommendations, weeds, and pests.
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from explainable_ai.knowledge_base.unified_knowledge_base import UnifiedAgriculturalKB

def populate_with_crop_yield_data(kb):
    """
    Manually populate the knowledge base with additional crop yield information
    """
    print("Populating with crop yield data...")
    
    # Example yield prediction records
    yield_records = [
        {
            "prediction_data": {
                "Region": "North",
                "Soil_Type": "Loam",
                "Crop": "Wheat",
                "Rainfall_mm": 450.0,
                "Temperature_Celsius": 22.0,
                "Fertilizer_Used": True,
                "Irrigation_Used": False,
                "Weather_Condition": "Sunny",
                "Days_to_Harvest": 125
            },
            "actual_yield": 6.8
        },
        {
            "prediction_data": {
                "Region": "South",
                "Soil_Type": "Clay",
                "Crop": "Rice",
                "Rainfall_mm": 1100.0,
                "Temperature_Celsius": 28.0,
                "Fertilizer_Used": True,
                "Irrigation_Used": True,
                "Weather_Condition": "Rainy",
                "Days_to_Harvest": 135
            },
            "actual_yield": 7.2
        },
        {
            "prediction_data": {
                "Region": "East",
                "Soil_Type": "Sandy",
                "Crop": "Maize",
                "Rainfall_mm": 800.0,
                "Temperature_Celsius": 25.0,
                "Fertilizer_Used": True,
                "Irrigation_Used": True,
                "Weather_Condition": "Cloudy",
                "Days_to_Harvest": 110
            },
            "actual_yield": 5.5
        }
    ]
    
    for record in yield_records:
        kb.add_domain_specific_information(
            'yield',
            prediction_data=record['prediction_data'],
            actual_yield=record['actual_yield']
        )

def populate_with_plant_disease_data(kb):
    """
    Manually populate the knowledge base with additional plant disease information
    """
    print("Populating with plant disease data...")
    
    # Additional disease information
    diseases = [
        {
            "disease_name": "Powdery Mildew",
            "symptoms": "White, powdery fungal growth on leaf surfaces, stems, and flowers. Leaves may yellow and drop prematurely.",
            "management_strategies": "Improve air circulation through proper spacing. Apply fungicides containing sulfur or potassium bicarbonate. Remove and destroy infected plant parts. Avoid overhead irrigation."
        },
        {
            "disease_name": "Fusarium Wilt",
            "symptoms": "Yellowing and wilting of lower leaves, often on one side of the plant. Vascular tissue shows brown discoloration.",
            "management_strategies": "Plant resistant varieties. Practice crop rotation with non-host crops. Soil solarization to reduce pathogen levels. Avoid overwatering and excessive nitrogen fertilization."
        },
        {
            "disease_name": "Downy Mildew",
            "symptoms": "Yellow spots on upper leaf surfaces with fuzzy, grayish growth on undersides. Common in cool, moist conditions.",
            "management_strategies": "Improve air circulation and reduce humidity. Apply fungicides preventatively. Remove infected plant material. Avoid overhead watering."
        }
    ]
    
    for disease in diseases:
        kb.add_domain_specific_information(
            'disease',
            disease_name=disease['disease_name'],
            symptoms=disease['symptoms'],
            management_strategies=disease['management_strategies']
        )

def populate_with_crop_recommendation_data(kb):
    """
    Manually populate the knowledge base with additional crop recommendation information
    """
    print("Populating with crop recommendation data...")
    
    # Additional crop recommendations
    recommendations = [
        {
            "crop_name": "Soybean",
            "soil_conditions": "Well-drained loam soil with pH 6.0-7.0, moderate organic matter",
            "weather_conditions": "Warm season crop requiring 20-28°C temperature, 600-800mm rainfall",
            "fertilizer_recommendation": "Inoculate with rhizobia bacteria for nitrogen fixation. Apply phosphorus and potassium based on soil test. Supplemental nitrogen may be needed on sandy soils."
        },
        {
            "crop_name": "Barley",
            "soil_conditions": "Well-drained soil with pH 6.0-7.5, good fertility",
            "weather_conditions": "Cool season crop tolerating frost, 300-500mm rainfall",
            "fertilizer_recommendation": "Apply nitrogen in split applications. Phosphorus and potassium before planting. Consider sulfur and micronutrient needs based on soil tests."
        },
        {
            "crop_name": "Sugarcane",
            "soil_conditions": "Deep, well-drained soil with pH 6.5-7.5, high organic matter",
            "weather_conditions": "Tropical crop requiring 25-35°C temperature, 1200-1500mm rainfall",
            "fertilizer_recommendation": "Apply organic manure before planting. Use NPK fertilizers in splits. Consider micronutrients like zinc and boron."
        }
    ]
    
    for rec in recommendations:
        kb.add_domain_specific_information(
            'recommendation',
            crop_name=rec['crop_name'],
            soil_conditions=rec['soil_conditions'],
            weather_conditions=rec['weather_conditions'],
            fertilizer_recommendation=rec['fertilizer_recommendation']
        )

def populate_with_pest_weed_data(kb):
    """
    Manually populate the knowledge base with additional pest and weed information
    """
    print("Populating with pest and weed data...")
    
    # Additional pest information
    pests_weeds = [
        {
            "category_name": "Colorado Potato Beetle",
            "description": "Large, oval beetle with yellowish-orange body and black stripes. Larvae are red with black heads and spots.",
            "management_strategies": "Crop rotation to disrupt life cycle. Hand-picking adults and larvae. Beneficial insects like ladybugs and parasitic wasps. Neem oil or spinosad for organic control. Chemical insecticides when thresholds exceeded."
        },
        {
            "category_name": "Johnsongrass",
            "description": "Perennial grass weed with long rhizomes. Stems are thick and solid with white midrib on leaves.",
            "management_strategies": "Prevent seed production through mowing. Tillage to fragment rhizomes (can spread if not done properly). Herbicides specific to grassy weeds. Competitive crop planting to suppress growth."
        },
        {
            "category_name": "Codling Moth",
            "description": "Small grayish moth whose larvae tunnel into fruit, especially apples and pears.",
            "management_strategies": "Pheromone traps for monitoring. Sanitation by removing fallen fruit. Beneficial insects like parasitic wasps. Timing of insecticide applications based on degree-day models."
        }
    ]
    
    for pw in pests_weeds:
        kb.add_domain_specific_information(
            'pest_weed',
            category_name=pw['category_name'],
            description=pw['description'],
            management_strategies=pw['management_strategies']
        )

def main():
    """
    Main function to demonstrate manual population of knowledge bases
    """
    print("Manual Population Guide for Agricultural Knowledge Bases")
    print("=" * 55)
    
    # Initialize the unified knowledge base
    print("Initializing unified agricultural knowledge base...")
    kb = UnifiedAgriculturalKB()
    
    # Populate each domain with additional data
    populate_with_crop_yield_data(kb)
    populate_with_plant_disease_data(kb)
    populate_with_crop_recommendation_data(kb)
    populate_with_pest_weed_data(kb)
    
    # Save all knowledge bases
    print("\nSaving knowledge bases...")
    kb.save_all_knowledge_bases()
    
    print("\nKnowledge bases populated and saved successfully!")
    
    # Demonstrate querying the populated knowledge base
    print("\nDemonstrating queries on populated knowledge base:")
    print("-" * 50)
    
    queries = [
        "How to manage powdery mildew on crops?",
        "What are the best conditions for growing soybeans?",
        "How to control Colorado potato beetle?",
        "What factors affect wheat yield?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = kb.query_knowledge_base(query, k=1)
        if results:
            result = results[0]
            print(f"  Domain: {result['domain']}")
            print(f"  Title: {result['metadata']['title']}")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Content: {result['content'][:200]}...")
        else:
            print("  No results found")

if __name__ == "__main__":
    main()