# Enhanced Agricultural Knowledge Bases

This directory contains enhanced knowledge bases for various agricultural domains including crop yield prediction, plant disease management, and crop recommendations.

## Directory Structure

```
knowledge_base/
├── __init__.py
├── crop_yield_knowledge_base.py
├── plant_disease_knowledge_base.py
├── crop_recommendation_knowledge_base.py
├── unified_knowledge_base.py
├── data_population_script.py
├── test_knowledge_bases.py
└── README.md
```

## Components

### 1. Crop Yield Knowledge Base
- Enhanced yield prediction based on environmental factors
- Analysis of crop-specific growing conditions
- Integration with existing crop yield datasets

### 2. Plant Disease Knowledge Base
- Comprehensive information on plant diseases
- Symptom identification and management strategies
- Integration with plant disease datasets

### 3. Crop Recommendation Knowledge Base
- Crop and fertilizer recommendations
- Soil and weather condition analysis
- Integration with crop recommendation datasets

### 4. Unified Knowledge Base
- Single interface to access all agricultural knowledge
- Cross-domain querying capabilities
- Simplified API for integration

## Usage

### Initializing the Knowledge Bases

```python
from explainable_ai.knowledge_base.unified_knowledge_base import UnifiedAgriculturalKB

# Initialize the unified knowledge base
kb = UnifiedAgriculturalKB()
```

### Querying the Knowledge Bases

```python
# Query across all domains
results = kb.query_knowledge_base("How to improve wheat yield?", k=3)

# Query a specific domain
results = kb.query_knowledge_base("Tomato disease symptoms", domain="disease", k=2)
```

### Saving and Loading Knowledge Bases

```python
# Save all knowledge bases
kb.save_all_knowledge_bases()

# Load all knowledge bases
kb.load_all_knowledge_bases()
```

## Data Population

The knowledge bases are automatically populated with information from the existing datasets in the project. To manually populate with additional information, run:

```bash
python explainable_ai/knowledge_base/data_population_script.py
```

## Testing

To test the functionality of the knowledge bases, run:

```bash
python explainable_ai/knowledge_base/test_knowledge_bases.py
```

## Integration with Main RAG System

The enhanced knowledge bases are automatically integrated with the main RAG system in `explainable_ai/rag_system.py`. The system will use the enhanced knowledge bases if they are available, falling back to the basic knowledge base if needed.