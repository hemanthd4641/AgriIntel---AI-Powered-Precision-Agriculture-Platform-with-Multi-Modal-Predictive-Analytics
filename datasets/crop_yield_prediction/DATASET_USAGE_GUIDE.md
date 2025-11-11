# Crop Yield Prediction Dataset Usage Guide

## Dataset Organization

Your crop yield prediction dataset should be organized in the following structure:

```
datasets/crop_yield_prediction/
├── historical_yield/
│   ├── historical_yield_data.csv
│   └── (other historical yield data files)
├── weather_data/
│   ├── weather_data.csv
│   └── (other weather data files)
├── soil_composition/
│   ├── soil_data.csv
│   └── (other soil data files)
├── farming_practices/
│   ├── farming_practices.csv
│   └── (other farming practices data files)
├── satellite_imagery/
│   ├── sentinel2/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── ndvi/
│   ├── landsat8/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── ndvi/
│   └── metadata/
└── README.md
```

## Data Types and Expected Columns

### Historical Yield Data
- **File location**: `datasets/crop_yield_prediction/historical_yield/`
- **Expected columns**:
  - `year`: The year of the yield record
  - `crop_type`: Type of crop (e.g., wheat, corn, rice)
  - `location`: Field or region identifier
  - `yield_tonnes_per_hectare`: Actual yield in tonnes per hectare

### Weather Data
- **File location**: `datasets/crop_yield_prediction/weather_data/`
- **Expected columns**:
  - `date`: Date of the weather record
  - `location`: Field or region identifier
  - `temperature_celsius`: Average temperature in Celsius
  - `rainfall_mm`: Rainfall in millimeters
  - `humidity_percent`: Humidity percentage
  - `sunlight_hours`: Hours of sunlight

### Soil Composition Data
- **File location**: `datasets/crop_yield_prediction/soil_composition/`
- **Expected columns**:
  - `location`: Field or region identifier
  - `ph_level`: Soil pH level
  - `nitrogen_ppm`: Nitrogen content in parts per million
  - `phosphorus_ppm`: Phosphorus content in parts per million
  - `potassium_ppm`: Potassium content in parts per million
  - `organic_matter_percent`: Organic matter percentage

### Farming Practices Data
- **File location**: `datasets/crop_yield_prediction/farming_practices/`
- **Expected columns**:
  - `season`: Growing season identifier
  - `crop_type`: Type of crop
  - `fertilizer_type`: Type of fertilizer used
  - `fertilizer_amount_kg_per_hectare`: Fertilizer amount in kg/hectare
  - `irrigation_method`: Irrigation method used (e.g., sprinkler, drip, flood)
  - `pesticide_used`: Whether pesticides were used (yes/no)

### Satellite Imagery Data
- **File location**: `datasets/crop_yield_prediction/satellite_imagery/`
- **Directory structure**:
  - `sentinel2/` - Sentinel-2 satellite imagery
  - `landsat8/` - Landsat-8 satellite imagery
  - `metadata/` - Image metadata files
- **Subdirectories**:
  - `raw/` - Unprocessed satellite imagery files
  - `processed/` - Preprocessed imagery (cloud removal, resizing, normalization)
  - `ndvi/` - NDVI calculated images
- **File formats**:
  - Raw imagery: GeoTIFF format
  - Processed imagery: GeoTIFF format
  - NDVI: GeoTIFF format
  - Metadata: JSON format

## Loading Data in Your Application

You can load the crop yield prediction data using the data loader utilities:

```python
from preprocessing.data_loader import load_crop_yield_data_by_type

# Load historical yield data
historical_data = load_crop_yield_data_by_type('historical_yield')

# Load weather data
weather_data = load_crop_yield_data_by_type('weather_data')

# Load soil composition data
soil_data = load_crop_yield_data_by_type('soil_composition')

# Load farming practices data
farming_data = load_crop_yield_data_by_type('farming_practices')

# Load satellite imagery data
satellite_data = load_satellite_imagery_data()
```

## Adding New Data

To add your own crop yield prediction data:

1. Place CSV files in the appropriate subdirectories:
   - Historical yield data → `datasets/crop_yield_prediction/historical_yield/`
   - Weather data → `datasets/crop_yield_prediction/weather_data/`
   - Soil composition data → `datasets/crop_yield_prediction/soil_composition/`
   - Farming practices data → `datasets/crop_yield_prediction/farming_practices/`
   - Satellite imagery → `datasets/crop_yield_prediction/satellite_imagery/`

2. Ensure your CSV files have the expected columns as listed above

3. For satellite imagery, organize files in the appropriate subdirectories (sentinel2/landsat8 → raw/processed/ndvi)

4. The data loader will automatically load the first CSV file it finds in each directory

## Data Quality Guidelines

- Ensure consistent naming for locations across all datasets
- Use consistent date formats (YYYY-MM-DD recommended)
- Handle missing values appropriately (use NaN or empty cells)
- Validate data ranges (e.g., pH should be between 0-14)
- Include units in column names where appropriate