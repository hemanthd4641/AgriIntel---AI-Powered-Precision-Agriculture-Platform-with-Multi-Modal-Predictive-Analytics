// Crop Yield Prediction JavaScript - Enhanced version with detailed insights

console.log('Crop Yield Prediction module loaded');

// API base URL (in a real app, this would be your Django backend)
const API_BASE_URL = 'http://127.0.0.1:8000/api';

// Initialize the predictions feature when the section is shown
function initPredictions() {
    console.log('Initializing Crop Yield Predictions feature');
    
    // Get DOM elements for predictions
    const predictionForm = document.getElementById('prediction-form');
    const predictionResults = document.getElementById('prediction-results');
    const predictionOutput = document.getElementById('prediction-output');
    
    // Set up event listeners for predictions
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            e.stopPropagation(); // Prevent event from bubbling up
            predictYield();
            return false; // Prevent default form submission
        });
    }
}

// Predict crop yield from manual data input
async function predictYield() {
    console.log('Predict Yield function called');
    
    // Get DOM elements
    const predictionResults = document.getElementById('prediction-results');
    const predictionOutput = document.getElementById('prediction-output');
    
    // Get form values
    const region = document.getElementById('region').value;
    const soilType = document.getElementById('soil-type').value;
    const crop = document.getElementById('crop').value;
    const rainfallMm = document.getElementById('rainfall-mm').value;
    const temperatureCelsius = document.getElementById('temperature-celsius').value;
    const fertilizerUsed = document.getElementById('fertilizer-used').value;
    const irrigationUsed = document.getElementById('irrigation-used').value;
    const weatherCondition = document.getElementById('weather-condition').value;
    const daysToHarvest = document.getElementById('days-to-harvest').value;
    
    console.log('Form values:', {
        region, soilType, crop, rainfallMm, temperatureCelsius, 
        fertilizerUsed, irrigationUsed, weatherCondition, daysToHarvest
    });
    
    // Validate required fields
    if (!region || !soilType || !crop || !rainfallMm || !temperatureCelsius || 
        !fertilizerUsed || !irrigationUsed || !weatherCondition || !daysToHarvest) {
        alert('Please fill in all fields');
        return;
    }
    
    // Show loading state
    showPredictionLoading();
    
    // Create data object for API request
    const requestData = {
        region: region,
        soil_type: soilType,
        crop: crop,
        rainfall_mm: parseFloat(rainfallMm),
        temperature_celsius: parseFloat(temperatureCelsius),
        fertilizer_used: fertilizerUsed === 'true',
        irrigation_used: irrigationUsed === 'true',
        weather_condition: weatherCondition,
        days_to_harvest: parseInt(daysToHarvest)
    };
    
    console.log('Request data:', requestData);
    
    try {
        // Send request to backend
        console.log('Sending request to:', `${API_BASE_URL}/predict-yield/`);
        const response = await fetch(`${API_BASE_URL}/predict-yield/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify(requestData)
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', [...response.headers.entries()]);
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (response.ok) {
            // Display results
            displayYieldResults(data);
        } else {
            showPredictionError(data.error || 'Unknown error occurred');
            if (data.missing_fields) {
                showPredictionError(`Missing fields: ${data.missing_fields.join(', ')}`);
            }
        }
    } catch (error) {
        console.error('Error predicting yield:', error);
        showPredictionError(`Error predicting yield: ${error.message}. Please try again.`);
    }
}

// Show loading state for predictions
function showPredictionLoading() {
    const predictionOutput = document.getElementById('prediction-output');
    const predictionResults = document.getElementById('prediction-results');
    
    if (predictionOutput) {
        predictionOutput.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Predicting crop yield based on provided conditions...</p>
                <p>Consulting knowledge base for enhanced insights...</p>
            </div>
        `;
    }
    if (predictionResults) {
        predictionResults.style.display = 'block';
    }
}

// Display yield prediction results with enhanced formatting
function displayYieldResults(data) {
    console.log('Displaying results:', data);
    
    // Get DOM elements
    const predictionOutput = document.getElementById('prediction-output');
    const predictionResults = document.getElementById('prediction-results');
    
    // Format the explanation to be more readable
    let explanationHtml = '';
    if (data.explanation) {
        explanationHtml = formatExplanation(data.explanation);
    }
    
    // Create enhanced results HTML
    let html = `
        <div class="prediction-result">
            <h4>Yield Prediction for ${data.input_data.Crop}</h4>
            <div class="result-grid">
                <div class="result-item">
                    <span class="result-label">Predicted Yield:</span>
                    <span class="result-value">${data.predicted_yield} tons/hectare</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Confidence:</span>
                    <span class="result-value">${(data.confidence_score * 100).toFixed(1)}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Prediction Date:</span>
                    <span class="result-value">${new Date(data.timestamp).toLocaleString()}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Days to Harvest:</span>
                    <span class="result-value">${data.input_data.Days_to_Harvest} days</span>
                </div>
            </div>
    `;
    
    // Add explanation section
    if (explanationHtml) {
        html += `
            <div class="explanation-section">
                <h5>Prediction Analysis</h5>
                ${explanationHtml}
            </div>
        `;
    }
    
    // Add enhanced insights section
    html += `
        <div class="insights-section">
            <h5>AI-Powered Yield Insights</h5>
            <div class="insights-tabs">
                <button class="tab-button active" onclick="switchYieldTab('yield-overview')">Overview</button>
                <button class="tab-button" onclick="switchYieldTab('yield-analysis')">Analysis</button>
                <button class="tab-button" onclick="switchYieldTab('yield-recommendations')">Recommendations</button>
            </div>
            <div class="tab-content">
                <div id="yield-overview" class="tab-pane active">
                    <div class="insight-card">
                        <h6>Prediction Summary</h6>
                        <p>${data.insights?.prediction_summary || `Based on the provided conditions, the predicted yield for ${data.input_data.Crop} is ${data.predicted_yield} tons per hectare.`}</p>
                        <div class="risk-assessment">
                            <span>Risk Level: ${data.insights?.risk_assessment || 'Moderate'}</span>
                            <span>Confidence: ${(data.confidence_score * 100).toFixed(1)}%</span>
                        </div>
                    </div>
    `;
    
    // Add key factors if available
    if (data.insights?.key_factors && data.insights.key_factors.length > 0) {
        html += `
                    <div class="insight-card">
                        <h6>Key Yield Factors</h6>
                        <ul>
        `;
        data.insights.key_factors.forEach(factor => {
            html += `<li>${factor}</li>`;
        });
        html += `
                        </ul>
                    </div>
        `;
    }
    
    html += `
                </div>
                <div id="yield-analysis" class="tab-pane">
    `;
    
    // Add detailed analysis if available
    if (data.insights?.llm_explanation) {
        html += `
                    <div class="insight-card">
                        <h6>Comprehensive Yield Analysis</h6>
                        <div class="analysis-content">${formatExplanation(data.insights.llm_explanation)}</div>
                    </div>
        `;
    } else {
        html += `
                    <div class="insight-card">
                        <h6>Environmental Impact Analysis</h6>
                        <ul>
                            <li>Temperature of ${data.input_data.Temperature_Celsius}°C is ${getTemperatureImpact(data.input_data.Temperature_Celsius)}</li>
                            <li>Rainfall of ${data.input_data.Rainfall_mm}mm is ${getRainfallImpact(data.input_data.Rainfall_mm)}</li>
                            <li>Soil type ${data.input_data.Soil_Type} provides ${getSoilImpact(data.input_data.Soil_Type)}</li>
                            <li>${data.input_data.Fertilizer_Used ? 'Fertilizer use' : 'No fertilizer use'} will impact nutrient availability</li>
                            <li>${data.input_data.Irrigation_Used ? 'Irrigation use' : 'No irrigation use'} affects water availability</li>
                        </ul>
                    </div>
        `;
    }
    
    html += `
                </div>
                <div id="yield-recommendations" class="tab-pane">
    `;
    
    // Add recommendations if available
    if (data.insights?.llm_recommendations) {
        html += `
                    <div class="insight-card">
                        <h6>AI-Powered Recommendations</h6>
                        <div class="recommendations-content">${formatExplanation(data.insights.llm_recommendations)}</div>
                    </div>
        `;
    } else if (data.insights?.recommendations && data.insights.recommendations.length > 0) {
        html += `
                    <div class="insight-card">
                        <h6>Yield Optimization Recommendations</h6>
                        <ul>
        `;
        data.insights.recommendations.forEach(rec => {
            html += `<li>${rec}</li>`;
        });
        html += `
                        </ul>
                    </div>
        `;
    } else {
        html += `
                    <div class="insight-card">
                        <h6>General Recommendations</h6>
                        <ul>
                            <li>Maintain consistent watering schedule</li>
                            <li>Monitor for pest and disease pressure</li>
                            <li>Apply appropriate fertilizers based on soil test results</li>
                            <li>Harvest at optimal maturity stage</li>
                        </ul>
                    </div>
        `;
    }
    
    html += `
                </div>
            </div>
        </div>
    `;
    
    // Add input data section
    html += `
        <div class="input-data-section">
            <h5>Input Data Used</h5>
            <div class="data-grid">
                <div class="data-item"><strong>Region:</strong> ${data.input_data.Region}</div>
                <div class="data-item"><strong>Soil Type:</strong> ${data.input_data.Soil_Type}</div>
                <div class="data-item"><strong>Crop:</strong> ${data.input_data.Crop}</div>
                <div class="data-item"><strong>Rainfall:</strong> ${data.input_data.Rainfall_mm} mm</div>
                <div class="data-item"><strong>Temperature:</strong> ${data.input_data.Temperature_Celsius}°C</div>
                <div class="data-item"><strong>Fertilizer Used:</strong> ${data.input_data.Fertilizer_Used ? 'Yes' : 'No'}</div>
                <div class="data-item"><strong>Irrigation Used:</strong> ${data.input_data.Irrigation_Used ? 'Yes' : 'No'}</div>
                <div class="data-item"><strong>Weather:</strong> ${data.input_data.Weather_Condition}</div>
                <div class="data-item"><strong>Days to Harvest:</strong> ${data.input_data.Days_to_Harvest}</div>
            </div>
        </div>
    `;
    
    html += `
        </div>
    `;
    
    // Ensure the results section is visible
    if (predictionResults) {
        predictionResults.style.display = 'block';
    }
    if (predictionOutput) {
        predictionOutput.innerHTML = html;
    }
    
    // Scroll to results
    if (predictionResults) {
        predictionResults.scrollIntoView({ behavior: 'smooth' });
    }
}

// Switch between tabs in the insights section
window.switchYieldTab = function(tabName) {
    // Hide all tab panes
    const tabPanes = document.querySelectorAll('.tab-pane');
    tabPanes.forEach(pane => pane.classList.remove('active'));
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => button.classList.remove('active'));
    
    // Show the selected tab pane
    const activePane = document.getElementById(tabName);
    if (activePane) {
        activePane.classList.add('active');
    }
    
    // Add active class to the clicked button
    event.currentTarget.classList.add('active');
};

// Format explanation content for better display
function formatExplanation(content) {
    if (!content) return '';
    return content
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/<br><br>/g, '</p><p>')
        .replace(/^<br>/, '')
        .replace(/<br>$/, '');
}

// Get temperature impact description
function getTemperatureImpact(temp) {
    if (temp < 5) return 'extremely cold - severe frost damage risk';
    if (temp < 10) return 'very cold - limited crop growth';
    if (temp < 15) return 'cold - slow growth for most crops';
    if (temp < 20) return 'cool - suitable for cool-season crops';
    if (temp < 25) return 'optimal - ideal for most crops';
    if (temp < 30) return 'warm - suitable for heat-tolerant crops';
    if (temp < 35) return 'hot - stress on many crops';
    return 'extremely hot - severe heat stress';
}

// Get rainfall impact description
function getRainfallImpact(rainfall) {
    if (rainfall < 20) return 'severely deficient - drought conditions';
    if (rainfall < 50) return 'deficient - supplemental irrigation needed';
    if (rainfall < 100) return 'adequate for most crops';
    if (rainfall < 150) return 'good - supports healthy growth';
    if (rainfall < 200) return 'excellent - optimal for high-yield crops';
    return 'excessive - risk of waterlogging';
}

// Get soil impact description
function getSoilImpact(soilType) {
    const impacts = {
        'Clay': 'high water retention but poor drainage - suitable for drought-tolerant crops',
        'Sandy': 'excellent drainage but low nutrient retention - needs frequent fertilization',
        'Loam': 'ideal balance of drainage and nutrient retention - optimal for most crops',
        'Silt': 'moderate water retention and fertility - good for many crops',
        'Peat': 'high organic matter but acidic - needs pH adjustment',
        'Chalky': 'alkaline with good drainage but may lack nutrients - needs nutrient supplementation'
    };
    return impacts[soilType] || 'suitable for crop growth with proper management';
}

// Show error message for predictions
function showPredictionError(message) {
    const predictionOutput = document.getElementById('prediction-output');
    const predictionResults = document.getElementById('prediction-results');
    
    if (predictionOutput) {
        predictionOutput.innerHTML = `
            <div class="error-message">
                <h4>Error</h4>
                <p>${message}</p>
                <p>Please check your input data and try again.</p>
            </div>
        `;
    }
    if (predictionResults) {
        predictionResults.style.display = 'block';
    }
}

// Get CSRF token for Django
function getCSRFToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
}

// Initialize when the predictions section is shown
document.addEventListener('DOMContentLoaded', function() {
    // Also listen for section change events
    document.addEventListener('sectionChanged', function(e) {
        if (e.detail.sectionId === 'predictions') {
            initPredictions();
        }
    });
    
    // Initialize if predictions section is already active
    if (document.getElementById('predictions') && document.getElementById('predictions').classList.contains('active')) {
        initPredictions();
    }
});