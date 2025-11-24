// Crop & Fertilizer Recommendation JavaScript - Enhanced version with detailed insights

console.log('Crop & Fertilizer Recommendation module loaded');

// API base URL (in a real app, this would be your Django backend)
const API_BASE_URL = 'http://127.0.0.1:8000/api';

// Make the functions globally accessible
window.getRecommendations = async function() {
    console.log('Get Recommendations function called');
    
    // Get DOM elements
    const recommendationResults = document.getElementById('recommendation-results');
    const recommendationOutput = document.getElementById('recommendation-output');
    
    // Get form values
    const soilNitrogen = document.getElementById('soil-nitrogen').value;
    const soilPhosphorus = document.getElementById('soil-phosphorus').value;
    const soilPotassium = document.getElementById('soil-potassium').value;
    const soilPh = document.getElementById('soil-ph').value;
    const temperature = document.getElementById('temperature').value;
    const humidity = document.getElementById('humidity').value;
    const rainfall = document.getElementById('rainfall').value;
    const location = document.getElementById('location').value;
    const season = document.getElementById('season').value;
    
    console.log('Form values:', {
        soilNitrogen, soilPhosphorus, soilPotassium, soilPh,
        temperature, humidity, rainfall, location, season
    });
    
    // Validate required fields
    if (!soilNitrogen || !soilPhosphorus || !soilPotassium || !soilPh ||
        !temperature || !humidity || !rainfall || !location || !season) {
        alert('Please fill in all fields');
        return;
    }
    
    // Show loading state
    showRecommendationLoading();
    
    // Create data object for API request
    const requestData = {
        soil_nitrogen: parseFloat(soilNitrogen),
        soil_phosphorus: parseFloat(soilPhosphorus),
        soil_potassium: parseFloat(soilPotassium),
        soil_ph: parseFloat(soilPh),
        temperature: parseFloat(temperature),
        humidity: parseFloat(humidity),
        rainfall: parseFloat(rainfall),
        location: location,
        season: season
    };
    
    console.log('Request data:', requestData);
    
    try {
        // Send request to backend
        console.log('Sending request to:', `${API_BASE_URL}/recommendations/combined/`);
        const response = await fetch(`${API_BASE_URL}/recommendations/combined/`, {
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
            window.displayRecommendations(data);
        } else {
            showRecommendationError(data.error || 'Unknown error occurred');
            if (data.missing_fields) {
                showRecommendationError(`Missing fields: ${data.missing_fields.join(', ')}`);
            }
        }
    } catch (error) {
        console.error('Error getting recommendations:', error);
        showRecommendationError(`Error getting recommendations: ${error.message}. Please try again.`);
    }
};

// Make displayRecommendations globally accessible
window.displayRecommendations = function(data) {
    const recommendationOutput = document.getElementById('recommendation-output');
    const recommendationResults = document.getElementById('recommendation-results');
    
    if (!data.recommendations || data.recommendations.length === 0) {
        recommendationOutput.innerHTML = '<p>No recommendations found. Please check your input values and try again.</p>';
        return;
    }
    
    // Create enhanced results HTML with tabbed interface
    let html = `
        <div class="prediction-result">
            <h4>Crop & Fertilizer Recommendations</h4>
    `;
    
    // Add top recommendation highlight
    const topRec = data.recommendations[0];
    const confidencePercent = (topRec.confidence * 100).toFixed(1);
    
    html += `
            <div class="result-grid">
                <div class="result-item">
                    <span class="result-label">Top Recommendation:</span>
                    <span class="result-value">${topRec.crop}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Confidence:</span>
                    <span class="result-value">${confidencePercent}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Recommended Fertilizer:</span>
                    <span class="result-value">${topRec.fertilizer}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Quantity:</span>
                    <span class="result-value">${topRec.quantity_kg_per_ha.toFixed(1)} kg/ha</span>
                </div>
            </div>
    `;
    
    // Add enhanced insights section with tabbed interface
    html += `
            <div class="insights-section">
                <h5>AI-Powered Recommendations Insights</h5>
                <div class="insights-tabs">
                    <button class="tab-button active" onclick="switchRecommendationTab('recommendations-overview')">Overview</button>
                    <button class="tab-button" onclick="switchRecommendationTab('recommendations-details')">Detailed Recommendations</button>
                    <button class="tab-button" onclick="switchRecommendationTab('recommendations-advice')">Expert Advice</button>
                </div>
                <div class="tab-content">
                    <div id="recommendations-overview" class="tab-pane active">
                        <div class="insight-card">
                            <h6>Recommendation Summary</h6>
                            <p>Based on your soil conditions and environmental factors, ${topRec.crop} is the top recommended crop with ${confidencePercent}% confidence.</p>
                            <div class="risk-assessment">
                                <span>Recommendation Strength: ${getConfidenceLevel(confidencePercent)}</span>
                                <span>Confidence: ${confidencePercent}%</span>
                            </div>
                        </div>
                        <div class="insight-card">
                            <h6>Key Environmental Factors</h6>
                            <ul>
                                <li>Soil nutrients: ${getNutrientStatus(topRec)}</li>
                                <li>Temperature conditions: ${getTemperatureStatus()}</li>
                                <li>Moisture levels: ${getMoistureStatus()}</li>
                            </ul>
                        </div>
                    </div>
                    <div id="recommendations-details" class="tab-pane">
    `;
    
    // Add detailed recommendations for all crops with dynamic confidence bars
    data.recommendations.forEach((rec, index) => {
        const recConfidence = (rec.confidence * 100).toFixed(1);
        const confidenceWidth = Math.min(recConfidence, 100); // Cap at 100%
        
        html += `
                        <div class="insight-card">
                            <h6>${index + 1}. ${rec.crop}</h6>
                            <div class="confidence-bar">
                                <div class="confidence-level" style="width: ${confidenceWidth}%; background-color: ${getConfidenceColor(recConfidence)}"></div>
                            </div>
                            <p>Confidence: ${recConfidence}%</p>
                            <p><strong>Recommended Fertilizer:</strong> ${rec.fertilizer}</p>
                            <p><strong>Quantity:</strong> ${rec.quantity_kg_per_ha.toFixed(1)} kg/ha</p>
                            ${rec.advice ? `<p><strong>Advice:</strong> ${formatRecommendationContent(rec.advice).substring(0, 100)}${rec.advice.length > 100 ? '...' : ''}</p>` : ''}
                        </div>
        `;
    });
    
    html += `
                    </div>
                    <div id="recommendations-advice" class="tab-pane">
                        <div class="insight-card">
                            <h6>Expert Recommendations</h6>
    `;
    
    // Add AI suggestions if available
    if (data.ai_suggestions && data.ai_enabled) {
        html += `
                            <div class="ai-suggestions">
                                <h6>🤖 AI-Powered Expert Suggestions</h6>
                                <div class="recommendations-content">
                                    ${formatRecommendationContent(data.ai_suggestions)}
                                </div>
                            </div>
        `;
    } else if (topRec.advice) {
        html += `
                            <div class="recommendations-content">
                                <p>${formatRecommendationContent(topRec.advice)}</p>
                            </div>
        `;
    } else {
        html += `
                            <p>Based on the analysis of your soil conditions and environmental factors, here are some general recommendations:</p>
                            <ul>
                                <li>Maintain soil pH between 6.0-7.0 for optimal nutrient uptake</li>
                                <li>Ensure adequate drainage to prevent waterlogging</li>
                                <li>Consider crop rotation to maintain soil health</li>
                                <li>Apply fertilizers based on soil test results</li>
                            </ul>
        `;
    }
    
    html += `
                        </div>
                    </div>
                </div>
            </div>
    `;
    
    // Add input data section with dynamic analysis
    html += `
            <div class="input-data-section">
                <h5>Input Data Analysis</h5>
                <div class="data-grid">
                    ${generateDynamicDataAnalysis()}
                </div>
            </div>
        </div>
    `;
    
    recommendationOutput.innerHTML = html;
    // Ensure the results section is visible
    recommendationResults.style.display = 'block';
    
    // Scroll to results
    recommendationResults.scrollIntoView({ behavior: 'smooth' });
};

// Initialize the recommendations feature when the section is shown
function initRecommendations() {
    console.log('Initializing Crop & Fertilizer Recommendations feature');
    
    // Get DOM elements for recommendations
    const recommendationForm = document.getElementById('recommendation-form');
    const recommendationResults = document.getElementById('recommendation-results');
    const recommendationOutput = document.getElementById('recommendation-output');
    
    // Set up event listeners for recommendations
    if (recommendationForm && !recommendationForm.dataset.initialized) {
        recommendationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            e.stopPropagation(); // Prevent event from bubbling up
            window.getRecommendations();
            return false; // Prevent default form submission
        });
        recommendationForm.dataset.initialized = 'true';
        console.log('Recommendation form initialized');
    }
}

// Show loading state for recommendations
function showRecommendationLoading() {
    const recommendationOutput = document.getElementById('recommendation-output');
    const recommendationResults = document.getElementById('recommendation-results');
    
    if (recommendationOutput) {
        recommendationOutput.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyzing soil conditions and environmental factors...</p>
                <p>Generating crop and fertilizer recommendations...</p>
            </div>
        `;
    }
    if (recommendationResults) {
        recommendationResults.style.display = 'block';
    }
}

// Show error message for recommendations
function showRecommendationError(message) {
    const recommendationOutput = document.getElementById('recommendation-output');
    const recommendationResults = document.getElementById('recommendation-results');
    
    if (recommendationOutput) {
        recommendationOutput.innerHTML = `
            <div class="error-message">
                <h4>Error</h4>
                <p>${message}</p>
            </div>
        `;
    }
    if (recommendationResults) {
        recommendationResults.style.display = 'block';
    }
}

// Get CSRF token for Django
function getCSRFToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
}

// Switch between tabs in the recommendations section
window.switchRecommendationTab = function(tabName) {
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

// Format recommendation content for better display
function formatRecommendationContent(content) {
    if (!content) return '';
    return content
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/<br><br>/g, '</p><p>')
        .replace(/^<br>/, '')
        .replace(/^(.*)$/, '<p>$1</p>');
}

// Get nutrient status based on levels
function getNutrientStatus(rec) {
    const n = rec.input_data?.Nitrogen || rec.soil_nitrogen || 0;
    const p = rec.input_data?.Phosphorus || rec.soil_phosphorus || 0;
    const k = rec.input_data?.Potassium || rec.soil_potassium || 0;
    
    let status = [];
    if (n < 50) status.push("Low N");
    else if (n > 150) status.push("High N");
    else status.push("Adequate N");
    
    if (p < 20) status.push("Low P");
    else if (p > 80) status.push("High P");
    else status.push("Adequate P");
    
    if (k < 50) status.push("Low K");
    else if (k > 200) status.push("High K");
    else status.push("Adequate K");
    
    return status.join(", ");
}

// Get temperature status
function getTemperatureStatus() {
    const temp = parseFloat(document.getElementById('temperature').value);
    if (temp < 10) return "Cool climate";
    if (temp > 30) return "Warm climate";
    return "Moderate climate";
}

// Get moisture status
function getMoistureStatus() {
    const humidity = parseFloat(document.getElementById('humidity').value);
    const rainfall = parseFloat(document.getElementById('rainfall').value);
    
    let status = [];
    if (humidity < 40) status.push("Low humidity");
    else if (humidity > 70) status.push("High humidity");
    else status.push("Moderate humidity");
    
    if (rainfall < 50) status.push("Low rainfall");
    else if (rainfall > 150) status.push("High rainfall");
    else status.push("Moderate rainfall");
    
    return status.join(", ");
}

// Generate dynamic data analysis based on input values
function generateDynamicDataAnalysis() {
    const nitrogen = parseFloat(document.getElementById('soil-nitrogen').value);
    const phosphorus = parseFloat(document.getElementById('soil-phosphorus').value);
    const potassium = parseFloat(document.getElementById('soil-potassium').value);
    const ph = parseFloat(document.getElementById('soil-ph').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const rainfall = parseFloat(document.getElementById('rainfall').value);
    const location = document.getElementById('location').value;
    const season = document.getElementById('season').value;
    
    let html = '';
    
    // Add soil analysis
    html += `<div class="data-item"><strong>Soil Analysis:</strong> ${getSoilAnalysis(nitrogen, phosphorus, potassium, ph)}</div>`;
    
    // Add climate analysis
    html += `<div class="data-item"><strong>Climate Suitability:</strong> ${getClimateAnalysis(temperature, humidity, rainfall)}</div>`;
    
    // Add location and season info
    html += `<div class="data-item"><strong>Location:</strong> ${location}</div>`;
    html += `<div class="data-item"><strong>Season:</strong> ${season}</div>`;
    
    return html;
}

// Get soil analysis based on nutrient levels
function getSoilAnalysis(nitrogen, phosphorus, potassium, ph) {
    let analysis = [];
    
    // Nitrogen analysis
    if (nitrogen < 50) {
        analysis.push("Low N");
    } else if (nitrogen > 150) {
        analysis.push("High N");
    } else {
        analysis.push("Adequate N");
    }
    
    // Phosphorus analysis
    if (phosphorus < 20) {
        analysis.push("Low P");
    } else if (phosphorus > 80) {
        analysis.push("High P");
    } else {
        analysis.push("Adequate P");
    }
    
    // Potassium analysis
    if (potassium < 50) {
        analysis.push("Low K");
    } else if (potassium > 200) {
        analysis.push("High K");
    } else {
        analysis.push("Adequate K");
    }
    
    // pH analysis
    if (ph < 6.0) {
        analysis.push("Acidic pH");
    } else if (ph > 7.5) {
        analysis.push("Alkaline pH");
    } else {
        analysis.push("Optimal pH");
    }
    
    return analysis.join(", ");
}

// Get climate analysis based on weather conditions
function getClimateAnalysis(temperature, humidity, rainfall) {
    let analysis = [];
    
    // Temperature analysis
    if (temperature < 10) {
        analysis.push("Cool climate");
    } else if (temperature > 30) {
        analysis.push("Warm climate");
    } else {
        analysis.push("Moderate climate");
    }
    
    // Humidity analysis
    if (humidity < 40) {
        analysis.push("Low humidity");
    } else if (humidity > 70) {
        analysis.push("High humidity");
    } else {
        analysis.push("Moderate humidity");
    }
    
    // Rainfall analysis
    if (rainfall < 50) {
        analysis.push("Low rainfall");
    } else if (rainfall > 150) {
        analysis.push("High rainfall");
    } else {
        analysis.push("Moderate rainfall");
    }
    
    return analysis.join(", ");
}

// Get confidence level description
function getConfidenceLevel(confidencePercent) {
    const confidence = parseFloat(confidencePercent);
    if (confidence >= 90) return "Very Strong";
    if (confidence >= 75) return "Strong";
    if (confidence >= 60) return "Moderate";
    if (confidence >= 40) return "Weak";
    return "Very Weak";
}

// Get confidence color based on percentage
function getConfidenceColor(confidencePercent) {
    const confidence = parseFloat(confidencePercent);
    if (confidence >= 80) return "#4caf50"; // Green
    if (confidence >= 60) return "#ff9800"; // Orange
    return "#f44336"; // Red
}

// Initialize when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initRecommendations);
} else {
    initRecommendations();
}

function initRecommendations() {
    console.log('Recommendations module initialized');
}