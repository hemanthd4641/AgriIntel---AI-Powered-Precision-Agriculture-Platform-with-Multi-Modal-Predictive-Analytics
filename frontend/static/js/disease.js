// Plant Disease Detection JavaScript - Enhanced version with detailed insights
// All functionality moved to main.js to avoid conflicts

console.log('Plant Disease Detection module loaded - functionality handled by main.js');

// API base URL (in a real app, this would be your Django backend)
const API_BASE_URL = 'http://127.0.0.1:8000/api';

// Initialize the disease detection feature when the section is shown
function initDiseaseDetection() {
    console.log('Initializing Plant Disease Detection feature');
    // All functionality is handled by main.js
}

// All other functions have been moved to main.js to avoid conflicts
                        <ul>
                            <li>Leaf discoloration or spotting</li>
                            <li>Wilting or stunted growth</li>
                            <li>Lesions on stems or fruits</li>
                            <li>Powdery or fuzzy growth on plant surfaces</li>
                        </ul>
                    </div>
                </div>
                <div id="disease-prevention" class="tab-pane">
                    <div class="insight-card">
                        <h6>Prevention Strategies</h6>
                        <ul>
                            <li>Practice crop rotation to reduce pathogen buildup</li>
                            <li>Ensure proper spacing between plants for air circulation</li>
                            <li>Avoid overhead irrigation to minimize leaf wetness</li>
                            <li>Remove and destroy infected plant material</li>
                            <li>Use disease-resistant varieties when available</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    html += `
        </div>
    `;
    
    if (diseaseOutput) {
        diseaseOutput.innerHTML = html;
    }
    // Ensure the results section is visible
    if (diseaseResults) {
        diseaseResults.style.display = 'block';
    }
}

// Get treatment advice for a disease with enhanced presentation
async function getTreatmentAdvice(diseaseName) {
    // Get DOM elements
    const adviceSection = document.getElementById('disease-advice');
    const adviceOutput = document.getElementById('advice-output');
    
    if (adviceSection && adviceOutput) {
        adviceSection.style.display = 'block';
        adviceOutput.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Generating treatment recommendations...</p>
                <p>Consulting knowledge base for enhanced insights...</p>
            </div>
        `;
        
        try {
            const response = await fetch(`${API_BASE_URL}/disease/advice/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    disease_name: diseaseName,
                    severity: 'moderate'
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Format the advice with enhanced presentation
                let html = `
                    <div class="advice-content">
                        <h4>AI-Powered Treatment Recommendations</h4>
                        <div class="insights-tabs">
                            <button class="tab-button active" onclick="switchDiseaseTab('treatment-overview')">Overview</button>
                            <button class="tab-button" onclick="switchDiseaseTab('treatment-options')">Treatment Options</button>
                            <button class="tab-button" onclick="switchDiseaseTab('treatment-prevention')">Prevention</button>
                        </div>
                        <div class="tab-content">
                            <div id="treatment-overview" class="tab-pane active">
                                <div class="insight-card">
                                    <h6>Treatment Summary</h6>
                                    <p>${formatAdviceContent(data.advice)}</p>
                                </div>
                `;
                
                // Add related documents if available
                if (data.related_documents && data.related_documents.length > 0) {
                    html += `
                                <div class="insight-card">
                                    <h6>Related Information</h6>
                                    <ul>
                    `;
                    data.related_documents.forEach(doc => {
                        html += `<li>${doc.title} (Relevance: ${(doc.similarity * 100).toFixed(1)}%)</li>`;
                    });
                    html += `
                                    </ul>
                                </div>
                    `;
                }
                
                html += `
                            </div>
                            <div id="treatment-options" class="tab-pane">
                                <div class="insight-card">
                                    <h6>Chemical Treatment Options</h6>
                                    <ul>
                                        <li>Apply appropriate fungicides as recommended for ${diseaseName}</li>
                                        <li>Follow label instructions for application rates and timing</li>
                                        <li>Rotate fungicide classes to prevent resistance development</li>
                                    </ul>
                                </div>
                                <div class="insight-card">
                                    <h6>Cultural Treatment Methods</h6>
                                    <ul>
                                        <li>Remove and destroy infected plant parts immediately</li>
                                        <li>Improve air circulation through proper plant spacing</li>
                                        <li>Adjust watering practices to reduce leaf wetness</li>
                                    </ul>
                                </div>
                            </div>
                            <div id="treatment-prevention" class="tab-pane">
                                <div class="insight-card">
                                    <h6>Long-term Prevention Strategies</h6>
                                    <ul>
                                        <li>Select disease-resistant crop varieties</li>
                                        <li>Implement proper crop rotation schedules</li>
                                        <li>Maintain soil health through organic amendments</li>
                                        <li>Monitor fields regularly for early disease detection</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                adviceOutput.innerHTML = html;
            } else {
                showDiseaseError(`Error getting advice: ${data.error || 'Unknown error occurred'}`);
            }
        } catch (error) {
            showDiseaseError(`Error getting advice: ${error.message}. Please try again.`);
        }
    }
}

// Switch between tabs in the insights section
window.switchDiseaseTab = function(tabName) {
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

// Format advice content for better display
function formatAdviceContent(content) {
    if (!content) return '';
    return content
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/<br><br>/g, '</p><p>')
        .replace(/^<br>/, '')
        .replace(/<br>$/, '');
}

// Get severity class based on confidence score
function getSeverityClass(confidence) {
    if (confidence >= 0.8) return 'high-risk';
    if (confidence >= 0.6) return 'moderate-risk';
    return 'low-risk';
}

// Get severity level description
function getSeverityLevel(confidence) {
    if (confidence >= 0.8) return 'High Risk';
    if (confidence >= 0.6) return 'Moderate Risk';
    return 'Low Risk';
}

// Show error message for disease detection
function showDiseaseError(message) {
    const diseaseOutput = document.getElementById('disease-output');
    const diseaseResults = document.getElementById('disease-results');
    
    if (diseaseOutput) {
        diseaseOutput.innerHTML = `
            <div class="error-message">
                <h4>Error</h4>
                <p>${message}</p>
                <p>Please check your image and try again.</p>
            </div>
        `;
    }
    if (diseaseResults) {
        diseaseResults.style.display = 'block';
    }
}

// Initialize when the disease detection section is shown
document.addEventListener('DOMContentLoaded', function() {
    // Also listen for section change events
    document.addEventListener('sectionChanged', function(e) {
        if (e.detail.sectionId === 'disease-detection') {
            initDiseaseDetection();
        }
    });
    
    // Initialize if disease detection section is already active
    if (document.getElementById('disease-detection') && document.getElementById('disease-detection').classList.contains('active')) {
        initDiseaseDetection();
    }
});