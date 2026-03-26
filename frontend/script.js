/**
 * Deepfake AI Detection System - Frontend JavaScript
 * Handles API communication and UI interactions
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';
let currentImageFile = null;
let currentHeatmapBase64 = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeDragAndDrop();
    checkBackendStatus();
    updateCharCount();
});

/**
 * Backend Health Check
 */
async function checkBackendStatus() {
    const statusElement = document.getElementById('backend-status-text');
    const statusBanner = document.getElementById('model-status');
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            const data = await response.json();
            statusElement.textContent = 'Online';
            statusElement.className = 'text-green-600 font-medium';
            statusBanner.classList.remove('hidden');
            statusBanner.className = 'mb-6 bg-green-50 border border-green-200 rounded-lg p-4 flex items-center justify-between';
        } else {
            throw new Error('Backend not responding');
        }
    } catch (error) {
        statusElement.textContent = 'Offline - Please start the backend server';
        statusElement.className = 'text-red-600 font-medium';
        statusBanner.classList.remove('hidden');
        statusBanner.className = 'mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center justify-between';
        showToast('error', 'Cannot connect to backend server. Please ensure it\'s running.');
    }
}

/**
 * Image Upload & Drag-Drop Handlers
 */
function initializeDragAndDrop() {
    const dropZone = document.getElementById('image-drop-zone');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-active'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-active'), false);
    });
    
    dropZone.addEventListener('drop', handleDrop, false);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleImageFile(files[0]);
    } else {
        showToast('error', 'Please upload an image file');
    }
}

function handleImageUpload(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleImageFile(files[0]);
    }
}

function handleImageFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showToast('error', 'Please upload a valid image file');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showToast('error', 'Image size must be less than 10MB');
        return;
    }
    
    currentImageFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('image-preview').src = e.target.result;
        document.getElementById('image-preview-container').classList.remove('hidden');
        document.getElementById('image-drop-zone').classList.add('hidden');
        
        // Enable detect button
        const btn = document.getElementById('detect-image-btn');
        btn.disabled = false;
        btn.className = 'w-full mt-4 bg-primary hover:bg-indigo-700 text-white py-3 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg';
        
        // Hide previous results
        hideImageResult();
    };
    reader.readAsDataURL(file);
    
    showToast('success', 'Image uploaded successfully!');
}

function removeImage() {
    currentImageFile = null;
    currentHeatmapBase64 = null;
    
    document.getElementById('image-input').value = '';
    document.getElementById('image-preview-container').classList.add('hidden');
    document.getElementById('image-drop-zone').classList.remove('hidden');
    document.getElementById('heatmap-container').classList.add('hidden');
    
    // Disable detect button
    const btn = document.getElementById('detect-image-btn');
    btn.disabled = true;
    btn.className = 'w-full mt-4 bg-gray-300 text-gray-500 py-3 rounded-lg font-semibold cursor-not-allowed';
    
    hideImageResult();
}

function hideImageResult() {
    document.getElementById('image-result').classList.add('hidden');
    document.getElementById('heatmap-container').classList.add('hidden');
}

/**
 * Image Detection
 */
async function detectImage() {
    if (!currentImageFile) {
        showToast('error', 'Please upload an image first');
        return;
    }
    
    const loadingDiv = document.getElementById('image-loading');
    const resultDiv = document.getElementById('image-result');
    const detectBtn = document.getElementById('detect-image-btn');
    
    // Show loading state
    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    detectBtn.disabled = true;
    detectBtn.className = 'w-full mt-4 bg-gray-300 text-gray-500 py-3 rounded-lg font-semibold cursor-not-allowed';
    
    try {
        const startTime = performance.now();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', currentImageFile);
        
        // Call API with heatmap option
        const response = await fetch(`${API_BASE_URL}/detect-image-with-heatmap`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Detection failed');
        }
        
        const data = await response.json();
        const endTime = performance.now();
        
        // Display result
        displayImageResult(data, (endTime - startTime) / 1000);
        
        // Store heatmap
        if (data.heatmap_base64) {
            currentHeatmapBase64 = data.heatmap_base64;
            document.getElementById('heatmap-image').src = currentHeatmapBase64;
            document.getElementById('heatmap-container').classList.remove('hidden');
        }
        
        showToast('success', `Analysis completed in ${((endTime - startTime)/1000).toFixed(2)}s`);
        
    } catch (error) {
        console.error('Error:', error);
        showToast('error', `Detection failed: ${error.message}`);
    } finally {
        // Hide loading state
        loadingDiv.classList.add('hidden');
        
        // Re-enable button if image still exists
        if (currentImageFile) {
            detectBtn.disabled = false;
            detectBtn.className = 'w-full mt-4 bg-primary hover:bg-indigo-700 text-white py-3 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg';
        }
    }
}

function displayImageResult(data, analysisTime) {
    const predictionEl = document.getElementById('image-prediction');
    const confidenceEl = document.getElementById('image-confidence');
    const confidenceBar = document.getElementById('image-confidence-bar');
    const timestampEl = document.getElementById('image-timestamp');
    const resultDiv = document.getElementById('image-result');
    
    // Set prediction with color
    predictionEl.textContent = data.prediction;
    predictionEl.style.color = data.color;
    
    // Set confidence
    confidenceEl.textContent = data.confidence_percentage;
    
    // Update confidence bar
    confidenceBar.style.width = `${data.confidence}%`;
    confidenceBar.style.backgroundColor = data.color;
    
    // Set timestamp
    const now = new Date();
    timestampEl.textContent = `${now.toLocaleDateString()} ${now.toLocaleTimeString()} (${analysisTime.toFixed(2)}s)`;
    
    // Show result
    resultDiv.classList.remove('hidden');
}

function toggleHeatmap() {
    const heatmapDisplay = document.getElementById('heatmap-display');
    const checkbox = document.getElementById('heatmap-toggle');
    
    if (checkbox.checked) {
        heatmapDisplay.classList.remove('hidden');
    } else {
        heatmapDisplay.classList.add('hidden');
    }
}

/**
 * Text Detection
 */
function updateCharCount() {
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    const detectBtn = document.getElementById('detect-text-btn');
    
    const length = textInput.value.length;
    charCount.textContent = `${length} character${length !== 1 ? 's' : ''}`;
    
    // Enable/disable button based on text length
    if (length >= 10 && length <= 5000) {
        detectBtn.disabled = false;
        detectBtn.className = 'w-full bg-primary hover:bg-indigo-700 text-white py-3 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg';
    } else {
        detectBtn.disabled = true;
        detectBtn.className = 'w-full bg-gray-300 text-gray-500 py-3 rounded-lg font-semibold cursor-not-allowed';
        
        if (length > 5000) {
            charCount.className = 'text-xs text-red-500 font-medium';
        } else {
            charCount.className = 'text-xs text-gray-500';
        }
    }
}

function fillSampleText(type) {
    const textInput = document.getElementById('text-input');
    
    if (type === 'human') {
        textInput.value = `The morning sun cast long shadows across the quiet street as I made my way to the corner café. 
There was something comforting about the familiar routine - the smell of freshly ground coffee beans, 
the gentle hum of conversation, the warmth of the ceramic mug in my hands. Mrs. Henderson, the owner, 
always remembered my order without asking. These small human connections, I've come to realize, are 
what make life worth living in this bustling city.`;
    } else if (type === 'ai') {
        textInput.value = `Artificial intelligence has revolutionized numerous industries in the twenty-first century. 
Machine learning algorithms process vast amounts of data to identify patterns and make predictions. 
These systems utilize neural networks with multiple layers to achieve increasingly sophisticated tasks. 
The applications range from healthcare diagnostics to autonomous vehicles, demonstrating the versatility 
of this transformative technology. Researchers continue to develop more advanced models with improved 
accuracy and efficiency.`;
    }
    
    updateCharCount();
    showToast('success', `Sample ${type} text loaded`);
}

async function detectText() {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();
    
    if (!text || text.length < 10) {
        showToast('error', 'Please enter at least 10 characters');
        return;
    }
    
    if (text.length > 5000) {
        showToast('error', 'Text exceeds maximum length of 5000 characters');
        return;
    }
    
    const loadingDiv = document.getElementById('text-loading');
    const resultDiv = document.getElementById('text-result');
    const detectBtn = document.getElementById('detect-text-btn');
    
    // Show loading state
    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    detectBtn.disabled = true;
    
    try {
        const startTime = performance.now();
        
        // Call API
        const response = await fetch(`${API_BASE_URL}/detect-text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Detection failed');
        }
        
        const data = await response.json();
        const endTime = performance.now();
        
        // Display result
        displayTextResult(data, (endTime - startTime) / 1000);
        
        showToast('success', `Analysis completed in ${((endTime - startTime)/1000).toFixed(2)}s`);
        
    } catch (error) {
        console.error('Error:', error);
        showToast('error', `Detection failed: ${error.message}`);
    } finally {
        // Hide loading state
        loadingDiv.classList.add('hidden');
        
        // Re-enable button
        detectBtn.disabled = false;
        detectBtn.className = 'w-full bg-primary hover:bg-indigo-700 text-white py-3 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg';
    }
}

function displayTextResult(data, analysisTime) {
    const predictionEl = document.getElementById('text-prediction');
    const confidenceEl = document.getElementById('text-confidence');
    const confidenceBar = document.getElementById('text-confidence-bar');
    const timestampEl = document.getElementById('text-timestamp');
    const resultDiv = document.getElementById('text-result');
    
    // Set prediction with color
    predictionEl.textContent = data.prediction;
    predictionEl.style.color = data.color;
    
    // Set confidence
    confidenceEl.textContent = data.confidence_percentage;
    
    // Update confidence bar
    confidenceBar.style.width = `${data.confidence}%`;
    confidenceBar.style.backgroundColor = data.color;
    
    // Set timestamp
    const now = new Date();
    timestampEl.textContent = `${now.toLocaleDateString()} ${now.toLocaleTimeString()} (${analysisTime.toFixed(2)}s)`;
    
    // Show result
    resultDiv.classList.remove('hidden');
}

/**
 * Toast Notification System
 */
function showToast(type, message) {
    const toast = document.getElementById('toast');
    const toastIcon = document.getElementById('toast-icon');
    const toastMessage = document.getElementById('toast-message');
    
    // Set icon and color based on type
    if (type === 'success') {
        toastIcon.className = 'fas fa-check-circle text-green-400';
        toast.className = 'fixed bottom-4 right-4 transform transition-all duration-300 bg-green-600 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-3';
    } else if (type === 'error') {
        toastIcon.className = 'fas fa-exclamation-circle text-red-400';
        toast.className = 'fixed bottom-4 right-4 transform transition-all duration-300 bg-red-600 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-3';
    } else {
        toastIcon.className = 'fas fa-info-circle text-blue-400';
        toast.className = 'fixed bottom-4 right-4 transform transition-all duration-300 bg-gray-800 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-3';
    }
    
    toastMessage.textContent = message;
    
    // Show toast
    toast.classList.remove('hidden');
    toast.classList.add('translate-y-0', 'opacity-100');
    
    // Auto-hide after 4 seconds
    setTimeout(() => {
        toast.classList.add('translate-y-2', 'opacity-0');
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 300);
    }, 4000);
}

/**
 * Utility Functions
 */
function getConfidenceColor(confidence) {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
}

// Auto-refresh backend status every 30 seconds
setInterval(() => {
    checkBackendStatus();
}, 30000);
