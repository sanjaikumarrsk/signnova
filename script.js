// --- Configuration ---
const SERVER_URL = 'http://127.0.0.1:5000/classify_gesture';
const INTERVAL_MS = 50; // Optimized prediction rate (20 FPS)
const STABILITY_COUNT = 6; // CRITICAL: Set to 40 for a 2-second stable hold (40 * 50ms = 2000ms)
const LERP_FACTOR = 0.2; 

// --- DOM Elements ---
const video = document.getElementById('webcam');
const canvasElement = document.getElementById('landmark-canvas');
const currentLetterLeft = document.getElementById('current-letter'); 
const currentLetterRight = document.getElementById('current-alphabet-right'); 
const currentWordDisplay = document.getElementById('current-word');
const currentSentenceDisplay = document.getElementById('current-sentence');

// Buttons
const resetBtn = document.getElementById('reset-btn');
const nextWordBtn = document.getElementById('next-word-btn'); 
const speakBtn = document.getElementById('speak-btn');
const modeToggleBtn = document.getElementById('mode-toggle');

const ttsStatus = document.getElementById('tts-status');

// --- State Variables ---
let drawingContext = null; 
let internalCanvas = null; 
let predictionTimeout = null;
let lastPrediction = '';
let stabilityCounter = 0;

let currentWord = '';
let currentSentence = '';
let isPaused = false; 
let lastLandmarks = []; 
let interpolatedLandmarks = []; 
let isDrawing = false;  
let isProcessingCooldown = false; // Prevents new letters from being added too quickly

const synth = window.speechSynthesis;

// --- Drawing Configuration (DYNAMIC SCALING) ---
const BASE_LINE_RATIO = 0.01; // 1% of the hand size
const BASE_DOT_RATIO = 0.015; // 1.5% of the hand size

const FINGER_COLORS = {
    'palm': 'rgb(200, 200, 200)', 'thumb': 'rgb(255, 80, 80)',    
    'index': 'rgb(80, 255, 80)', 'middle': 'rgb(255, 255, 80)', 
    'ring': 'rgb(80, 255, 255)', 'pinky': 'rgb(255, 80, 255)'     
};
const LANDMARK_COLOR_MAP = [
    'palm', 'thumb', 'thumb', 'thumb', 'thumb', 
    'index', 'index', 'index', 'index', 
    'middle', 'middle', 'middle', 'middle', 
    'ring', 'ring', 'ring', 'ring', 
    'pinky', 'pinky', 'pinky', 'pinky'
];
const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], 
    [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17] 
];


// =========================================================
// 1. INITIALIZATION: Start Webcam
// =========================================================
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.addEventListener('loadeddata', () => {
                canvasElement.width = video.videoWidth;
                canvasElement.height = video.videoHeight;
                drawingContext = canvasElement.getContext('2d');
                
                internalCanvas = document.createElement('canvas');
                internalCanvas.width = video.videoWidth;
                internalCanvas.height = video.videoHeight;

                startPredictionLoop();
                
                if (!isDrawing) {
                    drawLoop(); 
                    isDrawing = true;
                }
            });
        })
        .catch(err => {
            console.error("Error accessing webcam:", err);
            currentLetterLeft.textContent = "Error: Webcam access denied.";
        });

    // --- Event Listeners ---
    resetBtn.addEventListener('click', resetSentence);
    nextWordBtn.addEventListener('click', processNextWord); 
    speakBtn.addEventListener('click', () => speakText(currentSentence));
    modeToggleBtn.addEventListener('click', toggleDarkMode);
    
    updateDisplayOutputs();
}

// =========================================================
// 2. CORE LOGIC: Send Frame to Server
// =========================================================
function sendFrameToServer() {
    if (isPaused || !internalCanvas || video.paused || video.ended) {
        startPredictionLoop();
        return;
    }

    const context = internalCanvas.getContext('2d');
    context.drawImage(video, 0, 0, internalCanvas.width, internalCanvas.height); 

    internalCanvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');

        fetch(SERVER_URL, {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json()) 
        .then(data => {
            const predictedSign = data.gesture || 'N/A';
            const newLandmarks = data.landmarks || []; 

            lastLandmarks = newLandmarks; 

            handlePredictionResult(predictedSign);
            startPredictionLoop();
        })
        .catch(error => {
            console.error('Server error:', error);
            startPredictionLoop();
        });
    }, 'image/jpeg', 0.4); // Optimized: Reduced Quality (0.4) for faster transmission
}


// =========================================================
// 3. CONTINUOUS DRAWING LOOP (LERP FOR SMOOTHNESS)
// =========================================================
function drawLoop() {
    drawingContext.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (lastLandmarks.length > 0) {
        if (interpolatedLandmarks.length === 0) {
            interpolatedLandmarks = JSON.parse(JSON.stringify(lastLandmarks));
        }

        if (interpolatedLandmarks.length === lastLandmarks.length) {
            for (let i = 0; i < lastLandmarks.length; i++) {
                const target = lastLandmarks[i];
                const current = interpolatedLandmarks[i];

                // Linear Interpolation (LERP)
                current.x += (target.x - current.x) * LERP_FACTOR;
                current.y += (target.y - current.y) * LERP_FACTOR;
                current.z += (target.z - current.z) * LERP_FACTOR;
            }
        }
        
        drawLandmarks(interpolatedLandmarks); 
    }
    
    window.requestAnimationFrame(drawLoop);
}

// =========================================================
// 4. LANDMARK DRAWING LOGIC (DYNAMICALLY SCALED)
// =========================================================
function drawLandmarks(landmarks) {
    const width = canvasElement.width;
    const height = canvasElement.height;
    
    // --- 1. Calculate Dynamic Scale Factor ---
    if (landmarks.length < 6) return; // Ensure wrist (0) and index base (5) exist

    const wrist = landmarks[0];
    const indexBase = landmarks[5];
    
    // Calculate normalized distance between wrist and index finger base
    const handSizeNorm = Math.sqrt(
        Math.pow(wrist.x - indexBase.x, 2) + 
        Math.pow(wrist.y - indexBase.y, 2)
    );
    
    // Use the canvas width for a stable pixel reference
    const scaleFactor = handSizeNorm * width; 
    
    // Calculate actual pixel size for drawing
    const dynamicLineThickness = Math.max(1, scaleFactor * BASE_LINE_RATIO);
    const dynamicDotRadius = Math.max(2, scaleFactor * BASE_DOT_RATIO); // Min size 2 to stay visible


    // --- 2. Draw connections (lines) ---
    CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const startLm = landmarks[startIdx];
        const endLm = landmarks[endIdx];
        
        if (startLm && endLm) {
            const colorGroup = LANDMARK_COLOR_MAP[endIdx];
            const color = FINGER_COLORS[colorGroup] || FINGER_COLORS['palm'];
            
            drawingContext.strokeStyle = color;
            drawingContext.lineWidth = dynamicLineThickness; 
            
            drawingContext.beginPath();
            drawingContext.moveTo(startLm.x * width, startLm.y * height);
            drawingContext.lineTo(endLm.x * width, endLm.y * height);
            drawingContext.stroke();
        }
    });

    // --- 3. Draw landmarks (dots) ---
    landmarks.forEach((lm, index) => {
        const colorGroup = LANDMARK_COLOR_MAP[index];
        const color = FINGER_COLORS[colorGroup] || FINGER_COLORS['palm'];
        
        // Dot Fill
        drawingContext.fillStyle = color; 
        drawingContext.beginPath();
        drawingContext.arc(lm.x * width, lm.y * height, dynamicDotRadius, 0, 2 * Math.PI); 
        drawingContext.fill();
        
        // Dot Outline
        drawingContext.strokeStyle = 'white';
        drawingContext.lineWidth = 1;
        drawingContext.stroke();
    });
}


// =========================================================
// 5. RESULT HANDLING (2-SECOND STABILITY LOGIC)
// =========================================================
function handlePredictionResult(newSign) {
    currentLetterLeft.textContent = newSign;
    currentLetterRight.textContent = newSign;
    
    if (newSign === 'No Hand Detected' || newSign === 'N/A' || newSign.startsWith('ERROR')) {
        lastPrediction = '';
        stabilityCounter = 0;
        return;
    }

    if (newSign === lastPrediction) {
        stabilityCounter++;
    } else {
        lastPrediction = newSign;
        stabilityCounter = 1;
    }

    // Check if stability threshold (2 seconds) is met
    if (stabilityCounter === STABILITY_COUNT) {
        processStableSign(newSign);
    }
}

function processStableSign(stableSign) {
    // 1. If we are in the cooldown, ignore the stable sign.
    if (isProcessingCooldown) { 
        return; 
    }
    
    // Only process letters (length 1)
    if (stableSign.length === 1) { 
        currentWord += stableSign;
        
        // 2. Start the Cooldown period (500ms)
        isProcessingCooldown = true;
        setTimeout(() => {
            isProcessingCooldown = false;
        }, 500); 
    }
    
    updateDisplayOutputs();
    setTimeout(() => speakText(stableSign), 100); 
}


// =========================================================
// 6. CONTROL LOGIC (Next Word, Reset, UI)
// =========================================================

function processNextWord() {
    if (currentWord.length > 0) {
        currentSentence += currentWord + ' ';
        currentWord = '';
        updateDisplayOutputs();
        speakText('Space'); 
    }
}

function resetSentence() {
    currentWord = '';
    currentSentence = '';
    
    lastLandmarks = [];
    interpolatedLandmarks = [];
    drawingContext.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    updateDisplayOutputs();
    speakText('Sentence reset.');
}

function updateDisplayOutputs() {
    currentWordDisplay.textContent = currentWord || 'N/A';
    currentSentenceDisplay.textContent = currentSentence.trimEnd() || 'N/A';
    
    const displayedLetter = lastPrediction || 'N/A';
    currentLetterLeft.textContent = displayedLetter;
    currentLetterRight.textContent = displayedLetter;

    ttsStatus.textContent = '🔊 System Status: Ready';
}

function toggleDarkMode() {
    const body = document.body;
    body.classList.toggle('light-mode');
    
    const isLight = body.classList.contains('light-mode');
    modeToggleBtn.innerHTML = isLight 
        ? '<span role="img" aria-label="mode">🌙</span> Dark Mode'
        : '<span role="img" aria-label="mode">💡</span> Light Mode';
}

function speakText(text) {
    if (!synth.speaking) {
        const utterance = new SpeechSynthesisUtterance(text);
        
        utterance.onstart = () => {
            ttsStatus.textContent = '🎤 Speaking: "${text}"';
        };
        utterance.onend = () => {
            ttsStatus.textContent = '🔊 System Status: Ready';
        };

        synth.speak(utterance);
    }
}

function startPredictionLoop() {
    if (predictionTimeout) {
        clearTimeout(predictionTimeout);
    }
    predictionTimeout = setTimeout(sendFrameToServer, INTERVAL_MS);
}


// --- RUN THE APPLICATION ---
document.addEventListener('DOMContentLoaded', startWebcam);
