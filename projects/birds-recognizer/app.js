import * as featureExtraction from './modules/mfcc.js';
import * as modelTraining from './modules/training.js';

console.log('Birds Recognizer Loaded.');

// Global variables for recording.
let mediaRecorder;
let audioChunks = [];
let audioStream;

document.getElementById('micPermission').addEventListener('click', async () => {
  const statusElement = document.getElementById('micStatus');
  try {
    statusElement.textContent = "Requesting microphone access...";
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    statusElement.textContent = "Microphone access granted!";
    document.getElementById('recordingControls').style.display = 'block';
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(audioStream);
  } catch (err) {
    statusElement.textContent = `Error: ${err.message}`;
    console.error('Error accessing microphone:', err);
  }
});

// Start recording.
document.getElementById('startRecording').addEventListener('click', () => {
  audioChunks = [];
  mediaRecorder = new MediaRecorder(audioStream);
  mediaRecorder.addEventListener('dataavailable', event => {
    audioChunks.push(event.data);
  });
  mediaRecorder.addEventListener('stop', () => {
    document.getElementById('recordingStatus').textContent = 'Recording finished!';
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    document.getElementById('audioPlayback').src = audioUrl;
    document.getElementById('downloadLink').href = audioUrl;
    document.getElementById('audioPlayer').style.display = 'block';
    
    classifyAudioBlob(audioBlob);
  });
  mediaRecorder.start();
  document.getElementById('startRecording').disabled = true;
  document.getElementById('stopRecording').disabled = false;
  document.getElementById('recordingStatus').textContent = 'Recording...';
});

// Stop recording.
document.getElementById('stopRecording').addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    document.getElementById('startRecording').disabled = false;
    document.getElementById('stopRecording').disabled = true;
  }
});

/**
 * Prediction: Extract enhanced features from the recording and load a saved model for prediction.
 */
async function predict(audioBlob, model) {
  // Use enhanced MFCC features (60-dimensional)
  const mfccFeatures = await featureExtraction.extractEnhancedMfccFeatures(audioBlob);
  console.log(`Extracted ${mfccFeatures.length}D enhanced features for prediction`);
  
  const inputTensor = tf.tensor2d([mfccFeatures], [1, mfccFeatures.length]);
  const prediction = model.predict(inputTensor);
  const predictionArray = await prediction.array();

  const probabilities = predictionArray[0];
  const maxValue = Math.max(...probabilities);
  const maxIndex = probabilities.indexOf(maxValue);
  
  // Log prediction confidence for debugging
  console.log(`Prediction confidence: ${(maxValue * 100).toFixed(2)}%`);
  console.log(`Top 3 predictions:`, probabilities
    .map((prob, idx) => ({ class: idx, confidence: prob }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3));
  
//   console.log('Prediction:', predictionArray);
  return maxIndex;
}

// Wire the "Train Model" button.
document.getElementById('trainModelButton').addEventListener('click', async () => {
  try {
    // trainModel now returns an object
    const { model, classMap } = await modelTraining.trainModel();

    // --- Save the classMap as a downloadable file ---
    const classMapJson = JSON.stringify(Array.from(classMap.entries()));
    downloadTextFile('class-map.json', classMapJson); // Trigger download

    // Save the model itself (e.g., to files)
    await saveModelToFile(model); // Your existing function

    alert('Training complete! Model saved and class map stored.');
  } catch (err) {
    console.error('Error during training:', err);
    alert(`Error during training: ${err.message}. See console for details.`);
  }
});

document.getElementById('predictFileInput').addEventListener('change', async (event) => {
    const fileList = event.target.files;
    if(fileList && fileList.length > 0) {
      const audioBlob = fileList[0];
      await classifyAudioBlob(audioBlob);
    }
  });

async function classifyAudioBlob(audioBlob) {
  const model = await loadModelFromFile(
    './model/mfcc-model.json');

  // Call predict with the selected audio file.
  let predictedClass = await predict(audioBlob, model);

  const classMapResponse = await fetch('./model/class-map.json');
  if (!classMapResponse.ok) {
    throw new Error(`HTTP error loading class map! Status: ${classMapResponse.status}`);
  }
  const classMap = await classMapResponse.json(); // Expects [[name, index], ...]

  let predictedClassName = 'Unknown';
  // Convert the predicted class index to the corresponding class name.
  for (const entry of classMap) {
    if (entry[1] === predictedClass) {
      predictedClassName = entry[0]; // Get the name of the class
      break;
    }
  }

  console.log('Predicted class:', predictedClassName);
  const statusElement = document.getElementById('modelStatus');
  statusElement.textContent = 'Predicted class: ' + predictedClass;

  displayBirdImage(predictedClassName);
}

async function saveModelToLocalStorage(model) {
  console.log('Saving model to local storage...');
  await model.save('localstorage://mfcc-model');
  console.log('Model saved.');
}

async function loadModelFromLocalStorage() {
  console.log('Loading model from local storage...');
  const model = await tf.loadLayersModel('localstorage://mfcc-model');
  console.log('Model loaded.');

  return model;
}

async function saveModelToFile(model) {
  console.log('Saving model to files...');
  await model.save('downloads://mfcc-model');
  console.log('Model file download initiated.');
}

async function loadModelFromFile(modelUrl) {
  console.log('Loading model from files...');
  const statusElement = document.getElementById('modelStatus');
  statusElement.textContent = 'Loading model...';
  try {
    const model = await tf.loadLayersModel(modelUrl);
    console.log('Model loaded successfully from hosted files.');
    statusElement.textContent = 'Model loaded successfully!';
    return model;
  } catch (error) {
    console.error('Error loading model from files:', error);
    statusElement.textContent = `Error loading model: ${error.message}`;
    return null;
  }
}

/**
 * Triggers a browser download for the given text content as a file.
 * @param {string} filename - The desired name for the downloaded file.
 * @param {string} text - The text content to save in the file.
 */
function downloadTextFile(filename, text) {
  const element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
  console.log(`Download initiated for ${filename}`);
}

/**
 * Displays an image of the predicted bird.
 * @param {string} birdClassName - The class name of the bird to display
 */
function displayBirdImage(birdClassName) {
  let container = document.getElementById('birdImageContainer');
  if (container) {
    // Apply centering styles to container
    container.style.display = 'flex';
    container.style.justifyContent = 'center';
    container.style.alignItems = 'center';
    container.style.margin = '20px auto';
    container.style.width = '100%';
  }

  // Get or create the image element
  let birdImage = document.getElementById('birdImage');
  if (!birdImage) {
    birdImage = document.createElement('img');
    birdImage.id = 'birdImage';
    birdImage.style.width = '250px';
    birdImage.style.height = '250px';
    birdImage.style.objectFit = 'cover';
    birdImage.style.border = '1px solid #ccc';
    birdImage.style.borderRadius = '4px';
    birdImage.alt = 'Predicted bird';
    
    container.appendChild(birdImage);
  }
  
  // Set the image source based on the class name
  const imagePath = `./birds/${birdClassName.toLowerCase()}.png`;
  
  // Set the src and handle loading errors
  birdImage.src = imagePath;
  birdImage.onerror = function() {
    console.error(`Bird image not found: ${imagePath}`);
    birdImage.src = './birds/unknown.png';
    birdImage.alt = 'Bird image not available';
  };
  
  birdImage.style.display = 'block';
}