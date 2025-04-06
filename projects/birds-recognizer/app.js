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
    
    predict(audioBlob).then((predictedClass) => {
        console.log('Predicted class:', predictedClass);
    }).catch(err => {
        console.error('Error during prediction:', err);
    });
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
 * Prediction: Extract features from the recording and load a saved model for prediction.
 */
async function predict(audioBlob) {
  const mfccFeatures = await featureExtraction.simpleExtractMfccFeatures(audioBlob);
  const inputTensor = tf.tensor2d([mfccFeatures], [1, mfccFeatures.length]);
  const model = await tf.loadLayersModel('localstorage://my-mfcc-model');
  const prediction = model.predict(inputTensor);
  const predictionArray = await prediction.array();

  const probabilities = predictionArray[0];
  const maxValue = Math.max(...probabilities);
  const maxIndex = probabilities.indexOf(maxValue);
  
//   console.log('Prediction:', predictionArray);
  return maxIndex;
}

// Wire the "Train Model" button.
document.getElementById('trainModelButton').addEventListener('click', async () => {
  try {
    await modelTraining.trainModel();
    alert('Training complete!');
  } catch (err) {
    console.error('Error during training:', err);
    alert('Error during training. See console for details.');
  }
});

document.getElementById('predictFileInput').addEventListener('change', async (event) => {
    const fileList = event.target.files;
    if(fileList && fileList.length > 0) {
      // Use the first selected file as the audio blob.
      const audioBlob = fileList[0];
      // Call predict with the selected audio file.
        let predictedClass = await predict(audioBlob);
        console.log('Predicted class:', predictedClass);
    }
  });