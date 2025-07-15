import { simpleExtractMfccFeatures, extractEnhancedMfccFeatures } from './mfcc.js';

/**
 * Loads WAV files from a FileList, extracts MFCC features for each file,
 * and creates one-hot encoded labels based on class names derived from filenames.
 * Assumes filenames like 'classname_number.wav' (e.g., 'arctic_tern_1.wav').
 */
async function loadFilesAndExtractFeatures(fileList) {
  let features = [];
  let labelIndices = []; // Store numerical index for each sample first
  const classMap = new Map(); // Maps class name (string) to label index (number)
  let numClasses = 0;

  // Regex to extract class name from filename (e.g., "arctic_tern" from "arctic_tern_1.wav")
  const filenameRegex = /^([a-zA-Z_]+)_\d+\.wav$/;

  for (const file of fileList) {
    const match = file.name.match(filenameRegex);
    if (!match || !match[1]) {
      console.warn(`Skipping file with unexpected name format: ${file.name}`);
      continue; // Skip files that don't match the expected format
    }
    const className = match[1];

    let labelIndex;
    if (!classMap.has(className)) {
      // Assign a new index to a newly encountered class
      labelIndex = numClasses;
      classMap.set(className, labelIndex);
      numClasses++; // Increment the total count of unique classes
    } else {
      labelIndex = classMap.get(className);
    }

    try {
      // Use enhanced MFCC features (60-dimensional: 20 static + 20 delta + 20 delta-delta)
      const mfcc = await extractEnhancedMfccFeatures(file);
      if (mfcc && mfcc.length === 60) { // Validate 60-dimensional feature vector
          features.push(mfcc);
          labelIndices.push(labelIndex);
          console.log(`Processed ${file.name}: extracted ${mfcc.length}D features`);
      } else {
          console.warn(`Skipping file ${file.name} due to invalid MFCC output. Expected 60D, got ${mfcc ? mfcc.length : 'null'}`);
      }
    } catch (error) {
        console.error(`Error processing file ${file.name}:`, error);
        // Optionally skip this file or handle the error differently
    }
  }

  console.log(`Found ${numClasses} classes:`, Array.from(classMap.keys()));
  console.log('Class to index mapping:', classMap);

  if (features.length === 0) {
      throw new Error("No valid features extracted. Check file formats and content.");
  }

  return { features, labelIndices, numClasses, classMap };
}

/**
 * Loads WAV files from a FileList, extracts MFCC features for each file,
 * and creates one-hot encoded labels.
 */
export async function loadWavFilesAndExtractMFCC(fileList, labelIndex) {
  let features = [];
  let labels = [];
  for (const file of fileList) {
    // simpleExtractMfccFeatures returns a 13-element vector.
    const mfcc = await simpleExtractMfccFeatures(file);
    features.push(mfcc);
    let oneHot = new Array(numClasses).fill(0);
    oneHot[labelIndex] = 1;
    labels.push(oneHot);
  }
  return { features, labels };
}

/**
 * Processes files from a single file input, extracts features and labels,
 * and prepares tensors for training.
 * Expects a file input with id: 'allFilesInput'
 */
export async function prepareDataset() {
  const allFilesInput = document.getElementById('allFilesInput'); // Use single input ID
  if (!allFilesInput || !allFilesInput.files) {
      throw new Error("File input with id 'allFilesInput' not found or has no files.");
  }
  const fileList = allFilesInput.files;

  if (fileList.length === 0) {
      throw new Error("No files selected in 'allFilesInput'.");
  }

  const { features, labelIndices, numClasses, classMap } = await loadFilesAndExtractFeatures(fileList);

  if (numClasses < 2) {
      throw new Error(`Detected only ${numClasses} class(es). Need at least 2 for training.`);
  }

  // Convert features array to a 2D tensor
  const xs = tf.tensor2d(features); // shape: [numSamples, 60] - enhanced MFCC features

  // Convert numerical label indices to a 1D tensor
  const labelIndicesTensor = tf.tensor1d(labelIndices, 'int32');

  // Create one-hot encoded labels
  const ys = tf.oneHot(labelIndicesTensor, numClasses);

  // Clean up intermediate tensor
  labelIndicesTensor.dispose();

  // Return tensors and the number of classes found
  return { xs, ys, numClasses, classMap };
}

/**
 * Creates, compiles, and trains a robust sequential model.
 * The number of output units is determined dynamically from the dataset.
 */
export async function trainModel() {
    // Get data and the dynamically determined number of classes
    const { xs, ys, numClasses, classMap } = await prepareDataset();

    // Store classMap for later use during prediction (e.g., in localStorage or globally)
    // Example: localStorage.setItem('classMap', JSON.stringify(Array.from(classMap.entries())));
    // Or return it along with the model if needed elsewhere immediately.

    console.log(`Training model for ${numClasses} classes.`);

    // Build a more reliable model with additional regularization.
    const model = tf.sequential();

    // Input layer with BatchNormalization for 60-dimensional enhanced MFCC features
    model.add(tf.layers.batchNormalization({inputShape: [60]}));

    // First dense layer.
    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));
    // ... rest of the model definition remains the same ...
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.batchNormalization());

    // Second dense layer.
    model.add(tf.layers.dense({
      units: 32,
      activation: 'relu'
    }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.batchNormalization());

    // Output layer - uses the dynamically determined numClasses
    model.add(tf.layers.dense({
      units: numClasses, // Use dynamic number of classes
      activation: 'softmax'
    }));

    // Compile the model with a lower learning rate.
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    // Train the model with early stopping.
    await model.fit(xs, ys, {
      epochs: 100,
      batchSize: 8,
      validationSplit: 0.2,
      callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 })
    });

    console.log('Training complete!');

    // Dispose tensors to free memory
    xs.dispose();
    ys.dispose();

    // Return the trained model and potentially the classMap for prediction mapping
    return { model, classMap };
  }