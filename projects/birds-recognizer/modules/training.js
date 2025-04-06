import { simpleExtractMfccFeatures } from './mfcc.js';

// Global: Number of classes in the dataset.
const numClasses = 2;

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
 * Processes files for each class and concatenates them into training tensors.
 * Expects file inputs with ids: 'class0Input', 'class1Input', 'class2Input'
 */
export async function prepareDataset() {
  let features = [];
  let labels = [];

  const class0Files = document.getElementById('class0Input').files;
  const class1Files = document.getElementById('class1Input').files;

  async function processFiles(fileList, labelIndex) {
    const result = await loadWavFilesAndExtractMFCC(fileList, labelIndex);
    features = features.concat(result.features);
    labels = labels.concat(result.labels);
  }

  await processFiles(class0Files, 0);
  await processFiles(class1Files, 1);

  const xs = tf.tensor2d(features); // shape: [numSamples, 13]
  const ys = tf.tensor2d(labels);
  return { xs, ys };
}

/**
 * Creates, compiles, and trains a robust sequential model.
 */
export async function trainModel() {
    const { xs, ys } = await prepareDataset();
  
    // Build a more reliable model with additional regularization.
    const model = tf.sequential();
    
    // Input layer with BatchNormalization.
    model.add(tf.layers.batchNormalization({inputShape: [13]}));
    
    // First dense layer.
    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.batchNormalization());
    
    // Second dense layer.
    model.add(tf.layers.dense({
      units: 32,
      activation: 'relu'
    }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.batchNormalization());
    
    // Output layer.
    model.add(tf.layers.dense({
      units: numClasses,
      activation: 'softmax'
    }));
    
    // Compile the model with a lower learning rate.
    model.compile({
      optimizer: tf.train.adam(0.001),
      //optimizer: tf.train.sgd(0.01),
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
    await model.save('localstorage://my-mfcc-model');
  }