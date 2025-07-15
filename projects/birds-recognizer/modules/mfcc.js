'use strict';

import { fft } from './fft.js';

/**
 * Applies a Hamming window to a signal frame.
 */
export function applyHammingWindow(frame) {
  const N = frame.length;
  let windowed = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    windowed[n] = frame[n] * (0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (N - 1)));
  }
  return windowed;
}

/**
 * Creates a Mel filter bank.
 * Returns an array of NUM_FILTERS arrays of length (NFFT/2 + 1)
 */
export function createMelFilterBank(NUM_FILTERS, NFFT, sampleRate, lowFreq, highFreq) {
  const hzToMel = hz => 1125 * Math.log(1 + hz / 700);
  const melToHz = mel => 700 * (Math.exp(mel / 1125) - 1);

  const lowMel = hzToMel(lowFreq);
  const highMel = hzToMel(highFreq);
  const melPoints = new Array(NUM_FILTERS + 2);
  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = lowMel + (i * (highMel - lowMel)) / (NUM_FILTERS + 1);
  }

  const hzPoints = melPoints.map(mel => melToHz(mel));
  // Convert Hz to nearest FFT bin numbers
  const bin = hzPoints.map(hz => Math.floor((hz / sampleRate) * NFFT));

  let filterBank = [];
  const numBins = Math.floor(NFFT / 2) + 1;
  for (let m = 1; m <= NUM_FILTERS; m++) {
    let filter = new Array(numBins).fill(0);
    for (let k = bin[m - 1]; k < bin[m]; k++) {
      filter[k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1]);
    }
    for (let k = bin[m]; k < bin[m + 1]; k++) {
      filter[k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m]);
    }
    filterBank.push(filter);
  }
  return filterBank;
}

/**
 * Compute a simple Type-II Discrete Cosine Transform.
 * Only the first numCoeffs coefficients are returned.
 */
export function dct(vector, numCoeffs) {
  const N = vector.length;
  let result = new Array(numCoeffs).fill(0);
  for (let k = 0; k < numCoeffs; k++) {
    for (let n = 0; n < N; n++) {
      result[k] += vector[n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
    }
  }
  return result;
}

/**
 * Compute delta (first derivative) features from a sequence of feature vectors.
 * Uses a simple finite difference approximation.
 * @param {Array<Array<number>>} featureFrames - Array of feature vectors over time
 * @param {number} N - Window size for delta computation (default: 2)
 * @returns {Array<Array<number>>} Delta features for each frame
 */
export function computeDeltaFeatures(featureFrames, N = 2) {
  const numFrames = featureFrames.length;
  const numCoeffs = featureFrames[0].length;
  let deltaFeatures = [];

  for (let t = 0; t < numFrames; t++) {
    let delta = new Array(numCoeffs).fill(0);
    
    // Compute delta using finite difference with padding
    for (let i = 0; i < numCoeffs; i++) {
      let numerator = 0;
      let denominator = 0;
      
      for (let n = 1; n <= N; n++) {
        // Handle boundary conditions with padding
        const forwardIdx = Math.min(t + n, numFrames - 1);
        const backwardIdx = Math.max(t - n, 0);
        
        numerator += n * (featureFrames[forwardIdx][i] - featureFrames[backwardIdx][i]);
        denominator += 2 * n * n;
      }
      
      delta[i] = denominator > 0 ? numerator / denominator : 0;
    }
    
    deltaFeatures.push(delta);
  }
  
  return deltaFeatures;
}

/**
 * Apply cepstral mean normalization to MFCC features.
 * Subtracts the mean of each coefficient across all frames.
 * @param {Array<Array<number>>} mfccFrames - Array of MFCC vectors
 * @returns {Array<Array<number>>} Normalized MFCC features
 */
export function applyCepstralMeanNormalization(mfccFrames) {
  if (mfccFrames.length === 0) return mfccFrames;
  
  const numFrames = mfccFrames.length;
  const numCoeffs = mfccFrames[0].length;
  
  // Compute mean for each coefficient
  let means = new Array(numCoeffs).fill(0);
  for (let frame of mfccFrames) {
    for (let i = 0; i < numCoeffs; i++) {
      means[i] += frame[i];
    }
  }
  for (let i = 0; i < numCoeffs; i++) {
    means[i] /= numFrames;
  }
  
  // Subtract mean from each frame
  let normalizedFrames = [];
  for (let frame of mfccFrames) {
    let normalizedFrame = [];
    for (let i = 0; i < numCoeffs; i++) {
      normalizedFrame.push(frame[i] - means[i]);
    }
    normalizedFrames.push(normalizedFrame);
  }
  
  return normalizedFrames;
}

/**
 * Enhanced MFCC extractor with delta features and improved parameters.
 * Extracts 20 MFCC coefficients plus delta and delta-delta features (60 total).
 */
export async function extractEnhancedMfccFeatures(audioBlob) {
  // 1. Decode audio data
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioContext();
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const signal = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;

  // 2. Enhanced parameters
  const FRAME_SIZE = 2048; // Must be a power of 2 for FFT
  const HOP_SIZE = 512;
  const MFCC_COUNT = 20; // Increased from 13 to 20
  const NUM_FILTERS = 40; // Increased filter bank size
  const NFFT = FRAME_SIZE;

  // 3. Build Mel filter bank
  const melFilters = createMelFilterBank(NUM_FILTERS, NFFT, sampleRate, 0, sampleRate / 2);

  // 4. Process signal frame-by-frame to extract static MFCCs
  let mfccFrames = [];
  for (let start = 0; start < signal.length - FRAME_SIZE; start += HOP_SIZE) {
    let frame = signal.slice(start, start + FRAME_SIZE);
    let windowedFrame = applyHammingWindow(frame);
    
    // Compute FFT using our utility
    let fftResult = fft(windowedFrame);
    
    // Compute power spectrum for bins [0, NFFT/2]
    let powerSpectrum = [];
    for (let k = 0; k <= NFFT / 2; k++) {
      let re = fftResult.real[k];
      let im = fftResult.imag[k];
      powerSpectrum[k] = (re * re + im * im) / NFFT;
    }
    
    // Apply Mel filter bank
    let filterEnergies = new Array(NUM_FILTERS).fill(0);
    for (let m = 0; m < NUM_FILTERS; m++) {
      for (let k = 0; k < melFilters[m].length; k++) {
        filterEnergies[m] += powerSpectrum[k] * melFilters[m][k];
      }
      // Log energy to avoid log(0)
      filterEnergies[m] = Math.log(filterEnergies[m] + 1e-8);
    }
    
    // Compute DCT to get MFCCs
    let mfccs = dct(filterEnergies, MFCC_COUNT);
    mfccFrames.push(mfccs);
  }

  // 5. Apply cepstral mean normalization
  let normalizedMfccFrames = applyCepstralMeanNormalization(mfccFrames);

  // 6. Compute delta and delta-delta features
  let deltaFrames = computeDeltaFeatures(normalizedMfccFrames);
  let deltaDeltaFrames = computeDeltaFeatures(deltaFrames);

  // 7. Combine static, delta, and delta-delta features
  let enhancedFrames = [];
  for (let i = 0; i < normalizedMfccFrames.length; i++) {
    let combinedFrame = [
      ...normalizedMfccFrames[i],  // 20 static MFCC
      ...deltaFrames[i],           // 20 delta MFCC
      ...deltaDeltaFrames[i]       // 20 delta-delta MFCC
    ];
    enhancedFrames.push(combinedFrame);
  }

  // 8. Compute mean feature vector across frames (60 dimensions)
  let avgEnhancedMfcc = new Array(60).fill(0);
  enhancedFrames.forEach(frame => {
    for (let j = 0; j < 60; j++) {
      avgEnhancedMfcc[j] += frame[j];
    }
  });
  for (let j = 0; j < 60; j++) {
    avgEnhancedMfcc[j] /= enhancedFrames.length || 1;
  }
  
  console.log('Enhanced MFCC Features (60D):', avgEnhancedMfcc.slice(0, 5), '... (truncated)');
  console.log('Feature breakdown: Static(0-19), Delta(20-39), Delta-Delta(40-59)');
  
  return avgEnhancedMfcc;
}

/**
 * Simple MFCC extractor that decodes an audio Blob,
 * frames the signal, applies windowing, computes FFT power spectrum,
 * applies a Mel filter bank and DCT to extract MFCC features.
 * 
 * @deprecated Use extractEnhancedMfccFeatures for better performance
 */
export async function simpleExtractMfccFeatures(audioBlob) {
  // 1. Decode audio data
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioContext();
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const signal = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;

  // 2. Setup parameters
  const FRAME_SIZE = 2048; // Must be a power of 2 for FFT
  const HOP_SIZE = 512;
  const MFCC_COUNT = 13;
  const NUM_FILTERS = 26;
  const NFFT = FRAME_SIZE;

  // 3. Build Mel filter bank
  const melFilters = createMelFilterBank(NUM_FILTERS, NFFT, sampleRate, 0, sampleRate / 2);

  // 4. Process signal frame‐by‐frame
  let mfccFrames = [];
  for (let start = 0; start < signal.length - FRAME_SIZE; start += HOP_SIZE) {
    let frame = signal.slice(start, start + FRAME_SIZE);
    let windowedFrame = applyHammingWindow(frame);
    
    // Compute FFT using our utility
    let fftResult = fft(windowedFrame);
    
    // Compute power spectrum for bins [0, NFFT/2]
    let powerSpectrum = [];
    for (let k = 0; k <= NFFT / 2; k++) {
      let re = fftResult.real[k];
      let im = fftResult.imag[k];
      powerSpectrum[k] = (re * re + im * im) / NFFT;
    }
    
    // Apply Mel filter bank
    let filterEnergies = new Array(NUM_FILTERS).fill(0);
    for (let m = 0; m < NUM_FILTERS; m++) {
      for (let k = 0; k < melFilters[m].length; k++) {
        filterEnergies[m] += powerSpectrum[k] * melFilters[m][k];
      }
      // Log energy to avoid log(0)
      filterEnergies[m] = Math.log(filterEnergies[m] + 1e-8);
    }
    
    // Compute DCT to get MFCCs
    let mfccs = dct(filterEnergies, MFCC_COUNT);
    mfccFrames.push(mfccs);
  }

  // Optional: compute mean MFCC vector across frames
  let avgMfcc = new Array(MFCC_COUNT).fill(0);
  mfccFrames.forEach(frame => {
    for (let j = 0; j < MFCC_COUNT; j++) {
      avgMfcc[j] += frame[j];
    }
  });
  for (let j = 0; j < MFCC_COUNT; j++) {
    avgMfcc[j] /= mfccFrames.length || 1;
  }
  
  console.log('Average MFCC Features:', avgMfcc);
  return avgMfcc;
}