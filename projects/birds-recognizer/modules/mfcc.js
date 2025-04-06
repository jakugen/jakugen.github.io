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
 * Simple MFCC extractor that decodes an audio Blob,
 * frames the signal, applies windowing, computes FFT power spectrum,
 * applies a Mel filter bank and DCT to extract MFCC features.
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